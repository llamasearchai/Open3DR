"""
Medical WebSocket Router - Open3DReconstruction Medical AI Platform

This module provides real-time medical communication endpoints including
live patient monitoring, real-time AI analysis streaming, medical alerts,
and collaborative medical communication via WebSocket connections.
"""

from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
from datetime import datetime
import uuid

from loguru import logger
import numpy as np

from ...websocket import MedicalWebSocketManager, RealTimeAnalyzer, LiveMonitoringService
from ...security import HIPAACompliance, MedicalAudit
from ...utils import patient_data_anonymizer


# Router setup
router = APIRouter()
security = HTTPBearer()

# Medical WebSocket components
websocket_manager = MedicalWebSocketManager()
realtime_analyzer = RealTimeAnalyzer()
live_monitoring = LiveMonitoringService()
hipaa_compliance = HIPAACompliance()
medical_audit = MedicalAudit()


# Connection manager for tracking active medical connections
class MedicalConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.patient_connections: Dict[str, List[WebSocket]] = {}
        self.user_connections: Dict[str, WebSocket] = {}
        self.room_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str, connection_type: str = "general"):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.user_connections[user_id] = websocket
        
        # Log WebSocket connection for HIPAA audit
        await medical_audit.log_websocket_connection(
            user_id=user_id,
            connection_type=connection_type,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Medical WebSocket connected: {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections.remove(websocket)
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        
        # Remove from patient and room connections
        for patient_id, connections in self.patient_connections.items():
            if websocket in connections:
                connections.remove(websocket)
        
        for room_id, connections in self.room_connections.items():
            if websocket in connections:
                connections.remove(websocket)
        
        logger.info(f"Medical WebSocket disconnected: {user_id}")

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            await websocket.send_text(message)

    async def send_patient_update(self, message: str, patient_id: str):
        if patient_id in self.patient_connections:
            for websocket in self.patient_connections[patient_id]:
                await websocket.send_text(message)

    async def send_room_broadcast(self, message: str, room_id: str):
        if room_id in self.room_connections:
            for websocket in self.room_connections[room_id]:
                await websocket.send_text(message)

    async def broadcast_alert(self, alert_data: dict):
        message = json.dumps({
            "type": "medical_alert",
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        for websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except:
                # Handle disconnected clients
                pass

    def subscribe_to_patient(self, websocket: WebSocket, patient_id: str):
        if patient_id not in self.patient_connections:
            self.patient_connections[patient_id] = []
        self.patient_connections[patient_id].append(websocket)

    def subscribe_to_room(self, websocket: WebSocket, room_id: str):
        if room_id not in self.room_connections:
            self.room_connections[room_id] = []
        self.room_connections[room_id].append(websocket)


# Global connection manager
connection_manager = MedicalConnectionManager()


# Authentication for WebSocket
async def verify_websocket_token(token: str) -> dict:
    """Verify WebSocket authentication token."""
    if not token or len(token) < 20:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Simplified token validation
    return {"user_id": f"medical_user_{hash(token) % 10000}", "permissions": ["websocket", "monitoring"]}


@router.websocket("/live-monitoring/{patient_id}")
async def live_patient_monitoring(
    websocket: WebSocket,
    patient_id: str,
    token: str = Query(..., description="Authentication token")
):
    """
    Real-time patient monitoring WebSocket connection.
    
    Provides continuous streaming of:
    - Live vital signs data
    - Real-time AI analysis
    - Instant medical alerts
    - Trending analysis updates
    - Risk assessment changes
    """
    
    try:
        # Verify authentication
        user_data = await verify_websocket_token(token)
        user_id = user_data["user_id"]
        
        # Establish connection
        await connection_manager.connect(websocket, user_id, "patient_monitoring")
        connection_manager.subscribe_to_patient(websocket, patient_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "patient_id": patient_id,
            "user_id": user_id,
            "monitoring_active": True,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Start live monitoring service
        monitoring_task = asyncio.create_task(
            live_monitoring.start_patient_monitoring(patient_id, websocket)
        )
        
        try:
            while True:
                # Listen for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message["type"] == "update_thresholds":
                    await handle_threshold_update(patient_id, message["data"], websocket)
                elif message["type"] == "request_analysis":
                    await handle_analysis_request(patient_id, message["data"], websocket)
                elif message["type"] == "acknowledge_alert":
                    await handle_alert_acknowledgment(message["alert_id"], user_id, websocket)
                elif message["type"] == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()}))
                    
        except WebSocketDisconnect:
            # Cancel monitoring task
            monitoring_task.cancel()
            connection_manager.disconnect(websocket, user_id)
            
            # Log disconnection
            await medical_audit.log_websocket_disconnection(
                user_id=user_id,
                patient_id=patient_id,
                duration=(datetime.utcnow() - datetime.utcnow()).total_seconds(),
                timestamp=datetime.utcnow()
            )
            
    except Exception as e:
        logger.error(f"Live monitoring WebSocket error: {str(e)}")
        await websocket.close(code=1000)


@router.websocket("/medical-collaboration/{room_id}")
async def medical_collaboration_room(
    websocket: WebSocket,
    room_id: str,
    token: str = Query(..., description="Authentication token"),
    user_role: str = Query("physician", description="User role in collaboration")
):
    """
    Medical collaboration WebSocket for team communication.
    
    Enables real-time collaboration including:
    - Multi-disciplinary team communication
    - Shared medical case discussions
    - Real-time consultation
    - Collaborative diagnosis
    - Treatment plan development
    """
    
    try:
        # Verify authentication
        user_data = await verify_websocket_token(token)
        user_id = user_data["user_id"]
        
        # Establish connection
        await connection_manager.connect(websocket, user_id, "collaboration")
        connection_manager.subscribe_to_room(websocket, room_id)
        
        # Announce user joining the room
        join_message = {
            "type": "user_joined",
            "room_id": room_id,
            "user_id": user_id,
            "user_role": user_role,
            "timestamp": datetime.utcnow().isoformat()
        }
        await connection_manager.send_room_broadcast(json.dumps(join_message), room_id)
        
        try:
            while True:
                # Listen for collaboration messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Add user context to message
                message["sender_id"] = user_id
                message["sender_role"] = user_role
                message["timestamp"] = datetime.utcnow().isoformat()
                
                # Broadcast to room participants
                await connection_manager.send_room_broadcast(json.dumps(message), room_id)
                
                # Log collaboration activity
                await medical_audit.log_collaboration_activity(
                    room_id=room_id,
                    user_id=user_id,
                    message_type=message.get("type", "unknown"),
                    timestamp=datetime.utcnow()
                )
                
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket, user_id)
            
            # Announce user leaving
            leave_message = {
                "type": "user_left",
                "room_id": room_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            await connection_manager.send_room_broadcast(json.dumps(leave_message), room_id)
            
    except Exception as e:
        logger.error(f"Collaboration WebSocket error: {str(e)}")
        await websocket.close(code=1000)


@router.websocket("/medical-alerts")
async def medical_alerts_stream(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
    alert_types: str = Query("all", description="Comma-separated alert types to subscribe to")
):
    """
    Medical alerts WebSocket stream.
    
    Provides real-time medical alerts including:
    - Critical patient alerts
    - System-wide medical alerts
    - Emergency notifications
    - Clinical decision alerts
    - Equipment alerts
    """
    
    try:
        # Verify authentication
        user_data = await verify_websocket_token(token)
        user_id = user_data["user_id"]
        
        # Establish connection
        await connection_manager.connect(websocket, user_id, "alerts")
        
        # Parse alert type preferences
        subscribed_alert_types = alert_types.split(",") if alert_types != "all" else ["all"]
        
        # Send subscription confirmation
        await websocket.send_text(json.dumps({
            "type": "alert_subscription_confirmed",
            "subscribed_types": subscribed_alert_types,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Start alert monitoring
        alert_task = asyncio.create_task(
            stream_medical_alerts(websocket, subscribed_alert_types)
        )
        
        try:
            while True:
                # Listen for client acknowledgments
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "alert_acknowledged":
                    await handle_alert_acknowledgment(
                        message["alert_id"], 
                        user_id, 
                        websocket
                    )
                    
        except WebSocketDisconnect:
            alert_task.cancel()
            connection_manager.disconnect(websocket, user_id)
            
    except Exception as e:
        logger.error(f"Alerts WebSocket error: {str(e)}")
        await websocket.close(code=1000)


@router.websocket("/real-time-analysis")
async def real_time_medical_analysis(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
    analysis_type: str = Query("comprehensive", description="Type of real-time analysis")
):
    """
    Real-time medical AI analysis WebSocket.
    
    Provides streaming AI analysis including:
    - Live diagnostic analysis
    - Real-time image processing
    - Continuous pattern recognition
    - Instant clinical insights
    - Dynamic risk assessment
    """
    
    try:
        # Verify authentication
        user_data = await verify_websocket_token(token)
        user_id = user_data["user_id"]
        
        # Establish connection
        await connection_manager.connect(websocket, user_id, "real_time_analysis")
        
        # Send analysis stream started confirmation
        await websocket.send_text(json.dumps({
            "type": "analysis_stream_started",
            "analysis_type": analysis_type,
            "capabilities": ["diagnostic_ai", "imaging_analysis", "pattern_recognition"],
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        try:
            while True:
                # Listen for analysis requests
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "analyze_data":
                    # Process real-time analysis request
                    analysis_result = await realtime_analyzer.analyze_streaming_data(
                        data=message["data"],
                        analysis_type=analysis_type
                    )
                    
                    # Send analysis result
                    await websocket.send_text(json.dumps({
                        "type": "analysis_result",
                        "request_id": message.get("request_id"),
                        "result": analysis_result,
                        "confidence": analysis_result.get("confidence", 0.0),
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket, user_id)
            
    except Exception as e:
        logger.error(f"Real-time analysis WebSocket error: {str(e)}")
        await websocket.close(code=1000)


# Helper functions for WebSocket message handling
async def handle_threshold_update(patient_id: str, threshold_data: dict, websocket: WebSocket):
    """Handle monitoring threshold updates."""
    try:
        # Update monitoring thresholds
        await live_monitoring.update_patient_thresholds(patient_id, threshold_data)
        
        # Confirm update
        await websocket.send_text(json.dumps({
            "type": "thresholds_updated",
            "patient_id": patient_id,
            "new_thresholds": threshold_data,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to update thresholds: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }))


async def handle_analysis_request(patient_id: str, request_data: dict, websocket: WebSocket):
    """Handle on-demand analysis requests."""
    try:
        # Perform requested analysis
        analysis_result = await realtime_analyzer.analyze_patient_data(
            patient_id=patient_id,
            analysis_type=request_data.get("type", "comprehensive"),
            data=request_data.get("data")
        )
        
        # Send analysis result
        await websocket.send_text(json.dumps({
            "type": "analysis_complete",
            "patient_id": patient_id,
            "analysis_result": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Analysis failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }))


async def handle_alert_acknowledgment(alert_id: str, user_id: str, websocket: WebSocket):
    """Handle medical alert acknowledgments."""
    try:
        # Record alert acknowledgment
        await medical_audit.log_alert_acknowledgment(
            alert_id=alert_id,
            user_id=user_id,
            timestamp=datetime.utcnow()
        )
        
        # Confirm acknowledgment
        await websocket.send_text(json.dumps({
            "type": "alert_acknowledged",
            "alert_id": alert_id,
            "acknowledged_by": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to acknowledge alert: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }))


async def stream_medical_alerts(websocket: WebSocket, alert_types: List[str]):
    """Stream medical alerts to connected clients."""
    while True:
        try:
            # Check for new alerts (would integrate with alert system)
            new_alerts = await get_pending_alerts(alert_types)
            
            for alert in new_alerts:
                await websocket.send_text(json.dumps({
                    "type": "medical_alert",
                    "alert": alert,
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            # Wait before checking again
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Alert streaming error: {str(e)}")
            break


async def get_pending_alerts(alert_types: List[str]) -> List[dict]:
    """Get pending medical alerts (mock implementation)."""
    # Mock alert data - would integrate with real alert system
    return []


# WebSocket endpoint information
@router.get("/websocket-info")
async def get_websocket_info():
    """Get information about available WebSocket endpoints."""
    
    return {
        "websocket_endpoints": {
            "live_monitoring": {
                "path": "/ws/medical/live-monitoring/{patient_id}",
                "description": "Real-time patient monitoring stream",
                "features": ["vital_signs", "ai_analysis", "alerts", "trends"],
                "authentication": "required"
            },
            "collaboration": {
                "path": "/ws/medical/medical-collaboration/{room_id}",
                "description": "Medical team collaboration room",
                "features": ["team_chat", "case_discussion", "consultation"],
                "authentication": "required"
            },
            "alerts": {
                "path": "/ws/medical/medical-alerts",
                "description": "Medical alerts notification stream",
                "features": ["critical_alerts", "system_alerts", "acknowledgments"],
                "authentication": "required"
            },
            "real_time_analysis": {
                "path": "/ws/medical/real-time-analysis",
                "description": "Live AI analysis streaming",
                "features": ["diagnostic_ai", "image_analysis", "pattern_recognition"],
                "authentication": "required"
            }
        },
        "connection_statistics": {
            "active_connections": len(connection_manager.active_connections),
            "patient_monitoring_sessions": len(connection_manager.patient_connections),
            "collaboration_rooms": len(connection_manager.room_connections),
            "total_users_connected": len(connection_manager.user_connections)
        },
        "supported_features": [
            "real_time_monitoring",
            "medical_collaboration", 
            "instant_alerts",
            "ai_analysis_streaming",
            "hipaa_compliant_communication"
        ]
    } 