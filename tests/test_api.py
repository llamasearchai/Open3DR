"""
Comprehensive API tests for Open3D reconstruction platform.

Tests all FastAPI endpoints including reconstruction, simulation, 
agents, and WebSocket functionality.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
import json

from fastapi.testclient import TestClient
from httpx import AsyncClient
import websockets

from open3d.api import create_app
from open3d.core import Config, RenderConfig, Vector3
from . import MOCK_API_KEY, TEST_DATA_DIR, TEST_OUTPUT_DIR


class TestAPI:
    """Test suite for API endpoints."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application."""
        config = Config.load_default()
        config.api.cors_origins = ["*"]
        config.testing = True
        return create_app(config)

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    async def async_client(self, app):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers for tests."""
        return {"X-API-Key": MOCK_API_KEY}

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Open3D Reconstruction API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "docs_url" in data

    def test_docs_endpoint(self, client):
        """Test API documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Open3D Reconstruction API"
        assert "securitySchemes" in schema["components"]


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_health_detailed(self, client):
        """Test detailed health check."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "database" in data
        assert "redis" in data
        assert "gpu" in data
        assert "memory" in data

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200


class TestReconstructionEndpoints:
    """Test reconstruction API endpoints."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def reconstruction_request(self):
        """Sample reconstruction request."""
        return {
            "method": "instant_ngp",
            "config": {
                "resolution": 1024,
                "num_iterations": 5000,
                "learning_rate": 0.01,
            },
            "input_data": {
                "type": "images",
                "path": str(TEST_DATA_DIR / "sample_images"),
            },
            "output_settings": {
                "format": "ply",
                "quality": "high",
            }
        }

    def test_create_nerf_reconstruction(self, client, auth_headers, reconstruction_request):
        """Test creating NeRF reconstruction."""
        response = client.post(
            "/api/v1/reconstruction/nerf",
            json=reconstruction_request,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "reconstruction_id" in data
        assert data["status"] == "created"
        assert data["method"] == "instant_ngp"

    def test_create_gaussian_splatting_reconstruction(self, client, auth_headers):
        """Test creating Gaussian Splatting reconstruction."""
        request_data = {
            "method": "gaussian_splatting",
            "config": {
                "resolution": 1920,
                "num_iterations": 15000,
                "learning_rate": 0.001,
            },
            "input_data": {
                "type": "images",
                "path": str(TEST_DATA_DIR / "sample_images"),
            }
        }
        
        response = client.post(
            "/api/v1/reconstruction/gaussian-splatting",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "reconstruction_id" in data
        assert data["method"] == "gaussian_splatting"

    def test_get_reconstruction_status(self, client, auth_headers):
        """Test getting reconstruction status."""
        # First create a reconstruction
        reconstruction_request = {
            "method": "instant_ngp",
            "input_data": {"type": "images", "path": str(TEST_DATA_DIR)},
        }
        
        create_response = client.post(
            "/api/v1/reconstruction/nerf",
            json=reconstruction_request,
            headers=auth_headers
        )
        reconstruction_id = create_response.json()["reconstruction_id"]
        
        # Get status
        response = client.get(
            f"/api/v1/reconstruction/{reconstruction_id}/status",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["reconstruction_id"] == reconstruction_id
        assert "status" in data
        assert "progress" in data

    def test_download_reconstruction(self, client, auth_headers):
        """Test downloading reconstruction results."""
        # Mock a completed reconstruction
        reconstruction_id = "test-reconstruction-123"
        
        with patch("open3d.api.services.reconstruction_service.get_reconstruction") as mock_get:
            mock_get.return_value = {"status": "completed", "output_path": "test.ply"}
            
            response = client.get(
                f"/api/v1/reconstruction/{reconstruction_id}/download",
                headers=auth_headers
            )
            assert response.status_code == 200

    def test_list_reconstructions(self, client, auth_headers):
        """Test listing user reconstructions."""
        response = client.get(
            "/api/v1/reconstruction/list",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["reconstructions"], list)
        assert "total" in data

    def test_delete_reconstruction(self, client, auth_headers):
        """Test deleting reconstruction."""
        reconstruction_id = "test-reconstruction-123"
        
        response = client.delete(
            f"/api/v1/reconstruction/{reconstruction_id}",
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_unauthorized_access(self, client):
        """Test unauthorized access to reconstruction endpoints."""
        response = client.post(
            "/api/v1/reconstruction/nerf",
            json={"method": "instant_ngp"}
        )
        assert response.status_code == 401


class TestSimulationEndpoints:
    """Test simulation API endpoints."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def simulation_request(self):
        """Sample simulation request."""
        return {
            "scenario": "urban_driving",
            "duration": 60.0,
            "sensors": [
                {
                    "type": "camera",
                    "model": "pinhole",
                    "resolution": [1920, 1080],
                    "position": [0, 0, 1.5],
                    "rotation": [0, 0, 0],
                },
                {
                    "type": "lidar",
                    "model": "velodyne_hdl64e",
                    "position": [0, 0, 2.0],
                    "rotation": [0, 0, 0],
                }
            ],
            "environment": {
                "weather": "clear",
                "time_of_day": "noon",
                "traffic_density": 0.5,
            }
        }

    def test_create_simulation(self, client, auth_headers, simulation_request):
        """Test creating new simulation."""
        response = client.post(
            "/api/v1/simulation/create",
            json=simulation_request,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "simulation_id" in data
        assert data["status"] == "created"
        assert data["scenario"] == "urban_driving"

    def test_run_simulation(self, client, auth_headers):
        """Test running simulation scenario."""
        # First create simulation
        simulation_request = {
            "scenario": "highway_driving",
            "duration": 30.0,
            "sensors": [{"type": "camera"}]
        }
        
        create_response = client.post(
            "/api/v1/simulation/create",
            json=simulation_request,
            headers=auth_headers
        )
        simulation_id = create_response.json()["simulation_id"]
        
        # Run simulation
        response = client.post(
            f"/api/v1/simulation/{simulation_id}/run",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["simulation_id"] == simulation_id
        assert data["status"] in ["running", "started"]

    def test_get_simulation_status(self, client, auth_headers):
        """Test getting simulation status."""
        simulation_id = "test-simulation-123"
        
        response = client.get(
            f"/api/v1/simulation/{simulation_id}/status",
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_get_sensor_data(self, client, auth_headers):
        """Test retrieving sensor data from simulation."""
        simulation_id = "test-simulation-123"
        
        response = client.get(
            f"/api/v1/simulation/{simulation_id}/sensors",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "sensors" in data

    def test_stop_simulation(self, client, auth_headers):
        """Test stopping running simulation."""
        simulation_id = "test-simulation-123"
        
        response = client.post(
            f"/api/v1/simulation/{simulation_id}/stop",
            headers=auth_headers
        )
        assert response.status_code == 200


class TestAgentsEndpoints:
    """Test AI agents API endpoints."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def agent_request(self):
        """Sample agent request."""
        return {
            "agent_type": "reconstruction",
            "command": "Reconstruct a 3D model from these images with high quality settings",
            "context": {
                "input_path": str(TEST_DATA_DIR / "sample_images"),
                "quality": "high",
                "output_format": "ply"
            }
        }

    @patch("open3d.agents.ReconstructionAgent.process_command")
    async def test_process_agent_command(self, mock_process, client, auth_headers, agent_request):
        """Test processing agent command."""
        mock_process.return_value = {
            "status": "success",
            "result": "Reconstruction started with Gaussian Splatting method",
            "reconstruction_id": "test-123"
        }
        
        response = client.post(
            "/api/v1/agents/process",
            json=agent_request,
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "result" in data

    def test_get_agent_status(self, client, auth_headers):
        """Test getting agent status."""
        response = client.get(
            "/api/v1/agents/status",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "agents" in data

    def test_list_available_agents(self, client, auth_headers):
        """Test listing available agent types."""
        response = client.get(
            "/api/v1/agents/list",
            headers=auth_headers
        )
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["agents"], list)
        expected_agents = ["reconstruction", "simulation", "analysis"]
        for agent in expected_agents:
            assert any(a["type"] == agent for a in data["agents"])


class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""

    @pytest.mark.asyncio
    async def test_simulation_websocket(self):
        """Test real-time simulation WebSocket."""
        # This would require a running server, mock for now
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            mock_websocket.recv.return_value = json.dumps({
                "type": "sensor_data",
                "timestamp": 123456789,
                "data": {"camera": "base64_image_data"}
            })
            
            # Simulate WebSocket connection
            async with mock_connect("ws://localhost:8000/ws/simulation/123"):
                pass  # Connection successful

    @pytest.mark.asyncio
    async def test_reconstruction_progress_websocket(self):
        """Test reconstruction progress WebSocket."""
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            mock_websocket.recv.return_value = json.dumps({
                "type": "training_progress",
                "reconstruction_id": "test-123",
                "iteration": 5000,
                "loss": 0.025,
                "psnr": 28.5
            })
            
            async with mock_connect("ws://localhost:8000/ws/reconstruction/123"):
                pass


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    def test_invalid_api_key(self, client):
        """Test invalid API key handling."""
        response = client.post(
            "/api/v1/reconstruction/nerf",
            json={"method": "instant_ngp"},
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401

    def test_missing_api_key(self, client):
        """Test missing API key handling."""
        response = client.post(
            "/api/v1/reconstruction/nerf",
            json={"method": "instant_ngp"}
        )
        assert response.status_code == 401

    def test_invalid_json_payload(self, client, auth_headers):
        """Test invalid JSON payload handling."""
        response = client.post(
            "/api/v1/reconstruction/nerf",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_nonexistent_endpoint(self, client):
        """Test 404 handling for nonexistent endpoints."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client, auth_headers):
        """Test 405 handling for wrong HTTP methods."""
        response = client.put(
            "/api/v1/reconstruction/nerf",
            json={"method": "instant_ngp"},
            headers=auth_headers
        )
        assert response.status_code == 405

    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality."""
        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = client.get("/api/v1/health", headers=auth_headers)
            responses.append(response.status_code)
        
        # Should eventually hit rate limit
        assert any(status == 429 for status in responses)


class TestPerformance:
    """Performance and load tests."""

    @pytest.fixture
    def client(self):
        app = create_app()
        return TestClient(app)

    @pytest.mark.slow
    def test_concurrent_requests(self, client, auth_headers):
        """Test handling concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/api/v1/health", headers=auth_headers)
        
        # Test with 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in results)

    @pytest.mark.slow
    def test_large_payload_handling(self, client, auth_headers):
        """Test handling large request payloads."""
        # Create large but valid payload
        large_request = {
            "method": "instant_ngp",
            "config": {"resolution": 1024},
            "input_data": {
                "type": "images",
                "paths": [f"image_{i}.jpg" for i in range(1000)]  # Large list
            }
        }
        
        response = client.post(
            "/api/v1/reconstruction/nerf",
            json=large_request,
            headers=auth_headers
        )
        
        # Should handle large payload gracefully
        assert response.status_code in [200, 413, 422]  # Success or expected errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 