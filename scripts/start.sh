#!/bin/bash

# Open3D Reconstruction Platform - Startup Script
# This script helps users quickly start the platform in different modes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="production"
GPU_ENABLED=true
MONITORING=false
DEV_MODE=false

# Function to display usage
usage() {
    echo -e "${BLUE}Open3D Reconstruction Platform Startup Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE        Set deployment mode: production, development, testing (default: production)"
    echo "  -g, --gpu              Enable GPU support (default: true)"
    echo "  -n, --no-gpu           Disable GPU support"
    echo "  -M, --monitoring       Enable monitoring stack (Prometheus, Grafana)"
    echo "  -d, --dev              Enable development services (Jupyter, Streamlit)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Start in production mode"
    echo "  $0 --dev               # Start in development mode with dev services"
    echo "  $0 --monitoring        # Start with monitoring enabled"
    echo "  $0 --no-gpu            # Start without GPU support"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ENABLED=true
            shift
            ;;
        -n|--no-gpu)
            GPU_ENABLED=false
            shift
            ;;
        -M|--monitoring)
            MONITORING=true
            shift
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Starting Open3DReconstruction Platform${NC}"
echo -e "Mode: ${YELLOW}$MODE${NC}"
echo -e "GPU Enabled: ${YELLOW}$GPU_ENABLED${NC}"
echo -e "Monitoring: ${YELLOW}$MONITORING${NC}"
echo -e "Development: ${YELLOW}$DEV_MODE${NC}"
echo ""

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}docker-compose is not installed. Please install docker-compose first.${NC}"
    exit 1
fi

# Check for NVIDIA Docker runtime if GPU is enabled
if [ "$GPU_ENABLED" = true ]; then
    if ! docker info | grep -q nvidia; then
        echo -e "${YELLOW}NVIDIA Docker runtime not detected. GPU support may not work.${NC}"
        echo -e "${YELLOW}   Install NVIDIA Container Toolkit for GPU support.${NC}"
    fi
fi

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p data models output logs test-results notebooks

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    cat > .env << EOF
# Open3D Configuration
OPENAI_API_KEY=your-openai-api-key-here
ENVIRONMENT=$MODE

# Database Configuration
POSTGRES_DB=open3d
POSTGRES_USER=open3d
POSTGRES_PASSWORD=open3d_password

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
EOF
    echo -e "${YELLOW}Please edit .env file with your OpenAI API key${NC}"
fi

# Build Docker compose command
COMPOSE_CMD="docker-compose"
COMPOSE_PROFILES=""

# Add profiles based on options
if [ "$DEV_MODE" = true ]; then
    COMPOSE_PROFILES="$COMPOSE_PROFILES --profile dev"
fi

if [ "$MONITORING" = true ]; then
    COMPOSE_PROFILES="$COMPOSE_PROFILES --profile monitoring"
fi

# Start the platform
echo -e "${BLUE}Building Docker images...${NC}"
$COMPOSE_CMD $COMPOSE_PROFILES build

echo -e "${BLUE}Starting services...${NC}"
$COMPOSE_CMD $COMPOSE_PROFILES up -d

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to be ready...${NC}"
sleep 10

# Check if API is responding
if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
    echo -e "${GREEN}API is ready!${NC}"
else
    echo -e "${YELLOW}API is still starting up...${NC}"
fi

# Display service URLs
echo ""
echo -e "${GREEN}Open3DReconstruction Platform is running!${NC}"
echo ""
echo -e "${BLUE}Service URLs:${NC}"
echo -e "   API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "   API Health Check:  ${GREEN}http://localhost:8000/api/v1/health${NC}"

if [ "$DEV_MODE" = true ]; then
    echo -e "   Development API:   ${GREEN}http://localhost:8001${NC}"
    echo -e "   Jupyter Notebook:  ${GREEN}http://localhost:8888${NC}"
    echo -e "   Streamlit Dashboard: ${GREEN}http://localhost:8501${NC}"
fi

if [ "$MONITORING" = true ]; then
    echo -e "   Grafana Dashboard: ${GREEN}http://localhost:3000${NC} (admin/admin)"
    echo -e "   Prometheus:        ${GREEN}http://localhost:9090${NC}"
    echo -e "   Flower (Celery):   ${GREEN}http://localhost:5555${NC}"
fi

echo ""
echo -e "${BLUE}Management Commands:${NC}"
echo -e "   View logs:         ${YELLOW}docker-compose logs -f${NC}"
echo -e "   Stop services:     ${YELLOW}docker-compose down${NC}"
echo -e "   Restart services:  ${YELLOW}docker-compose restart${NC}"
echo -e "   Run tests:         ${YELLOW}docker-compose --profile test run --rm test-runner${NC}"

echo ""
echo -e "${BLUE}Quick Start Examples:${NC}"
echo -e "   1. 3D Reconstruction:"
echo -e "      ${YELLOW}curl -X POST http://localhost:8000/api/v1/reconstruction/nerf \\${NC}"
echo -e "      ${YELLOW}        -H 'Content-Type: application/json' \\${NC}"
echo -e "      ${YELLOW}        -H 'X-API-Key: your-api-key' \\${NC}"
echo -e "      ${YELLOW}        -d '{\"method\": \"instant_ngp\", \"input_data\": {\"type\": \"images\", \"path\": \"/data/images\"}}'${NC}"
echo ""
echo -e "   2. AI Agent Command:"
echo -e "      ${YELLOW}curl -X POST http://localhost:8000/api/v1/agents/process \\${NC}"
echo -e "      ${YELLOW}        -H 'Content-Type: application/json' \\${NC}"
echo -e "      ${YELLOW}        -H 'X-API-Key: your-api-key' \\${NC}"
echo -e "      ${YELLOW}        -d '{\"agent_type\": \"reconstruction\", \"command\": \"Reconstruct 3D model with high quality\"}'${NC}"

echo ""
echo -e "${GREEN}Platform ready for use!${NC}"

# Check if GPU is working (if enabled)
if [ "$GPU_ENABLED" = true ]; then
    echo ""
    echo -e "${BLUE}Checking GPU status...${NC}"
    if docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU support is working${NC}"
    else
        echo -e "${YELLOW}GPU support may not be working properly${NC}"
    fi
fi 