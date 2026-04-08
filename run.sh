#!/bin/bash
# Production-ready startup script for Network Routing System

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}======================================${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check Python installation
print_header "Checking Python Installation"
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION found"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
print_success "Virtual environment activated"

# Install/upgrade dependencies
print_header "Installing Dependencies"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
print_success "Dependencies installed"

# Create necessary directories
print_info "Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p logs
print_success "Directories ready"

# Choose startup mode
print_header "Select Startup Mode"
echo "1. API Server Only (http://localhost:8000)"
echo "2. Dashboard Only (http://localhost:8501)"
echo "3. Full Stack with Docker (both services)"
echo "4. Development Mode (API + Dashboard)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        print_header "Starting API Server"
        print_info "API will be available at http://localhost:8000"
        print_info "API Documentation at http://localhost:8000/docs"
        python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    2)
        print_header "Starting Dashboard"
        print_info "Dashboard will be available at http://localhost:8501"
        streamlit run app/dashboard/dashboard.py
        ;;
    3)
        print_header "Starting Full Stack with Docker"
        if ! command -v docker &> /dev/null; then
            print_error "Docker not found. Please install Docker."
            exit 1
        fi
        if ! command -v docker-compose &> /dev/null; then
            print_error "Docker Compose not found. Please install Docker Compose."
            exit 1
        fi
        print_info "Building Docker images..."
        docker-compose build
        print_info "Starting services..."
        docker-compose up -d
        sleep 3
        print_success "Services started!"
        print_info "API: http://localhost:8000 (docs: http://localhost:8000/docs)"
        print_info "Dashboard: http://localhost:8501"
        echo ""
        print_info "To view logs: docker-compose logs -f"
        print_info "To stop: docker-compose down"
        ;;
    4)
        print_header "Starting Development Mode"
        print_info "Starting API server in background..."
        python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload &
        API_PID=$!
        sleep 2
        print_success "API started (PID: $API_PID)"
        print_info "Starting Dashboard..."
        streamlit run app/dashboard/dashboard.py
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac
