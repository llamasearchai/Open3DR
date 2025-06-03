# Open3D Reconstruction Platform - Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libopencv-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (before other dependencies)
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional CUDA-specific packages
RUN pip install --no-cache-dir \
    cupy-cuda12x \
    tensorrt \
    onnxruntime-gpu

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as production

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgoogle-glog0v5 \
    libgflags2.2 \
    libeigen3-dev \
    libopencv-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user for security
RUN groupadd -r open3d && useradd -r -g open3d -s /bin/bash open3d
RUN mkdir -p /app /data /models /output && chown -R open3d:open3d /app /data /models /output

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=open3d:open3d . /app/

# Install the package in production mode
RUN pip install --no-deps -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/static /app/uploads && \
    chown -R open3d:open3d /app

# Switch to non-root user
USER open3d

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "open3d.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]

# Development stage
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    pre-commit

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to open3d user
USER open3d

# Override command for development
CMD ["python", "-m", "uvicorn", "open3d.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--factory"] 