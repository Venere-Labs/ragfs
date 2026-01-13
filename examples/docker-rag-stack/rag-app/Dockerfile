# RAG App Dockerfile
# Multi-stage build: Rust for ragfs + Python for Chainlit

# =============================================================================
# Stage 1: Build ragfs Python package with maturin
# =============================================================================
FROM rust:1.88-bookworm AS builder

# Install Python, maturin, and protobuf compiler
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    protobuf-compiler \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install maturin
RUN pip3 install --break-system-packages maturin

WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./

# Copy all workspace crates
COPY crates/ragfs ./crates/ragfs
COPY crates/ragfs-core ./crates/ragfs-core
COPY crates/ragfs-fuse ./crates/ragfs-fuse
COPY crates/ragfs-index ./crates/ragfs-index
COPY crates/ragfs-chunker ./crates/ragfs-chunker
COPY crates/ragfs-embed ./crates/ragfs-embed
COPY crates/ragfs-extract ./crates/ragfs-extract
COPY crates/ragfs-store ./crates/ragfs-store
COPY crates/ragfs-query ./crates/ragfs-query
COPY crates/ragfs-python ./crates/ragfs-python

# Build the wheel
WORKDIR /build/crates/ragfs-python
RUN maturin build --release --out /wheels

# =============================================================================
# Stage 2: Runtime Python image
# =============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built wheel from builder
COPY --from=builder /wheels/*.whl /wheels/

# Install ragfs wheel and other dependencies
COPY examples/docker-rag-stack/rag-app/requirements.txt .
RUN pip install --no-cache-dir /wheels/*.whl && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY examples/docker-rag-stack/rag-app/ .

# Create chainlit config directory
RUN mkdir -p .chainlit

# Expose port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Chainlit
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
