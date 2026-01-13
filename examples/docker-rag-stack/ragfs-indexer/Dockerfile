# RAGFS Indexer Dockerfile
# Multi-stage build for the Rust CLI

# Stage 1: Build the Rust binary
FROM rust:1.88-bookworm AS builder

WORKDIR /build

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build release binary
RUN cargo build --release -p ragfs

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary
COPY --from=builder /build/target/release/ragfs /usr/local/bin/ragfs

# Copy entrypoint script
COPY examples/docker-rag-stack/ragfs-indexer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create data directories
RUN mkdir -p /data/index /data/docs

# Set environment
ENV RUST_LOG=info
ENV RAGFS_DB_PATH=/data/index

ENTRYPOINT ["/entrypoint.sh"]
