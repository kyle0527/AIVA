# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl pkg-config libssl-dev rustc cargo git &&     rm -rf /var/lib/apt/lists/*

# Install maturin for PyO3 build
RUN pip install --no-cache-dir maturin==1.6.0

WORKDIR /app

# Copy source
COPY . /app

# Build rust extension and install
WORKDIR /app/services/features/function_crypto/rust_core
RUN maturin build --release --strip -i python3 &&     pip install target/wheels/crypto_engine-*.whl

# Back to app root
WORKDIR /app

# Install python deps (if any project-level requirements exist, adapt accordingly)
RUN pip install --no-cache-dir pytest

ENV PYTHONUNBUFFERED=1
CMD ["python","-m","services.features.function_crypto"]
