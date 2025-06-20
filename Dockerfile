# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  NUMBA_CACHE_DIR=/tmp/numba_cache \
  NUMBA_DISABLE_JIT=0 \
  TRANSFORMERS_CACHE=/tmp/transformers_cache \
  HF_HOME=/tmp/transformers_cache \
  HF_HUB_CACHE=/tmp/transformers_cache \
  PYTHONWARNINGS="ignore::UserWarning"

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  libsndfile1 \
  ffmpeg \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user for security with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/data/reference_voices /app/data/out /tmp/numba_cache /tmp/transformers_cache /home/appuser/.cache/huggingface && \
  chown -R appuser:appuser /app /tmp/numba_cache /tmp/transformers_cache /home/appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Make Docker entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use the Docker-optimized entrypoint
CMD ["./docker-entrypoint.sh"]