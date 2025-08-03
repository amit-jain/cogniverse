# Phoenix Docker image with Cogniverse customizations
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Phoenix and dependencies
RUN pip install --no-cache-dir \
    arize-phoenix>=4.0.0 \
    phoenix-evals>=0.4.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0 \
    opentelemetry-instrumentation>=0.41b0 \
    pandas \
    numpy

# Create data directories
RUN mkdir -p /data/traces /data/datasets /data/experiments /data/evaluations /data/logs

# Copy startup script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose Phoenix port
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:6006/health || exit 1

# Set environment variables
ENV PHOENIX_PORT=6006
ENV PHOENIX_HOST=0.0.0.0
ENV PHOENIX_WORKING_DIR=/data
ENV PHOENIX_ENABLE_PROMETHEUS=true
ENV PHOENIX_ENABLE_CORS=true
ENV PHOENIX_MAX_TRACES=100000

# Volume for data persistence
VOLUME ["/data"]

# Run Phoenix
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]