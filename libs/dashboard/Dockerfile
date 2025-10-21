# Cogniverse Dashboard Dockerfile
# Multi-stage build for Streamlit dashboard

FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY pyproject.toml ./
COPY libs/core ./libs/core
COPY libs/dashboard ./libs/dashboard

# Install dependencies with uv
RUN uv pip install --system -e ./libs/core
RUN uv pip install --system -e ./libs/dashboard

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cogniverse && \
    mkdir -p /app /data /logs && \
    chown -R cogniverse:cogniverse /app /data /logs

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=cogniverse:cogniverse libs/core ./libs/core
COPY --chown=cogniverse:cogniverse libs/dashboard ./libs/dashboard
COPY --chown=cogniverse:cogniverse config.yml ./config.yml

# Switch to non-root user
USER cogniverse

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=5)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create Streamlit config
RUN mkdir -p /home/cogniverse/.streamlit && \
    echo '[server]\nport = 8501\naddress = "0.0.0.0"\nheadless = true\n\n[theme]\nbase = "dark"\nprimaryColor = "#FF6B6B"\n' > /home/cogniverse/.streamlit/config.toml

# Run Streamlit dashboard
CMD ["streamlit", "run", "libs/dashboard/cogniverse_dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
