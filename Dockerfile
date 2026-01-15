# CIO-Agent FAB++ Evaluator
# Team AgentBusters - AgentBeats Competition

FROM python:3.11-slim

LABEL maintainer="AgentBusters Team"
LABEL description="CIO-Agent FAB++ Dynamic Finance Agent Benchmark"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -s /bin/bash cioagent && \
    chown -R cioagent:cioagent /app
USER cioagent

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app

# Default command
CMD ["cio-agent", "version"]

# Expose port for API (if needed)
EXPOSE 9109
