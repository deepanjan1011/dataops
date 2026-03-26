FROM python:3.11-slim

WORKDIR /app

# Create non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 user

# Install system dependencies (as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (as root, system-wide)
COPY --chown=user dataops_gym/server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (owned by user)
COPY --chown=user dataops_gym/ /app/dataops_gym/

# Switch to non-root user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Generate datasets at build time
RUN python -m dataops_gym.tasks.generate_datasets

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run the server
CMD ["uvicorn", "dataops_gym.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
