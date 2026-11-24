# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /workspace

# Install minimal OS dependencies for scikit-learn & xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only for compact image footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application layers
COPY backend ./backend
COPY ml ./ml
COPY data ./data
COPY frontend ./frontend

ENV PYTHONPATH="/workspace/backend:/workspace"
ENV PORT=10000

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:10000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:create_app --factory --app-dir backend --host 0.0.0.0 --port ${PORT:-10000}"]

