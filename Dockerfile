FROM python:3.11-slim

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Copy ML models
COPY yolov8n.pt .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]