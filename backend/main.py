"""
ITMS — Intelligent Traffic Management System
Entry point: uvicorn app:app --host 127.0.0.1 --port 8000

Install all dependencies first:
    pip install fastapi uvicorn ultralytics opencv-python python-multipart numpy
"""

import uvicorn
from backend.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,      # must be 1 — model threads are global
    )