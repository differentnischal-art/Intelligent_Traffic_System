"""
Compatibility entry point for the ITMS FastAPI backend.

The full backend lives in backend.server so both of these commands use the
same intelligent controller:

    uvicorn backend.server:app --host 127.0.0.1 --port 8000
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""

import uvicorn

from backend.server import app


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, workers=1)
