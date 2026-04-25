import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app import app

if __name__ == "__main__":
    print("🚀 Starting HandwrittenOCR API Server...")
    print("📡 The server will be available at:")
    print("   Local: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
