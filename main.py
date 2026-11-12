import sys
import os

# Ensure ml-service root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app
import config

if __name__ == "__main__":
    import uvicorn
    print(f"Starting LeafScan ML Microservice on {config.SERVICE_HOST}:{config.SERVICE_PORT}...")
    uvicorn.run("main:app", host=config.SERVICE_HOST, port=config.SERVICE_PORT, reload=False)
