#!/usr/bin/env python3
import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def train_model():
    """Train the ML model"""
    from app.ml_pipeline import MLPipeline
    pipeline = MLPipeline()
    pipeline.train_model()
    print("Model training completed!")

def start_server():
    """Start the FastAPI server"""
    subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [install|train|start|docker]")
        return
    
    command = sys.argv[1]
    
    if command == "install":
        install_dependencies()
    elif command == "train":
        train_model()
    elif command == "start":
        start_server()
    elif command == "docker":
        subprocess.run(["docker-compose", "up", "--build"])
    else:
        print("Unknown command. Use: install, train, start, or docker")

if __name__ == "__main__":
    main()