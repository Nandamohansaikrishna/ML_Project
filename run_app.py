# run.py
# Starts both FastAPI and Streamlit with one command
# run using :python run.py
import subprocess
import sys
import os
import time

def main():
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
        "app:app", "--reload", "--port", "8000"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )

    print("FastAPI  → http://localhost:8000")
    print("API Docs → http://localhost:8000/docs")

    time.sleep(3)

    ui = subprocess.Popen(
        [sys.executable, "-m", "streamlit",
        "run", "streamlit_app.py",
        "--server.port", "8501"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT
    )

    print("Streamlit → http://localhost:8501")

    try:
        api.wait()
        ui.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api.terminate()
        ui.terminate()
        api.wait()
        ui.wait()
        print("Both servers stopped.")

if __name__ == "__main__":
    main()