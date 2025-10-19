#!/usr/bin/env python3
"""
All-in-one launcher for WebGPU Minecraft Editor.
Starts both the FastAPI backend and Vite dev server, then opens browser.
"""
import os
import sys
import time
import signal
import subprocess
import webbrowser
from pathlib import Path


def load_env():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ[key] = value


def check_env():
    """Check environment and dependencies"""
    # Load .env file first
    load_env()

    print("="*60)
    print("  WebGPU Minecraft Editor - All-in-One Launcher")
    print("="*60)

    # Check for FAL_API_KEY
    if not os.environ.get("FAL_API_KEY") and not os.environ.get("FAL_KEY"):
        print("\n⚠️  WARNING: FAL_API_KEY not found in environment or .env file!")
        print("   Texture generation will not work without it.")
        print("   Create a .env file with: FAL_API_KEY=your_key_here\n")
    else:
        print("\n✓ FAL_API_KEY is set")

    # Check if required Python packages are installed
    try:
        import fastapi
        import uvicorn
        import httpx
        print("✓ Python dependencies found")
    except ImportError as e:
        print(f"\n❌ Missing Python dependency: {e}")
        print("   Install with: pip install fastapi uvicorn httpx python-multipart")
        sys.exit(1)

    # Check if npm is available
    if subprocess.run(["npm", "--version"], capture_output=True).returncode != 0:
        print("\n❌ npm not found. Please install Node.js")
        sys.exit(1)
    print("✓ npm found")

    print("")


def wait_for_server(url, timeout=30, service_name="Server"):
    """Wait for a server to be ready"""
    import urllib.request
    import urllib.error

    print(f"⏳ Waiting for {service_name} to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    print(f"✓ {service_name} is ready!")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(0.5)

    print(f"❌ {service_name} failed to start within {timeout} seconds")
    return False


def main():
    """Start all services and open browser"""
    check_env()

    processes = []

    def cleanup(signum=None, frame=None):
        """Clean up all child processes"""
        print("\n🛑 Shutting down...")
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except:
                proc.kill()
        print("✓ All services stopped")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        # Start FastAPI backend (using venv python)
        print("🚀 Starting FastAPI backend on http://localhost:8000")
        venv_python = Path("venv/bin/python").absolute()
        if not venv_python.exists():
            print(f"❌ Virtual environment not found at {venv_python}")
            print("   Run: python3 -m venv venv && source venv/bin/activate && uv pip install -r requirements.txt")
            sys.exit(1)

        backend_process = subprocess.Popen(
            [str(venv_python), "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(backend_process)

        # Wait for FastAPI to be ready
        if not wait_for_server("http://localhost:8000/health", service_name="FastAPI Backend"):
            cleanup()
            return

        # Start Vite dev server
        print("🚀 Starting Vite dev server on http://localhost:5173")
        vite_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd="webgpu",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(vite_process)

        # Wait for Vite to be ready
        if not wait_for_server("http://localhost:5173", service_name="Vite Dev Server"):
            cleanup()
            return

        # All services ready
        print("\n" + "="*60)
        print("  🎉 All services are running!")
        print("="*60)
        print("  Frontend:  http://localhost:5173")
        print("  Backend:   http://localhost:8000")
        print("  API Docs:  http://localhost:8000/docs")
        print("="*60)
        print("\n💡 Press Ctrl+C to stop all services\n")

        # Open browser
        webbrowser.open("http://localhost:5173")

        # Monitor processes
        while True:
            time.sleep(1)

            # Check if any process has died
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    name = "FastAPI" if i == 0 else "Vite"
                    print(f"\n❌ {name} process has stopped unexpectedly")
                    cleanup()
                    return

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()


if __name__ == "__main__":
    main()
