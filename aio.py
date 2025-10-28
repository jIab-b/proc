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
import argparse
from pathlib import Path


def load_env():
    """Load environment variables from .env file (small change to check)"""
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

    print("\n✓ API keys will be provided by user in the browser")

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


def launch_server():
    """Launch both FastAPI backend and Vite frontend servers asynchronously"""
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

    try:
        # Start FastAPI backend (using venv python)
        print("🚀 Starting FastAPI backend on http://localhost:8000")
        venv_python = Path("venv/bin/python").absolute()
        if not venv_python.exists():
            print(f"❌ Virtual environment not found at {venv_python}")
            print("   Run: python3 -m venv venv && source venv/bin/activate && uv pip install -r requirements.txt")
            sys.exit(1)

        backend_process = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"],
            cwd=str(Path.cwd()),
            env={"PATH": str(venv_python.parent)},  # Ensure venv bin in PATH
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True
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
        print("\n✅ Services started successfully in background")
        webbrowser.open("http://localhost:5173")

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()


def shutdown_server():
    """Shutdown both backend and frontend servers"""
    import httpx
    import subprocess
    import signal
    import os

    backend_shutdown = False
    frontend_shutdown = False

    # Try to shutdown backend gracefully
    try:
        print("🛑 Shutting down backend server...")
        with httpx.Client(timeout=5.0) as client:
            response = client.post("http://localhost:8000/shutdown")
            if response.status_code == 200:
                print("✓ Backend server shutdown successfully")
                backend_shutdown = True
            else:
                print(f"⚠️ Backend returned status {response.status_code}")
    except httpx.RequestError as e:
        print(f"⚠️ Could not shutdown backend gracefully: {e}")

    # Try to kill any remaining vite processes
    try:
        print("🛑 Checking for frontend processes...")
        result = subprocess.run(['pgrep', '-f', 'vite'], capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"✓ Killed Vite process (PID: {pid})")
                        frontend_shutdown = True
                    except (ProcessLookupError, OSError) as e:
                        print(f"⚠️ Could not kill Vite process {pid}: {e}")
        else:
            print("✓ No Vite processes found")
            frontend_shutdown = True
    except Exception as e:
        print(f"⚠️ Error checking for Vite processes: {e}")

    if backend_shutdown and frontend_shutdown:
        print("✅ All services shutdown successfully")
    else:
        print("⚠️ Some services may still be running")
        print("   You can check with: ps aux | grep -E '(python.*main.py|vite)'")


def main():
    """Start all services and open browser"""
    parser = argparse.ArgumentParser(description='WebGPU Minecraft Editor Launcher')
    parser.add_argument('--launch', action='store_true', help='Start backend + Vite asynchronously, open browser, then exit')
    parser.add_argument('--shutdown', action='store_true', help='Shutdown the running server')

    args = parser.parse_args()

    if args.launch:
        launch_server()
        return
    elif args.shutdown:
        shutdown_server()
        return

    # Original all-in-one functionality
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
