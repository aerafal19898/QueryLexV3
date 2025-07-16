"""
Entry point for the Legal Sanctions RAG application.
"""

import os
import sys
import logging
import dotenv
from pathlib import Path

# Add the project root directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Optionally, if needed, also add the app directory:
sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

# Load environment variables
dotenv.load_dotenv()

# Set OpenMP environment variable to fix thread warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

print("[RUN.PY] Starting application import...")
from app.main import app
print("[RUN.PY] Application imported successfully!")

if __name__ == "__main__":
    print("[RUN.PY] Starting main block...")
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Check if debug mode is enabled
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"[RUN.PY] ============================================")
    print(f"[RUN.PY] Starting QueryLexV4 server...")
    print(f"[RUN.PY] Host: {host}")
    print(f"[RUN.PY] Port: {port}")
    print(f"[RUN.PY] Debug mode: {debug}")
    print(f"[RUN.PY] ============================================")
    
    if debug:
        # Run in debug mode with Flask's development server
        print("[RUN.PY] Starting Flask development server...")
        app.run(host=host, port=port, debug=True)
    else:
        try:
            # Try to import waitress for production server
            from waitress import serve
            print(f"[RUN.PY] Starting production server on http://{host}:{port}")
            logging.info(f"Starting production server on http://{host}:{port}")
            serve(app, host=host, port=port, threads=8)
        except ImportError:
            # Fall back to Flask's built-in server
            print("[RUN.PY] Waitress not installed. Using Flask's development server.")
            logging.warning("Waitress not installed. Using Flask's development server.")
            logging.warning("This is not recommended for production use.")
            app.run(host=host, port=port, debug=False)