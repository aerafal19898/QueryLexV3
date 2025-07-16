#!/usr/bin/env python3
"""
Preload optimization script for QueryLex V4.
Run this script to preload heavy libraries and improve document processing performance.
"""

import os
import sys
import time
from pathlib import Path

# Add the project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("QueryLex V4 - Library Preload Optimization")
    print("=" * 60)
    print()
    
    # Set environment variable to enable preloading
    os.environ['PRELOAD_DOC_LIBS'] = 'true'
    
    print("This script will preload heavy document processing libraries")
    print("to improve runtime performance when processing documents.")
    print()
    print("Loading libraries...")
    print()
    
    start_time = time.time()
    
    try:
        # Import and trigger preloading
        from app.utils.preload_libraries import get_preloaded_libs
        
        # Force preload
        libs = get_preloaded_libs()
        
        if libs and libs.get('loaded'):
            elapsed = time.time() - start_time
            print()
            print("✓ Libraries preloaded successfully!")
            print(f"  Total time: {elapsed:.2f} seconds")
            print()
            print("To enable automatic preloading on startup, set the environment variable:")
            print("  PRELOAD_DOC_LIBS=true")
            print()
            print("You can add this to your .env file:")
            print("  echo 'PRELOAD_DOC_LIBS=true' >> .env")
            print()
        else:
            print()
            print("✗ Failed to preload libraries")
            if libs and 'error' in libs:
                print(f"  Error: {libs['error']}")
            print()
            print("Document processing will use lazy loading instead.")
            print()
    
    except Exception as e:
        print()
        print(f"✗ Error during preloading: {e}")
        print()
        print("Document processing will use lazy loading instead.")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    main()