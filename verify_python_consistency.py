#!/usr/bin/env python
"""
Verify Python version consistency across the environment.
"""

import sys
import os
import subprocess
import platform

def check_python_version():
    """Check Python version and environment details."""
    print("=" * 60)
    print("Python Environment Verification")
    print("=" * 60)
    
    # Current Python info
    print("\n1. Current Python Information:")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    print(f"   Architecture: {platform.machine()}")
    
    # Virtual environment check
    print("\n2. Virtual Environment Check:")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"   In virtual environment: {in_venv}")
    
    if in_venv:
        print(f"   Virtual env path: {sys.prefix}")
        venv_cfg = os.path.join(sys.prefix, 'pyvenv.cfg')
        if os.path.exists(venv_cfg):
            print(f"   Virtual env config: {venv_cfg}")
            with open(venv_cfg, 'r') as f:
                for line in f:
                    if 'version' in line or 'home' in line:
                        print(f"     {line.strip()}")
    
    # Check numpy compatibility
    print("\n3. Numpy Compatibility Check:")
    try:
        import numpy as np
        print(f"   ✅ Numpy version: {np.__version__}")
        print(f"   ✅ Numpy imported successfully")
        
        # Check numpy binary files
        numpy_path = os.path.dirname(np.__file__)
        core_path = os.path.join(numpy_path, 'core')
        if os.path.exists(core_path):
            pyd_files = [f for f in os.listdir(core_path) if f.endswith('.pyd')]
            if pyd_files:
                print(f"   Numpy binary files found:")
                for f in pyd_files[:3]:  # Show first 3
                    print(f"     - {f}")
                    # Check if it matches Python version
                    if f'cp{sys.version_info.major}{sys.version_info.minor}' in f:
                        print(f"       ✅ Matches Python {sys.version_info.major}.{sys.version_info.minor}")
                    else:
                        print(f"       ❌ Does not match Python {sys.version_info.major}.{sys.version_info.minor}")
    except ImportError as e:
        print(f"   ❌ Numpy import failed: {e}")
    
    # Check unstructured
    print("\n4. Unstructured Library Check:")
    try:
        import unstructured
        print(f"   ✅ Unstructured imported successfully")
        
        # Try to import partition_pdf
        try:
            from unstructured.partition.pdf import partition_pdf
            print(f"   ✅ partition_pdf imported successfully")
        except Exception as e:
            print(f"   ❌ partition_pdf import failed: {e}")
            
    except ImportError as e:
        print(f"   ❌ Unstructured import failed: {e}")
    
    # Check PyPDF2
    print("\n5. PyPDF2 Check:")
    try:
        import PyPDF2
        print(f"   ✅ PyPDF2 version: {PyPDF2.__version__}")
        print(f"   ✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"   ❌ PyPDF2 import failed: {e}")
    
    print("\n" + "=" * 60)
    
    # Recommendations
    print("\nRecommendations:")
    if sys.version_info.major == 3 and sys.version_info.minor == 12:
        print("✅ Python 3.12 detected - Good!")
    else:
        print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("   Consider using Python 3.12 for best compatibility")
    
    if in_venv:
        print("✅ Running in virtual environment - Good!")
    else:
        print("⚠️  Not running in virtual environment")
        print("   Run: venv\\Scripts\\activate")

if __name__ == "__main__":
    check_python_version()