"""
Tesseract OCR configuration for Windows
"""
import os
import platform

def configure_tesseract():
    """Configure Tesseract path for Windows systems"""
    if platform.system() == "Windows":
        # Common Tesseract installation paths on Windows
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\andri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        ]
        
        # Find and set Tesseract path
        for path in tesseract_paths:
            if os.path.exists(path):
                # Set for pytesseract
                try:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = path
                except ImportError:
                    pass
                
                # Set environment variables for unstructured
                os.environ['TESSERACT_PATH'] = path
                os.environ['TESSERACT_CMD'] = path
                
                # Add Tesseract directory to PATH
                tesseract_dir = os.path.dirname(path)
                current_path = os.environ.get('PATH', '')
                if tesseract_dir not in current_path:
                    os.environ['PATH'] = f"{tesseract_dir};{current_path}"
                
                print(f"[TESSERACT] Configured Tesseract at: {path}")
                return path
        
        print("[TESSERACT] Warning: Tesseract not found in common Windows paths")
        print("[TESSERACT] Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    
    return None  # Non-Windows systems usually have Tesseract in PATH

if __name__ == "__main__":
    configure_tesseract()