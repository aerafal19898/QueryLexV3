"""
Main application file for the Legal Sanctions RAG system.
"""

print("[MAIN] Starting imports...")

print("[MAIN] Importing Flask and core dependencies...")
from flask import Flask, render_template, request, jsonify, session, after_this_request, g, redirect, url_for, Response, stream_with_context
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token, jwt_required, get_jwt_identity, verify_jwt_in_request, decode_token
from functools import wraps
import os
import uuid
import time
import re
import json
from werkzeug.utils import secure_filename
from datetime import datetime
import io
import tempfile
import secrets
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tesseract_config import configure_tesseract

print("[MAIN] Importing document processing libraries...")
print("[MAIN] This may take a moment as some libraries are large...")

print("[MAIN] Importing HuggingFace and ML libraries...")
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch

print("[MAIN] Document processing libraries will be imported when needed...")

print("[MAIN] Importing NLTK...")
import nltk

print("[MAIN] Importing environment and utility libraries...")
import dotenv
from app.utils.supabase_client import SupabaseVectorClient, SupabaseCollection
print("[MAIN] Importing pytesseract (OCR library)...")
try:
    import pytesseract
    import platform
    
    # Configure Tesseract path for Windows
    if platform.system() == "Windows":
        # Check common Windows installation paths
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\andri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"[MAIN] Tesseract found at: {path}")
                break
        else:
            print("[MAIN] Warning: Tesseract not found in common Windows paths")
            print("[MAIN] OCR features may not work. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    PYTESSERACT_AVAILABLE = True
    print("[MAIN] pytesseract imported successfully")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("[MAIN] pytesseract not available (optional)")

# Load environment variables
print("[MAIN] Loading environment variables...")
dotenv.load_dotenv()

# Configure Tesseract for Windows
print("[MAIN] Configuring Tesseract OCR...")
configure_tesseract()

# Download necessary NLTK data
print("[MAIN] Downloading NLTK data (this may take a moment on first run)...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    # For English only
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    print("[MAIN] NLTK data downloaded successfully")
except Exception as e:
    print(f"[MAIN] Warning: NLTK download error: {str(e)}")

# Local imports
print("[MAIN] Importing application configuration...")
from app.config import (
    DOCUMENTS_DIR, 
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_API_BASE,
    MODEL_PROVIDER,
    MODEL_NAME,
    SECRET_KEY,
    DEFAULT_DATASETS,
    ROLE_PERMISSIONS,
    AVAILABLE_MODELS,
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_KEY
)

print("[MAIN] Importing utility services...")
from app.utils.openrouter_client import OpenRouterClient
from app.utils.chat_service import SupabaseChatService

print("[MAIN] All imports completed successfully!")

print("[MAIN] Creating Flask app...")
app = Flask(__name__)
app.secret_key = SECRET_KEY
# Configure session to be permanent and last for 31 days
app.config['PERMANENT_SESSION_LIFETIME'] = 60 * 60 * 24 * 31  # 31 days in seconds
CORS(app)
print("[MAIN] Flask app created successfully")

# Initialize Supabase Vector Client
print("[MAIN] Initializing Supabase client...")
try:
    supabase_client = SupabaseVectorClient(use_service_key=True)
    print("[MAIN] Successfully connected to Supabase")
except Exception as e:
    print(f"[MAIN] Warning: Could not connect to Supabase: {e}")
    supabase_client = None

# Configure SSL for HuggingFace embeddings
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize embeddings lazily (only when needed)
print(f"[MAIN] Setting up lazy embeddings initialization with model: {EMBEDDING_MODEL}")
embeddings = None  # Will be initialized when first needed

# Document processing libraries - background preloading with health checks
print("[MAIN] Document processing libraries will be loaded in background after startup")
partition_pdf = None
PyPDF2 = None
doc_processing_status = {
    'initialized': False,
    'loading': False,
    'error': None,
    'progress': 'Not started',
    'libraries': {
        'unstructured': {'loaded': False, 'error': None},
        'PyPDF2': {'loaded': False, 'error': None},
        'pdf2image': {'loaded': False, 'error': None},
        'pytesseract': {'loaded': False, 'error': None},
        'cleaners': {'loaded': False, 'error': None}
    }
}

def load_document_processing_libraries():
    """Load document processing libraries in background thread."""
    global partition_pdf, PyPDF2, doc_processing_status
    
    doc_processing_status['loading'] = True
    doc_processing_status['progress'] = 'Starting library loading...'
    
    try:
        # Load unstructured library (after fixing Python version)
        doc_processing_status['progress'] = 'Loading unstructured library...'
        print("[BACKGROUND] Loading unstructured library...")
        try:
            from unstructured.partition.pdf import partition_pdf as _partition_pdf
            partition_pdf = _partition_pdf
            doc_processing_status['libraries']['unstructured']['loaded'] = True
            print("[BACKGROUND] ✓ Unstructured library loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load unstructured library: {e}"
            doc_processing_status['libraries']['unstructured']['error'] = error_msg
            print(f"[BACKGROUND] ✗ {error_msg}")
            partition_pdf = None
        
        # Load PyPDF2 as fallback
        doc_processing_status['progress'] = 'Loading PyPDF2...'
        print("[BACKGROUND] Loading PyPDF2...")
        try:
            import PyPDF2 as _PyPDF2
            PyPDF2 = _PyPDF2
            doc_processing_status['libraries']['PyPDF2']['loaded'] = True
            print("[BACKGROUND] ✓ PyPDF2 loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load PyPDF2: {e}"
            doc_processing_status['libraries']['PyPDF2']['error'] = error_msg
            print(f"[BACKGROUND] ✗ {error_msg}")
            PyPDF2 = None
        
        # Load PDF processing dependencies
        doc_processing_status['progress'] = 'Loading PDF processing dependencies...'
        print("[BACKGROUND] Loading PDF processing dependencies...")
        try:
            import pdf2image
            import pytesseract
            import platform
            
            # Configure Tesseract path for Windows
            if platform.system() == "Windows":
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\andri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                ]
                
                for path in tesseract_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        # Also set environment variable for unstructured
                        os.environ['TESSERACT_PATH'] = path
                        print(f"[BACKGROUND] Tesseract configured at: {path}")
                        break
            
            doc_processing_status['libraries']['pdf2image']['loaded'] = True
            doc_processing_status['libraries']['pytesseract']['loaded'] = True
            print("[BACKGROUND] ✓ PDF processing dependencies loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load PDF processing dependencies: {e}"
            doc_processing_status['libraries']['pdf2image']['error'] = error_msg
            doc_processing_status['libraries']['pytesseract']['error'] = error_msg
            print(f"[BACKGROUND] ✗ {error_msg}")
        
        # Load cleaners
        doc_processing_status['progress'] = 'Loading unstructured cleaners...'
        print("[BACKGROUND] Loading unstructured cleaners...")
        try:
            from unstructured.cleaners.core import clean_extra_whitespace
            doc_processing_status['libraries']['cleaners']['loaded'] = True
            print("[BACKGROUND] ✓ Unstructured cleaners loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load unstructured cleaners: {e}"
            doc_processing_status['libraries']['cleaners']['error'] = error_msg
            print(f"[BACKGROUND] ✗ {error_msg}")
        
        # Check if we have at least one working library
        if partition_pdf is not None or PyPDF2 is not None:
            doc_processing_status['initialized'] = True
            doc_processing_status['progress'] = 'Document processing libraries ready'
            print("[BACKGROUND] Document processing libraries initialization complete")
        else:
            doc_processing_status['error'] = "No document processing libraries could be loaded"
            doc_processing_status['progress'] = 'Failed to load any document processing libraries'
            print("[BACKGROUND] ✗ Failed to load any document processing libraries")
        
    except Exception as e:
        doc_processing_status['error'] = f"Background loading failed: {e}"
        doc_processing_status['progress'] = f'Background loading failed: {e}'
        print(f"[BACKGROUND] ✗ Background loading failed: {e}")
    
    finally:
        doc_processing_status['loading'] = False

def get_document_processing_libs():
    """Get document processing libraries with status checking."""
    global partition_pdf, PyPDF2, doc_processing_status
    
    return partition_pdf, PyPDF2

def get_embeddings():
    """Get embeddings instance, initializing if needed."""
    global embeddings
    if embeddings is None:
        print(f"[EMBEDDINGS] Initializing HuggingFace embeddings with model: {EMBEDDING_MODEL}...")
        print("[EMBEDDINGS] NOTE: This may take a few minutes on first run.")
        print("[EMBEDDINGS] If this is the first time running, the model will be downloaded...")
        print("[EMBEDDINGS] Download progress is not shown, but the process is running...")
        print("[EMBEDDINGS] Please wait, this can take 2-5 minutes depending on your internet connection...")
        print("[EMBEDDINGS] Model size: ~1.3GB for BAAI/bge-large-en-v1.5")
        print("[EMBEDDINGS] Progress: Starting download/initialization...")
        
        # Document processing libraries are loaded separately when needed
        
        import time
        import threading
        import sys
        
        # Progress indicator
        def progress_indicator():
            chars = "|/-\\"
            idx = 0
            while not stop_progress:
                print(f"\r[EMBEDDINGS] Loading model... {chars[idx % len(chars)]}", end='', flush=True)
                time.sleep(0.2)
                idx += 1
        
        stop_progress = False
        progress_thread = threading.Thread(target=progress_indicator)
        progress_thread.daemon = True
        progress_thread.start()
        
        start_time = time.time()
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        finally:
            stop_progress = True
            progress_thread.join(timeout=1)
        
        end_time = time.time()
        print(f"\r[EMBEDDINGS] Embeddings initialized successfully in {end_time - start_time:.2f} seconds")
    return embeddings

print("[MAIN] Embeddings will be initialized on first use")

# Direct use of Supabase client - no wrapper functions needed

# Initialize the OpenRouter client with the new DeepSeek model
print(f"[MAIN] Initializing OpenRouter client with model: deepseek/deepseek-chat-v3-0324:free")
llm_client = OpenRouterClient(
    api_key=OPENROUTER_API_KEY, 
    api_base=OPENROUTER_API_BASE,
    model="deepseek/deepseek-chat-v3-0324:free"
)
print("[MAIN] OpenRouter client initialized")

# Check for GPU
print("[MAIN] Checking for GPU availability...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[MAIN] Using device: {device}")

# Initialize JWT manager for authentication
print("[MAIN] Initializing JWT manager...")
jwt = JWTManager(app)
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_SECRET_KEY", SECRET_KEY)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRES", 60 * 60))  # 1 hour
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRES", 60 * 60 * 24 * 30))  # 30 days
print("[MAIN] JWT manager configured")

# Import and initialize additional components
print("[MAIN] Importing additional utility components...")
from app.models.user import User
from app.utils.audit_logger import AuditLogger
from app.utils.feedback_service import SupabaseFeedbackService
from app.utils.credit_system import CreditSystem
from app.utils.encryption import DocumentEncryption
from app.utils.secure_processor import SecureDocumentProcessor

# Initialize user management
print("[MAIN] Initializing user management...")
user_manager = User()

# Initialize audit logger
from app.config import ENABLE_AUDIT_LOGGING, MAIL_SERVER, MAIL_PORT, MAIL_USERNAME, MAIL_PASSWORD
from app.config import MAIL_DEFAULT_SENDER, MAIL_FEEDBACK_RECIPIENT, MAIL_USE_TLS, DOCUMENT_ENCRYPTION_KEY

# Create audit logger
audit_logger = AuditLogger(enabled=os.environ.get("ENABLE_AUDIT_LOGGING", "True").lower() == "true")

# Initialize feedback service
feedback_manager = SupabaseFeedbackService(
    smtp_server=os.environ.get("MAIL_SERVER", MAIL_SERVER),
    smtp_port=int(os.environ.get("MAIL_PORT", MAIL_PORT)),
    smtp_username=os.environ.get("MAIL_USERNAME", MAIL_USERNAME),
    smtp_password=os.environ.get("MAIL_PASSWORD", MAIL_PASSWORD),
    sender_email=os.environ.get("MAIL_DEFAULT_SENDER", MAIL_DEFAULT_SENDER),
    recipient_email=os.environ.get("MAIL_FEEDBACK_RECIPIENT", MAIL_FEEDBACK_RECIPIENT),
    use_tls=os.environ.get("MAIL_USE_TLS", str(MAIL_USE_TLS)).lower() == "true"
)

# Initialize credit system
credit_system = CreditSystem()

# Initialize secure document components
# Use the Fernet key generation if needed
if not os.environ.get("DOCUMENT_ENCRYPTION_KEY"):
    # Make sure we use the fixed DOCUMENT_ENCRYPTION_KEY from config.py
    print("Using auto-generated encryption key")
else:
    print("Using environment variable for encryption key")

document_encryption = DocumentEncryption(key=os.environ.get("DOCUMENT_ENCRYPTION_KEY", DOCUMENT_ENCRYPTION_KEY))
secure_processor = SecureDocumentProcessor(
    encryption_handler=document_encryption,
    embedding_model=EMBEDDING_MODEL,
    device=device
)

# Initialize chat storage
try:
    chat_storage = SupabaseChatService(use_service_key=True)
    print("Successfully connected to Supabase for chat storage")
except Exception as e:
    print(f"Warning: Could not connect to Supabase for chat storage: {e}")
    chat_storage = None

# Initialize default dataset on app start
DEFAULT_DATASET_NAME = None

def initialize_default_dataset():
    """Initialize default dataset from the first available dataset in Supabase."""
    global DEFAULT_DATASET_NAME
    try:
        if supabase_client:
            collections = supabase_client.list_collections()
            collections = [{"name": c["name"]} for c in collections]
            if collections:
                # Set the first available dataset as default
                DEFAULT_DATASET_NAME = collections[0]["name"]
                print(f"Default dataset set to: {DEFAULT_DATASET_NAME}")
            else:
                print("No datasets available - users will see 'No datasets available' message")
                DEFAULT_DATASET_NAME = None
        else:
            print("Supabase client not available - cannot set default dataset")
            DEFAULT_DATASET_NAME = None
    except Exception as e:
        print(f"Error initializing default dataset: {e}")
        DEFAULT_DATASET_NAME = None

# Initialize default dataset
print("[MAIN] Initializing default dataset...")
initialize_default_dataset()
print("[MAIN] Default dataset initialization complete")

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset metadata is now stored in Supabase collections
# The load_dataset_metadata and save_dataset_metadata functions have been removed
# as all metadata operations now use supabase_client.update_collection_metadata()

# Configure file uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300MB total limit for dataset documents (no per-file limit)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_response_with_bold_headers(response):
    """Ensure section headers are bolded in the response."""
    if not isinstance(response, str):
        return response
    
    # Apply bold formatting to common section headers
    formatted_response = response
    
    # Main section headers
    formatted_response = formatted_response.replace('SOURCES:', '**SOURCES:**')
    formatted_response = formatted_response.replace('ANALYSIS:', '**ANALYSIS:**')
    formatted_response = formatted_response.replace('APPLICABLE PROVISIONS:', '**APPLICABLE PROVISIONS:**')
    formatted_response = formatted_response.replace('CONCLUSION:', '**CONCLUSION:**')
    
    # Alternative formats
    formatted_response = formatted_response.replace('Sources:', '**Sources:**')
    formatted_response = formatted_response.replace('Analysis:', '**Analysis:**')
    formatted_response = formatted_response.replace('Applicable Provisions:', '**Applicable Provisions:**')
    formatted_response = formatted_response.replace('Conclusion:', '**Conclusion:**')
    
    # Numbered sections
    formatted_response = formatted_response.replace('1. SOURCES:', '1. **SOURCES:**')
    formatted_response = formatted_response.replace('2. ANALYSIS:', '2. **ANALYSIS:**')
    formatted_response = formatted_response.replace('3. APPLICABLE PROVISIONS:', '3. **APPLICABLE PROVISIONS:**')
    formatted_response = formatted_response.replace('4. CONCLUSION:', '4. **CONCLUSION:**')
    
    return formatted_response

def robust_extract_text_from_pdf(pdf_path):
    """Extract text from PDF using multiple methods for better reliability."""
    global doc_processing_status
    
    # Store all extracted text
    all_text = []
    extraction_success = False
    
    # Check if libraries are still loading
    if doc_processing_status['loading']:
        print(f"[DOC_PROCESSING] Libraries still loading: {doc_processing_status['progress']}")
        print(f"[DOC_PROCESSING] Waiting for background loading to complete...")
        # Wait a reasonable amount of time for background loading
        import time
        max_wait = 300  # 5 minutes maximum wait
        wait_time = 0
        while doc_processing_status['loading'] and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            print(f"[DOC_PROCESSING] Still waiting... ({wait_time}s elapsed, status: {doc_processing_status['progress']})")
        
        if doc_processing_status['loading']:
            print(f"[DOC_PROCESSING] Timeout waiting for libraries to load. Proceeding with available libraries.")
    
    # Get document processing libraries
    partition_pdf_func, PyPDF2_lib = get_document_processing_libs()
    
    # METHOD 1: Using unstructured library (works well for many PDFs)
    if partition_pdf_func is not None:
        try:
            print(f"Trying unstructured partition for {pdf_path}")
            
            # Ensure Tesseract is configured for Windows
            import platform
            if platform.system() == "Windows" and 'TESSERACT_PATH' not in os.environ:
                tesseract_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r"C:\Users\andri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                ]
                for path in tesseract_paths:
                    if os.path.exists(path):
                        os.environ['TESSERACT_PATH'] = path
                        if PYTESSERACT_AVAILABLE:
                            pytesseract.pytesseract.tesseract_cmd = path
                        print(f"[EXTRACT] Tesseract configured at: {path}")
                        break
            
            # Try without extract_images_in_pdf parameter first
            try:
                elements = partition_pdf_func(pdf_path, strategy="hi_res")
            except TypeError:
                # If that fails, try with older version parameters
                elements = partition_pdf_func(pdf_path)
                
            for element in elements:
                if hasattr(element, 'text') and element.text:
                    all_text.append(element.text)
                    extraction_success = True
        except Exception as e:
            print(f"Error with unstructured partition: {str(e)}")
    else:
        print(f"[DOC_PROCESSING] Unstructured library not available, skipping to PyPDF2")
    
    # If first method got text, return it
    if extraction_success and len(''.join(all_text).strip()) > 100:
        print(f"Successfully extracted text using unstructured partition")
        return all_text
    
    # METHOD 2: Using PyPDF2 (good for text-based PDFs)
    if PyPDF2_lib is not None:
        try:
            print(f"Trying PyPDF2 for {pdf_path}")
            
            pdf_text = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2_lib.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            pdf_text.append(f"Page {page_num+1}: {text}")
                            extraction_success = True
            
            # If we got text, add it to all_text
            if extraction_success:
                all_text.extend(pdf_text)
                print(f"Successfully extracted text using PyPDF2")
        except Exception as e:
            print(f"Error with PyPDF2: {str(e)}")
    else:
        print(f"[DOC_PROCESSING] PyPDF2 library not available, skipping to OCR")
    
    # If we still don't have text and neither library is available, provide clear error
    if not extraction_success and partition_pdf_func is None and PyPDF2_lib is None:
        print(f"[DOC_PROCESSING] ERROR: No PDF processing libraries are available!")
        print(f"[DOC_PROCESSING] Library status: {doc_processing_status}")
        return []
    
    # METHOD 3: Using OCR if available and needed
    if not extraction_success and PYTESSERACT_AVAILABLE:
        try:
            print(f"Trying OCR for {pdf_path}")
            # Convert PDF to images and OCR
            from pdf2image import convert_from_path
            
            ocr_text = []
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                if text and text.strip():
                    ocr_text.append(f"Page {i+1} (OCR): {text}")
                    extraction_success = True
            
            if extraction_success:
                all_text.extend(ocr_text)
                print(f"Successfully extracted text using OCR")
        except Exception as e:
            print(f"Error with OCR: {str(e)}")
    
    # If no text was extracted, add a message
    if not extraction_success or not all_text:
        print(f"Could not extract any text from {pdf_path}")
        return []
    
    return all_text

@app.route('/api/health/document-processing')
def health_check():
    """Check the status of document processing library loading."""
    global doc_processing_status
    return jsonify(doc_processing_status)

@app.route('/')
def index():
    """Render the main chat interface."""
    # Mark the session as permanent so it persists beyond browser close
    session.permanent = True
    
    # Ensure session has a session_id (used for tracking user sessions)
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Optionally, you can track last_active_chat here if you want to reopen the last chat automatically
    last_active_chat = session.get('last_active_chat', None)
    
    # Get available datasets for the UI
    try:
        collections = supabase_client.list_collections()
        available_datasets = [{"name": c["name"], "description": f"Dataset: {c['name']}"} for c in collections]
        # If no datasets available, add a placeholder
        if not available_datasets:
            available_datasets = [{"name": "", "description": "No datasets available"}]
    except Exception as e:
        print(f"Error loading datasets for UI: {e}")
        available_datasets = [{"name": "", "description": "No datasets available"}]
    
    return render_template('index.html', 
                          datasets=available_datasets,
                          default_dataset=DEFAULT_DATASET_NAME,
                          available_models=AVAILABLE_MODELS,
                          last_active_chat=last_active_chat)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return AI responses with relevant context."""
    data = request.json
    user_message = data.get('message', '')
    dataset_name = data.get('dataset', DEFAULT_DATASET_NAME or 'EU-Sanctions')
    model_name = data.get('model', MODEL_NAME)  # Get the selected model
    history = data.get('history', [])
    
    # Validate that the dataset exists
    if supabase_client:
        try:
            collections = supabase_client.list_collections()
            available_datasets = [c["name"] for c in collections]
            
            if dataset_name not in available_datasets:
                if available_datasets:
                    dataset_name = available_datasets[0]
                    print(f"[CHAT] Dataset not found, using first available: '{dataset_name}'")
                else:
                    dataset_name = None
        except Exception as e:
            print(f"[CHAT] Error validating dataset: {e}")
    
    # Get session ID for this conversation
    if 'session_id' not in session:
        session.permanent = True
        session['session_id'] = str(uuid.uuid4())
    
    # Handle the case where no datasets are available
    if not dataset_name:
        return jsonify({
            'response': "No datasets are currently available. Please upload documents to create a dataset first.",
            'context': "No datasets available",
            'dataset': "None",
            'raw_context': "No datasets available"
        })
    
    # Retrieve context from Supabase based on user query
    collection = supabase_client.get_or_create_collection(dataset_name)
    
    # Handle the case where collection is None or empty
    if not collection:
        results = {
            'documents': [["Database connection error. Please check your Supabase configuration."]],
            'metadatas': [[{"source": "System", "page": 0}]]
        }
    else:
        # Get the document count to determine how many results we can request
        doc_count = collection.count()
        n_results = min(5, max(1, doc_count))  # Request at most 5, but at least 1 if available
        
        # Handle the case of an empty collection
        if doc_count == 0:
            results = {
                'documents': [["No documents found in the selected dataset. Please upload documents or select a different dataset."]],
                'metadatas': [[{"source": "System", "page": 0}]]
            }
        else:
            # Query the collection directly using embeddings
            query_embedding = get_embeddings().embed_query(user_message)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
    
    context = '\n\n'.join(results['documents'][0])
    
    # Extract sources information for better citation
    sources = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_info = {
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", meta.get("part", 0)),
            "snippet": doc[:150] + "..." if len(doc) > 150 else doc  # Short preview
        }
        sources.append(source_info)
    
    system_message = f"""You are a legal expert.
    
    # Instructions
    - Use the provided context to analyze the user's question thoroughly
    - Implement chain-of-thought reasoning by breaking down your analysis step by step
    - First carefully examine the relevant sections from the provided context
    - Think about what legal principles apply to this situation
    - Consider multiple perspectives and interpretations if applicable
    - Draw connections between different parts of the context
    - Formulate a comprehensive and legally sound analysis
    - Cite specific articles, sections, or provisions when possible
    - Clearly separate your reasoning process from your final conclusion
    - If you don't know the answer or it's not in the context, state this clearly
    
    # Output Format
    Structure your response with these sections, using **bold** formatting for section titles:
    1. **SOURCES**: A brief bulleted list of the most relevant source documents you're drawing from
    2. **ANALYSIS**: Your step-by-step reasoning about the question (this should be detailed)
    3. **APPLICABLE PROVISIONS**: Specific articles, sections, or legal provisions that apply
    4. **CONCLUSION**: Your final answer based on the analysis
    
    Important: Always format section headers in bold using **TITLE**: format
    
    # Context
    {context}
    """
    
    # Format the conversation history
    formatted_history = []
    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            formatted_history.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        # Log the full prompt for debugging
        print("\n--- LLM PROMPT DEBUG ---")
        print("System message:")
        print(system_message)
        print("\nChat history:")
        print(formatted_history[-10:] if formatted_history else [])
        print("--- END LLM PROMPT DEBUG ---\n")
        # Call LLM API with enhanced prompt
        response = llm_client.generate_with_rag(
            query=user_message,
            context=system_message,  # Pass the complete system message with instructions
            chat_history=formatted_history[-10:] if formatted_history else [],  # Use last 5 turns (10 messages)
            temperature=0.1  # Lower temperature for more analytical responses
        )
        # Ensure section headers are bolded in Markdown
        response = format_response_with_bold_headers(response)
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        response = "I'm sorry, I encountered an error processing your request. Please try again later."
        
    # Add source information to the response context
    source_context = "\n\n".join([f"Source: {s['source']} (Page/Section: {s['page']})\nPreview: {s['snippet']}" for s in sources])
    
    return jsonify({
        'response': response,
        'context': source_context,  # Use the formatted source context with citation info
        'dataset': dataset_name,
        'raw_context': context  # Include the raw context as well if needed
    })

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Return available datasets."""
    # List collections in Supabase
    try:
        collections = supabase_client.list_collections()
        
        # Debug output
        print(f"Found Supabase collections: {[c['name'] for c in collections]}")
        
        # Combine with default datasets
        all_datasets = []
        
        # Add custom datasets first
        for collection in collections:
            name = collection["name"]
            # Count documents in collection
            try:
                coll = supabase_client.get_collection(name)
                if coll:
                    # Get all documents from collection to count unique sources
                    results = coll.get(include=["metadatas"])
                    unique_sources = set()
                    if results and "metadatas" in results:
                        for meta in results["metadatas"]:
                            if meta:  # Check if metadata is not None
                                source = meta.get("source") or meta.get("file") or meta.get("filename")
                                if source:
                                    unique_sources.add(source)
                    doc_count = len(unique_sources)  # Count unique documents
                else:
                    doc_count = 0
                
                # Get metadata from Supabase collection
                supabase_metadata = collection.get("metadata", {})
                
                # If description is missing or is the default, create a better one
                stored_description = supabase_metadata.get("description", "")
                if not stored_description or stored_description.startswith("Custom dataset with"):
                    # Build a descriptive text from other metadata
                    description_parts = []
                    if supabase_metadata.get("topic"):
                        description_parts.append(f"Topic: {supabase_metadata.get('topic')}")
                    if supabase_metadata.get("author"):
                        description_parts.append(f"by {supabase_metadata.get('author')}")
                    
                    if description_parts:
                        description = " - ".join(description_parts)
                    else:
                        description = f"Custom dataset with {doc_count} entries"
                else:
                    description = stored_description
                    
                author = supabase_metadata.get("author")
                topic = supabase_metadata.get("topic")
                linkedin_url = supabase_metadata.get("linkedin_url")
                custom_instructions = supabase_metadata.get("custom_instructions")
                last_update_date = supabase_metadata.get("last_update_date")
                # Use document_count from metadata if available, otherwise use our calculated doc_count
                document_count = supabase_metadata.get("document_count", doc_count)
                all_datasets.append({
                    "name": name, 
                    "description": description,
                    "author": author,
                    "topic": topic,
                    "linkedin_url": linkedin_url,
                    "custom_instructions": custom_instructions,
                    "last_update_date": last_update_date,
                    "document_count": document_count,
                    "is_custom": True
                })
            except Exception as e:
                print(f"Error getting collection info for {name}: {str(e)}")
                supabase_metadata = collection.get("metadata", {})
                description = supabase_metadata.get("description", "Custom dataset")
                all_datasets.append({
                    "name": name, 
                    "description": description,
                    "is_custom": True
                })
        
        # Then add default datasets that aren't already in the list
        for dataset in DEFAULT_DATASETS:
            if not any(d['name'] == dataset['name'] for d in all_datasets):
                dataset["is_custom"] = False
                all_datasets.append(dataset)
        
        return jsonify(all_datasets)
    except Exception as e:
        print(f"Error fetching datasets: {str(e)}")
        # Fallback to default datasets
        return jsonify([{"name": "Default", "description": "Default dataset (error occurred)", "is_custom": False}])

@app.route('/api/datasets', methods=['POST'])
def create_dataset():
    """Create a new dataset."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        dataset_name = data.get('name')
        if not dataset_name:
            return jsonify({"error": "Dataset name is required"}), 400

        # Sanitize the dataset name
        sanitized_name = ''.join(c if c.isalnum() or c in '-_' else '-' for c in dataset_name)
        sanitized_name = sanitized_name.strip('-_')
        
        if len(sanitized_name) > 60:
            sanitized_name = sanitized_name[:60]
        if len(sanitized_name) < 3:
            sanitized_name = f"dataset-{int(time.time())}"
        if not sanitized_name[0].isalnum():
            sanitized_name = 'x' + sanitized_name[1:]
        if not sanitized_name[-1].isalnum():
            sanitized_name = sanitized_name[:-1] + 'x'

        # Check if dataset already exists
        try:
            if supabase_client:
                supabase_client.get_collection(sanitized_name)
                return jsonify({"error": f"Dataset '{sanitized_name}' already exists"}), 400
        except Exception:
            # Collection doesn't exist, which is what we want
            pass

        # Create the collection with metadata in Supabase
        if supabase_client:
            current_time = datetime.now().isoformat()
            # If description is empty or starts with "Custom dataset with", store empty to let get_datasets generate a better one
            final_description = data.get('description', '')
            if not final_description or final_description.startswith("Custom dataset with"):
                final_description = ""
            
            collection_data = supabase_client.create_collection(
                name=sanitized_name,
                description=final_description,
                author=data.get('author', ''),
                topic=data.get('topic', ''),
                linkedin_url=data.get('linkedin_url', ''),
                custom_instructions=data.get('custom_instructions', ''),
                created_at=current_time,
                last_update_date=current_time,
                document_count=0
            )
        else:
            return jsonify({"error": "Database connection not available"}), 500

        return jsonify({
            "success": True,
            "message": f"Dataset '{sanitized_name}' created successfully",
            "dataset": {
                "name": sanitized_name,
                "description": data.get('description', ''),
                "author": data.get('author', ''),
                "topic": data.get('topic', ''),
                "linkedin_url": data.get('linkedin_url', ''),
                "custom_instructions": data.get('custom_instructions', ''),
                "is_custom": True,
                "document_count": 0,
                "last_update_date": current_time
            }
        })

    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return jsonify({"error": f"Failed to create dataset: {str(e)}"}), 500

@app.route('/api/datasets/<dataset_name>', methods=['DELETE'])
def delete_dataset(dataset_name):
    """Delete a dataset by name."""
    if dataset_name in [d['name'] for d in DEFAULT_DATASETS]:
        return jsonify({"error": "Cannot delete default dataset"}), 400
    
    deleted_from_database = False
    deletion_messages = []
    
    try:
        # Attempt to delete from Supabase
        if supabase_client:
            success = supabase_client.delete_collection(dataset_name)
            if success:
                deleted_from_database = True 
                deletion_messages.append(f"Collection '{dataset_name}' deleted from Supabase.")
                print(f"Successfully deleted collection '{dataset_name}' from Supabase.")
            else:
                error_msg = f"Failed to delete collection '{dataset_name}' from Supabase"
                deletion_messages.append(error_msg)
                print(error_msg)
        else:
            deleted_from_database = True
            msg = f"Supabase not available, cannot delete collection '{dataset_name}'"
            deletion_messages.append(msg)
            print(msg)
    except Exception as e:
        error_msg = f"Error deleting collection '{dataset_name}': {str(e)}"
        deletion_messages.append(error_msg)
        print(error_msg)
            
    # Metadata is now stored in Supabase collection, no need to delete from local file
    deleted_from_metadata = True
    msg = f"Dataset '{dataset_name}' metadata is managed in Supabase (no local metadata to delete)."
    deletion_messages.append(msg)
    print(msg)

    if deleted_from_database and deleted_from_metadata:
        return jsonify({"success": True, "message": f"Dataset '{dataset_name}' successfully deleted. Details: {' '.join(deletion_messages)}"})
    else:
        # Even if one part failed but the other succeeded, it's a partial success from user POV if dataset is gone from list.
        # The key is whether it will be gone from the UI. If metadata is deleted, it will be.
        if deleted_from_metadata:
             return jsonify({"success": True, "message": f"Dataset '{dataset_name}' removed. Status: {' '.join(deletion_messages)}"})
        else:
            # This implies metadata deletion failed, which is less likely.
            return jsonify({"error": f"Failed to fully delete dataset '{dataset_name}'. Status: {' '.join(deletion_messages)}"}), 500

@app.route('/api/datasets/<dataset_name>', methods=['PUT'])
def update_dataset_metadata_route(dataset_name):
    """Update metadata for an existing dataset."""
    print(f"[DEBUG] Incoming update for dataset_name: '{dataset_name}'")
    data_to_update = request.json
    if not data_to_update:
        return jsonify({"error": "No data provided for update"}), 400

    # Check if dataset exists in Supabase
    try:
        if supabase_client:
            # Try to get the collection to check if it exists
            collection_data = supabase_client._get_collection_data(dataset_name)
            exists_in_supabase = True
            # Get document count from Supabase
            doc_count = supabase_client.count_documents(dataset_name)
        else:
            return jsonify({"error": "Database connection not available"}), 500
    except Exception as e:
        print(f"[DEBUG] Error checking Supabase collection: {e}")
        return jsonify({"error": "Dataset not found"}), 404

    # Also check if it's a default dataset, which shouldn't be "updated" via this mechanism
    if any(d['name'] == dataset_name for d in DEFAULT_DATASETS):
        return jsonify({"error": "Default datasets cannot be modified."}), 403

    # Update metadata in Supabase
    try:
        current_time = datetime.now().isoformat()
        
        # Prepare metadata updates
        metadata_updates = {
            "description": data_to_update.get("description", ""),
            "author": data_to_update.get("author", ""),
            "topic": data_to_update.get("topic", ""),
            "linkedin_url": data_to_update.get("linkedin_url", ""),
            "custom_instructions": data_to_update.get("custom_instructions", ""),
            "last_update_date": current_time,
            "document_count": doc_count
        }
        
        # Update collection metadata in Supabase
        success = supabase_client.update_collection_metadata(dataset_name, **metadata_updates)
        
        if not success:
            return jsonify({"error": "Failed to update dataset metadata"}), 500
        
        return jsonify({"success": True, "message": f"Dataset '{dataset_name}' updated successfully."})
        
    except Exception as e:
        print(f"[DEBUG] Error updating dataset metadata: {e}")
        return jsonify({"error": "Failed to update dataset metadata"}), 500

@app.route('/api/process-documents', methods=['POST'])
def process_documents():
    """Process uploaded documents and create embeddings."""
    # This would handle file uploads in production
    # For now, we'll process the existing documents
    
    dataset_name = request.json.get('dataset_name', 'Default')
    
    # Create or get collection
    collection = supabase_client.get_or_create_collection(dataset_name)
    
    if not collection:
        return jsonify({
            "status": "error",
            "message": "Database connection not available",
            "dataset": dataset_name
        })
    
    # Process PDF files
    documents = []
    metadatas = []
    ids = []
    
    # Check if directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        return jsonify({
            "status": "error",
            "message": f"Documents directory not found. Created: {DOCUMENTS_DIR}. Please add PDF files to this directory.",
            "dataset": dataset_name
        })
    
    # List document files (PDF and TXT)
    doc_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(('.pdf', '.txt'))][:10]
    
    # Handle case where no documents exist
    if not doc_files:
        # Add some sample data for testing
        sample_text = "This is a sample document for the EU Sanctions dataset. The European Union imposes sanctions or restrictive measures in pursuit of the specific objectives of the Common Foreign and Security Policy (CFSP). Sanctions are preventative, non-punitive instruments which aim to bring about a change in policy or activity by targeting non-EU countries, entities, and individuals responsible for the malign behavior at stake."
        documents.append(sample_text)
        metadatas.append({"source": "sample_document.txt", "page": 1})
        ids.append(f"sample_document_0_1")
    
    # Process document files
    for i, doc_file in enumerate(doc_files):
        file_path = os.path.join(DOCUMENTS_DIR, doc_file)
        
        try:
            if doc_file.endswith('.pdf'):
                # Use robust PDF extraction method
                extracted_texts = robust_extract_text_from_pdf(file_path)
                
                # Process extracted texts
                for j, text in enumerate(extracted_texts):
                    if text and len(text) > 50:  # Skip very short segments
                        documents.append(text)
                        metadatas.append({"source": doc_file, "page": j})
                        ids.append(f"{doc_file}_{i}_{j}")
            elif doc_file.endswith('.txt'):
                # Read text file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Split text into paragraphs or chunks
                chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
                
                for j, chunk in enumerate(chunks):
                    if len(chunk) > 50:  # Skip very short segments
                        documents.append(chunk)
                        metadatas.append({"source": doc_file, "part": j})
                        ids.append(f"{doc_file}_{i}_{j}")
                        
                # If no chunks (single paragraph), add the whole text
                if not chunks and len(text) > 50:
                    documents.append(text)
                    metadatas.append({"source": doc_file, "part": 0})
                    ids.append(f"{doc_file}_{i}_0")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Add to Supabase with embeddings
    for i in range(0, len(documents), 100):  # Batch processing
        batch_docs = documents[i:i+100]
        batch_meta = metadatas[i:i+100]
        batch_ids = ids[i:i+100]
        
        # Generate embeddings for the batch with SSL error handling
        try:
            batch_embeddings = get_embeddings().embed_documents(batch_docs)
        except Exception as e:
            if "EOF occurred in violation of protocol" in str(e) or "SSL" in str(e):
                print(f"SSL error generating embeddings, retrying individually with delay: {e}")
                batch_embeddings = []
                for doc in batch_docs:
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            time.sleep(1)  # Wait 1 second between attempts
                            doc_embedding = get_embeddings().embed_documents([doc])
                            batch_embeddings.extend(doc_embedding)
                            break
                        except Exception as retry_e:
                            if attempt == max_retries - 1:
                                print(f"Failed to generate embedding for document after {max_retries} attempts: {retry_e}")
                                # Use zero vector as fallback
                                batch_embeddings.append([0.0] * 1024)  # 1024-dim embeddings for BAAI/bge-large-en-v1.5
                            else:
                                print(f"Retry {attempt + 1}/{max_retries} failed: {retry_e}")
                                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
        
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
    
    # Store dataset description in Supabase
    dataset_description = request.json.get('dataset_description', '')
    if dataset_description:
        supabase_client.update_collection_metadata(
            dataset_name, 
            description=dataset_description,
            last_update_date=datetime.now().isoformat()
        )
    
    return jsonify({
        "status": "success",
        "message": f"Processed {len(documents)} text chunks from {len(doc_files)} documents",
        "dataset": dataset_name
    })

@app.route('/api/upload-documents', methods=['POST'])
def upload_documents():
    """Handle file uploads and process them."""
    def generate_progress():
        try:
            if 'files' not in request.files:
                yield json.dumps({"status": "error", "message": "No files uploaded"}) + "\n"
                return
            
            # Get dataset name and sanitize it for ChromaDB requirements
            dataset_name = request.form.get('dataset_name', 'New Dataset')
            dataset_description = request.form.get('dataset_description', '')
            dataset_author = request.form.get('dataset_author', '')
            dataset_topic = request.form.get('dataset_topic', '')
            dataset_linkedin = request.form.get('dataset_linkedin', '')
            dataset_custom_instructions = request.form.get('dataset_custom_instructions', '')
            
            # Sanitize the dataset name
            sanitized_name = ''.join(c if c.isalnum() or c in '-_' else '-' for c in dataset_name)
            sanitized_name = sanitized_name.strip('-_')
            
            if len(sanitized_name) > 60:
                sanitized_name = sanitized_name[:60]
            if len(sanitized_name) < 3:
                sanitized_name = f"dataset-{int(time.time())}"
            if not sanitized_name[0].isalnum():
                sanitized_name = 'x' + sanitized_name[1:]
            if not sanitized_name[-1].isalnum():
                sanitized_name = sanitized_name[:-1] + 'x'
            
            dataset_name = sanitized_name
            
            # Check if dataset already exists
            try:
                if supabase_client:
                    supabase_client.get_collection(dataset_name)
                    yield json.dumps({"status": "error", "message": f"Dataset '{dataset_name}' already exists"}) + "\n"
                    return
            except Exception:
                # Collection doesn't exist, which is what we want
                pass

            # Create collection
            if supabase_client:
                collection_data = supabase_client.create_collection(
                    name=sanitized_name,
                    description=dataset_description,
                    author=dataset_author,
                    topic=dataset_topic,
                    linkedin_url=dataset_linkedin,
                    custom_instructions=dataset_custom_instructions
                )
                collection = SupabaseCollection(supabase_client, sanitized_name, collection_data)
            else:
                yield json.dumps({"status": "error", "message": "Database connection not available"}) + "\n"
                return
            
            # Process uploaded files
            files = request.files.getlist('files')
            total_documents = len(files)
            documents = []
            metadatas = []
            ids = []
            
            for i, file in enumerate(files):
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    try:
                        if filename.endswith('.pdf'):
                            extracted_texts = robust_extract_text_from_pdf(file_path)
                            for j, text in enumerate(extracted_texts):
                                if text and len(text) > 50:
                                    documents.append(text)
                                    metadatas.append({"source": filename, "page": j + 1})
                                    ids.append(f"{filename}_{i}_{j}")
                        elif filename.endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read()
                            chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
                            for j, chunk in enumerate(chunks):
                                if len(chunk) > 50:
                                    documents.append(chunk)
                                    metadatas.append({"source": filename, "page": 1})
                                    ids.append(f"{filename}_{i}_{j}")
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue
                    
                    # Clean up the file after processing
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Error removing temporary file {file_path}: {str(e)}")
            
            # Add to Supabase in batches with embeddings
            for i in range(0, len(documents), 100):
                batch_docs = documents[i:i+100]
                batch_meta = metadatas[i:i+100]
                batch_ids = ids[i:i+100]
                
                # Generate embeddings for the batch with SSL error handling
                try:
                    batch_embeddings = get_embeddings().embed_documents(batch_docs)
                except Exception as e:
                    if "EOF occurred in violation of protocol" in str(e) or "SSL" in str(e):
                        print(f"SSL error generating embeddings, retrying individually with delay: {e}")
                        batch_embeddings = []
                        for doc in batch_docs:
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    time.sleep(1)  # Wait 1 second between attempts
                                    doc_embedding = get_embeddings().embed_documents([doc])
                                    batch_embeddings.extend(doc_embedding)
                                    break
                                except Exception as retry_e:
                                    if attempt == max_retries - 1:
                                        print(f"Failed to generate embedding for document after {max_retries} attempts: {retry_e}")
                                        # Use zero vector as fallback
                                        batch_embeddings.append([0.0] * 1024)  # 1024-dim embeddings for BAAI/bge-large-en-v1.5
                                    else:
                                        print(f"Retry {attempt + 1}/{max_retries} failed: {retry_e}")
                                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
                
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            
            # Update dataset metadata in Supabase
            current_time = datetime.now().isoformat()
            # If description is empty or starts with "Custom dataset with", store empty to let get_datasets generate a better one
            final_description = dataset_description
            if not dataset_description or dataset_description.startswith("Custom dataset with"):
                final_description = ""
            
            # Update collection metadata in Supabase
            supabase_client.update_collection_metadata(
                dataset_name,
                description=final_description,
                author=dataset_author,
                topic=dataset_topic,
                linkedin_url=dataset_linkedin,
                custom_instructions=dataset_custom_instructions,
                created_at=current_time,
                last_update_date=current_time,
                document_count=total_documents
            )
            
            # Small delay to ensure metadata is written to disk
            time.sleep(0.1)
            
            yield json.dumps({"progress": 1.0, "status": "Processing complete!"}) + "\n"
            
        except Exception as e:
            print(f"Error in upload_documents: {str(e)}")
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
    
    return Response(stream_with_context(generate_progress()), mimetype='application/x-ndjson')

# Folder API routes
@app.route('/api/folders', methods=['GET'])
def list_folders_route():
    """List all folders."""
    try:
        folders = chat_storage.list_folders()
        return jsonify(folders)
    except Exception as e:
        print(f"Error listing folders: {e}")
        return jsonify([])

@app.route('/api/folders', methods=['POST'])
def create_folder_route():
    """Create a new folder."""
    try:
        data = request.json
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': 'Missing folder name'}), 400
        folder_id = chat_storage.create_folder(name)
        return jsonify({'success': True, 'id': folder_id, 'name': name})
    except Exception as e:
        print(f"Error creating folder: {e}")
        return jsonify({'error': 'Failed to create folder'}), 500

@app.route('/api/folders/<folder_id>', methods=['PUT'])
def rename_folder_route(folder_id):
    """Rename a folder."""
    data = request.json
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({'error': 'Missing new folder name'}), 400

    # Update folder
    success = chat_storage.update_folder(folder_id, new_name)
    if not success:
        return jsonify({'error': 'Folder not found'}), 404

    return jsonify({'success': True})

@app.route('/api/folders/<folder_id>', methods=['DELETE'])
def delete_folder_route(folder_id):
    """Delete a folder and move its chats to the default folder."""
    if folder_id == 'default':
        return jsonify({'error': 'Cannot delete the default folder'}), 400

    # First, move all chats in this folder to the default folder
    chats = chat_storage.list_chats()
    for chat in chats:
        if chat.get('folder_id') == folder_id:
            chat_storage.move_chat_to_folder(chat['id'], 'default')

    # Then delete the folder
    success = chat_storage.delete_folder(folder_id)
    if not success:
        return jsonify({'error': 'Folder not found'}), 404

    return jsonify({'success': True})

# Chat API routes
@app.route('/api/chats', methods=['GET'])
def list_chats():
    """List all chats with message counts and proper timestamps."""
    try:
        folder_id = request.args.get('folder_id')
        chats = chat_storage.list_chats(folder_id)
    except Exception as e:
        print(f"Error listing chats: {e}")
        return jsonify([])
    
    # Enhance each chat with message count and format timestamps
    enhanced_chats = []
    for chat in chats:
        # Get message count for this chat
        try:
            messages = chat_storage.client.table("messages").select("id").eq("chat_id", chat["id"]).execute()
            message_count = len(messages.data) if messages.data else 0
        except Exception as e:
            print(f"Error getting message count for chat {chat['id']}: {e}")
            message_count = 0
        
        # Format timestamp - convert to Unix timestamp for JavaScript
        try:
            from datetime import datetime
            import time
            
            # Parse the ISO timestamp from Supabase
            if chat.get('updated_at'):
                dt = datetime.fromisoformat(chat['updated_at'].replace('Z', '+00:00'))
                updated_timestamp = int(dt.timestamp())
            else:
                updated_timestamp = int(time.time())  # Current time as fallback
                
            # Also format created_at
            if chat.get('created_at'):
                dt = datetime.fromisoformat(chat['created_at'].replace('Z', '+00:00'))
                created_timestamp = int(dt.timestamp())
            else:
                created_timestamp = updated_timestamp
        except Exception as e:
            print(f"Error formatting timestamp for chat {chat['id']}: {e}")
            import time
            updated_timestamp = int(time.time())
            created_timestamp = updated_timestamp
        
        # Create enhanced chat object
        enhanced_chat = {
            **chat,
            'message_count': message_count,
            'updated_at': updated_timestamp,
            'created_at': created_timestamp
        }
        enhanced_chats.append(enhanced_chat)
    
    return jsonify(enhanced_chats)

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat."""
    try:
        data = request.json
        title = data.get('title')
        folder_id = data.get('folder_id', 'default')
        dataset = data.get('dataset', DEFAULT_DATASET_NAME)
        
        chat_data = chat_storage.create_chat(title, folder_id, dataset)
        
        return jsonify({"id": chat_data["id"]})
    except Exception as e:
        print(f"Error creating chat: {e}")
        return jsonify({"error": "Failed to create chat"}), 500

@app.route('/api/chats/<chat_id>/move', methods=['POST'])
def move_chat_to_folder_route(chat_id):
    """Move a chat to a specified folder."""
    data = request.json
    folder_id = data.get('folder_id')

    if folder_id is None:
        return jsonify({"error": "Missing folder_id"}), 400

    # Optional: Check if the target folder exists (though chat_storage.move currently doesn't)
    # folders = chat_storage.list_folders()
    # if not any(f['id'] == folder_id for f in folders):
    #     return jsonify({"error": f"Target folder {folder_id} not found"}), 404

    success = chat_storage.move_chat_to_folder(chat_id, folder_id)
    if not success:
        # This implies chat_id was not found by chat_storage.update_chat
        return jsonify({"error": f"Chat {chat_id} not found or failed to update."}), 404 
    
    return jsonify({"success": True, "message": f"Chat {chat_id} moved to folder {folder_id}"})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a chat by ID."""
    # Validate chat_id format
    if not chat_id or chat_id == '[object Object]' or chat_id == 'undefined':
        return jsonify({"error": "Invalid chat ID format"}), 400
    
    try:
        chat = chat_storage.get_chat(chat_id)
        if chat is None:
            return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        print(f"Error getting chat {chat_id}: {e}")
        return jsonify({"error": "Failed to retrieve chat. Please try again."}), 500
    
    # Validate and sanitize messages to ensure they're correctly formatted
    if 'messages' in chat:
        # Log message count for debugging
        print(f"Chat {chat_id} has {len(chat['messages'])} messages")
        
        # Filter out any messages with missing/invalid content
        valid_messages = []
        for msg in chat['messages']:
            # Ensure messages have required fields
            if 'role' not in msg or 'content' not in msg or not msg['content']:
                print(f"Skipping invalid message in chat {chat_id}: {msg}")
                continue
                
            # Ensure role is valid
            if msg['role'] not in ['user', 'assistant', 'system']:
                print(f"Invalid role in message: {msg['role']}")
                msg['role'] = 'system'  # Default to system for invalid roles
                
            # Ensure assistant messages have proper formatting
            if msg['role'] == 'assistant' and not msg.get('metadata', {}).get('isSystem', False):
                # Check if AI response has the expected format
                has_sources = "SOURCES:" in msg['content']
                has_analysis = "ANALYSIS:" in msg['content']
                
                # If message doesn't have expected format, log it
                if not (has_sources or has_analysis):
                    print(f"Assistant message lacks expected sections: {msg['content'][:100]}...")
            
            valid_messages.append(msg)
            
        chat['messages'] = valid_messages
    
    return jsonify(chat)

@app.route('/api/chats/<chat_id>', methods=['PUT'])
def update_chat_route(chat_id):
    """Update a chat."""
    data = request.json
    success = chat_storage.update_chat(chat_id, **data)
    
    if not success:
        return jsonify({"error": "Chat not found"}), 404
    
    return jsonify({"success": True})

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat and return the next most recent chat."""
    result = chat_storage.delete_chat(chat_id)
    
    if not result:
        return jsonify({"error": "Failed to delete chat"}), 404
    
    # Get the next most recent chat
    all_chats = chat_storage.list_chats()
    next_chat = all_chats[0] if all_chats else None
    
    return jsonify({
        "success": True,
        "next_chat": next_chat,
        "message": "Chat deleted successfully"
    })

@app.route('/api/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    """Add a message to a chat and stream the response."""
    # Capture request data before streaming
    data = request.json
    print(f"[CHAT_API] Request data: {data}")
    user_message = data.get('message', '')
    dataset_name = data.get('dataset')
    model_name = data.get('model', MODEL_NAME)  # Get the selected model
    
    print("[CHAT_API] Attempting to get chat from storage...")
    chat = chat_storage.get_chat(chat_id)
    if chat is None:
        print(f"[CHAT_API] Chat {chat_id} not found.")
        return jsonify({"error": "Chat not found"}), 404
    print(f"[CHAT_API] Chat {chat_id} retrieved.")
    
    # Use the chat's dataset if not specified, fallback to default dataset
    if not dataset_name:
        dataset_name = chat.get('dataset', DEFAULT_DATASET_NAME or 'Default')
    
    # Validate that the dataset exists
    available_datasets = []
    if supabase_client:
        try:
            collections = supabase_client.list_collections()
            available_datasets = [c["name"] for c in collections]
        except Exception as e:
            print(f"[CHAT_API] Error getting datasets: {e}")
    
    # If the dataset doesn't exist, use the first available one or DEFAULT_DATASET_NAME
    if dataset_name not in available_datasets:
        print(f"[CHAT_API] Dataset '{dataset_name}' not found in available datasets: {available_datasets}")
        if available_datasets:
            dataset_name = available_datasets[0]
            print(f"[CHAT_API] Using first available dataset: '{dataset_name}'")
        elif DEFAULT_DATASET_NAME:
            dataset_name = DEFAULT_DATASET_NAME
            print(f"[CHAT_API] Using default dataset: '{dataset_name}'")
        else:
            return jsonify({"error": "No datasets available. Please create a dataset first."}), 400
    
    # Update the chat's dataset if it changed
    if dataset_name != chat.get('dataset'):
        chat_storage.update_chat(chat_id, **{"dataset": dataset_name})
    
    # Update the model if it changed
    if model_name != chat.get('model', MODEL_NAME):
        chat_storage.update_chat(chat_id, **{"model": model_name})
    
    # Load dataset-specific custom instructions from Supabase
    custom_instructions_for_dataset = ""
    try:
        collection = supabase_client.get_collection(dataset_name)
        if collection and hasattr(collection, 'collection_data'):
            metadata = collection.collection_data.get('metadata', {})
            custom_instructions_for_dataset = metadata.get("custom_instructions", "")
    except Exception as e:
        print(f"[CHAT_API] Error loading custom instructions for {dataset_name}: {e}")
    
    # Add user message
    print("[CHAT_API] Attempting to add user message to storage...")
    chat_storage.add_message(chat_id, "user", user_message)
    print("[CHAT_API] User message added to storage.")
    
    # Add initial assistant message for streaming
    chat_storage.add_message(chat_id, "assistant", "Generating response...", {"streaming": True})
    
    # Extract chat history for context
    history = []
    
    # Add user and assistant messages
    for msg in chat.get('messages', []):
        if msg['role'] in ['user', 'assistant']:
            history.append({"role": msg['role'], "content": msg['content']})
    
    # Import the document processor for advanced querying
    from app.utils.document_processor import LegalDocumentProcessor
    
    # Initialize document processor with the same settings as configured globally
    doc_processor = LegalDocumentProcessor(
        embedding_model=EMBEDDING_MODEL,
        device=device
    )
    
    # Use advanced query with hybrid search and reranking
    results = doc_processor.query_dataset(
        dataset_name=dataset_name,
        query=user_message,
        n_results=8,  # Increased from 5 to 8 for better context
        use_hybrid_search=True,
        use_reranking=True
    )
    
    # Format the context from retrieved documents
    context = ""
    if results and results['documents']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            source = meta.get("source", "Unknown")
            page = meta.get("page", meta.get("part", 0))
            context += f"Source: {source} (Page {page})\n{doc}\n\n"
    
    # Create enhanced prompt for LLM with Chain of Thought reasoning
    if custom_instructions_for_dataset:
        # Use ONLY custom instructions as the complete system prompt
        system_message = custom_instructions_for_dataset
        print(f"[CHAT_API] Using ONLY custom instructions as system prompt for dataset {dataset_name}")
    else:
        # Use default system message with context
        system_message = f"""You are a legal expert.
        
        # Instructions
        - Use the provided context to analyze the user's question thoroughly
        - Implement chain-of-thought reasoning by breaking down your analysis step by step
        - First carefully examine the relevant sections from the provided context
        - Think about what legal principles apply to this situation
        - Consider multiple perspectives and interpretations if applicable
        - Draw connections between different parts of the context
        - Formulate a comprehensive and legally sound analysis
        - Cite specific articles, sections, or provisions when possible
        - Clearly separate your reasoning process from your final conclusion
        - If you don't know the answer or it's not in the context, state this clearly
        
        # Output Format
        Structure your response with these sections, using **bold** formatting for section titles:
        1. **SOURCES**: A brief bulleted list of the most relevant source documents you're drawing from
        2. **ANALYSIS**: Your step-by-step reasoning about the question (this should be detailed)
        3. **APPLICABLE PROVISIONS**: Specific articles, sections, or legal provisions that apply
        4. **CONCLUSION**: Your final answer based on the analysis
        
        Important: Always format section headers in bold using **TITLE**: format
        
        # Context
        {context}
        """
    
    def generate():
        # Initialize variables in the outer scope
        full_response = ""
        saved_response_length = 0
        
        # Prepare messages for the LLM
        messages = []
        
        # Add system message first
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        # Add chat history (excluding system messages)
        for msg in history:
            if msg['role'] != 'system':  # Skip system messages as we already added it
                messages.append(msg)
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Stream the LLM response
        for chunk in llm_client.stream_with_rag(
            query=user_message,
            context=context,  # Pass the actual context here
            chat_history=messages,
            temperature=0.3
        ):
            print(f"[CHAT_API] In generate(): Received chunk: {chunk[:50]}...")
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            if len(full_response) - saved_response_length > 200:
                update_streaming_message(chat_id, full_response)
                saved_response_length = len(full_response)
                print(f"[CHAT_API] In generate(): Updated streaming message, now at {saved_response_length} chars")
        
        print("[CHAT_API] In generate(): Finished iterating llm_client.stream_with_rag")
        # Yield a completion event 
        yield f"data: {json.dumps({'done': True})}\n\n"
        
        # Make sure to save the final complete response with bold formatting
        formatted_response = format_response_with_bold_headers(full_response)
        finalize_message(chat_id, formatted_response)
    
    print("[CHAT_API] Returning streaming response object.")
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/save-last-chat', methods=['POST'])
def save_last_chat():
    """Save the last active chat ID in the user's session."""
    data = request.json
    chat_id = data.get('chat_id')
    
    if not chat_id:
        return jsonify({"error": "No chat ID provided"}), 400
    
    # Make sure session is permanent
    session.permanent = True
    
    # Save the chat ID in the session
    session['last_active_chat'] = chat_id
    
    return jsonify({"success": True})

# Authentication, user management, and security routes

# Helper function to get current user and verify permissions
def get_current_user():
    """Get the current authenticated user."""
    try:
        # Verify JWT token is valid
        verify_jwt_in_request()
        
        # Get user ID from JWT token
        user_id = get_jwt_identity()
        
        # Get user from database
        user = user_manager.get_user(user_id)
        
        return user
    except Exception:
        return None

def check_permission(permission):
    """Check if the current user has the specified permission."""
    user = get_current_user()
    
    if not user:
        return False
    
    role = user.get("role")
    if not role:
        return False
    
    permissions = ROLE_PERMISSIONS.get(role, [])
    return permission in permissions

# Authentication middleware for routes
def auth_required(permission=None):
    """Decorator for routes that require authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = get_current_user()
            
            if not user:
                return jsonify({"error": "Authentication required"}), 401
            
            # Check if user has the required permission
            if permission and not check_permission(permission):
                return jsonify({"error": "Permission denied"}), 403
            
            # Add user to flask.g
            g.user = user
            
            # Log the access
            audit_logger.log_access(
                user_id=user.get("id"),
                resource_id=request.path,
                resource_type="api",
                action=request.method.lower(),
                status="success",
                ip_address=request.remote_addr,
                session_id=request.cookies.get('session')
            )
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Auth routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.json
    
    # Validate required fields
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data or not data[field]:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # Extract fields
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Create user
    user_id = user_manager.create_user(username, email, password)
    
    if not user_id:
        return jsonify({"error": "Username or email already exists"}), 400
    
    # Log the registration
    audit_logger.log_authentication(
        user_id=user_id,
        action="register",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session')
    )
    
    return jsonify({
        "success": True,
        "message": "User registered successfully"
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Log in a user."""
    data = request.json
    
    # Validate required fields
    if 'username_or_email' not in data or not data['username_or_email']:
        return jsonify({"error": "Username or email is required"}), 400
    
    if 'password' not in data or not data['password']:
        return jsonify({"error": "Password is required"}), 400
    
    # Extract fields
    username_or_email = data.get('username_or_email')
    password = data.get('password')
    
    # Authenticate user
    success, user, error_message = user_manager.authenticate(username_or_email, password)
    
    if not success:
        # Log failed login attempt
        audit_logger.log_authentication(
            user_id=None,
            action="login",
            status="failure",
            ip_address=request.remote_addr,
            session_id=request.cookies.get('session'),
            details={"error": error_message, "attempted_login": username_or_email}
        )
        
        return jsonify({"error": error_message}), 401
    
    # Generate tokens
    tokens = user_manager.generate_tokens(user["id"])
    
    # Log successful login
    audit_logger.log_authentication(
        user_id=user["id"],
        action="login",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session')
    )
    
    # Return user data and tokens
    return jsonify({
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "profile": user.get("profile", {})
        },
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"]
    })

@app.route('/api/auth/refresh', methods=['POST'])
def refresh_token():
    """Refresh the access token using a refresh token."""
    try:
        # Get the refresh token from the request
        refresh_token = request.json.get('refresh_token')
        
        if not refresh_token:
            return jsonify({"error": "Refresh token is required"}), 400
        
        try:
            # Decode the refresh token to get the user ID
            token_data = decode_token(refresh_token)
            user_id = token_data["sub"]
            
            # Check if the user exists and is active
            user = user_manager.get_user(user_id)
            
            if not user or not user.get("is_active", False):
                return jsonify({"error": "User not found or inactive"}), 401
            
            # Generate a new access token
            tokens = user_manager.generate_tokens(user_id)
            
            # Log token refresh
            audit_logger.log_authentication(
                user_id=user_id,
                action="token_refresh",
                status="success",
                ip_address=request.remote_addr,
                session_id=request.cookies.get('session')
            )
            
            return jsonify({
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"]
            })
            
        except Exception as e:
            return jsonify({"error": "Invalid refresh token"}), 401
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@auth_required()
def logout():
    """Log out a user by blacklisting their tokens."""
    # In a real implementation, we would add the tokens to a blacklist
    # For now, we'll just log the logout
    user = g.user
    
    audit_logger.log_authentication(
        user_id=user.get("id"),
        action="logout",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session')
    )
    
    return jsonify({"success": True})

# User management routes
@app.route('/api/users/me', methods=['GET'])
@auth_required()
def get_current_user_profile():
    """Get the current user's profile."""
    user = g.user
    
    # Get user credits
    credits = credit_system.get_user_balance(user["id"])
    
    return jsonify({
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "credits": credits,
        "profile": user.get("profile", {})
    })

@app.route('/api/users/me', methods=['PUT'])
@auth_required()
def update_current_user_profile():
    """Update the current user's profile."""
    user = g.user
    data = request.json
    
    # Only allow updating specific fields
    allowed_fields = ['profile']
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    
    # Update user
    success = user_manager.update_user(user["id"], update_data)
    
    if not success:
        return jsonify({"error": "Failed to update user"}), 500
    
    return jsonify({"success": True})

@app.route('/api/users/change-password', methods=['POST'])
@auth_required()
def change_password():
    """Change the current user's password."""
    user = g.user
    data = request.json
    
    if 'current_password' not in data or not data['current_password']:
        return jsonify({"error": "Current password is required"}), 400
    
    if 'new_password' not in data or not data['new_password']:
        return jsonify({"error": "New password is required"}), 400
    
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    # Change password
    success, error_message = user_manager.change_password(user["id"], current_password, new_password)
    
    if not success:
        return jsonify({"error": error_message}), 400
    
    # Log password change
    audit_logger.log_data_event(
        user_id=user["id"],
        resource_id=user["id"],
        resource_type="user",
        action="change_password",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session')
    )
    
    return jsonify({"success": True})

# Secure document routes
@app.route('/api/secure-documents/upload', methods=['POST'])
@auth_required("write")
def upload_secure_document():
    """Upload and process a document securely."""
    user = g.user
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Get dataset name from request
    dataset_name = request.form.get('dataset', 'Default')
    
    # Ensure we have enough credits
    processing_cost = 5  # Example: 5 credits per document
    if not credit_system.check_can_afford(user["id"], processing_cost):
        return jsonify({"error": "Insufficient credits"}), 402  # Payment Required
    
    try:
        # Process document in memory without saving to disk
        result = secure_processor.memoryless_processing(
            file_obj=file.stream,
            dataset_name=dataset_name,
            original_filename=file.filename,
            user_id=user["id"]
        )
        
        # Deduct credits for document processing
        success, new_balance = credit_system.deduct_usage(
            user_id=user["id"],
            amount=processing_cost,
            feature="document_processing",
            description=f"Processing document: {file.filename}"
        )
        
        # Log document upload
        audit_logger.log_data_event(
            user_id=user["id"],
            resource_id=result["encrypted_id"],
            resource_type="document",
            action="upload",
            status="success",
            ip_address=request.remote_addr,
            session_id=request.cookies.get('session'),
            details={"dataset": dataset_name, "filename": file.filename}
        )
        
        return jsonify({
            "status": "success",
            "message": "Document processed securely",
            "document_id": result["encrypted_id"],
            "dataset": dataset_name,
            "credits_used": processing_cost,
            "credits_remaining": new_balance
        })
        
    except Exception as e:
        # Log error
        audit_logger.log_exception(
            exception=e,
            user_id=user["id"],
            resource_type="document",
            action="upload",
            ip_address=request.remote_addr,
            session_id=request.cookies.get('session')
        )
        
        return jsonify({"error": f"Error processing document: {str(e)}"}), 500

@app.route('/api/secure-documents/<document_id>', methods=['GET'])
@auth_required("read")
def get_secure_document(document_id):
    """Get temporary access to a secure document."""
    user = g.user
    
    try:
        # Get secure access to the document
        access = secure_processor.get_document_securely(
            encrypted_id=document_id,
            user_id=user["id"],
            max_age_seconds=300  # 5 minutes
        )
        
        # Log document access
        audit_logger.log_access(
            user_id=user["id"],
            resource_id=document_id,
            resource_type="document",
            action="access",
            status="success",
            ip_address=request.remote_addr,
            session_id=request.cookies.get('session')
        )
        
        return jsonify({
            "access_token": access["access_token"],
            "original_name": access["original_name"],
            "mime_type": access["mime_type"],
            "expires_at": access["expires_at"]
        })
        
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
        
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
        
    except Exception as e:
        return jsonify({"error": f"Error accessing document: {str(e)}"}), 500

@app.route('/api/secure-documents/stream/<access_token>', methods=['GET'])
def stream_secure_document(access_token):
    """Stream a secure document using an access token."""
    try:
        # Stream the document
        file, filename, mimetype = secure_processor.stream_document_securely(access_token)
        
        # Set up response
        response = Response(
            stream_with_context(file.read()),
            mimetype=mimetype
        )
        
        # Set content disposition header to make the browser download the file
        response.headers["Content-Disposition"] = f'inline; filename="{filename}"'
        
        # Make sure to close the file when the response is complete
        @after_this_request
        def cleanup(response):
            file.close()
            return response
        
        return response
        
    except PermissionError:
        return jsonify({"error": "Invalid or expired access token"}), 403
        
    except Exception as e:
        return jsonify({"error": f"Error streaming document: {str(e)}"}), 500

@app.route('/api/secure-documents/<document_id>', methods=['DELETE'])
@auth_required("delete")
def delete_secure_document(document_id):
    """Delete a secure document."""
    user = g.user
    
    try:
        # Delete the document
        success = secure_processor.delete_document_securely(
            encrypted_id=document_id,
            user_id=user["id"]
        )
        
        if not success:
            return jsonify({"error": "Document not found"}), 404
        
        # Log document deletion
        audit_logger.log_data_event(
            user_id=user["id"],
            resource_id=document_id,
            resource_type="document",
            action="delete",
            status="success",
            ip_address=request.remote_addr,
            session_id=request.cookies.get('session')
        )
        
        return jsonify({"success": True})
        
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
        
    except Exception as e:
        return jsonify({"error": f"Error deleting document: {str(e)}"}), 500

# Credit system routes
@app.route('/api/credits/balance', methods=['GET'])
@auth_required()
def get_credit_balance():
    """Get the current user's credit balance."""
    user = g.user
    
    # Get user credits
    credits = credit_system.get_user_balance(user["id"])
    
    return jsonify({
        "credits": credits
    })

@app.route('/api/credits/history', methods=['GET'])
@auth_required()
def get_credit_history():
    """Get the current user's credit transaction history."""
    user = g.user
    
    # Get transaction history
    history = credit_system.get_transaction_history(user["id"])
    
    return jsonify(history)

@app.route('/api/credits/packages', methods=['GET'])
def get_credit_packages():
    """Get available credit packages for purchase."""
    # Get credit packages
    packages = credit_system.get_credit_packages()
    
    return jsonify(packages)

@app.route('/api/credits/purchase', methods=['POST'])
@auth_required()
def purchase_credits():
    """Purchase credits."""
    user = g.user
    data = request.json
    
    if 'package_id' not in data:
        return jsonify({"error": "Package ID is required"}), 400
    
    package_id = data.get('package_id')
    payment_ref = data.get('payment_ref')  # Optional payment reference
    
    # Purchase the package
    success, message, credits_added = credit_system.purchase_credit_package(
        user_id=user["id"],
        package_id=package_id,
        payment_ref=payment_ref
    )
    
    if not success:
        return jsonify({"error": message}), 400
    
    # Get new balance
    new_balance = credit_system.get_user_balance(user["id"])
    
    # Log the purchase
    audit_logger.log_data_event(
        user_id=user["id"],
        resource_id=package_id,
        resource_type="credit_package",
        action="purchase",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session'),
        details={"credits_added": credits_added, "payment_ref": payment_ref}
    )
    
    return jsonify({
        "success": True,
        "message": message,
        "credits_added": credits_added,
        "new_balance": new_balance
    })

# Feedback routes
@app.route('/api/user-feedback', methods=['POST'])
def submit_user_feedback():
    """Submit feedback."""
    data = request.json
    
    # Try to get user ID from token if available
    try:
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
    except Exception:
        user_id = None
    
    # Check required fields
    if 'feedback_type' not in data:
        return jsonify({"error": "Feedback type is required"}), 400
    
    if 'content' not in data:
        return jsonify({"error": "Feedback content is required"}), 400
    
    feedback_type = data.get('feedback_type')
    content = data.get('content')
    rating = data.get('rating')
    
    # Validate the feedback
    is_valid, error_message = feedback_manager.validate_feedback(content, feedback_type, rating)
    
    if not is_valid:
        return jsonify({"error": error_message}), 400
    
    # Sanitize the content
    sanitized_content = feedback_manager.sanitize_feedback(content)
    
    # Submit the feedback
    feedback_id = feedback_manager.submit_feedback(
        user_id=user_id,
        feedback_type=feedback_type,
        content=sanitized_content,
        rating=rating,
        metadata=data.get('metadata')
    )
    
    # Log the feedback submission
    audit_logger.log_data_event(
        user_id=user_id,
        resource_id=feedback_id,
        resource_type="feedback",
        action="submit",
        status="success",
        ip_address=request.remote_addr,
        session_id=request.cookies.get('session')
    )
    
    return jsonify({
        "success": True,
        "feedback_id": feedback_id
    })

# --- DATASET DOCUMENT MANAGEMENT ENDPOINTS ---
from flask import send_from_directory

@app.route('/api/datasets/<dataset_name>/documents', methods=['GET'])
def list_dataset_documents(dataset_name):
    """List all documents in a dataset (by Supabase collection metadata)."""
    print(f"[DEBUG] list_dataset_documents called for: {dataset_name}")
    try:
        if not supabase_client:
            print("[DEBUG] No supabase_client available")
            return jsonify([]), 200
            
        collection = supabase_client.get_collection(name=dataset_name)
        if not collection:
            return jsonify({"documents": [], "error": "Collection not found"}), 404
            
        results = collection.get(include=["metadatas"])  # Only metadatas, ids always returned
        total_chunks = len(results.get("ids", []))
        print(f"[DEBUG] Found {total_chunks} total chunks in collection {dataset_name}")
        print(f"[DEBUG] Supabase collection metadatas for {dataset_name}:")
        for meta, doc_id in zip(results["metadatas"], results["ids"]):
            print(f"  id={doc_id} meta={meta}")
        docs = {}
        for meta, doc_id in zip(results["metadatas"], results["ids"]):
            filename = meta.get("source") or meta.get("file") or meta.get("filename")
            if not filename:
                print(f"[DEBUG] Skipping chunk {doc_id} - no filename in metadata: {meta}")
                continue
            if filename not in docs:
                docs[filename] = {"filename": filename, "ids": [], "id": filename}
            docs[filename]["ids"].append(doc_id)
        doc_list = list(docs.values())
        print(f"[DEBUG] Processed {len(doc_list)} unique documents from {total_chunks} chunks")
        if not doc_list:
            # Return empty array to match expected format
            return jsonify([]), 200
        return jsonify(doc_list)
    except Exception as e:
        print(f"Error listing documents for dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 200

@app.route('/api/datasets/<dataset_name>/documents', methods=['POST'])
def upload_documents_to_dataset(dataset_name):
    """Upload documents to an existing dataset (append to Chroma collection)."""
    def generate_progress():
        try:
            if 'files' not in request.files:
                yield json.dumps({"status": "error", "message": "No files uploaded"}) + "\n"
                return

            # Get form data (metadata fields)
            dataset_description = request.form.get('dataset_description', '')
            dataset_author = request.form.get('dataset_author', '')
            dataset_topic = request.form.get('dataset_topic', '')
            dataset_linkedin = request.form.get('dataset_linkedin', '')
            dataset_custom_instructions = request.form.get('dataset_custom_instructions', '')
            
            # Save uploaded files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], timestamp)
            os.makedirs(upload_path, exist_ok=True)
            uploaded_files = request.files.getlist('files')
            saved_files = []
            
            # Initial progress update
            yield json.dumps({"progress": 0.0, "status": "Starting upload..."}) + "\n"
            time.sleep(0.5)  # Simulate processing delay
            
            for file in uploaded_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(upload_path, filename)
                    file.save(file_path)
                    saved_files.append(filename)
            
            if not saved_files:
                yield json.dumps({"status": "error", "message": "No valid files uploaded"}) + "\n"
                return

            # Get collection
            collection = supabase_client.get_or_create_collection(dataset_name)
            if not collection:
                yield json.dumps({"status": "error", "message": "Database connection not available"}) + "\n"
                return
            
            # Process and add to Supabase
            documents = []
            metadatas = []
            ids = []
            total_documents = len(saved_files)  # This is the number of actual files uploaded
            
            # Get existing metadata from Supabase
            current_metadata = {}
            existing_doc_count = 0
            try:
                existing_collection = supabase_client.get_collection(dataset_name)
                if existing_collection and hasattr(existing_collection, 'collection_data'):
                    current_metadata = existing_collection.collection_data.get('metadata', {})
                    existing_doc_count = current_metadata.get("document_count", 0)
            except Exception as e:
                print(f"Could not retrieve existing metadata for {dataset_name}: {e}")
            
            # Update progress for file processing
            yield json.dumps({"progress": 0.2, "status": "Processing files..."}) + "\n"
            time.sleep(0.5)  # Simulate processing delay
            
            # Keep track of unique documents
            unique_documents = set()
            
            for i, doc_file in enumerate(saved_files):
                file_path = os.path.join(upload_path, doc_file)
                try:
                    if doc_file.endswith('.pdf'):
                        extracted_texts = robust_extract_text_from_pdf(file_path)
                        for j, text in enumerate(extracted_texts):
                            if text and len(text) > 50:
                                documents.append(text)
                                metadatas.append({"source": doc_file, "page": j + 1})
                                ids.append(f"{doc_file}_{i}_{j}")
                                unique_documents.add(doc_file)  # Add to unique documents set
                    elif doc_file.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                        chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
                        for j, chunk in enumerate(chunks):
                            if len(chunk) > 50:
                                documents.append(chunk)
                                metadatas.append({"source": doc_file, "page": 1})
                                ids.append(f"{doc_file}_{i}_{j}")
                                unique_documents.add(doc_file)  # Add to unique documents set
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                
                # Update progress during file processing
                progress = 0.2 + (0.4 * (i + 1) / total_documents)
                yield json.dumps({"progress": progress, "status": f"Processing file {i+1} of {total_documents}..."}) + "\n"
                time.sleep(0.5)  # Simulate processing delay
            
            # Update progress for Chroma processing
            yield json.dumps({"progress": 0.6, "status": "Adding to database..."}) + "\n"
            time.sleep(0.5)  # Simulate processing delay
            
            # Add to Supabase in batches with embeddings
            total_batches = (len(documents) + 99) // 100
            for i in range(0, len(documents), 100):
                batch_docs = documents[i:i+100]
                batch_meta = metadatas[i:i+100]
                batch_ids = ids[i:i+100]
                
                # Generate embeddings for the batch with SSL error handling
                try:
                    batch_embeddings = get_embeddings().embed_documents(batch_docs)
                except Exception as e:
                    if "EOF occurred in violation of protocol" in str(e) or "SSL" in str(e):
                        print(f"SSL error generating embeddings, retrying individually with delay: {e}")
                        batch_embeddings = []
                        for doc in batch_docs:
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    time.sleep(1)  # Wait 1 second between attempts
                                    doc_embedding = get_embeddings().embed_documents([doc])
                                    batch_embeddings.extend(doc_embedding)
                                    break
                                except Exception as retry_e:
                                    if attempt == max_retries - 1:
                                        print(f"Failed to generate embedding for document after {max_retries} attempts: {retry_e}")
                                        # Use zero vector as fallback
                                        batch_embeddings.append([0.0] * 1024)  # 1024-dim embeddings for BAAI/bge-large-en-v1.5
                                    else:
                                        print(f"Retry {attempt + 1}/{max_retries} failed: {retry_e}")
                                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
                
                # Add documents to collection with error handling
                try:
                    success = collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                    if not success:
                        error_msg = f"Failed to add batch {i//100 + 1} to database"
                        print(error_msg)
                        yield json.dumps({"status": "error", "message": error_msg}) + "\n"
                        return
                except Exception as add_error:
                    error_msg = f"Error adding batch {i//100 + 1} to database: {str(add_error)}"
                    print(error_msg)
                    yield json.dumps({"status": "error", "message": error_msg}) + "\n"
                    return
                
                # Update progress during batch processing
                batch_progress = 0.6 + (0.3 * (i // 100 + 1) / total_batches)
                yield json.dumps({"progress": batch_progress, "status": f"Processing batch {i//100 + 1} of {total_batches}..."}) + "\n"
                time.sleep(0.5)  # Simulate processing delay
            
            # Update dataset metadata with the correct document count
            # Always update metadata, whether it's a new or existing dataset
            # Get current unique documents from collection
            try:
                results = collection.get(include=["metadatas"])
                current_unique_sources = set()
                
                # Debug logging
                print(f"Retrieved {len(results.get('metadatas', []))} total chunks from collection")
                
                # Count unique source files from all chunks
                if results and "metadatas" in results:
                    for meta in results["metadatas"]:
                        if meta:  # Check if metadata is not None
                            source = meta.get("source") or meta.get("file") or meta.get("filename")
                            if source:
                                current_unique_sources.add(source)
                
                print(f"Found {len(current_unique_sources)} unique source files in collection")
                
                # Note: saved_files are already added to the collection above,
                # so they should already be in current_unique_sources.
                # We don't need to add them again.
                
            except Exception as e:
                print(f"Error counting documents: {e}")
                # Fallback: count the newly uploaded files
                current_unique_sources = set(saved_files)
            
            # Update metadata in Supabase
            current_time = datetime.now().isoformat()
            
            # If description is empty or starts with "Custom dataset with", store empty to let get_datasets generate a better one
            final_description = dataset_description
            if not dataset_description or dataset_description.startswith("Custom dataset with"):
                final_description = ""
            
            # Prepare metadata update - preserve existing metadata when fields are empty
            metadata_update = {
                "document_count": len(current_unique_sources),
                "last_update_date": current_time
            }
            
            # Only update metadata fields if they are provided (not empty)
            # Otherwise, preserve existing values
            if dataset_description:
                metadata_update["description"] = final_description
            elif "description" in current_metadata:
                metadata_update["description"] = current_metadata["description"]
                
            if dataset_author:
                metadata_update["author"] = dataset_author
            elif "author" in current_metadata:
                metadata_update["author"] = current_metadata["author"]
                
            if dataset_topic:
                metadata_update["topic"] = dataset_topic
            elif "topic" in current_metadata:
                metadata_update["topic"] = current_metadata["topic"]
                
            if dataset_linkedin:
                metadata_update["linkedin_url"] = dataset_linkedin
            elif "linkedin_url" in current_metadata:
                metadata_update["linkedin_url"] = current_metadata["linkedin_url"]
                
            if dataset_custom_instructions:
                metadata_update["custom_instructions"] = dataset_custom_instructions
            elif "custom_instructions" in current_metadata:
                metadata_update["custom_instructions"] = current_metadata["custom_instructions"]
            
            # Add created_at timestamp if this is a new dataset
            if "created_at" not in current_metadata:
                metadata_update["created_at"] = current_time
            else:
                metadata_update["created_at"] = current_metadata["created_at"]
            
            # Update collection metadata in Supabase
            supabase_client.update_collection_metadata(dataset_name, **metadata_update)
            
            # Small delay to ensure metadata is written to disk
            time.sleep(0.1)
            
            # Final progress update with success message
            yield json.dumps({"progress": 1.0, "status": f"Processed {len(documents)} text chunks from {total_documents} documents. Upload complete!", "success": True}) + "\n"
            
        except Exception as e:
            print(f"Error uploading documents to dataset {dataset_name}: {e}")
            yield json.dumps({"status": "error", "message": f"Server error: {str(e)}"}) + "\n"
    
    return Response(stream_with_context(generate_progress()), mimetype='application/x-ndjson')

@app.route('/api/datasets/<dataset_name>/documents/<doc_id>', methods=['DELETE'])
def delete_document_from_dataset(dataset_name, doc_id):
    """Delete a document and all its chunks from a dataset."""
    try:
        collection = supabase_client.get_collection(name=dataset_name)
        if not collection:
            return jsonify({"success": False, "message": "Collection not found."}), 404
        # Find all ids for this document (by filename)
        results = collection.get(include=["metadatas"])  # Only metadatas, ids always returned
        ids_to_delete = [id_ for meta, id_ in zip(results["metadatas"], results["ids"]) if meta.get("source") == doc_id or meta.get("file") == doc_id or meta.get("filename") == doc_id]
        if not ids_to_delete:
            return jsonify({"success": False, "message": "No chunks found for this document."}), 404
        
        # Delete the chunks
        collection.delete(ids=ids_to_delete)
        
        # Update dataset metadata in Supabase with new document count
        # Get current unique documents from collection after deletion
        results = collection.get(include=["metadatas"])
        current_unique_sources = set()
        for meta in results["metadatas"]:
            source = meta.get("source")
            if source:
                current_unique_sources.add(source)
        
        # Update metadata in Supabase
        supabase_client.update_collection_metadata(
            dataset_name,
            document_count=len(current_unique_sources),
            last_update_date=datetime.now().isoformat()
        )
        
        return jsonify({"success": True, "message": f"Deleted document {doc_id} and its chunks."})
    except Exception as e:
        print(f"Error deleting document {doc_id} from dataset {dataset_name}: {e}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/datasets/check-name')
def check_dataset_name():
    """Check if a dataset name is already taken."""
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({"taken": False, "error": "No name provided"}), 400
    
    if supabase_client:
        try:
            supabase_client.get_collection(name)
            return jsonify({"taken": True})
        except Exception:
            return jsonify({"taken": False})
    else:
        return jsonify({"taken": False, "error": "Database connection not available"})

# Add an alias for the update endpoint to support /update
@app.route('/api/datasets/<dataset_name>/update', methods=['PUT'])
def update_dataset_metadata_route_alias(dataset_name):
    print(f"[DEBUG] Alias route hit for dataset_name: '{dataset_name}'")
    return update_dataset_metadata_route(dataset_name)

# Add these functions before the add_message route
def update_streaming_message(chat_id: str, content: str) -> None:
    """Update a streaming message in the chat storage."""
    try:
        chat = chat_storage.get_chat(chat_id)
        if chat and chat.get('messages'):
            # Find the last message (which should be the streaming one)
            last_message = chat['messages'][-1]
            if last_message['role'] == 'assistant':
                # Update message content using Supabase service
                chat_storage.update_message(
                    message_id=last_message['id'],
                    content=content,
                    metadata={'streaming': True}
                )
    except Exception as e:
        print(f"Error updating streaming message: {str(e)}")

def finalize_message(chat_id: str, content: str) -> None:
    """Finalize a streaming message in the chat storage."""
    try:
        chat = chat_storage.get_chat(chat_id)
        if chat and chat.get('messages'):
            # Find the last message (which should be the streaming one)
            last_message = chat['messages'][-1]
            if last_message['role'] == 'assistant':
                # Update message content using Supabase service
                chat_storage.update_message(
                    message_id=last_message['id'],
                    content=content,
                    metadata={'streaming': False}
                )
    except Exception as e:
        print(f"Error finalizing message: {str(e)}")

# Start background thread to load document processing libraries
print("[MAIN] Starting background thread for document processing libraries...")
import threading
background_thread = threading.Thread(target=load_document_processing_libraries, daemon=True)
background_thread.start()
print("[MAIN] Background thread started. Libraries will load while app serves requests.")
print("[MAIN] Check /api/health/document-processing for loading status.")

# Note: The application is started from run.py, not from this file directly
print("[MAIN] ============================================")
print("[MAIN] QueryLexV4 application fully loaded!")
print("[MAIN] ============================================")
print("[MAIN] Ready to serve requests.")
print("[MAIN] The app is now ready to start!")