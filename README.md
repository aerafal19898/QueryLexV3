# QueryLex V4 – Secure Legal‑RAG Platform ⚖️🤖

A secure, AI-powered legal document processing and query platform built with enterprise-grade security and compliance in mind. QueryLex V4 enables legal professionals to upload, process, and query legal documents using advanced Retrieval-Augmented Generation (RAG) technology.

---

## 🚀 What's New in V4

- **Supabase-First Architecture:** All chat, document, and dataset storage is now handled via Supabase (Postgres + pgvector). No more ChromaDB by default.
- **Enhanced AI Performance:** Improved document chunking (1024 tokens), hybrid search (vector + keyword), and semantic reranking for better accuracy.
- **Datasets as Collections:** Datasets are managed in the Supabase `collections` table. Documents are chunked and stored in the `documents` table with vector embeddings.
- **Real-time Chat:** Chat history and messages are stored in Supabase with proper timestamp handling and message counting.
- **Feedback System:** User feedback is stored in Supabase with migration from local JSON files.
- **Enhanced UI:** Bold formatting for AI response headers, improved dataset selection, and better error handling.
- **Document Processing:** Advanced legal document chunking with metadata extraction for articles, citations, and legal entities.
- **Windows Support:** Full Windows compatibility with Tesseract OCR integration.

---

## ✨ Core Features

### Document Processing
- **Advanced PDF Processing** – Multiple extraction methods (PyPDF2, Unstructured, OCR)
- **Intelligent Chunking** – Legal document-aware chunking with semantic boundaries
- **Metadata Extraction** – Automatic extraction of articles, citations, dates, and legal entities
- **OCR Support** – Tesseract OCR integration for scanned documents
- **Batch Processing** – Efficient batch processing with retry logic

### AI-Powered Search
- **Hybrid Search** – Combines vector similarity with keyword matching
- **Semantic Reranking** – Cross-encoder models for improved relevance
- **Context-Aware Responses** – 8-chunk context with optimized temperature
- **Streaming Responses** – Real-time token streaming with OpenRouter
- **Multiple LLM Support** – Integration with various language models

### User Experience
- **Real-time Chat Interface** – Interactive chat with document context
- **Dataset Management** – Organize documents into collections
- **Drag-and-Drop Upload** – Easy document upload with progress tracking
- **Source Citations** – All responses include source chunk references
- **Organized Workspace** – Folder-based organization system

### Security & Compliance
- **EU Data Locality** – All data stored in EU Supabase cluster with pgvector
- **Document Encryption** – Fernet-encrypted documents and embeddings
- **PII Protection** – Advanced PII detection and scrubbing
- **Comprehensive Audit Logging** – Track all system activities
- **GDPR Compliance** – Built-in data protection features
- **Secure Document Processing** – SSL retry logic and batch processing

---

## 🏗 Project Structure

```
QueryLexV4/
├── app/
│   ├── main.py                    # Flask API with chat, document, and dataset endpoints
│   ├── config.py                  # Configuration and environment variables
│   ├── models/                    # Data models (user, etc.)
│   │   ├── __init__.py
│   │   └── user.py
│   └── utils/
│       ├── supabase_client.py     # Supabase vector database operations
│       ├── document_processor.py  # AI-powered document chunking and embedding
│       ├── chat_service.py        # Chat management with Supabase storage
│       ├── feedback_service.py    # Feedback collection and storage
│       ├── openrouter_client.py   # LLM API integration
│       ├── credit_system.py       # Usage tracking and billing
│       ├── audit_logger.py        # Security and compliance logging
│       ├── encryption.py          # Document encryption utilities
│       └── secure_processor.py    # Secure document processing
├── data/                          # Local storage for backups and migration
├── static/                        # Frontend assets (CSS, JS, images)
├── templates/                     # Jinja2 HTML templates
├── tesseract_config.py           # Windows Tesseract configuration
├── generate_keys.py              # Security key generation utility
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker configuration
├── Dockerfile                    # Container configuration
└── supabase_schema.sql          # Database schema for Supabase setup
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (tested with Python 3.12)
- Node.js (for frontend dependencies)
- Git

### System Dependencies

#### Windows
```bash
# Install Tesseract OCR
winget install UB-Mannheim.TesseractOCR

# Install Poppler for PDF processing
winget install poppler
```

#### Linux (Ubuntu/Debian)
```bash
# Install Tesseract OCR
sudo apt-get install tesseract-ocr

# Install Poppler for PDF processing
sudo apt-get install poppler-utils
```

#### macOS
```bash
# Install Tesseract OCR
brew install tesseract

# Install Poppler for PDF processing
brew install poppler
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/QueryLexV4.git
cd QueryLexV4
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Node.js dependencies:**
```bash
npm install
```

4. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
# API Keys
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Security Keys (generate using generate_keys.py)
SECRET_KEY=your_flask_secret_key
JWT_SECRET_KEY=your_jwt_secret_key
DOCUMENT_ENCRYPTION_KEY=your_32_byte_encryption_key

# Database (Supabase Vector Database)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Model Configuration
MODEL_PROVIDER=openrouter
MODEL_NAME=meta-llama/llama-3.3-70b-instruct
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Feature Flags
DEBUG=false
ENCRYPTION_ENABLED=true
SECURE_PROCESSING=true
```

5. **Generate security keys:**
```bash
python generate_keys.py
```

6. **Set up Supabase:**
   - Create a Supabase project with pgvector extension
   - Run the SQL schema from `supabase_schema.sql`
   - Configure environment variables
   - See `SUPABASE_SETUP.md` for detailed instructions

7. **Start the development server:**
```bash
python app/main.py
```

The application will be available at `http://localhost:5000`

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual container
docker build -t querylex-v4 .
docker run -p 5000:5000 querylex-v4
```

---

## 📚 Usage

### Document Upload
1. Click "Upload Documents" in the interface
2. Select PDF files to upload
3. Choose or create a dataset
4. Wait for processing to complete

### Querying Documents
1. Select a dataset from the dropdown
2. Type your question in the chat interface
3. Receive AI-generated responses with source citations
4. View context chunks and metadata

### Managing Datasets
- Create new datasets for different document collections
- Delete datasets when no longer needed
- The system automatically selects valid datasets

---

## ⚡ Performance Optimization

### Document Processing Performance

**Document processing libraries use background preloading** with health checks and fallback mechanisms:

- **App startup:** Immediate (no delays)
- **Background loading:** Libraries load in separate thread after startup
- **Health monitoring:** `/api/health/document-processing` endpoint shows loading status
- **Fallback system:** Multiple PDF processing methods with automatic fallback

**Processing methods (in order of preference):**
1. `unstructured` library with Tesseract OCR - For comprehensive document analysis
2. `PyPDF2` - Reliable for text-based PDFs
3. `pdf2image` + `pytesseract` - OCR for image-based PDFs (fallback)

**Benefits:**
- ✅ Instant application startup
- ✅ Libraries load in background while app serves requests
- ✅ Real-time status monitoring via health check endpoint
- ✅ Automatic fallback if primary library fails
- ✅ SSL retry logic for embedding generation
- ✅ Timeout protection prevents indefinite waiting

---

## 🔒 Security Features

- **End-to-end Encryption** for all documents using Fernet
- **Supabase RLS** (Row Level Security) for data access control
- **JWT Authentication** for secure API access
- **Secure Document Processing** with SSL retry logic
- **PII Detection and Scrubbing** before storage
- **Audit Logging** for all system activities
- **Vector Search Security** with encrypted embeddings

---

## 🤖 AI Features

- **Hybrid Search:** Combines vector similarity with keyword matching
- **Semantic Reranking:** Cross-encoder models for improved relevance
- **Legal Document Processing:** Specialized chunking for legal text
- **Metadata Extraction:** Automatic extraction of articles, citations, dates
- **Streaming Responses:** Real-time token streaming with OpenRouter
- **Context-Aware:** 8-chunk context with optimized temperature (0.3)
- **Bold Headers:** Formatted responses with section highlighting

---

## 🐛 Troubleshooting

### Common Issues

**Tesseract not found on Windows:**
- Ensure Tesseract is installed: `winget install UB-Mannheim.TesseractOCR`
- Check the installation path in `tesseract_config.py`

**SSL errors during embedding generation:**
- The system includes automatic SSL retry logic
- Check your internet connection and firewall settings

**Document processing fails:**
- Verify system dependencies are installed (Tesseract, Poppler)
- Check the `/api/health/document-processing` endpoint
- Review console logs for specific error messages

**Supabase connection issues:**
- Verify your Supabase credentials in `.env`
- Ensure the pgvector extension is enabled
- Check the database schema is properly set up

---

## 📊 API Endpoints

### Document Management
- `POST /api/upload` - Upload documents
- `GET /api/datasets` - List available datasets
- `DELETE /api/datasets/<name>` - Delete a dataset

### Chat Interface
- `POST /api/chat` - Send chat message and get streaming response
- `GET /api/chats` - List user chats
- `GET /api/chats/<chat_id>` - Get chat details

### Health & Status
- `GET /api/health/document-processing` - Check processing library status
- `GET /api/health` - General health check

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black app/
flake8 app/
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Related Documentation

- [Supabase Setup Guide](SUPABASE_SETUP.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Google Cloud Run Setup](GOOGLE_CLOUD_RUN.md)

---

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

**Built with ❤️ for the legal profession**