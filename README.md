# QueryLex V4 – Secure Legal‑RAG Marketplace ⚖️🤖

A secure, EU-hosted marketplace where legal professionals can buy, sell, and chat with Retrieval‑Augmented‑Generation (RAG) models. Built with enterprise-grade security and compliance in mind.

---

## 🚀 What's New in V4
- **Supabase-First Architecture:** All chat, document, and dataset storage is now handled via Supabase (Postgres + pgvector). No more ChromaDB by default.
- **Enhanced AI Performance:** Improved document chunking (1024 tokens), hybrid search (vector + keyword), and semantic reranking for better accuracy.
- **Datasets as Collections:** Datasets are managed in the Supabase `collections` table. Documents are chunked and stored in the `documents` table with vector embeddings.
- **Real-time Chat:** Chat history and messages are stored in Supabase with proper timestamp handling and message counting.
- **Feedback System:** User feedback is stored in Supabase with migration from local JSON files.
- **Enhanced UI:** Bold formatting for AI response headers, improved dataset selection, and better error handling.
- **Document Processing:** Advanced legal document chunking with metadata extraction for articles, citations, and legal entities.
- **Version:** This is QueryLex V4 with significant architecture improvements.

---

## ✨ Core Features

### Marketplace & Creator Tools
- **Marketplace / Leaderboard** – Sort by usage and ratings (OpenRouter style)
- **Creator Wizard** – Simple upload ▶ tag ▶ price ▶ publish workflow
- **Immutable Versioning** – Document bundles are versioned with secure hashes
- **Revenue Sharing** – Creators earn 70% of revenue (30% platform fee)

### User Experience
- **Free Tier** – 3 free requests per month (rate-limited to 1 req/min)
- **Credit System** – Top-up credits (1 € = 1 000 tokens)
- **Real-time Streaming** – Chat responses stream tokens in real-time
- **Source Citations** – All responses include source chunk hashes
- **Organized Workspace** – Drag-and-drop folder organization

### Security & Compliance
- **EU Data Locality** – All data stored in EU Supabase cluster with pgvector
- **Document Encryption** – Fernet-encrypted documents and embeddings
- **PII Protection** – ≥ 95% recall/precision PII scrubber (Piiranha + regex)
- **Comprehensive Audit Logging** – Track all system activities
- **GDPR Compliance** – Built-in data protection features
- **Secure Document Processing** – SSL retry logic and batch processing for reliability

## 🏗 Project Structure

```
app/
├── main.py              # Flask API with chat, document, and dataset endpoints
├── config.py            # Configuration and environment variables
├── models/              # Data models (user, etc.)
└── utils/
    ├── supabase_client.py    # Supabase vector database operations
    ├── document_processor.py # AI-powered document chunking and embedding
    ├── chat_service.py       # Chat management with Supabase storage
    ├── feedback_service.py   # Feedback collection and storage
    ├── openrouter_client.py  # LLM API integration
    ├── credit_system.py      # Usage tracking and billing
    ├── audit_logger.py       # Security and compliance logging
    └── encryption.py         # Document encryption utilities
data/                    # Local storage for backups and migration
static/                  # Frontend assets (CSS, JS, images)
templates/               # Jinja2 HTML templates
supabase_schema.sql      # Database schema for Supabase setup
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/aerafal19898/QueryLexv2.git
cd QueryLexv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
# API Keys
OPENROUTER_API_KEY=your_key
STRIPE_SECRET_KEY=your_key
STRIPE_WEBHOOK_SECRET=your_key

# Security
SECRET_KEY=generate_secure_key
JWT_SECRET_KEY=generate_secure_key
DOCUMENT_ENCRYPTION_KEY=generate_32_byte_key

# Database (Supabase Vector Database)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_key

# Feature Flags
DEBUG=false
ENCRYPTION_ENABLED=true
SECURE_PROCESSING=true
```

4. Set up Supabase (see SUPABASE_SETUP.md for detailed instructions):
   - Create a Supabase project with pgvector extension
   - Run the SQL schema from `supabase_schema.sql`
   - Configure environment variables
   - The system will auto-create default datasets on first run

5. Start the development server:
```bash
python run.py
```

## 💳 Billing & Payouts

- **Stripe Connect** integration for secure payments
- **Automatic Payouts** to creators
- **Full Refund** capability for administrators
- **Usage-based Billing** with credit system

## 🔒 Security Features

- **End-to-end Encryption** for all documents using Fernet
- **Supabase RLS** (Row Level Security) for data access control
- **JWT Authentication** for secure API access
- **Secure Document Processing** with SSL retry logic
- **PII Detection and Scrubbing** before storage
- **Audit Logging** for all system activities
- **Vector Search Security** with encrypted embeddings

## 🤖 AI Features

- **Hybrid Search:** Combines vector similarity with keyword matching
- **Semantic Reranking:** Cross-encoder models for improved relevance
- **Legal Document Processing:** Specialized chunking for legal text
- **Metadata Extraction:** Automatic extraction of articles, citations, dates
- **Streaming Responses:** Real-time token streaming with OpenRouter
- **Context-Aware:** 8-chunk context with optimized temperature (0.3)
- **Bold Headers:** Formatted responses with section highlighting

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines before submitting pull requests.

## 📄 License

MIT License - See LICENSE file for details