# Supabase Setup Guide for QueryLex V4 🗃️

This comprehensive guide explains how to set up Supabase as the vector database for QueryLex V4, providing scalable document storage and vector search capabilities.

---

## Prerequisites

1. **Supabase Account**: Sign up at https://supabase.com
2. **Python Environment**: Python 3.8+ with updated requirements.txt installed
3. **OpenRouter API Key**: For LLM integration
4. **Basic SQL Knowledge**: For database schema setup

---

## Step 1: Create a Supabase Project

### 1.1 Project Creation

1. **Log in** to your Supabase dashboard
2. **Click "New Project"**
3. **Choose your organization**
4. **Enter project details**:
   - **Name**: `querylex-v4-vector-db` (or your preferred name)
   - **Database Password**: Generate a strong password and save it securely
   - **Region**: Choose the region closest to your users
     - For EU compliance: Select a EU region (e.g., `eu-west-1`)
     - For US users: Select a US region (e.g., `us-east-1`)
5. **Click "Create new project"**

### 1.2 Wait for Project Initialization

The project creation process takes 2-5 minutes. You'll see a progress indicator.

---

## Step 2: Enable Vector Extension

### 2.1 Enable pgvector Extension

1. **Navigate to SQL Editor** in the left sidebar
2. **Run the following command**:

```sql
-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2.2 Apply Database Schema

1. **Copy the contents** of `supabase_schema.sql` from your project
2. **Paste it into the SQL Editor**
3. **Click "Run"** to execute all commands

The schema includes:
- **collections**: Dataset/collection management
- **documents**: Document chunks with vector embeddings
- **chats**: Chat session management
- **messages**: Individual chat messages
- **folders**: Chat organization
- **feedback**: User feedback collection
- **Vector functions**: Similarity search functions

---

## Step 3: Configure Environment Variables

### 3.1 Find Your Supabase Credentials

1. **Go to your project dashboard**
2. **Navigate to "Settings" → "API"**
3. **Copy the following values**:
   - **URL**: `https://your-project-id.supabase.co`
   - **anon public key**: For client-side operations
   - **service_role key**: For server-side operations (keep secret!)

### 3.2 Update Your .env File

Add the following to your `.env` file:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-public-key
SUPABASE_SERVICE_KEY=your-service-role-key

# Security Keys (generate using generate_keys.py)
SECRET_KEY=your-flask-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
DOCUMENT_ENCRYPTION_KEY=your-32-byte-encryption-key

# API Configuration
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Model Configuration
MODEL_PROVIDER=openrouter
MODEL_NAME=meta-llama/llama-3.3-70b-instruct
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Feature Flags
DEBUG=false
ENCRYPTION_ENABLED=true
SECURE_PROCESSING=true
```

---

## Step 4: Vector Dimensions Configuration

### 4.1 Check Your Embedding Model

The schema is configured for **1024-dimensional vectors**. Verify your embedding model:

- **BAAI/bge-large-en-v1.5**: 1024 dimensions ✅ (default)
- **all-MiniLM-L6-v2**: 384 dimensions
- **text-embedding-ada-002**: 1536 dimensions

### 4.2 Update Dimensions (if needed)

If you're using a different embedding model, update the schema:

```sql
-- For 384 dimensions (all-MiniLM-L6-v2)
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(384);

-- Update the match function
DROP FUNCTION IF EXISTS match_documents;
CREATE OR REPLACE FUNCTION match_documents(
    collection_id UUID,
    query_embedding vector(384),  -- Change this number
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    FROM documents
    WHERE documents.collection_id = match_documents.collection_id
        AND 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
```

---

## Step 5: Security Configuration

### 5.1 Row Level Security (RLS)

For production deployments, configure RLS policies:

```sql
-- Enable RLS on tables
ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE folders ENABLE ROW LEVEL SECURITY;

-- Example: Public read access for collections
CREATE POLICY "Public collections access" ON collections 
FOR SELECT USING (true);

-- Example: Authenticated user access for chats
CREATE POLICY "User chats access" ON chats 
FOR ALL USING (auth.uid() = user_id);

-- Example: Admin access for all tables
CREATE POLICY "Admin full access" ON collections 
FOR ALL USING (
    auth.jwt() ->> 'role' = 'admin'
);
```

### 5.2 Service Role Permissions

Ensure your service role has the necessary permissions:

```sql
-- Grant permissions to service role
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
```

---

## Step 6: Test the Connection

### 6.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Test Database Connection

Create a test script `test_supabase.py`:

```python
import os
from dotenv import load_dotenv
from app.utils.supabase_client import SupabaseVectorClient

load_dotenv()

def test_connection():
    try:
        # Test connection
        client = SupabaseVectorClient(use_service_key=True)
        print("✅ Successfully connected to Supabase")
        
        # Test creating a collection
        collection = client.get_or_create_collection("test_collection")
        print(f"✅ Collection created/retrieved: {collection}")
        
        # Test embedding functionality
        from app.utils.document_processor import LegalDocumentProcessor
        processor = LegalDocumentProcessor("BAAI/bge-large-en-v1.5")
        print("✅ Document processor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
```

Run the test:

```bash
python test_supabase.py
```

### 6.3 Start Application

```bash
python app/main.py
```

Check the console output for:
```
Successfully connected to Supabase
Document processor initialized
```

---

## Step 7: Performance Optimization

### 7.1 Vector Indexing

For large datasets, optimize vector search performance:

```sql
-- Create HNSW index for better performance (recommended for 50k+ vectors)
CREATE INDEX CONCURRENTLY idx_documents_embedding_hnsw ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- For smaller datasets, use IVFFlat
CREATE INDEX CONCURRENTLY idx_documents_embedding_ivfflat ON documents 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Optimize metadata searches
CREATE INDEX CONCURRENTLY idx_documents_metadata_gin ON documents 
USING gin (metadata);

-- Index collection lookups
CREATE INDEX CONCURRENTLY idx_documents_collection_id ON documents (collection_id);
```

### 7.2 Connection Pooling

For high-traffic applications:

```env
# Add to your environment
SUPABASE_DB_POOL_SIZE=20
SUPABASE_DB_MAX_OVERFLOW=30
SUPABASE_DB_POOL_TIMEOUT=30
```

### 7.3 Query Optimization

Monitor and optimize slow queries:

```sql
-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Monitor slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

---

## Step 8: Monitoring and Maintenance

### 8.1 Supabase Dashboard Monitoring

Monitor the following metrics:

1. **Database Size**: Track storage usage
2. **API Requests**: Monitor request volume
3. **Performance Metrics**: Query execution times
4. **Error Rates**: Track failed requests

### 8.2 Application Health Checks

Implement health checks:

```python
# In your application
@app.route('/api/health/database')
def health_check():
    try:
        client = SupabaseVectorClient(use_service_key=True)
        # Simple query to test connection
        result = client.client.table("collections").select("id").limit(1).execute()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
```

### 8.3 Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h your-supabase-host -U postgres -d your-database > "backup_${DATE}.sql"

# Compress and store
gzip "backup_${DATE}.sql"
aws s3 cp "backup_${DATE}.sql.gz" s3://your-backup-bucket/
```

---

## Step 9: Migration from Existing Data

### 9.1 ChromaDB Migration

If migrating from ChromaDB:

```python
import chromadb
import json
from app.utils.supabase_client import SupabaseVectorClient

def migrate_from_chromadb():
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="./data/chroma")
    
    # Connect to Supabase
    supabase_client = SupabaseVectorClient(use_service_key=True)
    
    # Migrate each collection
    for collection_name in chroma_client.list_collections():
        print(f"Migrating {collection_name}...")
        
        # Get ChromaDB collection
        chroma_collection = chroma_client.get_collection(collection_name)
        results = chroma_collection.get(include=["documents", "metadatas", "embeddings"])
        
        # Create Supabase collection
        supabase_collection = supabase_client.get_or_create_collection(collection_name)
        
        # Migrate documents in batches
        batch_size = 100
        for i in range(0, len(results["documents"]), batch_size):
            batch_docs = results["documents"][i:i+batch_size]
            batch_embeddings = results["embeddings"][i:i+batch_size]
            batch_metadatas = results["metadatas"][i:i+batch_size]
            
            # Insert into Supabase
            supabase_client.add_documents(
                collection_name=collection_name,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        print(f"✅ Migrated {len(results['documents'])} documents")
```

### 9.2 Local Data Migration

Migrate existing local data:

```python
def migrate_local_data():
    # Migrate user data
    # Migrate chat history
    # Migrate feedback data
    pass
```

---

## Step 10: Production Deployment

### 10.1 Environment Configuration

```env
# Production environment
NODE_ENV=production
FLASK_ENV=production
DEBUG=false

# Supabase Production Settings
SUPABASE_URL=https://your-prod-project.supabase.co
SUPABASE_ANON_KEY=your-prod-anon-key
SUPABASE_SERVICE_KEY=your-prod-service-key

# Security
ENCRYPTION_ENABLED=true
SECURE_PROCESSING=true
JWT_SECRET_KEY=your-strong-jwt-secret
```

### 10.2 Database Optimization

```sql
-- Production optimizations
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET pg_stat_statements.track = 'all';

-- Restart required after system changes
SELECT pg_reload_conf();
```

### 10.3 Monitoring Setup

```python
# Add to your application
import logging
from supabase import create_client

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Monitor database connections
def monitor_db_connections():
    client = create_client(supabase_url, supabase_service_key)
    result = client.rpc('get_db_stats').execute()
    return result.data
```

---

## Troubleshooting

### Common Issues

**1. "vector extension not found"**
```sql
-- Solution: Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;
```

**2. "dimension mismatch"**
```bash
# Check your embedding model dimensions
python -c "from transformers import AutoModel; print(AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').config.hidden_size)"
```

**3. "authentication failed"**
```bash
# Verify your credentials
curl -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
     -H "apikey: $SUPABASE_ANON_KEY" \
     "$SUPABASE_URL/rest/v1/collections"
```

**4. "RLS policy violation"**
```sql
-- Temporarily disable RLS for testing
ALTER TABLE collections DISABLE ROW LEVEL SECURITY;
```

**5. "Connection timeout"**
```env
# Increase timeout settings
SUPABASE_TIMEOUT=60
SUPABASE_RETRY_COUNT=3
```

### Performance Issues

**1. Slow vector search**
```sql
-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM documents 
ORDER BY embedding <=> '[0.1,0.2,...]' 
LIMIT 10;
```

**2. High memory usage**
```python
# Reduce batch sizes
EMBEDDING_BATCH_SIZE=50
DOCUMENT_CHUNK_SIZE=512
```

**3. Connection pool exhaustion**
```env
# Increase pool size
SUPABASE_DB_POOL_SIZE=50
SUPABASE_DB_MAX_OVERFLOW=100
```

---

## Security Considerations

### 1. API Key Management

```bash
# Use environment-specific keys
# Development
SUPABASE_SERVICE_KEY=your-dev-key

# Production
SUPABASE_SERVICE_KEY=your-prod-key

# Never commit keys to version control
echo "*.env" >> .gitignore
```

### 2. Network Security

```sql
-- Restrict access by IP (if needed)
CREATE POLICY "IP restriction" ON collections 
FOR ALL USING (
    inet_client_addr() << '192.168.1.0/24'::inet
);
```

### 3. Data Encryption

```python
# Encrypt sensitive data before storing
from cryptography.fernet import Fernet

def encrypt_document(content):
    key = os.environ.get('DOCUMENT_ENCRYPTION_KEY')
    fernet = Fernet(key)
    return fernet.encrypt(content.encode()).decode()
```

---

## Support and Resources

### Documentation Links

- **Supabase Docs**: https://supabase.com/docs
- **pgvector Docs**: https://github.com/pgvector/pgvector
- **Vector Search Guide**: https://supabase.com/docs/guides/database/extensions/pgvector

### Common Commands

```bash
# Check Supabase status
supabase status

# Reset database
supabase db reset

# Generate types
supabase gen types typescript --local > types/supabase.ts

# Run migrations
supabase db push
```

### Getting Help

1. **Check the logs** in your application
2. **Review Supabase dashboard** for errors
3. **Test with curl** to isolate issues
4. **Check network connectivity** to Supabase
5. **Verify environment variables** are set correctly

---

## Next Steps

After successful setup:

1. ✅ **Test document upload** and processing
2. ✅ **Verify vector search** functionality
3. ✅ **Test chat creation** and messaging
4. ✅ **Configure monitoring** and alerting
5. ✅ **Set up backup** procedures
6. ✅ **Deploy to production** environment
7. ✅ **Monitor performance** and optimize as needed

**Your QueryLex V4 with Supabase is now ready for production! 🚀**