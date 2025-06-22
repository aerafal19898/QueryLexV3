# Supabase Setup Guide for QueryLex

This guide explains how to set up Supabase as the vector database for QueryLex, replacing ChromaDB.

## Prerequisites

1. A Supabase account (sign up at https://supabase.com)
2. Python environment with the updated requirements.txt installed

## Step 1: Create a Supabase Project

1. Log in to your Supabase dashboard
2. Click "New Project"
3. Choose your organization
4. Enter project details:
   - Name: `querylex-vector-db` (or your preferred name)
   - Database Password: Generate a strong password and save it
   - Region: Choose the region closest to your users (for EU compliance, choose a EU region)
5. Click "Create new project"

## Step 2: Enable Vector Extension

1. Go to your project dashboard
2. Navigate to "SQL Editor" in the left sidebar
3. Run the SQL commands from `supabase_schema.sql`:

```sql
-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Copy and paste the entire contents of supabase_schema.sql here
```

Alternatively, you can:
1. Copy the contents of `supabase_schema.sql`
2. Paste it into the SQL Editor
3. Click "Run" to execute all commands

## Step 3: Configure Environment Variables

Add the following environment variables to your `.env` file:

```env
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key

# Optional: Keep ChromaDB settings for backward compatibility
CHROMA_DIR=./data/chroma
```

To find your Supabase keys:
1. Go to your project dashboard
2. Navigate to "Settings" → "API"
3. Copy the "URL" and "anon public" key
4. Copy the "service_role" key (keep this secret!)

## Step 4: Update Vector Dimensions

The schema is currently set up for 1024-dimensional vectors. If your embedding model uses different dimensions, update the schema:

1. Check your embedding model dimensions:
   - `BAAI/bge-large-en-v1.5`: 1024 dimensions ✓ (current setting)
   - `all-MiniLM-L6-v2`: 384 dimensions
   - `text-embedding-ada-002`: 1536 dimensions

2. If you need to change dimensions, run this SQL:

```sql
-- For 384 dimensions (all-MiniLM-L6-v2)
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(384);
DROP FUNCTION IF EXISTS match_documents;
CREATE OR REPLACE FUNCTION match_documents(
    collection_id UUID,
    query_embedding vector(384),  -- Change this number
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
-- ... rest of function remains the same
```

## Step 5: Configure Row Level Security (Optional)

For production deployments, you may want to configure more restrictive RLS policies:

```sql
-- Example: Restrict access to authenticated users with specific roles
CREATE POLICY "Collections access for specific roles" ON collections 
FOR ALL USING (
    auth.jwt() ->> 'role' IN ('admin', 'user')
);
```

## Step 6: Test the Connection

1. Install the updated dependencies:
```bash
pip install -r requirements.txt
```

2. Start your application:
```bash
python run.py
```

3. Check the console output for:
```
Successfully connected to Supabase
```

If you see an error, verify your environment variables and network connectivity.

## Step 7: Migrate Existing Data (Optional)

If you have existing ChromaDB data to migrate:

1. Export your ChromaDB collections:
```python
import chromadb
import json

client = chromadb.PersistentClient(path="./data/chroma")
collections = client.list_collections()

for collection in collections:
    coll = client.get_collection(collection.name)
    results = coll.get(include=["documents", "metadatas", "embeddings"])
    
    # Save to JSON for migration
    with open(f"{collection.name}_export.json", "w") as f:
        json.dump(results, f)
```

2. Import to Supabase using the QueryLex upload interface or API

## Performance Tuning

For better performance with large datasets:

1. **Indexing**: The schema includes IVFFlat indexes. For very large datasets, consider creating HNSW indexes:

```sql
-- Drop existing index
DROP INDEX IF EXISTS idx_documents_embedding;

-- Create HNSW index (better for large datasets)
CREATE INDEX idx_documents_embedding_hnsw ON documents 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

2. **Connection Pooling**: For high-traffic applications, consider using connection pooling:

```env
# Add to your environment
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
```

## Monitoring and Maintenance

1. **Monitor Usage**: Check your Supabase dashboard regularly for:
   - Database size
   - API requests
   - Performance metrics

2. **Backup Strategy**: Supabase automatically backs up your database, but consider:
   - Exporting critical collections regularly
   - Setting up automated backups of metadata

3. **Cost Optimization**:
   - Monitor your usage on the Supabase dashboard
   - Consider upgrading to Pro for better performance if needed
   - Use database optimizations for large datasets

## Troubleshooting

### Common Issues:

1. **"vector extension not found"**
   - Ensure you've enabled the vector extension in Step 2

2. **"dimension mismatch"**
   - Check that your vector dimensions match the schema
   - Update the schema if using a different embedding model

3. **"authentication failed"**
   - Verify your SUPABASE_URL and keys are correct
   - Check that your service key has the necessary permissions

4. **"RLS policy violation"**
   - Ensure your RLS policies allow the operations you're trying to perform
   - Consider temporarily disabling RLS for testing

### Performance Issues:

1. **Slow similarity search**
   - Ensure proper indexing is in place
   - Consider using HNSW instead of IVFFlat for large datasets
   - Check if you need to increase the `ef_search` parameter

2. **High memory usage**
   - Reduce batch sizes in document processing
   - Consider processing documents in smaller chunks

## Security Considerations

1. **API Keys**: Never expose your service key in client-side code
2. **RLS Policies**: Implement proper row-level security for production
3. **SSL**: Always use HTTPS in production
4. **Audit Logging**: Enable Supabase audit logging for compliance

## Chat and Folder Migration

The schema now includes chat and folder management:

### Tables Added:
- **folders**: Chat folder organization
- **chats**: Chat sessions with metadata
- **messages**: Individual chat messages

### Migration Notes:
- Existing local chat data in `data/chats/` can be migrated using the chat service
- The application automatically falls back to local storage if Supabase is unavailable
- Chat functionality maintains backward compatibility

## Next Steps

After successful setup:

1. Test all existing QueryLex functionality
2. Upload some sample documents to verify the vector search works
3. Test chat creation, messaging, and folder organization
4. Configure monitoring and alerting
5. Set up your production deployment

For questions or issues, refer to:
- Supabase Documentation: https://supabase.com/docs
- pgvector Documentation: https://github.com/pgvector/pgvector