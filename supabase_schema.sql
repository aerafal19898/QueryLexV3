-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create collections table (equivalent to ChromaDB collections)
CREATE TABLE IF NOT EXISTS collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create documents table (equivalent to ChromaDB documents)
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1024), -- Adjust dimension based on your embedding model
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON documents(collection_id);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_collections_name ON collections(name);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_documents(
    collection_id UUID,
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE d.collection_id = match_documents.collection_id
        AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_collections_updated_at BEFORE UPDATE ON collections FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policies for collections
CREATE POLICY "Collections are viewable by everyone" ON collections FOR SELECT USING (true);
CREATE POLICY "Collections are insertable by authenticated users" ON collections FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Collections are updatable by authenticated users" ON collections FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Collections are deletable by authenticated users" ON collections FOR DELETE USING (auth.role() = 'authenticated');

-- Create policies for documents  
CREATE POLICY "Documents are viewable by everyone" ON documents FOR SELECT USING (true);
CREATE POLICY "Documents are insertable by authenticated users" ON documents FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Documents are updatable by authenticated users" ON documents FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Documents are deletable by authenticated users" ON documents FOR DELETE USING (auth.role() = 'authenticated');

-- Insert some default collections if they don't exist
INSERT INTO collections (name, metadata) 
VALUES 
    ('EU-Sanctions', '{"description": "European Union sanctions regulations and guidelines"}'),
    ('US-Sanctions', '{"description": "United States sanctions regulations and guidelines"}'),
    ('UN-Sanctions', '{"description": "United Nations sanctions regulations and guidelines"}')
ON CONFLICT (name) DO NOTHING;

-- Create a function to get collection by name
CREATE OR REPLACE FUNCTION get_collection_by_name(collection_name TEXT)
RETURNS TABLE (
    id UUID,
    name TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT c.id, c.name, c.metadata, c.created_at, c.updated_at
    FROM collections c
    WHERE c.name = collection_name;
END;
$$;

-- Create a function to count documents in a collection
CREATE OR REPLACE FUNCTION count_documents_in_collection(collection_id UUID)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    doc_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO doc_count
    FROM documents d
    WHERE d.collection_id = count_documents_in_collection.collection_id;
    
    RETURN doc_count;
END;
$$;

-- Create a function to get unique sources in a collection
CREATE OR REPLACE FUNCTION get_unique_sources_in_collection(collection_name TEXT)
RETURNS TEXT[]
LANGUAGE plpgsql
AS $$
DECLARE
    collection_uuid UUID;
    sources TEXT[];
BEGIN
    -- Get collection ID
    SELECT id INTO collection_uuid
    FROM collections
    WHERE name = collection_name;
    
    IF collection_uuid IS NULL THEN
        RETURN ARRAY[]::TEXT[];
    END IF;
    
    -- Get unique sources
    SELECT ARRAY_AGG(DISTINCT (metadata->>'source')) INTO sources
    FROM documents
    WHERE collection_id = collection_uuid
        AND metadata->>'source' IS NOT NULL;
    
    RETURN COALESCE(sources, ARRAY[]::TEXT[]);
END;
$$;

-- Create a function to delete documents by source
CREATE OR REPLACE FUNCTION delete_documents_by_source(
    collection_name TEXT,
    source_name TEXT
)
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    collection_uuid UUID;
    deleted_count INTEGER;
BEGIN
    -- Get collection ID
    SELECT id INTO collection_uuid
    FROM collections
    WHERE name = collection_name;
    
    IF collection_uuid IS NULL THEN
        RETURN 0;
    END IF;
    
    -- Delete documents and get count
    DELETE FROM documents
    WHERE collection_id = collection_uuid
        AND metadata->>'source' = source_name;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$;

-- Create folders table for chat organization
CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create chats table
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    folder_id TEXT REFERENCES folders(id) ON DELETE SET NULL,
    dataset TEXT,
    model TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for chat tables
CREATE INDEX IF NOT EXISTS idx_chats_folder_id ON chats(folder_id);
CREATE INDEX IF NOT EXISTS idx_chats_created_at ON chats(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- Enable RLS for chat tables
ALTER TABLE folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Create policies for folders
CREATE POLICY "Folders are viewable by everyone" ON folders FOR SELECT USING (true);
CREATE POLICY "Folders are insertable by authenticated users" ON folders FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Folders are updatable by authenticated users" ON folders FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Folders are deletable by authenticated users" ON folders FOR DELETE USING (auth.role() = 'authenticated');

-- Create policies for chats
CREATE POLICY "Chats are viewable by everyone" ON chats FOR SELECT USING (true);
CREATE POLICY "Chats are insertable by authenticated users" ON chats FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Chats are updatable by authenticated users" ON chats FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Chats are deletable by authenticated users" ON chats FOR DELETE USING (auth.role() = 'authenticated');

-- Create policies for messages
CREATE POLICY "Messages are viewable by everyone" ON messages FOR SELECT USING (true);
CREATE POLICY "Messages are insertable by authenticated users" ON messages FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Messages are updatable by authenticated users" ON messages FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Messages are deletable by authenticated users" ON messages FOR DELETE USING (auth.role() = 'authenticated');

-- Add triggers for updated_at timestamps
CREATE TRIGGER update_folders_updated_at BEFORE UPDATE ON folders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_chats_updated_at BEFORE UPDATE ON chats FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default folder
INSERT INTO folders (id, name) 
VALUES ('default', 'Default')
ON CONFLICT (id) DO NOTHING;

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('bug', 'feature', 'general', 'suggestion', 'other')),
    content TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'in_progress', 'resolved', 'closed')),
    metadata JSONB DEFAULT '{}',
    status_history JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for feedback table
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);

-- Enable RLS for feedback table
ALTER TABLE feedback ENABLE ROW LEVEL SECURITY;

-- Create policies for feedback
CREATE POLICY "Feedback is viewable by everyone" ON feedback FOR SELECT USING (true);
CREATE POLICY "Feedback is insertable by everyone" ON feedback FOR INSERT WITH CHECK (true);
CREATE POLICY "Feedback is updatable by authenticated users" ON feedback FOR UPDATE USING (auth.role() = 'authenticated');
CREATE POLICY "Feedback is deletable by authenticated users" ON feedback FOR DELETE USING (auth.role() = 'authenticated');

-- Add trigger for updated_at timestamp
CREATE TRIGGER update_feedback_updated_at BEFORE UPDATE ON feedback FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();