-- Chunks table: Text segments with embeddings for vector search
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Chunk content
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',  -- text, table, equation, figure_caption, reference
    
    -- Position in document
    chunk_index INTEGER NOT NULL,  -- Sequential order within document
    page_number INTEGER,
    section_title VARCHAR(512),
    
    -- Chunking metadata
    token_count INTEGER,
    char_count INTEGER,
    
    -- Vector embedding (384 dimensions for BAAI/bge-small-en-v1.5)
    embedding vector(384),
    
    -- Additional context
    metadata JSONB,  -- Store additional chunk-specific metadata
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_chunk ON chunks(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);

-- Vector similarity search index (HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (comment out HNSW above and uncomment below for different trade-off)
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON chunks 
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON chunks 
USING gin(to_tsvector('english', content));
