-- Documents table: Metadata about each ingested PDF
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(512) NOT NULL,
    title VARCHAR(1024),
    authors TEXT[],  -- Array of author names
    abstract TEXT,
    publication_year INTEGER,
    venue VARCHAR(512),  -- Journal/Conference name
    
    -- Storage references
    s3_key VARCHAR(512) NOT NULL,  -- MinIO object key
    pdf_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hash for deduplication
    
    -- Processing metadata
    num_pages INTEGER,
    file_size_bytes BIGINT,
    processing_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    error_message TEXT,
    
    -- UDR (Unified Document Representation) - full structured content as JSON
    udr_data JSONB,
    
    -- Timestamps
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_pdf_hash ON documents(pdf_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents USING gin(to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_abstract ON documents USING gin(to_tsvector('english', abstract));

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
