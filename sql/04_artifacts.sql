-- Artifacts index: Tables, figures, equations extracted from papers
-- Used for specific queries like "What are the accuracy metrics in Paper X?"
CREATE TABLE IF NOT EXISTS artifacts_index (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Artifact identification
    artifact_type VARCHAR(50) NOT NULL,  -- table, figure, equation, algorithm
    artifact_number VARCHAR(50),  -- e.g., "Table 1", "Figure 2", "Equation 3"
    caption TEXT,
    
    -- Position
    page_number INTEGER,
    section_title VARCHAR(512),
    
    -- Content
    content TEXT,  -- Extracted text/data (CSV for tables, LaTeX for equations, etc.)
    structured_data JSONB,  -- Parsed structured representation (e.g., table as JSON)
    
    -- Storage reference for images
    s3_key VARCHAR(512),  -- MinIO key for figure/diagram images
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_artifacts_document_id ON artifacts_index(document_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts_index(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifacts_page ON artifacts_index(page_number);
CREATE INDEX IF NOT EXISTS idx_artifacts_caption_fts ON artifacts_index 
USING gin(to_tsvector('english', caption));
CREATE INDEX IF NOT EXISTS idx_artifacts_content_fts ON artifacts_index 
USING gin(to_tsvector('english', content));
