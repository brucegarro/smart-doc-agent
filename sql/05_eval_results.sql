-- Evaluation results: Store extracted metrics from papers
-- Optimized for queries like "Compare F1-scores across papers"
CREATE TABLE IF NOT EXISTS eval_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Metric identification
    metric_name VARCHAR(255) NOT NULL,  -- e.g., "accuracy", "F1-score", "BLEU"
    metric_value NUMERIC(10, 6),
    metric_value_text VARCHAR(255),  -- For non-numeric values like "89.2±1.3"
    
    -- Context
    dataset VARCHAR(255),  -- Which dataset/benchmark
    model_name VARCHAR(255),  -- Which model/method
    task VARCHAR(255),  -- What task (classification, NER, etc.)
    
    -- Position in paper
    page_number INTEGER,
    table_reference VARCHAR(100),  -- e.g., "Table 3"
    section_title VARCHAR(512),
    
    -- Additional context
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for metric queries
CREATE INDEX IF NOT EXISTS idx_eval_document_id ON eval_results(document_id);
CREATE INDEX IF NOT EXISTS idx_eval_metric_name ON eval_results(metric_name);
CREATE INDEX IF NOT EXISTS idx_eval_dataset ON eval_results(dataset);
CREATE INDEX IF NOT EXISTS idx_eval_model ON eval_results(model_name);
CREATE INDEX IF NOT EXISTS idx_eval_metric_value ON eval_results(metric_value);
CREATE INDEX IF NOT EXISTS idx_eval_composite ON eval_results(metric_name, dataset, document_id);
