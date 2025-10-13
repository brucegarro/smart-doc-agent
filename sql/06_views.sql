-- Useful views for common queries

-- Document statistics view
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    d.id,
    d.filename,
    d.title,
    d.processing_status,
    d.num_pages,
    COUNT(DISTINCT c.id) as chunk_count,
    COUNT(DISTINCT a.id) as artifact_count,
    COUNT(DISTINCT e.id) as eval_metric_count,
    d.created_at,
    d.processed_at
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN artifacts_index a ON d.id = a.document_id
LEFT JOIN eval_results e ON d.id = e.document_id
GROUP BY d.id;

-- Metric comparison view
CREATE OR REPLACE VIEW metric_comparison AS
SELECT 
    d.id as document_id,
    d.title,
    d.filename,
    e.metric_name,
    e.metric_value,
    e.dataset,
    e.model_name,
    e.task
FROM eval_results e
JOIN documents d ON e.document_id = d.id
WHERE e.metric_value IS NOT NULL
ORDER BY e.metric_name, e.dataset, e.metric_value DESC;
