ALTER TABLE scans ADD COLUMN result_storage TEXT;
CREATE INDEX idx_scans_model_completed ON scans(json_extract(model_snapshot,'$.model.id'),created_at DESC)
WHERE status='completed' AND deleted_at IS NULL;
