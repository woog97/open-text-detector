CREATE TABLE paper_runs(id TEXT PRIMARY KEY, status TEXT NOT NULL, model_snapshot TEXT NOT NULL, filters TEXT NOT NULL, total INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE UNIQUE INDEX one_active_paper_run ON paper_runs((1)) WHERE status IN ('running','stopping');
CREATE TABLE paper_run_items(run_id TEXT NOT NULL REFERENCES paper_runs(id), ordinal INTEGER NOT NULL, paper_id TEXT NOT NULL, scan_id TEXT REFERENCES scans(id), status TEXT NOT NULL DEFAULT 'pending', error TEXT, PRIMARY KEY(run_id,ordinal));
CREATE INDEX paper_run_pending ON paper_run_items(run_id,status,ordinal);
