ALTER TABLE deployment_run_member
ADD COLUMN endpoint TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN member_status TEXT NOT NULL DEFAULT 'planned'
CHECK (member_status IN (
    'planned', 'starting', 'ready', 'stopped',
    'failed', 'cancelled'
));

ALTER TABLE deployment_run_member
ADD COLUMN pid INTEGER CHECK (pid IS NULL OR pid > 0);

ALTER TABLE deployment_run_member
ADD COLUMN argv_json TEXT NOT NULL DEFAULT '[]';

ALTER TABLE deployment_run_member
ADD COLUMN target_model_path TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN draft_model_path TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN started_at TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN ready_at TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN finished_at TEXT;

ALTER TABLE deployment_run_member
ADD COLUMN exit_code INTEGER;

ALTER TABLE deployment_run_member
ADD COLUMN stdout TEXT NOT NULL DEFAULT '';

ALTER TABLE deployment_run_member
ADD COLUMN stderr TEXT NOT NULL DEFAULT '';

ALTER TABLE deployment_run_member
ADD COLUMN forced_kill INTEGER NOT NULL DEFAULT 0
CHECK (forced_kill IN (0, 1));

ALTER TABLE deployment_run_member
ADD COLUMN cleanup_error TEXT;

CREATE INDEX idx_deployment_run_member_status
ON deployment_run_member(member_status, deployment_run_id);
