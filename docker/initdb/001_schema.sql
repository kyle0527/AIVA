
-- 掃描記錄表
CREATE TABLE IF NOT EXISTS scans (
  id VARCHAR(255) PRIMARY KEY,                     -- 掃描ID
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  targets JSONB NOT NULL,                          -- 目標配置
  status VARCHAR(50) NOT NULL,                     -- 掃描狀態
  
  -- 掃描配置和結果
  config JSONB,                                    -- 掃描配置
  summary JSONB,                                   -- 掃描摘要
  error_info TEXT,                                 -- 錯誤資訊
  
  -- 統計資訊
  total_tasks INTEGER DEFAULT 0,
  completed_tasks INTEGER DEFAULT 0,
  failed_tasks INTEGER DEFAULT 0,
  
  -- 時間資訊
  started_at TIMESTAMP,
  completed_at TIMESTAMP
);

-- 報告表
CREATE TABLE IF NOT EXISTS reports (
  id SERIAL PRIMARY KEY,
  report_id VARCHAR(255) UNIQUE NOT NULL,          -- 報告ID
  scan_id VARCHAR(255) NOT NULL,                   -- 關聯的掃描ID
  
  -- 報告資訊
  title VARCHAR(500),
  format VARCHAR(50) NOT NULL,                     -- 格式 (json, html, pdf, markdown)
  status VARCHAR(50) NOT NULL DEFAULT 'generating', -- 生成狀態
  
  -- 內容
  content TEXT,                                    -- 報告內容
  file_path VARCHAR(1000),                         -- 檔案路徑
  file_size BIGINT,                                -- 檔案大小
  
  -- 時間戳
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- 外鍵約束
  FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

-- 任務執行記錄表（用於監控和統計）
CREATE TABLE IF NOT EXISTS task_executions (
  id SERIAL PRIMARY KEY,
  task_id VARCHAR(255) NOT NULL,
  scan_id VARCHAR(255) NOT NULL,
  worker_id VARCHAR(255),
  
  -- 任務資訊
  task_type VARCHAR(100) NOT NULL,                 -- 任務類型 (xss, sqli, ssrf)
  target_url TEXT NOT NULL,
  
  -- 執行狀態
  status VARCHAR(50) NOT NULL DEFAULT 'pending',   -- pending, running, completed, failed
  
  -- 時間和效能
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  duration_seconds NUMERIC(10,3),
  
  -- 結果統計
  findings_count INTEGER DEFAULT 0,
  error_message TEXT,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- 外鍵約束
  FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

-- 更新後的 findings 表，與 SQLAlchemy 模型對應
CREATE TABLE IF NOT EXISTS findings (
  id SERIAL PRIMARY KEY,                           -- 自增主鍵
  finding_id VARCHAR(255) UNIQUE NOT NULL,         -- 業務主鍵
  scan_id VARCHAR(255) NOT NULL,                   -- 掃描ID
  task_id VARCHAR(255) NOT NULL,                   -- 任務ID
  
  -- 漏洞資訊
  vulnerability_name VARCHAR(255) NOT NULL,        -- 漏洞名稱
  severity VARCHAR(50) NOT NULL,                   -- 嚴重程度
  confidence VARCHAR(50) NOT NULL,                 -- 信心度
  cwe VARCHAR(50),                                 -- CWE編號
  
  -- 目標資訊
  target_url TEXT NOT NULL,                        -- 目標URL
  target_parameter VARCHAR(255),                   -- 參數名稱
  target_method VARCHAR(10),                       -- HTTP方法
  
  -- 狀態和原始數據
  status VARCHAR(50) NOT NULL,                     -- 狀態
  raw_data TEXT NOT NULL,                          -- 完整JSON數據
  
  -- 時間戳
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 創建索引以優化查詢效能
CREATE INDEX IF NOT EXISTS idx_findings_finding_id ON findings(finding_id);
CREATE INDEX IF NOT EXISTS idx_findings_scan_id ON findings(scan_id);
CREATE INDEX IF NOT EXISTS idx_findings_task_id ON findings(task_id);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);
CREATE INDEX IF NOT EXISTS idx_findings_created_at ON findings(created_at);

-- 觸發器：自動更新 updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_findings_updated_at 
    BEFORE UPDATE ON findings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 為 scans 表創建索引
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);

-- 為 scans 表創建 updated_at 觸發器
CREATE TRIGGER update_scans_updated_at 
    BEFORE UPDATE ON scans 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 為 reports 表創建索引
CREATE INDEX IF NOT EXISTS idx_reports_report_id ON reports(report_id);
CREATE INDEX IF NOT EXISTS idx_reports_scan_id ON reports(scan_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_format ON reports(format);
CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);

-- 為 reports 表創建 updated_at 觸發器
CREATE TRIGGER update_reports_updated_at 
    BEFORE UPDATE ON reports 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 為 task_executions 表創建索引
CREATE INDEX IF NOT EXISTS idx_task_executions_task_id ON task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_scan_id ON task_executions(scan_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON task_executions(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_task_type ON task_executions(task_type);
CREATE INDEX IF NOT EXISTS idx_task_executions_started_at ON task_executions(started_at);

-- 創建一些有用的視圖
CREATE OR REPLACE VIEW scan_statistics AS
SELECT 
    s.id as scan_id,
    s.status,
    s.created_at,
    s.completed_at,
    s.total_tasks,
    s.completed_tasks,
    s.failed_tasks,
    COALESCE(f.total_findings, 0) as total_findings,
    COALESCE(f.high_severity_findings, 0) as high_severity_findings,
    COALESCE(f.medium_severity_findings, 0) as medium_severity_findings,
    COALESCE(f.low_severity_findings, 0) as low_severity_findings
FROM scans s
LEFT JOIN (
    SELECT 
        scan_id,
        COUNT(*) as total_findings,
        SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_severity_findings,
        SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium_severity_findings,
        SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END) as low_severity_findings
    FROM findings
    GROUP BY scan_id
) f ON s.id = f.scan_id;

-- 任務效能統計視圖
CREATE OR REPLACE VIEW task_performance_stats AS
SELECT 
    task_type,
    status,
    COUNT(*) as task_count,
    AVG(duration_seconds) as avg_duration_seconds,
    MIN(duration_seconds) as min_duration_seconds,
    MAX(duration_seconds) as max_duration_seconds,
    AVG(findings_count) as avg_findings_per_task
FROM task_executions
WHERE completed_at IS NOT NULL
GROUP BY task_type, status;
