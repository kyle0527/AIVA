-- Enhanced Schema for Asset and Vulnerability Lifecycle Management
-- 此腳本新增資產與漏洞生命週期管理功能
-- 執行順序：001_schema.sql -> 002_enhanced_schema.sql

-- ============================================================================
-- 1. 資產表 (Assets) - 統一管理所有被掃描的目標資產
-- ============================================================================
CREATE TABLE IF NOT EXISTS assets (
  id SERIAL PRIMARY KEY,
  asset_id VARCHAR(255) UNIQUE NOT NULL,           -- 資產唯一識別碼
  
  -- 資產基本資訊
  name VARCHAR(500),                                -- 資產名稱
  type VARCHAR(100) NOT NULL,                       -- 資產類型 (url, repository, host, container, api_endpoint)
  value TEXT NOT NULL,                              -- 資產值 (URL, IP, 儲存庫路徑等)
  description TEXT,                                 -- 資產描述
  
  -- 業務與環境上下文
  business_criticality VARCHAR(50) DEFAULT 'medium', -- 業務重要性 (critical, high, medium, low)
  environment VARCHAR(50) DEFAULT 'development',     -- 環境 (production, staging, development, testing)
  owner VARCHAR(255),                                -- 負責人/團隊
  tags JSONB DEFAULT '[]',                           -- 自訂標籤
  
  -- 技術資訊
  technology_stack JSONB DEFAULT '{}',               -- 技術堆疊資訊
  metadata JSONB DEFAULT '{}',                       -- 額外的元資料
  
  -- 狀態與時間
  status VARCHAR(50) DEFAULT 'active',               -- 狀態 (active, archived, deleted)
  first_discovered_at TIMESTAMP,                     -- 首次發現時間
  last_scanned_at TIMESTAMP,                         -- 最後掃描時間
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 建立索引
CREATE INDEX IF NOT EXISTS idx_assets_asset_id ON assets(asset_id);
CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(type);
CREATE INDEX IF NOT EXISTS idx_assets_business_criticality ON assets(business_criticality);
CREATE INDEX IF NOT EXISTS idx_assets_environment ON assets(environment);
CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
CREATE INDEX IF NOT EXISTS idx_assets_owner ON assets(owner);

-- 為 assets 表建立 updated_at 觸發器
CREATE TRIGGER update_assets_updated_at 
    BEFORE UPDATE ON assets 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 2. 漏洞表 (Vulnerabilities) - 去重後的漏洞總表
-- ============================================================================
CREATE TABLE IF NOT EXISTS vulnerabilities (
  id SERIAL PRIMARY KEY,
  vulnerability_id VARCHAR(255) UNIQUE NOT NULL,    -- 漏洞唯一識別碼
  
  -- 漏洞基本資訊
  name VARCHAR(255) NOT NULL,                       -- 漏洞名稱
  vulnerability_type VARCHAR(100) NOT NULL,         -- 漏洞類型 (xss, sqli, ssrf 等)
  severity VARCHAR(50) NOT NULL,                    -- 嚴重程度 (critical, high, medium, low)
  confidence VARCHAR(50) NOT NULL,                  -- 信心度 (high, medium, low)
  
  -- 標準參考
  cwe VARCHAR(50),                                  -- CWE 編號
  cve VARCHAR(50),                                  -- CVE 編號 (如適用)
  owasp_category VARCHAR(100),                      -- OWASP 類別
  
  -- 關聯資產
  asset_id VARCHAR(255) NOT NULL,                   -- 關聯的資產 ID
  
  -- 位置資訊
  location JSONB NOT NULL,                          -- 漏洞位置 (URL, 參數, 程式碼路徑等)
  
  -- 漏洞詳細資訊
  description TEXT,                                 -- 漏洞描述
  impact TEXT,                                      -- 影響說明
  remediation TEXT,                                 -- 修復建議
  
  -- 風險評估
  cvss_score NUMERIC(3,1),                          -- CVSS 分數
  risk_score NUMERIC(5,2),                          -- 計算後的風險分數
  exploitability VARCHAR(50) DEFAULT 'medium',      -- 可利用性 (high, medium, low)
  business_impact VARCHAR(50),                      -- 業務影響評估
  
  -- 生命週期狀態
  status VARCHAR(50) DEFAULT 'new',                 -- 狀態 (new, open, in_progress, fixed, risk_accepted, false_positive, wont_fix)
  resolution VARCHAR(50),                           -- 解決方式
  
  -- 時間追蹤
  first_detected_at TIMESTAMP NOT NULL,             -- 首次檢測時間
  last_detected_at TIMESTAMP NOT NULL,              -- 最後檢測時間
  fixed_at TIMESTAMP,                               -- 修復時間
  verified_fixed_at TIMESTAMP,                      -- 驗證修復時間
  
  -- 處理資訊
  assigned_to VARCHAR(255),                         -- 指派給
  sla_deadline TIMESTAMP,                           -- SLA 截止時間
  notes TEXT,                                       -- 備註
  
  -- 關聯與根因
  root_cause_vulnerability_id VARCHAR(255),         -- 根本原因的漏洞 ID (若為衍生漏洞)
  related_vulnerability_ids JSONB DEFAULT '[]',     -- 相關漏洞 ID 列表
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- 外鍵約束
  FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE
);

-- 建立索引
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_vulnerability_id ON vulnerabilities(vulnerability_id);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_asset_id ON vulnerabilities(asset_id);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_severity ON vulnerabilities(severity);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_status ON vulnerabilities(status);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_type ON vulnerabilities(vulnerability_type);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_first_detected ON vulnerabilities(first_detected_at);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_last_detected ON vulnerabilities(last_detected_at);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_assigned_to ON vulnerabilities(assigned_to);
CREATE INDEX IF NOT EXISTS idx_vulnerabilities_business_impact ON vulnerabilities(business_impact);

-- 為 vulnerabilities 表建立 updated_at 觸發器
CREATE TRIGGER update_vulnerabilities_updated_at 
    BEFORE UPDATE ON vulnerabilities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 3. 修改 Findings 表 - 新增與 Vulnerabilities 的關聯
-- ============================================================================
-- 新增 vulnerability_id 欄位到現有的 findings 表
ALTER TABLE findings 
ADD COLUMN IF NOT EXISTS vulnerability_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS asset_id VARCHAR(255);

-- 新增外鍵約束
ALTER TABLE findings 
ADD CONSTRAINT fk_findings_vulnerability 
FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(vulnerability_id) ON DELETE SET NULL;

ALTER TABLE findings 
ADD CONSTRAINT fk_findings_asset 
FOREIGN KEY (asset_id) REFERENCES assets(asset_id) ON DELETE CASCADE;

-- 新增索引
CREATE INDEX IF NOT EXISTS idx_findings_vulnerability_id ON findings(vulnerability_id);
CREATE INDEX IF NOT EXISTS idx_findings_asset_id ON findings(asset_id);

-- ============================================================================
-- 4. 漏洞歷史表 (Vulnerability History) - 追蹤狀態變更
-- ============================================================================
CREATE TABLE IF NOT EXISTS vulnerability_history (
  id SERIAL PRIMARY KEY,
  vulnerability_id VARCHAR(255) NOT NULL,
  
  -- 變更資訊
  changed_by VARCHAR(255),                          -- 變更者
  change_type VARCHAR(50) NOT NULL,                 -- 變更類型 (status_change, severity_change, assignment, etc.)
  old_value TEXT,                                   -- 舊值
  new_value TEXT,                                   -- 新值
  comment TEXT,                                     -- 變更說明
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- 外鍵約束
  FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(vulnerability_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vuln_history_vuln_id ON vulnerability_history(vulnerability_id);
CREATE INDEX IF NOT EXISTS idx_vuln_history_created_at ON vulnerability_history(created_at);

-- ============================================================================
-- 5. 漏洞標籤表 (Vulnerability Tags) - 靈活的分類系統
-- ============================================================================
CREATE TABLE IF NOT EXISTS vulnerability_tags (
  id SERIAL PRIMARY KEY,
  vulnerability_id VARCHAR(255) NOT NULL,
  tag VARCHAR(100) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- 外鍵約束
  FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities(vulnerability_id) ON DELETE CASCADE,
  
  -- 唯一約束 (同一個漏洞不能有重複標籤)
  UNIQUE(vulnerability_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_vuln_tags_vuln_id ON vulnerability_tags(vulnerability_id);
CREATE INDEX IF NOT EXISTS idx_vuln_tags_tag ON vulnerability_tags(tag);

-- ============================================================================
-- 6. 增強的視圖 (Enhanced Views)
-- ============================================================================

-- 資產風險概覽視圖
CREATE OR REPLACE VIEW asset_risk_overview AS
SELECT 
    a.asset_id,
    a.name as asset_name,
    a.type as asset_type,
    a.business_criticality,
    a.environment,
    a.owner,
    COUNT(DISTINCT v.id) as total_vulnerabilities,
    SUM(CASE WHEN v.severity = 'CRITICAL' AND v.status IN ('new', 'open', 'in_progress') THEN 1 ELSE 0 END) as critical_open,
    SUM(CASE WHEN v.severity = 'HIGH' AND v.status IN ('new', 'open', 'in_progress') THEN 1 ELSE 0 END) as high_open,
    SUM(CASE WHEN v.severity = 'MEDIUM' AND v.status IN ('new', 'open', 'in_progress') THEN 1 ELSE 0 END) as medium_open,
    SUM(CASE WHEN v.severity = 'LOW' AND v.status IN ('new', 'open', 'in_progress') THEN 1 ELSE 0 END) as low_open,
    AVG(v.risk_score) as avg_risk_score,
    MAX(v.last_detected_at) as last_vulnerability_detected,
    a.last_scanned_at
FROM assets a
LEFT JOIN vulnerabilities v ON a.asset_id = v.asset_id
GROUP BY a.asset_id, a.name, a.type, a.business_criticality, a.environment, a.owner, a.last_scanned_at;

-- 漏洞趨勢視圖
CREATE OR REPLACE VIEW vulnerability_trends AS
SELECT 
    DATE(first_detected_at) as detection_date,
    severity,
    status,
    COUNT(*) as vulnerability_count
FROM vulnerabilities
GROUP BY DATE(first_detected_at), severity, status
ORDER BY detection_date DESC, severity;

-- MTTR (Mean Time To Resolve) 統計視圖
CREATE OR REPLACE VIEW mttr_statistics AS
SELECT 
    v.severity,
    v.vulnerability_type,
    a.business_criticality,
    a.environment,
    COUNT(*) as total_fixed,
    AVG(EXTRACT(EPOCH FROM (v.fixed_at - v.first_detected_at))/3600) as avg_hours_to_fix,
    MIN(EXTRACT(EPOCH FROM (v.fixed_at - v.first_detected_at))/3600) as min_hours_to_fix,
    MAX(EXTRACT(EPOCH FROM (v.fixed_at - v.first_detected_at))/3600) as max_hours_to_fix
FROM vulnerabilities v
JOIN assets a ON v.asset_id = a.asset_id
WHERE v.status = 'fixed' AND v.fixed_at IS NOT NULL
GROUP BY v.severity, v.vulnerability_type, a.business_criticality, a.environment;

-- 開放漏洞 SLA 追蹤視圖
CREATE OR REPLACE VIEW sla_tracking AS
SELECT 
    v.vulnerability_id,
    v.name,
    v.severity,
    v.status,
    v.assigned_to,
    v.sla_deadline,
    a.asset_id,
    a.name as asset_name,
    a.business_criticality,
    a.environment,
    CASE 
        WHEN v.sla_deadline IS NULL THEN 'no_sla'
        WHEN v.sla_deadline < CURRENT_TIMESTAMP THEN 'overdue'
        WHEN v.sla_deadline < CURRENT_TIMESTAMP + INTERVAL '24 hours' THEN 'due_soon'
        ELSE 'on_track'
    END as sla_status,
    EXTRACT(EPOCH FROM (v.sla_deadline - CURRENT_TIMESTAMP))/3600 as hours_until_deadline
FROM vulnerabilities v
JOIN assets a ON v.asset_id = a.asset_id
WHERE v.status IN ('new', 'open', 'in_progress')
ORDER BY v.sla_deadline ASC NULLS LAST;

-- ============================================================================
-- 7. 有用的函數
-- ============================================================================

-- 計算資產的綜合風險分數
CREATE OR REPLACE FUNCTION calculate_asset_risk_score(asset_id_param VARCHAR)
RETURNS NUMERIC AS $$
DECLARE
    risk_score NUMERIC;
BEGIN
    SELECT 
        COALESCE(SUM(
            CASE v.severity
                WHEN 'CRITICAL' THEN 10.0
                WHEN 'HIGH' THEN 7.0
                WHEN 'MEDIUM' THEN 4.0
                WHEN 'LOW' THEN 1.0
                ELSE 0
            END * 
            CASE a.business_criticality
                WHEN 'critical' THEN 2.0
                WHEN 'high' THEN 1.5
                WHEN 'medium' THEN 1.0
                WHEN 'low' THEN 0.5
                ELSE 1.0
            END
        ), 0)
    INTO risk_score
    FROM vulnerabilities v
    JOIN assets a ON v.asset_id = a.asset_id
    WHERE v.asset_id = asset_id_param 
    AND v.status IN ('new', 'open', 'in_progress');
    
    RETURN risk_score;
END;
$$ LANGUAGE plpgsql;

-- 自動設定 SLA 截止時間
CREATE OR REPLACE FUNCTION set_vulnerability_sla()
RETURNS TRIGGER AS $$
BEGIN
    -- 如果沒有設定 SLA，根據嚴重程度自動設定
    IF NEW.sla_deadline IS NULL THEN
        NEW.sla_deadline := CASE NEW.severity
            WHEN 'CRITICAL' THEN NEW.first_detected_at + INTERVAL '24 hours'
            WHEN 'HIGH' THEN NEW.first_detected_at + INTERVAL '7 days'
            WHEN 'MEDIUM' THEN NEW.first_detected_at + INTERVAL '30 days'
            WHEN 'LOW' THEN NEW.first_detected_at + INTERVAL '90 days'
            ELSE NEW.first_detected_at + INTERVAL '30 days'
        END;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_vulnerability_sla_trigger
    BEFORE INSERT ON vulnerabilities
    FOR EACH ROW EXECUTE FUNCTION set_vulnerability_sla();

-- 記錄漏洞狀態變更
CREATE OR REPLACE FUNCTION log_vulnerability_change()
RETURNS TRIGGER AS $$
BEGIN
    -- 記錄狀態變更
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO vulnerability_history (vulnerability_id, change_type, old_value, new_value)
        VALUES (NEW.vulnerability_id, 'status_change', OLD.status, NEW.status);
    END IF;
    
    -- 記錄嚴重程度變更
    IF OLD.severity IS DISTINCT FROM NEW.severity THEN
        INSERT INTO vulnerability_history (vulnerability_id, change_type, old_value, new_value)
        VALUES (NEW.vulnerability_id, 'severity_change', OLD.severity, NEW.severity);
    END IF;
    
    -- 記錄指派變更
    IF OLD.assigned_to IS DISTINCT FROM NEW.assigned_to THEN
        INSERT INTO vulnerability_history (vulnerability_id, change_type, old_value, new_value)
        VALUES (NEW.vulnerability_id, 'assignment_change', OLD.assigned_to, NEW.assigned_to);
    END IF;
    
    -- 自動記錄修復時間
    IF OLD.status != 'fixed' AND NEW.status = 'fixed' AND NEW.fixed_at IS NULL THEN
        NEW.fixed_at := CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER log_vulnerability_change_trigger
    BEFORE UPDATE ON vulnerabilities
    FOR EACH ROW EXECUTE FUNCTION log_vulnerability_change();

-- ============================================================================
-- 8. 初始資料與示例
-- ============================================================================

-- 插入一些常見的標籤
INSERT INTO vulnerability_tags (vulnerability_id, tag) 
SELECT DISTINCT vulnerability_id, 'needs_review' 
FROM vulnerabilities 
WHERE status = 'new' 
ON CONFLICT DO NOTHING;

COMMENT ON TABLE assets IS '資產表：統一管理所有被掃描的目標資產，包含業務上下文';
COMMENT ON TABLE vulnerabilities IS '漏洞表：去重後的漏洞總表，管理完整生命週期';
COMMENT ON TABLE vulnerability_history IS '漏洞歷史表：追蹤所有狀態和屬性變更';
COMMENT ON TABLE vulnerability_tags IS '漏洞標籤表：靈活的分類和過濾系統';
