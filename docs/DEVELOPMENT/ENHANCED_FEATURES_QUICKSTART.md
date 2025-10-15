# AIVA 增強功能快速入門指南

## 概述

AIVA 平台已升級為企業級攻擊面管理(ASPM)解決方案，新增以下核心能力：

✅ **資產與漏洞生命週期管理** - 從發現到修復的完整追蹤  
✅ **程式碼層面根因分析** - 識別共用元件導致的多個漏洞  
✅ **SAST-DAST 關聯分析** - 驗證靜態分析發現的真實可利用性  
✅ **業務驅動的風險評估** - 整合業務重要性和環境上下文  
✅ **智慧去重與合併** - 相同漏洞在多次掃描中只保留一條記錄  

---

## 快速開始

### 1. 資料庫遷移

首先，執行增強版 Schema 來啟用新功能：

```bash
# 如果使用 Docker
docker exec -i aiva_postgres psql -U postgres -d aiva_db < docker/initdb/002_enhanced_schema.sql

# 或直接連接資料庫
psql -U postgres -d aiva_db -f docker/initdb/002_enhanced_schema.sql
```

**驗證遷移成功：**

```sql
-- 檢查新表是否已建立
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('assets', 'vulnerabilities', 'vulnerability_history', 'vulnerability_tags');

-- 應該返回 4 筆記錄
```

### 2. 基本使用 - 資產管理

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from services.integration.aiva_integration.reception.lifecycle_manager import AssetVulnerabilityManager

# 建立資料庫連接
engine = create_engine("postgresql://user:password@localhost:5432/aiva_db")
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# 初始化管理器
manager = AssetVulnerabilityManager(session)

# 註冊資產（包含業務上下文）
asset = manager.register_asset(
    asset_value="https://api.example.com",
    asset_type="url",
    name="Production API",
    business_criticality="critical",  # critical/high/medium/low
    environment="production",          # production/staging/development/testing
    owner="security-team@example.com",
    tags=["api", "payment", "pci-dss"],
    technology_stack={
        "framework": "Django 4.2",
        "language": "Python 3.11",
        "database": "PostgreSQL 15"
    }
)

print(f"資產已註冊: {asset.asset_id}")
```

### 3. 處理掃描發現 - 自動去重

```python
from services.aiva_common.schemas import FindingPayload

# 假設從掃描引擎獲取了 findings
for finding in findings:  # type: FindingPayload
    # 自動去重並管理生命週期
    vulnerability, is_new = manager.process_finding(finding, asset.asset_id)
    
    if is_new:
        print(f"新漏洞: {vulnerability.name} ({vulnerability.severity})")
    else:
        print(f"已知漏洞再次發現: {vulnerability.vulnerability_id}")
        # 系統會自動更新 last_detected_at
        # 如果之前已修復，會重新開啟

session.commit()
```

### 4. 漏洞生命週期管理

```python
# 更新漏洞狀態
manager.update_vulnerability_status(
    vulnerability_id="vuln_abc123",
    new_status="in_progress",  # new/open/in_progress/fixed/risk_accepted/false_positive
    changed_by="john.doe@example.com",
    comment="已開始修復，預計 3 天完成"
)

# 指派漏洞
manager.assign_vulnerability(
    vulnerability_id="vuln_abc123",
    assigned_to="alice@example.com",
    changed_by="manager@example.com"
)

# 添加標籤
manager.add_vulnerability_tag("vuln_abc123", "urgent")
manager.add_vulnerability_tag("vuln_abc123", "requires_review")

session.commit()
```

### 5. 查詢與統計

```python
# 獲取資產的所有開放漏洞（按風險分數排序）
vulnerabilities = manager.get_asset_vulnerabilities(
    asset_id="asset_xyz789",
    include_fixed=False
)

print(f"開放漏洞數: {len(vulnerabilities)}")
for vuln in vulnerabilities[:5]:  # 前 5 個最高風險
    print(f"  - {vuln.name} ({vuln.severity}) 風險分數: {vuln.risk_score}")

# 獲取逾期漏洞
overdue = manager.get_overdue_vulnerabilities()
print(f"\n⚠️ 逾期漏洞: {len(overdue)} 個")

# 計算平均修復時間 (MTTR)
mttr_critical = manager.calculate_mttr(severity="CRITICAL", days=30)
print(f"\n嚴重漏洞 MTTR: {mttr_critical['avg_hours']:.1f} 小時")
print(f"  - 最快: {mttr_critical['min_hours']:.1f} 小時")
print(f"  - 最慢: {mttr_critical['max_hours']:.1f} 小時")
```

### 6. 高級分析 - 根因分析

```python
from services.integration.aiva_integration.analysis.vuln_correlation_analyzer import VulnerabilityCorrelationAnalyzer

analyzer = VulnerabilityCorrelationAnalyzer()

# 將 findings 轉換為字典格式
finding_dicts = [
    {
        "finding_id": f.finding_id,
        "vulnerability_type": f.vulnerability.name.value,
        "severity": f.vulnerability.severity.value,
        "location": {
            "code_file": "api/users.py",  # SAST 需要
            "function_name": "get_user",
            "line_number": 45
        }
    }
    for f in findings
]

# 程式碼層面根因分析
root_cause_result = analyzer.analyze_code_level_root_cause(finding_dicts)

print("\n🎯 根因分析結果:")
for root_cause in root_cause_result["root_causes"]:
    print(f"\n共用元件: {root_cause['component_type']} '{root_cause['component_name']}'")
    print(f"  檔案: {root_cause['file_path']}")
    print(f"  影響漏洞數: {root_cause['affected_vulnerabilities']}")
    print(f"  嚴重程度分布: {root_cause['severity_distribution']}")
    print(f"  建議: {root_cause['recommendation']}")

print(f"\n修復效率提升: {root_cause_result['summary']['fix_efficiency']}")
```

### 7. SAST-DAST 關聯分析

```python
# 混合 SAST 和 DAST 發現
mixed_findings = [
    {
        "finding_id": "sast_001",
        "scan_type": "sast",
        "vulnerability_type": "sql_injection",
        "severity": "HIGH",
        "location": {
            "code_file": "api/users.py",
            "line_number": 45,
            "function_name": "get_user_by_id"
        }
    },
    {
        "finding_id": "dast_042",
        "scan_type": "dast",
        "vulnerability_type": "sqli",
        "severity": "HIGH",
        "location": {
            "url": "https://api.example.com/users",
            "parameter": "id",
            "method": "GET"
        }
    }
]

# 執行關聯分析
correlation_result = analyzer.analyze_sast_dast_correlation(mixed_findings)

print("\n✅ SAST-DAST 關聯分析:")
print(f"已驗證資料流: {len(correlation_result['confirmed_flows'])}")
print(f"確認率: {correlation_result['summary']['confirmation_rate']}%")

for flow in correlation_result["confirmed_flows"]:
    print(f"\n已驗證漏洞: {flow['vulnerability_type']}")
    print(f"  Source (DAST): {flow['source']['location']} - {flow['source']['parameter']}")
    print(f"  Sink (SAST): {flow['sink']['location']}:{flow['sink']['line']}")
    print(f"  影響: {flow['impact']} (已提升)")
    print(f"  建議: {flow['recommendation']}")
```

---

## 實用查詢範例

### SQL 查詢

```sql
-- 1. 查看資產風險概覽
SELECT * FROM asset_risk_overview 
WHERE business_criticality IN ('critical', 'high')
ORDER BY avg_risk_score DESC;

-- 2. 查看逾期漏洞
SELECT * FROM sla_tracking 
WHERE sla_status = 'overdue'
ORDER BY hours_until_deadline;

-- 3. 漏洞趨勢（過去 7 天）
SELECT * FROM vulnerability_trends 
WHERE detection_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY detection_date DESC;

-- 4. MTTR 統計
SELECT * FROM mttr_statistics
WHERE environment = 'production'
ORDER BY avg_hours_to_fix DESC;

-- 5. 計算特定資產的風險分數
SELECT calculate_asset_risk_score('asset_abc123');
```

### Python 查詢

```python
from sqlalchemy import func
from services.integration.aiva_integration.reception.models_enhanced import (
    Asset, Vulnerability, VulnerabilityHistory
)

# 1. 查詢高風險資產
high_risk_assets = session.query(Asset).join(Vulnerability).filter(
    Vulnerability.status.in_(['new', 'open', 'in_progress']),
    Vulnerability.severity.in_(['CRITICAL', 'HIGH']),
    Asset.environment == 'production'
).distinct().all()

# 2. 統計各狀態的漏洞數
status_counts = session.query(
    Vulnerability.status,
    func.count(Vulnerability.id)
).group_by(Vulnerability.status).all()

# 3. 查詢特定人員負責的漏洞
assigned_vulns = session.query(Vulnerability).filter(
    Vulnerability.assigned_to == 'alice@example.com',
    Vulnerability.status.in_(['new', 'open', 'in_progress'])
).order_by(Vulnerability.sla_deadline.asc()).all()

# 4. 查詢漏洞變更歷史
history = session.query(VulnerabilityHistory).filter(
    VulnerabilityHistory.vulnerability_id == 'vuln_abc123'
).order_by(VulnerabilityHistory.created_at.desc()).all()

for entry in history:
    print(f"{entry.created_at}: {entry.change_type} - {entry.old_value} -> {entry.new_value}")
```

---

## 整合到現有掃描流程

在你的掃描協調器中整合新功能：

```python
# 在 services/core/aiva_core/app.py 或協調器中

from services.integration.aiva_integration.reception.lifecycle_manager import AssetVulnerabilityManager
from services.integration.aiva_integration.analysis.vuln_correlation_analyzer import VulnerabilityCorrelationAnalyzer

async def process_scan_with_lifecycle(scan_config):
    # 1. 執行掃描（現有邏輯）
    findings = await execute_scan(scan_config)
    
    # 2. 註冊資產並處理漏洞（新增）
    session = get_db_session()
    manager = AssetVulnerabilityManager(session)
    
    asset = manager.register_asset(
        asset_value=scan_config['target_url'],
        asset_type='url',
        business_criticality=scan_config.get('business_criticality', 'medium'),
        environment=scan_config.get('environment', 'development')
    )
    
    # 3. 處理每個 finding
    for finding in findings:
        vulnerability, is_new = manager.process_finding(finding, asset.asset_id)
        if is_new:
            # 觸發通知
            await notify_new_vulnerability(vulnerability)
    
    # 4. 執行分析
    analyzer = VulnerabilityCorrelationAnalyzer()
    finding_dicts = [convert_to_dict(f) for f in findings]
    
    correlation = analyzer.analyze_correlations(finding_dicts)
    root_cause = analyzer.analyze_code_level_root_cause(finding_dicts)
    sast_dast = analyzer.analyze_sast_dast_correlation(finding_dicts)
    
    session.commit()
    session.close()
    
    return {
        'asset': asset,
        'vulnerabilities': findings,
        'analysis': {
            'correlation': correlation,
            'root_cause': root_cause,
            'sast_dast': sast_dast
        }
    }
```

---

## 最佳實踐

### 1. 資產註冊

- **務必提供業務上下文**：`business_criticality` 和 `environment` 直接影響風險評分
- **使用一致的命名**：資產名稱應該清晰且易於識別
- **標籤系統**：使用標籤進行靈活分類（如 "pci-dss", "hipaa", "gdpr"）

### 2. 漏洞管理

- **及時更新狀態**：當開始修復時立即更新為 `in_progress`
- **記錄變更原因**：使用 `comment` 參數記錄狀態變更的理由
- **使用標籤**：為特殊情況添加標籤（如 "false_positive", "wont_fix", "urgent"）

### 3. 分析與報告

- **定期執行根因分析**：識別系統性問題
- **優先修復根本原因**：一次修復解決多個漏洞
- **關注 SAST-DAST 確認率**：低確認率可能表示 SAST 規則需要調整

### 4. 效能優化

- **使用資料庫視圖**：預定義的視圖（如 `asset_risk_overview`）已優化查詢效能
- **批次處理**：處理大量 findings 時使用批次提交
- **索引使用**：所有關鍵欄位都已建立索引，確保查詢條件使用這些欄位

---

## 常見問題 (FAQ)

**Q: 舊的掃描資料會怎樣？**  
A: `002_enhanced_schema.sql` 不會影響現有的 `findings` 表。新增的欄位使用 `ADD COLUMN IF NOT EXISTS`，可以安全執行。

**Q: 如何處理誤報？**  
A: 使用 `update_vulnerability_status()` 將狀態設為 `false_positive`，並添加 comment 說明原因。

**Q: 漏洞去重的邏輯是什麼？**  
A: 基於「資產 ID + 漏洞類型 + 位置」生成唯一識別碼。相同的組合視為同一個漏洞。

**Q: 風險分數如何計算？**  
A: `基礎分數(嚴重程度) × 信心度乘數 × 業務重要性乘數`。公式在 `lifecycle_manager.py` 中可調整。

**Q: SLA 自動計算規則？**  
A: CRITICAL=24小時, HIGH=7天, MEDIUM=30天, LOW=90天。可在觸發器函數中修改。

**Q: 如何與現有 API 整合？**  
A: 參考 `services/integration/aiva_integration/examples/enhanced_scan_integration.py` 的完整範例。

---

## 進階主題

### 自訂風險評分模型

修改 `lifecycle_manager.py` 中的 `_calculate_initial_risk_score()` 方法：

```python
def _calculate_initial_risk_score(self, finding, asset_id):
    # 你的自訂邏輯
    base_score = self._get_base_score(finding.vulnerability.severity)
    
    # 可以加入更多因素
    exploitability_factor = self._get_exploitability_factor(finding)
    asset_value_factor = self._get_asset_value(asset_id)
    threat_intel_factor = self._check_threat_intelligence(finding)
    
    return base_score * exploitability_factor * asset_value_factor * threat_intel_factor
```

### 整合外部威脅情報

```python
# 在處理 finding 時檢查外部威脅情報
def enrich_with_threat_intel(vulnerability):
    if vulnerability.cve:
        # 查詢 NIST NVD
        nvd_data = query_nvd(vulnerability.cve)
        vulnerability.cvss_score = nvd_data.get('cvss_score')
        
        # 查詢是否有公開 exploit
        exploit_db = query_exploit_db(vulnerability.cve)
        if exploit_db:
            vulnerability.exploitability = 'high'
            vulnerability.add_tag('known_exploit')
```

---

## 路線圖

### 已完成 ✅
- 資產與漏洞生命週期管理
- 程式碼層面根因分析
- SAST-DAST 關聯分析
- 業務上下文整合

### 進行中 🔄
- 攻擊路徑分析自然語言推薦
- 風險評估引擎業務上下文深度整合

### 規劃中 📋
- API 安全測試模組
- AI 驅動的漏洞驗證代理
- SIEM 整合與通知機制
- EASM 探索階段
- 行動應用安全測試 (MAST)

---

## 技術支援

- 📖 完整文檔: `ENHANCEMENT_IMPLEMENTATION_REPORT.md`
- 💻 範例程式碼: `services/integration/aiva_integration/examples/`
- 🐛 問題回報: 請在專案 Issues 中回報
- 📧 聯絡: security-team@example.com

---

**最後更新**: 2025年10月14日  
**版本**: v2.0-enhanced
