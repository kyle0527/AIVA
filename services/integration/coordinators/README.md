# Integration Coordinators - 雙閉環協調器系統

## 概述

基於 **AIVA 雙閉環架構設計**，Integration 協調器負責：

1. **內循環（Internal Loop）**: 收集執行數據 → 優化測試策略 → 回饋 Core
2. **外循環（External Loop）**: 整理漏洞數據 → 生成報告 → 提交客戶 → 學習改進

## 架構設計

### 數據流向

```
┌─────────────────────────────────────────────────────────────────┐
│                         內循環（優化）                          │
│                                                                 │
│  Core (決策)                                                    │
│    ↓ 任務                                                       │
│  Features (執行)                                                │
│    ↓ 原始結果                                                   │
│  Integration Coordinator (分析)                                 │
│    ↓ 優化建議                                                   │
│  Core (調整策略) ←──────────────────────┐                       │
│                                         │                       │
└─────────────────────────────────────────┼───────────────────────┘
                                          │
┌─────────────────────────────────────────┼───────────────────────┐
│                         外循環（報告）  │                       │
│                                         │                       │
│  Integration (彙總)                     │                       │
│    ↓ 報告數據                           │                       │
│  Report Generator (格式化)              │                       │
│    ↓ PDF/JSON/SARIF                     │                       │
│  客戶/Bug Bounty 平台                   │                       │
│    ↓ 回饋（接受/拒絕/補充）             │                       │
│  Core (學習調整) ───────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 核心組件

### 1. BaseCoordinator（基礎協調器）

所有具體協調器的抽象基類，定義通用的數據處理流程：

```python
class BaseCoordinator(ABC):
    async def collect_result(result_dict) -> Dict:
        # 1. 驗證結果格式
        # 2. 存儲原始數據
        # 3. 提取內循環數據
        # 4. 提取外循環數據
        # 5. 驗證漏洞真實性
        # 6. 生成 Core 反饋
        # 7. 發送反饋
```

**職責**:
- ✅ 資料驗證（Pydantic Schema）
- ✅ 去重處理（避免重複）
- ✅ 數據存儲（時序DB + 文檔DB + 緩存）
- ✅ 漏洞驗證（自動化驗證邏輯）
- ✅ 反饋生成（MQ 發送）

### 2. 具體協調器

#### XSSCoordinator
- **模組**: `function_xss`
- **特性**: 
  - Payload 類型分析（script/event/svg/img/iframe）
  - 注入上下文識別（HTML body/attribute/script）
  - CSP 檢測和繞過建議
  - WAF 識別和規避策略

#### SQLiCoordinator（待實現）
- **模組**: `function_sqli`
- **特性**:
  - 注入類型分析（Union/Error/Time/Boolean/NoSQL）
  - 數據庫指紋識別（MySQL/PostgreSQL/MSSQL）
  - Payload 編碼策略優化
  - 數據提取效率分析

#### WebScannerCoordinator（待實現）
- **模組**: `function_web_scanner`
- **特性**:
  - 目錄掃描效率分析
  - 子域名發現模式
  - 端口服務識別
  - 技術棧檢測

#### DDoSCoordinator（待實現）
- **模組**: `function_ddos`
- **特性**:
  - 壓力測試結果分析
  - 目標承載能力評估
  - Rate Limiting 識別
  - 攻擊向量效果比較

## 數據模型

### Features 返回的標準格式

基於業界標準（SARIF、OWASP、Bug Bounty 平台）：

```python
{
    "task_id": "uuid",
    "feature_module": "function_xss",
    "timestamp": "ISO8601",
    "duration_ms": 1234,
    "status": "completed|failed|timeout",
    "success": bool,
    
    "target": {
        "url": str,
        "endpoint": str,
        "method": str,
        "parameters": dict
    },
    
    "findings": [
        {
            "id": "uuid",
            "vulnerability_type": str,
            "severity": "critical|high|medium|low",
            "cvss_score": float,
            "cwe_id": str,
            "owasp_category": str,
            
            "title": str,
            "description": str,
            "evidence": {
                "payload": str,
                "request": str,
                "response": str,
                "confidence": float
            },
            
            "poc": {
                "steps": [str],
                "curl_command": str,
                "exploit_code": str
            },
            
            "impact": {
                "confidentiality": str,
                "integrity": str,
                "availability": str
            },
            
            "remediation": {
                "recommendation": str,
                "references": [str],
                "effort": str
            },
            
            "bounty_info": {
                "eligible": bool,
                "estimated_value": str,
                "program_relevance": float
            }
        }
    ],
    
    "statistics": {
        "payloads_tested": int,
        "requests_sent": int,
        "false_positives_filtered": int,
        "success_rate": float
    },
    
    "performance": {
        "avg_response_time_ms": float,
        "rate_limit_hits": int,
        "retries": int,
        "network_errors": int
    }
}
```

### Coordinator 返回的數據

```python
{
    "status": "success|duplicate|error",
    "task_id": str,
    
    # 內循環：優化數據
    "internal_loop": {
        "payload_efficiency": {payload_type: success_rate},
        "successful_patterns": [str],
        "recommended_concurrency": int,
        "recommended_timeout_ms": int,
        "strategy_adjustments": dict,
        "priority_adjustments": dict
    },
    
    # 外循環：報告數據
    "external_loop": {
        "total_findings": int,
        "critical_count": int,
        "high_count": int,
        "verified_findings": int,
        "bounty_eligible_count": int,
        "estimated_total_value": str,
        "owasp_coverage": dict,
        "cwe_distribution": dict,
        "findings": [Finding]
    },
    
    # 驗證結果
    "verification": [
        {
            "finding_id": str,
            "verified": bool,
            "confidence": float,
            "verification_method": str,
            "notes": str
        }
    ],
    
    # Core 反饋
    "feedback": {
        "execution_success": bool,
        "findings_count": int,
        "high_value_findings": int,
        "optimization_suggestions": OptimizationData,
        "recommended_next_actions": [str],
        "continue_testing": bool,
        "learning_data": dict
    }
}
```

## 使用範例

### 基礎使用

```python
from services.integration.coordinators import XSSCoordinator

# 1. 初始化協調器
coordinator = XSSCoordinator()

# 2. 處理 Features 返回的結果
result = await coordinator.collect_result(feature_result_dict)

# 3. 使用內循環數據（Core 自動收到）
optimization = result['internal_loop']
print(f"建議併發數: {optimization['recommended_concurrency']}")
print(f"Payload 效率: {optimization['payload_efficiency']}")

# 4. 使用外循環數據（生成報告）
report = result['external_loop']
print(f"高危漏洞: {report['high_count']}")
print(f"賞金預估: {report['estimated_total_value']}")
```

### 整合 MQ/DB/Cache

```python
from services.integration.coordinators import XSSCoordinator
from infrastructure.mq import RabbitMQClient
from infrastructure.db import PostgreSQLClient
from infrastructure.cache import RedisClient

# 初始化客戶端
mq_client = RabbitMQClient()
db_client = PostgreSQLClient()
cache_client = RedisClient()

# 初始化協調器（自動處理存儲和發送）
coordinator = XSSCoordinator(
    mq_client=mq_client,
    db_client=db_client,
    cache_client=cache_client
)

# 處理結果（自動發送反饋給 Core）
result = await coordinator.collect_result(feature_result_dict)
```

## 內循環數據用途

### 1. 優化測試策略

| 數據 | 用途 | 示例 |
|------|------|------|
| `payload_efficiency` | 優先使用成功率高的 payload | script_tag: 0.85 → 優先使用 |
| `successful_patterns` | 記錄成功模式供 ML 學習 | "html_body:script_tag:none" |
| `performance.avg_response_time_ms` | 動態調整超時設置 | 150ms → 設置 300ms 超時 |
| `rate_limit_hits` | 調整請求速率 | >0 → 降低併發數 |

### 2. 資源優化

```python
# 根據性能指標自動調整
if rate_limit_hits > 0:
    concurrency -= 2  # 降低併發
    rate_limit += 5   # 增加延遲

if avg_response_time < 100:
    concurrency += 5  # 提高併發
```

### 3. 智能決策

```python
# 根據成功模式選擇 payload
top_payloads = sorted_by_efficiency(payload_efficiency)[:5]
next_test_payloads = generate_variants(top_payloads)
```

## 外循環數據用途

### 1. 漏洞驗證標準

- ✅ **證據完整性**: Request + Response + Screenshot
- ✅ **可重現性**: 清晰的 PoC 步驟
- ✅ **置信度評分**: 0.0 - 1.0（0.7+ 為驗證通過）
- ✅ **誤報過濾**: 自動檢測編碼、CSP 等

### 2. 報告類型

#### 技術報告（開發團隊）
```
- 漏洞詳情、PoC、修復建議
- CVSS 評分、CWE 分類、OWASP 映射
- 影響範圍、利用難度
```

#### 管理報告（管理層）
```
- 風險評分、嚴重程度分布
- 合規性檢查（PCI-DSS、SOC2）
- 修復優先級、投入預估
```

#### Bug Bounty 報告（客戶/平台）
```
- 清晰的 PoC（步驟 + Curl + 截圖）
- 影響說明、賞金預估
- 相關性評分、提交就緒度
```

### 3. 持續改進循環

```
客戶回饋 → 學習數據
├─ 接受 → 提高該類型檢測權重
├─ 拒絕（誤報）→ 調整過濾閾值
└─ 需補充 → 優化證據收集邏輯
```

## 數據存儲策略

### 1. 時序數據（InfluxDB/TimescaleDB）
- 性能指標、響應時間、成功率
- 用於趨勢分析、異常檢測

### 2. 文檔數據（MongoDB/Elasticsearch）
- 完整結果、Finding 詳情、證據
- 用於全文搜索、相似性分析

### 3. 關聯數據（PostgreSQL）
- 任務關聯、漏洞去重、統計
- 用於報告生成、資料分析

### 4. 緩存數據（Redis）
- 實時反饋、Payload 排名、去重
- 用於快速決策、即時優化

## 擴展指南

### 創建新的協調器

```python
from services.integration.coordinators import BaseCoordinator

class MyCoordinator(BaseCoordinator):
    def __init__(self, **kwargs):
        super().__init__(feature_module="function_my_feature", **kwargs)
    
    async def _extract_optimization_data(self, result):
        # 實現：分析性能、payload 效率、策略建議
        return OptimizationData(...)
    
    async def _extract_report_data(self, result):
        # 實現：整理漏洞、統計分類、賞金信息
        return ReportData(...)
    
    async def _verify_findings(self, result):
        # 實現：驗證漏洞真實性、過濾誤報
        return [VerificationResult(...)]
```

## 參考標準

- **SARIF**: Static Analysis Results Interchange Format
- **OWASP**: OWASP Testing Guide v4.2
- **CVSS**: Common Vulnerability Scoring System 3.1
- **CWE**: Common Weakness Enumeration
- **HackerOne/Bugcrowd**: Bug Bounty 平台標準
- **OpenTelemetry**: 可觀測性標準

## 下一步

1. ✅ BaseCoordinator 和 XSSCoordinator 完成
2. ⬜ 實現 SQLiCoordinator
3. ⬜ 實現 WebScannerCoordinator
4. ⬜ 實現 DDoSCoordinator
5. ⬜ 整合 MQ/DB/Cache 客戶端
6. ⬜ 整合測試和驗證
7. ⬜ 性能優化和壓力測試
