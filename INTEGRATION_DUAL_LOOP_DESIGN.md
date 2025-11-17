"""Integration 模組 - 雙閉環數據收集與協調設計

基於 AIVA 雙閉環架構：
1. **內循環（Internal Loop）**: Core ↔ Features ↔ Integration
   - 目的：優化測試策略、自我學習、即時調整
   
2. **外循環（External Loop）**: Integration → 報告輸出 → 客戶 → 回饋 → Core
   - 目的：漏洞驗證、報告生成、客戶回饋、持續改進

## 數據流向分析

### 內循環（Internal Loop）
```
Core (決策)
  ↓ 發送測試任務
Features (執行)
  ↓ 返回原始結果
Integration (收集+分析)
  ↓ 回饋優化建議
Core (調整策略)
```

### 外循環（External Loop）  
```
Integration (彙總結果)
  ↓ 生成報告
Report Generator (格式化)
  ↓ 輸出報告
客戶/Bug Bounty 平台
  ↓ 回饋（接受/拒絕/需補充）
Core (學習調整)
```

---

## Features 返回的數據結構（參考業界標準）

基於 OWASP、SARIF (Static Analysis Results Interchange Format)、
CVE 標準和 Bug Bounty 平台要求（HackerOne、Bugcrowd）：

### 1. 通用結果結構（所有 Features 共用）
```python
{
    "task_id": "uuid",
    "feature_module": "function_xss|function_sqli|...",
    "timestamp": "ISO8601",
    "duration_ms": 1234,
    
    # 執行狀態
    "status": "completed|failed|timeout",
    "success": true,
    
    # 目標信息
    "target": {
        "url": "https://example.com",
        "endpoint": "/api/users",
        "method": "POST",
        "parameters": {...}
    },
    
    # 測試結果
    "findings": [
        {
            "id": "finding-uuid",
            "vulnerability_type": "xss_reflected|sqli_union|...",
            "severity": "critical|high|medium|low|info",
            "cvss_score": 7.5,  # CVSS 3.1
            "cwe_id": "CWE-79",
            "owasp_category": "A03:2021-Injection",
            
            # 漏洞詳情
            "title": "Reflected XSS in search parameter",
            "description": "User input is reflected without sanitization",
            "evidence": {
                "payload": "<script>alert('XSS')</script>",
                "request": "...",
                "response": "...",
                "matched_pattern": "...",
                "confidence": 0.95  # 0-1
            },
            
            # PoC（Proof of Concept）
            "poc": {
                "steps": ["1. Navigate to...", "2. Enter payload..."],
                "curl_command": "curl -X POST ...",
                "exploit_code": "...",
                "screenshot_path": "/path/to/evidence.png"
            },
            
            # 影響評估
            "impact": {
                "confidentiality": "high|medium|low",
                "integrity": "high|medium|low", 
                "availability": "high|medium|low",
                "scope_changed": false
            },
            
            # 修復建議
            "remediation": {
                "recommendation": "Implement input validation and output encoding",
                "references": [
                    "https://owasp.org/...",
                    "https://cwe.mitre.org/..."
                ],
                "effort": "low|medium|high"
            },
            
            # Bug Bounty 特定
            "bounty_info": {
                "eligible": true,
                "estimated_value": "$500-$2000",
                "program_relevance": 0.9
            }
        }
    ],
    
    # 統計信息（內循環優化用）
    "statistics": {
        "payloads_tested": 100,
        "requests_sent": 120,
        "false_positives_filtered": 5,
        "time_per_payload_ms": 10.5,
        "success_rate": 0.85
    },
    
    # 性能指標（內循環優化用）
    "performance": {
        "avg_response_time_ms": 150,
        "max_response_time_ms": 500,
        "rate_limit_hits": 0,
        "retries": 2,
        "network_errors": 0
    },
    
    # 錯誤信息（如果失敗）
    "errors": [
        {
            "code": "TIMEOUT",
            "message": "Request timeout after 30s",
            "recoverable": true
        }
    ]
}
```

---

## Integration 協調接口設計

### 基礎協調器（Base Coordinator）
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime
import uuid

class BaseCoordinator(ABC):
    """基礎協調器 - 所有具體協調器的父類"""
    
    def __init__(self, mq_client, db_client, cache_client):
        self.mq_client = mq_client
        self.db_client = db_client
        self.cache_client = cache_client
        
    async def collect_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """收集 Features 返回的結果"""
        # 1. 驗證結果格式
        validated = await self._validate_result(result)
        
        # 2. 存儲原始結果（內循環）
        await self._store_raw_result(validated)
        
        # 3. 提取優化數據（內循環）
        optimization_data = await self._extract_optimization_data(validated)
        
        # 4. 提取報告數據（外循環）
        report_data = await self._extract_report_data(validated)
        
        # 5. 驗證漏洞真實性
        verification = await self._verify_findings(validated)
        
        # 6. 回饋給 Core
        feedback = await self._generate_feedback(
            optimization_data, 
            verification
        )
        
        return {
            "internal_loop": optimization_data,
            "external_loop": report_data,
            "verification": verification,
            "feedback": feedback
        }
    
    @abstractmethod
    async def _validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """驗證結果格式"""
        pass
    
    @abstractmethod
    async def _extract_optimization_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取內循環優化數據"""
        pass
    
    @abstractmethod
    async def _extract_report_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """提取外循環報告數據"""
        pass
    
    @abstractmethod
    async def _verify_findings(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """驗證漏洞真實性"""
        pass
```

---

## 內循環數據用途

### 1. 優化測試策略
- **Payload 效率**: 哪些 payload 成功率高 → 優先使用
- **響應時間**: 哪些端點慢 → 調整超時設置
- **誤報率**: 哪些檢測邏輯誤報多 → 調整閾值
- **成功模式**: 記錄成功案例 → 機器學習訓練

### 2. 資源優化
- **並發控制**: 根據目標響應調整併發數
- **Rate Limiting**: 學習目標的速率限制
- **重試策略**: 優化重試時機和次數

### 3. 智能調整
- **動態 Payload 選擇**: 根據目標特徵選擇 payload
- **優先級排序**: 優先測試高風險端點
- **自適應超時**: 根據歷史響應時間調整

---

## 外循環數據用途

### 1. 漏洞驗證
- **證據完整性**: Request/Response/Screenshot
- **可重現性**: PoC 步驟是否清晰
- **置信度**: 檢測邏輯的信心分數
- **誤報過濾**: 自動識別和過濾誤報

### 2. 報告生成
- **技術報告**: 給開發團隊的詳細報告
  * 漏洞描述、PoC、修復建議
  * CVSS、CWE、OWASP 分類
  
- **管理報告**: 給管理層的風險概覽
  * 嚴重程度分布、風險評分
  * 合規性檢查（OWASP Top 10、PCI-DSS）
  
- **Bug Bounty 報告**: 給平台/客戶的提交
  * 清晰的 PoC、影響說明
  * 賞金預估、相關性評分

### 3. 持續改進
- **客戶回饋**: 接受/拒絕/需補充 → 調整檢測邏輯
- **賞金數據**: 記錄實際賞金 → 優化價值評估
- **誤報追蹤**: 記錄誤報案例 → 訓練過濾模型

---

## 數據存儲策略（參考 ELK、時序數據庫）

### 1. 時序數據（InfluxDB/TimescaleDB）
- 性能指標、響應時間、成功率
- 用於趨勢分析、性能監控

### 2. 文檔數據（MongoDB/Elasticsearch）
- 完整結果、Finding 詳情
- 用於全文搜索、相似性分析

### 3. 關聯數據（PostgreSQL）
- 任務關聯、漏洞去重
- 用於統計分析、報告生成

### 4. 緩存數據（Redis）
- 即時反饋、Payload 排名
- 用於快速決策、實時優化

---

## 下一步實現計劃

1. **創建 BaseCoordinator 抽象類**
2. **實現具體協調器**（XSS、SQLi、Web、DDoS）
3. **設計數據驗證 Schema**（Pydantic）
4. **實現漏洞驗證邏輯**
5. **集成報告生成器**
6. **建立回饋循環機制**
"""
