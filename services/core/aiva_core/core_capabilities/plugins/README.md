# 🔌 Plugins - 插件系統

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 模組概述](#-模組概述)
  - [核心能力](#核心能力)
- [📂 檔案列表](#-檔案列表)
- [🔧 核心組件](#-核心組件)
  - [EnhancedCapabilityRegistry - 增強能力註冊中心](#enhancedcapabilityregistry---增強能力註冊中心)
  - [AISummaryPlugin - AI 摘要插件](#aisummaryplugin---ai-摘要插件)
- [🏗️ 插件架構](#-插件架構)
- [🚀 使用範例](#-使用範例)
  - [註冊和使用插件](#註冊和使用插件)
  - [自定義插件](#自定義插件)
  - [能力編排](#能力編排)
  - [插件統計](#插件統計)
- [🎯 AI 摘要生成示例](#-ai-摘要生成示例)
- [📊 插件管理](#-插件管理)
  - [啟用/禁用插件](#啟用禁用插件)
  - [插件版本管理](#插件版本管理)
- [📚 相關文檔](#-相關文檔)

---

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: v2.1.2  
> **狀態**: ✅ 生產就緒  
> **最後更新**: 2025-12-20  
> **代碼量**: 1 個 Python 檔案，約 617 行代碼  
> **角色**: AIVA 的「擴展中樞」- 可插拔的智能分析模組

---

## 🎯 模組概述

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [插件架構](#插件架構)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**Plugins** 子模組提供可插拔的能力擴展系統，支援動態註冊和管理各種功能插件。核心插件包括 AI 摘要生成、能力編排和智能分析等。

### 核心能力
1. **動態註冊** - 運行時動態註冊插件能力
2. **能力編排** - 智能編排多個能力協同工作
3. **AI 摘要** - 基於 AI 的結果摘要生成
4. **插件管理** - 統一的插件生命週期管理

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **ai_summary_plugin.py** | 617 | AI 摘要插件 - 智能分析和能力註冊 | ✅ 生產 |

**總計**: 約 617 行代碼

---

## 🔧 核心組件

### EnhancedCapabilityRegistry - 增強能力註冊中心

**檔案**: `ai_summary_plugin.py` (617 行)

整合 v1 能力註冊和 AI 模組智能編排的統一註冊系統。

#### 核心類別

```python
class EnhancedCapabilityRegistry:
    """增強的能力註冊中心 - 整合 v1 和 AI 模組功能
    
    功能:
    - 基礎能力註冊 (來自 v1)
    - 智能編排系統 (來自 AI 模組)
    - 插件元數據管理
    - 統計和性能追蹤
    """
    
    def __init__(self):
        # 基礎註冊表 (來自 v1)
        self._capabilities: Dict[str, Dict[str, Any]] = {}
        
        # 智能編排系統 (來自 AI 模組)
        self._orchestration_rules: Dict[str, Dict[str, Any]] = {}
        self._capability_dependencies: Dict[str, List[str]] = {}
        
        # 插件元數據系統
        self._plugin_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 統計和性能追蹤
        self._stats = {
            'total_registrations': 0,
            'successful_executions': 0,
            'failed_executions': 0
        }
    
    def register_capability(
        self,
        name: str,
        handler: Callable,
        metadata: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None
    ):
        """註冊能力"""
        
    def execute_capability(
        self,
        name: str,
        context: Dict[str, Any]
    ) -> Any:
        """執行能力"""
        
    def orchestrate(
        self,
        capabilities: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """編排多個能力"""
```

---

### AISummaryPlugin - AI 摘要插件

```python
class AISummaryPlugin:
    """AI 摘要插件
    
    功能:
    - 掃描結果智能摘要
    - 漏洞分析和建議
    - 風險評估報告
    """
    
    def __init__(self, registry: EnhancedCapabilityRegistry):
        self.registry = registry
        self._register_capabilities()
    
    def _register_capabilities(self):
        """註冊插件能力"""
        
        # 註冊摘要生成能力
        self.registry.register_capability(
            name="generate_summary",
            handler=self.generate_summary,
            metadata={
                "description": "生成 AI 增強的掃描結果摘要",
                "version": "1.0.0",
                "author": "AIVA Team"
            }
        )
        
        # 註冊風險評估能力
        self.registry.register_capability(
            name="assess_risk",
            handler=self.assess_risk,
            metadata={
                "description": "評估整體安全風險",
                "version": "1.0.0"
            }
        )
    
    async def generate_summary(
        self,
        findings: List[Dict]
    ) -> Dict[str, Any]:
        """生成智能摘要"""
        
    async def assess_risk(
        self,
        findings: List[Dict]
    ) -> Dict[str, Any]:
        """風險評估"""
```

---

## 🏗️ 插件架構

```
┌─────────────────────────────────────────────────┐
│         EnhancedCapabilityRegistry              │
│     (增強能力註冊中心)                           │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐  ┌──────────────┐           │
│  │ Capability   │  │ Orchestration│           │
│  │ Registry     │  │ Engine       │           │
│  └──────┬───────┘  └──────┬───────┘           │
│         │                  │                   │
│         └────────┬─────────┘                   │
│                  │                             │
└──────────────────┼─────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼─────┐      ┌─────▼────┐
    │ AI       │      │ Custom   │
    │ Summary  │      │ Plugins  │
    │ Plugin   │      │          │
    └──────────┘      └──────────┘
```

---

## 🚀 使用範例

### 註冊和使用插件

```python
from core_capabilities.plugins import (
    EnhancedCapabilityRegistry,
    AISummaryPlugin
)

# 1. 創建註冊中心
registry = EnhancedCapabilityRegistry()

# 2. 註冊 AI 摘要插件
ai_plugin = AISummaryPlugin(registry)

# 3. 使用插件能力
findings = [
    {"name": "SQL Injection", "severity": "high", ...},
    {"name": "XSS", "severity": "medium", ...}
]

# 生成摘要
summary = await registry.execute_capability(
    name="generate_summary",
    context={"findings": findings}
)

print(summary)
# 輸出:
# {
#   "total_findings": 2,
#   "critical_issues": 1,
#   "key_findings": [...],
#   "recommendations": [...],
#   "executive_summary": "發現 1 個高危 SQL 注入漏洞..."
# }
```

### 自定義插件

```python
class CustomPlugin:
    """自定義插件示例"""
    
    def __init__(self, registry: EnhancedCapabilityRegistry):
        self.registry = registry
        self._register()
    
    def _register(self):
        """註冊能力"""
        
        # 註冊自定義能力
        self.registry.register_capability(
            name="custom_analysis",
            handler=self.analyze,
            metadata={
                "description": "自定義分析功能",
                "version": "1.0.0"
            },
            dependencies=["generate_summary"]  # 依賴其他能力
        )
    
    async def analyze(self, data: Dict) -> Dict:
        """自定義分析邏輯"""
        # 可以調用其他已註冊的能力
        summary = await self.registry.execute_capability(
            "generate_summary",
            {"findings": data.get("findings", [])}
        )
        
        # 執行自定義邏輯
        custom_result = self._custom_logic(data)
        
        return {
            "summary": summary,
            "custom": custom_result
        }
    
    def _custom_logic(self, data: Dict) -> Dict:
        """實現自定義邏輯"""
        return {"processed": True}

# 使用
custom_plugin = CustomPlugin(registry)
result = await registry.execute_capability(
    "custom_analysis",
    {"findings": findings}
)
```

### 能力編排

```python
# 編排多個能力協同工作
orchestration_result = await registry.orchestrate(
    capabilities=[
        "generate_summary",
        "assess_risk",
        "custom_analysis"
    ],
    context={
        "findings": findings,
        "target": "https://example.com"
    }
)

# 結果包含所有能力的輸出
print(orchestration_result)
# {
#   "generate_summary": {...},
#   "assess_risk": {...},
#   "custom_analysis": {...},
#   "execution_time": 1.23,
#   "success": True
# }
```

### 插件統計

```python
# 查看插件統計信息
stats = registry.get_statistics()

print(f"已註冊能力: {stats['total_registrations']}")
print(f"成功執行: {stats['successful_executions']}")
print(f"失敗次數: {stats['failed_executions']}")
print(f"成功率: {stats['success_rate']:.2%}")

# 查看能力依賴圖
dependencies = registry.get_dependency_graph()
print(json.dumps(dependencies, indent=2))
```

---

## 🎯 AI 摘要生成示例

```python
# 使用 AI 插件生成智能摘要
summary = await ai_plugin.generate_summary(findings)

# 摘要示例
{
  "executive_summary": """
    本次掃描發現 127 個安全問題，其中 3 個為嚴重級別，
    15 個為高危級別。主要風險集中在身份驗證繞過和 
    SQL 注入漏洞。建議立即修復嚴重級別問題。
  """,
  "key_findings": [
    {
      "title": "SQL 注入漏洞",
      "severity": "critical",
      "count": 3,
      "impact": "可能導致數據庫完全洩露",
      "recommendation": "使用參數化查詢"
    },
    {
      "title": "身份驗證繞過",
      "severity": "high",
      "count": 2,
      "impact": "未授權訪問管理功能",
      "recommendation": "加強認證機制"
    }
  ],
  "risk_assessment": {
    "overall_risk": "high",
    "score": 8.3,
    "factors": {
      "vulnerability_severity": 9.0,
      "exploitability": 8.5,
      "business_impact": 8.0
    }
  },
  "recommendations": [
    "立即修復 SQL 注入漏洞（3 個）",
    "實施 WAF 保護",
    "加強輸入驗證",
    "定期安全審計"
  ],
  "trends": {
    "compared_to_last_scan": "+15%",
    "most_common_vuln_type": "SQL Injection",
    "improvement_areas": ["認證機制", "輸入驗證"]
  }
}
```

---

## 📊 插件管理

### 啟用/禁用插件

```python
# 禁用插件
registry.disable_capability("generate_summary")

# 啟用插件
registry.enable_capability("generate_summary")

# 檢查狀態
is_enabled = registry.is_capability_enabled("generate_summary")
```

### 插件版本管理

```python
# 註冊多個版本
registry.register_capability(
    name="generate_summary_v1",
    handler=summary_v1,
    metadata={"version": "1.0.0"}
)

registry.register_capability(
    name="generate_summary_v2",
    handler=summary_v2,
    metadata={"version": "2.0.0"}
)

# 使用指定版本
result = await registry.execute_capability("generate_summary_v2", context)
```

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Processing 子模組](../processing/README.md) - 結果處理
- [Dialog 子模組](../dialog/README.md) - 對話助理
- [Cognitive Core - Decision](../../cognitive_core/decision/README.md) - 決策引擎

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
