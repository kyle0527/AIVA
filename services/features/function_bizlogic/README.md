# 💼 業務邏輯漏洞檢測

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                  業務邏輯檢測架構                            │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │  Logic Engine  │  Workflow   │
│  Interface    │             │               │   Analysis   │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  logic_test    │  flow_map   │
│ .BIZLOGIC_TEST│             │  ─────────────  │      │      │
│      │        │             │   - workflow   │      ↓      │
│      └────────┼─────────────┼─  - race_cond  │  Anomaly    │
│               │             │   - rate_limit │  Detection  │
│               ↓             │       ↓        │             │
│         FindingPayload      │   Pattern      │             │
│         (aiva_common)       │   Violation    │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **流程分析** - 理解業務工作流程
2. **異常測試** - 執行非正常操作序列
3. **邊界測試** - 測試極限和邊界條件
4. **競態檢測** - 並發操作異常檢測

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.BIZLOGIC_TEST,
    payload={
        "workflow_endpoints": ["/cart/add", "/checkout", "/payment"],
        "test_scenarios": ["race_condition", "workflow_bypass", "limit_bypass"],
        "concurrent_requests": 10
    }
)
```

## 🔧 核心能力
- **工作流程分析**: 自動理解業務邏輯
- **競態條件**: 並發請求異常檢測
- **限制繞過**: 速率限制和額度繞過

## 🎯 後續發展
- [ ] **AI 流程學習** - 自動學習業務邏輯模式
- [ ] **時序分析** - 基於時間的邏輯漏洞
