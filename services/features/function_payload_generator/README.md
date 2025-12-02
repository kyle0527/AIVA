# 🎯 載荷生成器模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   載荷生成架構                              │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │ Payload Engine │  Template   │
│  Interface    │             │               │   Library    │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  payload_gen   │  template   │
│.PAYLOAD_CREATE│             │  ─────────────  │  database   │
│      │        │             │   - xss        │      │      │
│      └────────┼─────────────┼─  - sqli       │      ↓      │
│               │             │   - cmd_inj    │  Custom     │
│               ↓             │       ↓        │  Generation │
│         TaskResult          │   Encoding     │             │
│         (aiva_common)       │   & Bypass     │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **載荷類型選擇** - 根據攻擊類型選擇模板
2. **目標適配** - 針對目標環境客製化
3. **編碼處理** - 應用繞過和編碼技術
4. **載荷輸出** - 生成可用的攻擊載荷

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.PAYLOAD_CREATE,
    payload={
        "payload_type": "xss",  # xss|sqli|cmd_inj|xxe
        "target_context": "form_input",
        "bypass_filters": ["waf", "xss_filter"],
        "encoding": "url_double"
    }
)
```

## 🔧 核心能力
- **多類型載荷**: 支援主要攻擊向量
- **智能編碼**: 自動選擇適當編碼方式
- **繞過技術**: WAF 和過濾器規避

## 🎯 後續發展
- [ ] **AI 生成** - 機器學習輔助載荷創建
- [ ] **零日載荷** - 最新攻擊技術整合