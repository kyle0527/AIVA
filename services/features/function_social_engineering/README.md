# 🎭 社會工程模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                  社會工程分析架構                            │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │  SocEng Engine │  OSINT      │
│  Interface    │             │               │  Integration │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  soceng_mgr    │  osint_db   │
│ .SOCENG_RECON │             │  ─────────────  │  social_api │
│      │        │             │   - profiling  │      │      │
│      └────────┼─────────────┼─  - pretext    │      ↓      │
│               │             │   - attack_vec │  Target     │
│               ↓             │       ↓        │  Profile    │
│         TaskResult          │   Attack       │  Database   │
│         (aiva_common)       │   Scenarios    │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **目標研究** - OSINT 收集目標資訊
2. **心理分析** - 分析目標的行為模式
3. **攻擊向量** - 設計可能的社工攻擊方案
4. **風險評估** - 評估組織的社工風險等級

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.SOCENG_RECON,
    payload={
        "target_domain": "example.com",
        "target_individuals": ["john.doe@example.com"],
        "information_sources": ["linkedin", "social_media", "public_records"],
        "attack_simulation": False  # 僅分析，不執行
    }
)
```

## 🔧 核心能力
- **OSINT 整合**: 自動化資訊收集
- **人員檔案**: 建立目標人員行為模式
- **攻擊模擬**: 預測可能的社工攻擊向量
- **培訓建議**: 提供社工防護培訓建議

## ⚠️ 倫理使用
- 僅限授權的安全測試
- 不得用於實際欺騙或詐騙
- 遵守當地法律法規

## 🎯 後續發展
- [ ] **AI 人格建模** - 更精準的心理分析
- [ ] **多語言支援** - 跨文化社工分析
- [ ] **防護建議** - 自動化防護措施建議