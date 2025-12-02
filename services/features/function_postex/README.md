# 🎯 後滲透模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   後滲透操作架構                            │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │ PostEx Engine  │  Technique  │
│  Interface    │             │               │   Database   │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  postex_mgr    │  mitre_db   │
│ .POSTEX_EXEC  │             │  ─────────────  │  tactic_map │
│      │        │             │   - privilege  │      │      │
│      └────────┼─────────────┼─  - lateral    │      ↓      │
│               │             │   - persist    │  TTPs       │
│               ↓             │       ↓        │  Mapping    │
│         TaskResult          │   MITRE        │             │
│         (aiva_common)       │   ATT&CK       │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **初始評估** - 確認已獲得的存取權限
2. **權限提升** - 嘗試獲得更高權限
3. **橫向移動** - 在網路中擴展影響範圍
4. **持久化** - 建立持續存取機制

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.POSTEX_EXECUTE,
    payload={
        "target_session": "session_id_123",
        "techniques": ["privilege_escalation", "lateral_movement"],
        "scope_limitation": "subnet_only",
        "stealth_mode": True
    }
)
```

## 🔧 核心能力
- **MITRE ATT&CK 對應**: 標準戰術技術程序
- **權限提升**: Windows/Linux 權限提升技術
- **橫向移動**: 網路內部移動技術
- **隱蔽性**: 避免檢測的技術

## ⚠️ 安全使用
- 僅限授權的滲透測試
- 嚴格控制影響範圍
- 詳細記錄所有操作

## 🎯 後續發展
- [ ] **EDR 規避** - 現代端點檢測規避技術
- [ ] **容器化環境** - Docker/K8s 後滲透技術
- [ ] **雲端環境** - AWS/Azure 橫向移動