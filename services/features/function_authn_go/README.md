# 🔐 權限驗證模組 (Go)

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   權限驗證檢測架構                           │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.go  │  Auth Engine   │  Session    │
│  Interface    │             │               │  Management  │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  auth_bypass   │  session    │
│  .AUTHN_TEST  │             │  ─────────────  │  tracking   │
│      │        │             │   - session    │      │      │
│      └────────┼─────────────┼─  - token      │      ↓      │
│               │             │   - cookie     │  Validation │
│               ↓             │       ↓        │    Store    │
│         FindingPayload      │   Bypass       │             │
│         (aiva_common)       │   Detection    │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **接收指令** - AI 下達 `AUTHN_TEST` 命令
2. **認證機制分析** - 識別驗證方式和會話管理
3. **繞過測試** - 執行多種繞過技術
4. **結果分析** - 評估繞過成功率和影響

## 🚀 支援指令

### AI 命令系統
```python
from aiva_common import AICommand, CommandType

command = AICommand(
    command_type=CommandType.AUTHN_TEST,
    payload={
        "target_endpoint": "/admin/dashboard",
        "auth_mechanisms": ["session", "jwt", "oauth"],
        "test_depth": "comprehensive"
    }
)
```

### 判斷使用時機
- ✅ **適用**: 管理介面、API 端點、用戶會話
- ⚠️ **注意**: 避免影響正常用戶會話

## 🎯 後續發展
- [ ] **JWT 增強** - 更多 token 攻擊技術
- [ ] **OAuth 2.0** - 現代認證協議支援
- [ ] **MFA 繞過** - 多因子認證測試