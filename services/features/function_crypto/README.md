# 🔐 密碼學分析模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   密碼學分析架構                             │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │  Crypto Engine │  Algorithm  │
│  Interface    │             │               │   Database   │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  crypto_test   │  cipher_db  │
│ .CRYPTO_AUDIT │             │  ─────────────  │      │      │
│      │        │             │   - weak_algo  │      ↓      │
│      └────────┼─────────────┼─  - key_reuse  │  Weakness   │
│               │             │   - entropy    │  Analysis   │
│               ↓             │       ↓        │             │
│         FindingPayload      │   Strength     │             │
│         (aiva_common)       │   Assessment   │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **演算法識別** - 檢測使用的加密算法
2. **強度分析** - 評估金鑰長度和隨機性
3. **實作檢查** - 發現常見實作錯誤
4. **弱點評估** - 產生安全建議

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.CRYPTO_AUDIT,
    payload={
        "target_service": "https://api.example.com",
        "check_algorithms": ["ssl_tls", "jwt", "passwords"],
        "analysis_depth": "comprehensive"
    }
)
```

## 🔧 核心能力
- **弱演算法檢測**: MD5, SHA1, DES 等過時算法
- **金鑰管理**: 硬編碼金鑰和金鑰重用檢測
- **隨機性分析**: 熵不足和可預測性檢測

## 🎯 後續發展
- [ ] **量子安全** - 後量子密碼學算法支援
- [ ] **側信道攻擊** - 時序攻擊檢測