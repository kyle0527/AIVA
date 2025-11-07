# AIVA 實際狀況快速開始指南 🚀

> **更新日期**: 2025年11月7日  
> **目的**: 提供 AIVA 實際可用功能的真實指南  
> **重要**: 本指南基於實際測試結果，不包含無法使用的功能

---

## 🎯 在開始之前

### ⚠️ **重要提醒**
AIVA 當前為**研究原型階段**，以下是實際狀況：
- ✅ **可用**: AI 對話系統、基礎架構、代碼研究
- ❌ **不可用**: 大部分安全檢測功能、自動化掃描
- 🚧 **開發中**: 核心檢測邏輯、Bug Bounty 功能

---

## 🚀 5分鐘實際體驗

### Step 1: 環境準備
```bash
# 進入 AIVA 目錄
cd /path/to/AIVA-git

# 確保 Python 環境
python --version  # 需要 Python 3.8+
```

### Step 2: 測試 AI 對話助手 (✅ 可用)
```bash
python -c "
import asyncio
import sys
sys.path.append('.')
from services.core.aiva_core.dialog.assistant import AIVADialogAssistant

async def test():
    print('🤖 正在初始化 AI 對話助手...')
    assistant = AIVADialogAssistant()
    
    queries = [
        '系統狀況如何？',
        '你能做什麼？',
        '有哪些可用的功能？'
    ]
    
    for query in queries:
        print(f'\\n❓ {query}')
        result = await assistant.process_user_input(query)
        print(f'🤖 {result[\"message\"][:150]}...')

asyncio.run(test())
"
```

### Step 3: 檢查系統能力 (✅ 可用)
```bash
python -c "
import asyncio
import sys
sys.path.append('.')
from services.integration.capability.registry import CapabilityRegistry

async def check():
    print('📊 檢查系統註冊的安全檢測能力...')
    registry = CapabilityRegistry()
    caps = await registry.list_capabilities()
    
    print(f'註冊的檢測能力: {len(caps) if caps else 0} 個')
    if not caps or len(caps) == 0:
        print('💡 這表示安全檢測功能尚未實現')
    
asyncio.run(check())
"
```

### Step 4: 了解限制 (❌ 無法使用的功能示例)
```bash
# 以下指令會失敗，展示當前限制：

echo "嘗試導入 SQL 注入檢測模組..."
python -c "
try:
    from services.features.function_sqli import SmartDetectionManager
    print('✅ SQL 注入模組可用')
except Exception as e:
    print(f'❌ SQL 注入模組無法使用: {e}')
"

echo -e "\\n總結: 大部分安全檢測功能仍在開發中"
```

---

## 🎯 實際可以做什麼

### ✅ **推薦用途**
1. **學習架構設計**: 研究兩階段智能分離架構
2. **AI 整合參考**: 了解 AI 對話系統實現方式  
3. **代碼研究**: 分析多語言安全工具整合思路
4. **開發參與**: 參與修復模組依賴問題

### ❌ **不建議用途**
1. **實際安全測試**: 檢測功能不完整
2. **Bug Bounty 工作**: 缺乏真正的漏洞發現能力
3. **生產環境使用**: 穩定性和可靠性不足

---

## 🔧 開發者快速上手

### 想要貢獻代碼？
```bash
# 1. 檢查模組依賴問題
python -c "
import sys
sys.path.append('.')
try:
    from services.features import *
    print('✅ Features 模組可用')
except Exception as e:
    print(f'❌ Features 模組問題: {e}')
    print('💡 這是一個可以修復的問題！')
"

# 2. 查看實際能力評估
cat AIVA_REALISTIC_CAPABILITY_ASSESSMENT.md

# 3. 了解架構設計
ls -la services/  # 五大模組結構
```

### 改進優先級建議
1. **修復模組依賴** - 讓基礎檢測功能可以導入
2. **實現一個完整檢測** - 建議從 SQL 注入開始  
3. **建立實際測試** - 用真實靶場驗證功能
4. **完善 AI 決策** - 替換佔位符代碼

---

## 📚 更多資源

- **實際能力評估**: [AIVA_REALISTIC_CAPABILITY_ASSESSMENT.md](AIVA_REALISTIC_CAPABILITY_ASSESSMENT.md)
- **架構分析**: [AIVA_TWO_PHASE_ARCHITECTURE_ANALYSIS.md](AIVA_TWO_PHASE_ARCHITECTURE_ANALYSIS.md)  
- **網路研究**: [AIVA_NETWORK_ARCHITECTURE_RESEARCH_SUPPLEMENT.md](AIVA_NETWORK_ARCHITECTURE_RESEARCH_SUPPLEMENT.md)

---

## 💡 總結

AIVA 是一個有創新架構設計的安全測試框架原型，但當前**不適合實際安全測試使用**。

**如果你是**:
- 🎓 **學習者**: 可以研究架構設計和 AI 整合思路
- 👨‍💻 **開發者**: 可以參與修復和完善功能  
- 🔍 **安全研究員**: 建議等待功能完善後再使用
- 💰 **Bug Bounty 獵人**: 目前不建議使用，功能不完整

**AIVA 的價值在於創新的設計思路，而不是當前的實現完整度。**