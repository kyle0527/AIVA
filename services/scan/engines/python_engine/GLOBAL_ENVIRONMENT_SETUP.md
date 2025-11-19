# Python Engine - 全域環境安裝指南

> **目的**: 確保 Python Engine 在全域環境中正確配置，避免虛擬環境依賴問題  
> **適用場景**: 多專案共享依賴、系統級工具、開發環境統一配置  
> **最後更新**: 2025-11-19  
> **驗證狀態**: ✅ 已在 Juice Shop 靶場驗證通過

---

## 📑 目錄

- [📋 為什麼需要全域安裝？](#-為什麼需要全域安裝)
- [🔧 安裝步驟](#-安裝步驟)
- [🧪 功能驗證測試](#-功能驗證測試)
- [📦 完整依賴清單](#-完整依賴清單)
- [🔍 故障排查](#-故障排查)
- [✅ 驗證檢查清單](#-驗證檢查清單)
- [🎯 測試結果參考](#-測試結果參考)
- [🔗 相關文檔](#-相關文檔)
- [💡 最佳實踐](#-最佳實踐)

---

## 📋 為什麼需要全域安裝？

### 問題背景

在開發過程中發現虛擬環境存在以下問題：

1. **依賴不一致**: 不同專案的 venv 可能缺少關鍵依賴
2. **Playwright 衝突**: 瀏覽器驅動路徑在 venv 中不穩定
3. **維護成本**: 每個專案都需要獨立安裝和管理依賴
4. **測試困難**: 快速驗證時需要頻繁切換環境

### 全域安裝優勢

✅ **一次安裝，處處可用**: 所有專案共享同一套依賴  
✅ **穩定性高**: 系統級安裝路徑固定，不易出錯  
✅ **測試方便**: 可以直接使用 `python` 命令測試  
✅ **瀏覽器驅動統一**: Playwright 瀏覽器只需安裝一次

---

## 🔧 安裝步驟

### 1. 確認 Python 版本

```powershell
# 檢查 Python 版本（需要 3.11+）
python --version
# 應顯示: Python 3.13.x 或更高

# 確認 pip 可用
python -m pip --version
```

### 2. 安裝核心依賴

```powershell
# HTTP 和網絡相關
python -m pip install httpx aiohttp

# HTML 解析（關鍵！）
python -m pip install beautifulsoup4 lxml

# 異步和數據處理
python -m pip install pydantic

# Playwright（動態渲染）
python -m pip install playwright
```

### 3. 安裝 Playwright 瀏覽器

```powershell
# 安裝 Chromium 瀏覽器驅動
playwright install chromium

# 如果需要完整依賴（Linux/WSL）
playwright install --with-deps chromium
```

### 4. 驗證安裝

```powershell
# 測試 BeautifulSoup
python -c "from bs4 import BeautifulSoup; print('✅ BeautifulSoup 可用')"

# 測試 Playwright
python -c "from playwright.async_api import async_playwright; print('✅ Playwright 可用')"

# 測試 httpx
python -c "import httpx; print('✅ httpx 可用')"
```

**預期輸出**:
```
✅ BeautifulSoup 可用
✅ Playwright 可用
✅ httpx 可用
```

---

## 🧪 功能驗證測試

### 快速測試（5 頁）

```powershell
cd C:\D\fold7\AIVA-git

$env:PYTHONPATH="C:\D\fold7\AIVA-git"

python -c "
import asyncio
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def test():
    print('🎯 Python Engine 快速測試')
    
    request = ScanStartPayload(
        scan_id='test_quick',
        targets=['http://localhost:3000'],
        strategy='deep',
    )
    
    orchestrator = ScanOrchestrator()
    result = await orchestrator.execute_scan(request)
    
    print(f'✅ 資產: {len(result.assets)}')
    print(f'✅ URL: {result.summary.urls_found}')
    print(f'✅ 表單: {result.summary.forms_found}')

asyncio.run(test())
"
```

### 完整驗證（20 頁）

```powershell
cd C:\D\fold7\AIVA-git

$env:PYTHONPATH="C:\D\fold7\AIVA-git"

python -c "
import asyncio
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def test():
    print('='*80)
    print('🎯 Python Engine 完整驗證')
    print('='*80)
    print('目標: http://localhost:3000 (Juice Shop)')
    print('策略: deep (max_pages=20, 動態掃描)')
    print()
    
    request = ScanStartPayload(
        scan_id='test_full',
        targets=['http://localhost:3000'],
        strategy='deep',
    )
    
    print('🚀 開始掃描...')
    orchestrator = ScanOrchestrator()
    result = await orchestrator.execute_scan(request)
    
    print()
    print('✅ 掃描完成')
    print(f'  資產: {len(result.assets)}')
    print(f'  URL: {result.summary.urls_found}')
    print(f'  表單: {result.summary.forms_found}')
    
    # 資產類型統計
    types = {}
    for a in result.assets:
        types[a.type] = types.get(a.type, 0) + 1
    
    print()
    print('📋 資產類型 (前 5):')
    for t, c in sorted(types.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f'  {t:15s}: {c:4d}')
    
    print('='*80)

asyncio.run(test())
"
```

**預期結果**:
```
✅ 資產: 1400-1500 個
✅ URL: 20 個
✅ 表單: 20-30 個
✅ Playwright 成功初始化
✅ JS 腳本提取正常
✅ 無 BeautifulSoup 導入錯誤
```

---

## 📦 完整依賴清單

### 最小依賴（必須）

```txt
beautifulsoup4>=4.12.0    # HTML 解析
lxml>=4.9.0               # XML/HTML 解析器
playwright>=1.41.0        # 動態渲染
httpx>=0.26.0             # HTTP 客戶端
pydantic>=2.5.0           # 數據驗證
```

### 推薦依賴（建議）

```txt
aiohttp>=3.9.0            # 異步 HTTP
pika>=1.3.0               # RabbitMQ 客戶端
python-dotenv>=1.0.0      # 環境變數管理
```

### 開發依賴（可選）

```txt
pytest>=7.4.0             # 測試框架
pytest-asyncio>=0.21.0    # 異步測試
black>=23.0.0             # 代碼格式化
ruff>=0.1.0               # 代碼檢查
```

---

## 🔍 故障排查

### 問題 1: BeautifulSoup 導入失敗

**錯誤訊息**:
```
name 'BeautifulSoup' is not defined
```

**解決方案**:
```powershell
# 確認已安裝
python -m pip list | Select-String beautifulsoup

# 重新安裝
python -m pip install --force-reinstall beautifulsoup4 lxml

# 驗證
python -c "from bs4 import BeautifulSoup; print('OK')"
```

### 問題 2: Playwright 瀏覽器未安裝

**錯誤訊息**:
```
Executable doesn't exist at ...
```

**解決方案**:
```powershell
# 安裝瀏覽器
playwright install chromium

# 查看已安裝瀏覽器
playwright show
```

### 問題 3: pip 路徑錯誤

**錯誤訊息**:
```
Fatal error in launcher: Unable to create process
```

**解決方案**:
```powershell
# 使用 python -m pip 代替 pip
python -m pip install beautifulsoup4

# 或修復 pip
python -m ensurepip --upgrade
```

### 問題 4: PYTHONPATH 未設置

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'services'
```

**解決方案**:
```powershell
# 設置 PYTHONPATH（每次運行前）
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# 或添加到環境變數（永久）
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\D\fold7\AIVA-git', 'User')
```

---

## ✅ 驗證檢查清單

運行完整安裝後，請確認以下項目：

- [ ] Python 版本 3.11+
- [ ] BeautifulSoup4 已安裝並可導入
- [ ] lxml 已安裝
- [ ] Playwright 已安裝並可導入
- [ ] Chromium 瀏覽器驅動已安裝
- [ ] httpx 已安裝
- [ ] pydantic 已安裝
- [ ] 快速測試通過（5 頁）
- [ ] 完整測試通過（20 頁）
- [ ] 無 BeautifulSoup 導入錯誤
- [ ] JS 腳本提取正常工作

---

## 🎯 測試結果參考

### 成功案例（2025-11-19）

**測試環境**:
- OS: Windows 11
- Python: 3.13.0
- 目標: Juice Shop (localhost:3000)

**測試結果**:
```
✅ Playwright initialized successfully
✅ Created chromium browser: browser_0
✅ Extracted 34 dynamic contents from http://localhost:3000/
✅ Inline script: 0 sinks, 4 patterns
✅ External script ...remote.js: 2 sinks, 10 patterns
✅ 資產總數: 1498
✅ URL 數: 20
✅ 表單數: 25
✅ JS 相關資產: 64
```

**資產類型分布**:
- link: 1154
- ajax_endpoint: 175
- api_call: 162
- form: 7

---

## 🔗 相關文檔

- **主 README**: [README.md](./README.md)
- **修復報告**: [FIX_COMPLETION_REPORT.md](./FIX_COMPLETION_REPORT.md)
- **驗證計劃**: [VALIDATION_TEST_PLAN.md](./VALIDATION_TEST_PLAN.md)

---

## 💡 最佳實踐

### 開發環境

1. **使用全域 Python**: 避免虛擬環境依賴問題
2. **固定版本**: 使用 `requirements-freeze.txt` 鎖定版本
3. **定期更新**: 每月檢查並更新依賴

### 測試流程

1. **安裝完成後立即驗證**: 避免後續問題
2. **使用快速測試**: 日常開發使用 5 頁測試
3. **完整測試前提交**: 確保功能正常

### 故障處理

1. **檢查版本**: 確認 Python 和依賴版本
2. **查看日誌**: 使用 `DEBUG` 級別查看詳細錯誤
3. **重新安裝**: 使用 `--force-reinstall` 強制重裝

---

**備註**: 本指南基於實際驗證經驗編寫，所有步驟已在 Juice Shop 靶場環境中測試通過。
