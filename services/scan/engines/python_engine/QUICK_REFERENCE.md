# Python Engine - 快速參考卡

## 📑 目錄

- [🚀 安裝（3 步驟）](#-安裝3-步驟)
- [🧪 測試（1 命令）](#-測試1-命令)
- [🐛 故障排查](#-故障排查)
- [📊 驗證結果](#-驗證結果)
- [🔗 文檔導航](#-文檔導航)

---

## 🚀 安裝（3 步驟）

```powershell
# 1. 安裝依賴
python -m pip install beautifulsoup4 lxml playwright httpx pydantic

# 2. 安裝瀏覽器
playwright install chromium

# 3. 驗證
python -c "from bs4 import BeautifulSoup; print('✅')"
```

詳細: [全域環境安裝指南](./GLOBAL_ENVIRONMENT_SETUP.md)

---

## 🧪 測試（1 命令）

```powershell
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git"
python -c "import asyncio; from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator; from services.aiva_common.schemas import ScanStartPayload; asyncio.run((lambda: ScanOrchestrator().execute_scan(ScanStartPayload(scan_id='test', targets=['http://localhost:3000'], strategy='deep')))())"
```

預期: 1400+ 資產, 20 URL, 20+ 表單

---

## 🐛 故障排查

### BeautifulSoup 錯誤
```powershell
python -m pip install --force-reinstall beautifulsoup4 lxml
```

### Playwright 錯誤
```powershell
playwright install chromium
```

### PYTHONPATH 錯誤
```powershell
$env:PYTHONPATH="C:\D\fold7\AIVA-git"
```

詳細: [故障排查](./GLOBAL_ENVIRONMENT_SETUP.md#-故障排查)

---

## 📊 驗證結果（2025-11-19）

- ✅ 資產: 1498
- ✅ URL: 20
- ✅ 表單: 25
- ✅ JS 資產: 64
- ✅ Playwright: 正常
- ✅ BeautifulSoup: 正常

---

## 🔗 文檔導航

- 📘 [README](./README.md) - 完整文檔
- ⚙️ [全域安裝](./GLOBAL_ENVIRONMENT_SETUP.md) - 必讀
- 🔧 [BeautifulSoup 修復](./BEAUTIFULSOUP_FIX.md) - 故障案例
- 📦 [依賴清單](./requirements-global.txt) - 安裝清單
