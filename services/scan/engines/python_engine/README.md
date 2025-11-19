# Python Engine - Phase1 主力爬蟲引擎

> **角色定位**: Phase1 核心引擎 - 靜態/動態爬取、表單發現、API分析  
> **技術優勢**: 完整的爬蟲生態、Playwright 動態渲染、豐富的分析工具  
> **當前狀態**: ✅ 90% 功能完成 + ✅ 核心修復完成 + ✅ 全域環境驗證通過  
> **最後更新**: 2025-11-19 (BeautifulSoup 修復 + 全域環境配置)  
> **驗證狀態**: ✅ 已在 Juice Shop 靶場完整驗證通過

---

## 🚀 快速開始

### 全域環境安裝（推薦）⭐

詳細指南: **[全域環境安裝指南](./GLOBAL_ENVIRONMENT_SETUP.md)**

```powershell
# 1. 安裝核心依賴
python -m pip install beautifulsoup4 lxml playwright httpx pydantic

# 2. 安裝瀏覽器驅動
playwright install chromium

# 3. 驗證安裝
python -c "from bs4 import BeautifulSoup; print('✅ BeautifulSoup OK')"
python -c "from playwright.async_api import async_playwright; print('✅ Playwright OK')"

# 4. 運行測試
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git"
python -c "
import asyncio
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def test():
    request = ScanStartPayload(scan_id='test', targets=['http://localhost:3000'], strategy='deep')
    result = await (ScanOrchestrator()).execute_scan(request)
    print(f'✅ 資產: {len(result.assets)}, URL: {result.summary.urls_found}, 表單: {result.summary.forms_found}')

asyncio.run(test())
"
```

**預期結果**: 
```
✅ 資產: 1400-1500, URL: 20, 表單: 20-30
✅ Playwright 成功初始化
✅ 無 BeautifulSoup 錯誤
```

---

## 📋 功能特性

### Phase1 核心能力

| 功能類別 | 完成度 | 說明 |
|---------|--------|------|
| **靜態爬取** | ✅ 90% | HTML 解析、鏈接提取、深度控制 |
| **表單發現** | ✅ 95% | 登入/註冊/搜尋表單識別和參數提取 |
| **API 分析** | ✅ 90% | RESTful API 端點發現、參數挖掘 |
| **動態渲染** | ✅ 85% | Playwright 整合、JavaScript 執行 |
| **JS 分析** | ✅ 90% | API 端點提取、敏感資訊檢測 |
| **指紋識別** | ✅ 85% | Web 服務器、框架、CMS 識別 |
| **Phase0 整合** | ⚠️ 70% | 待驗證 Rust 結果利用 |

### 最新優化 (2025-11-19)

| 優化 | 參考 | 狀態 |
|------|------|------|
| **Asset 去重** | Rust A4 (HashSet) | ✅ 完成 |
| **錯誤處理** | Rust A3 (match) | ✅ 完成 |
| **Timeout 處理** | Rust 超時機制 | ✅ 完成 |

詳細修復內容請參閱: [FIX_COMPLETION_REPORT.md](./FIX_COMPLETION_REPORT.md)

---

## 🏗️ 架構組件

### 核心模組

```
python_engine/
├── worker.py                    # Phase1 Worker (RabbitMQ 訂閱)
├── scan_orchestrator.py         # 掃描編排器 (636 lines)
├── scan_context.py              # 掃描上下文管理 (✅ 新增去重邏輯)
│
├── core_crawling_engine/        # 靜態爬蟲引擎
│   ├── http_client_hi.py       # HTTP 客戶端 (✅ 改進錯誤處理)
│   ├── static_content_parser.py # HTML 解析器
│   └── url_queue_manager.py    # URL 隊列管理
│
├── dynamic_engine/              # 動態渲染引擎
│   ├── headless_browser_pool.py # Playwright 瀏覽器池
│   ├── dynamic_content_extractor.py # 動態內容提取
│   ├── ajax_api_handler.py     # AJAX 端點捕獲
│   └── js_interaction_simulator.py # 互動模擬
│
├── info_gatherer/               # 資訊收集
│   ├── javascript_source_analyzer.py # JS 源碼分析
│   └── sensitive_info_detector.py    # 敏感資訊檢測
│
└── fingerprint_manager.py       # 指紋識別
```

### 支持組件

- **authentication_manager.py** - 認證管理 (Basic/Token/Cookie)
- **header_configuration.py** - HTTP 頭配置
- **strategy_controller.py** - 掃描策略控制
- **scope_manager.py** - 掃描範圍管理

---

## 💻 使用方式

### 1. 通過 Worker (推薦)

Python Worker 會自動監聽 RabbitMQ Phase1 任務:

```python
# 啟動 Worker
python worker.py
```

### 2. 直接調用 (測試用)

```python
from scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

orchestrator = ScanOrchestrator()

request = ScanStartPayload(
    scan_id="test_001",
    targets=["http://localhost:3000"],
    strategy="FAST",  # 或 "DEEP"
    max_depth=2,
)

result = await orchestrator.execute_scan(request)

print(f"發現 {len(result.assets)} 個資產")
print(f"表單: {result.summary.forms_found}")
print(f"API: {result.summary.apis_found}")
```

### 掃描策略

| 策略 | 用途 | 動態渲染 | 深度 | 時間 |
|------|------|---------|------|------|
| **FAST** | 快速掃描 | ❌ | 1-2 | ~10秒 |
| **DEEP** | 深度分析 | ✅ | 3-4 | ~30秒 |
| **AGGRESSIVE** | 完整掃描 | ✅ | 5+ | ~60秒+ |

---

## 🎯 與其他引擎的配合

### Phase0 → Phase1 流程

```
1. Rust Engine (Phase0) - 178ms
   ├─ 端點發現: 40個 (字典爆破)
   ├─ JS 分析: 71 findings
   ├─ 技術棧: Angular, jQuery
   └─ 風險評估: Critical × 4
   
2. Python Engine (Phase1) - ~10-30秒
   ├─ 利用 Phase0 端點 (避免重複)
   ├─ 深度表單分析
   ├─ API 參數挖掘
   ├─ 動態內容提取
   └─ 完整資產清單
```

### 與 TypeScript/Go 協同

- **TypeScript**: 處理 SPA 應用 (Python 不擅長)
- **Go**: 處理 SSRF/CSPM/SCA (Python 不支援)
- **Python**: 處理傳統 Web 應用 (最強)

---

## 📊 性能特徵

### 掃描效率

| 指標 | FAST 模式 | DEEP 模式 |
|------|----------|----------|
| **掃描時間** | ~10秒 | ~30秒 |
| **內存使用** | ~50MB | ~100MB |
| **並發數** | 2-4 | 2-3 |
| **適合目標** | 傳統 Web | 複雜應用 |

### 與 Rust 對比

| 項目 | Rust | Python |
|------|------|--------|
| **速度** | ⭐⭐⭐⭐⭐ (178ms) | ⭐⭐⭐ (~10s) |
| **內存** | ⭐⭐⭐⭐⭐ (5MB) | ⭐⭐⭐ (50MB) |
| **功能** | ⭐⭐⭐ (基礎) | ⭐⭐⭐⭐⭐ (完整) |
| **動態** | ❌ 不支援 | ✅ Playwright |
| **表單** | ❌ 不支援 | ✅ 完整支援 |

---

## 🧪 測試驗證

### 驗證計劃

詳細測試計劃請參閱: [VALIDATION_TEST_PLAN.md](./VALIDATION_TEST_PLAN.md)

#### Test 1: 單靶場靜態爬取
```bash
# 目標: Juice Shop (localhost:3000)
# 預期: 發現表單、API端點

pytest test_validation.py::test_single_target
```

#### Test 2: 動態渲染 (SPA)
```bash
# 目標: Juice Shop (Angular SPA)
# 預期: Playwright 成功載入

pytest test_validation.py::test_dynamic_rendering
```

#### Test 3: 去重驗證 (新增)
```bash
# 測試: Asset 去重邏輯
# 預期: 重複 Asset 被過濾

pytest test_validation.py::test_deduplication
```

#### Test 4: 錯誤處理驗證 (新增)
```bash
# 測試: HTTP 超時、錯誤不中斷
# 預期: 單個失敗不影響整體

pytest test_validation.py::test_error_handling
```

---

## ⚙️ 依賴需求

### Python 版本
- **Python**: 3.11+（推薦 3.13+）

### 核心依賴
```txt
beautifulsoup4>=4.12.0 # HTML 解析 (⚠️ 必須！)
lxml>=4.9.0            # XML/HTML 解析器
playwright>=1.41.0     # 動態渲染
httpx>=0.26.0          # HTTP 客戶端
pydantic>=2.5.0        # 數據驗證
aiohttp>=3.9.0         # 異步 HTTP
```

### 安裝方式

#### 選項 1: 全域安裝（推薦）

詳細指南請參閱: **[全域環境安裝指南](./GLOBAL_ENVIRONMENT_SETUP.md)** ⭐

```powershell
# 安裝核心依賴
python -m pip install beautifulsoup4 lxml playwright httpx pydantic

# 安裝 Playwright 瀏覽器
playwright install chromium

# 驗證安裝
python -c "from bs4 import BeautifulSoup; print('✅ OK')"
python -c "from playwright.async_api import async_playwright; print('✅ OK')"
```

#### 選項 2: 虛擬環境安裝

```bash
cd services/scan/engines/python_engine

# 創建虛擬環境
python -m venv .venv

# 激活虛擬環境
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt

# 安裝 Playwright 瀏覽器
playwright install chromium
```

**注意**: 虛擬環境可能遇到 Playwright 路徑問題，建議使用全域安裝。

---

## 🐛 已知問題與修復

### ✅ 已修復 (2025-11-19)

| 問題 | 修復方案 | 參考 |
|------|---------|------|
| **Asset 重複** | HashSet 去重 | Rust A4 |
| **HTTP 錯誤中斷** | 具體異常處理 | Rust A3 |
| **Timeout 未單獨處理** | TimeoutException | Rust 超時 |

### ⚠️ 待驗證

| 問題 | 計劃 | 優先級 |
|------|------|-------|
| Phase0 結果利用率低 | 實際測試驗證 | 中 |
| 大規模目標性能 | 並行優化 | 低 |
| 內存使用優化 | 資源管理改進 | 低 |

---

## 📈 優化路線圖

### 短期 (1-2 週)

- [x] ✅ Asset 去重優化
- [x] ✅ 錯誤處理增強
- [ ] ⏳ 實際靶場驗證
- [ ] ⏳ Phase0 整合測試

### 中期 (2-4 週)

- [ ] 性能優化 (並行處理)
- [ ] 內存使用優化
- [ ] 更多錯誤場景處理
- [ ] 完整測試覆蓋

### 長期 (1-2 月)

- [ ] 與 TypeScript/Go 協同優化
- [ ] 智能引擎選擇
- [ ] 自適應策略調整

---

## 🔗 相關文檔

### 安裝和配置 ⭐ **必讀**
- **全域環境安裝指南**: [GLOBAL_ENVIRONMENT_SETUP.md](./GLOBAL_ENVIRONMENT_SETUP.md)
- **BeautifulSoup 修復記錄**: [BEAUTIFULSOUP_FIX.md](./BEAUTIFULSOUP_FIX.md)

### 開發和維護
- **修復報告**: [FIX_COMPLETION_REPORT.md](./FIX_COMPLETION_REPORT.md)
- **驗證計劃**: [VALIDATION_TEST_PLAN.md](./VALIDATION_TEST_PLAN.md)
- **架構分析**: [../ENGINE_COMPLETION_ANALYSIS.md](../ENGINE_COMPLETION_ANALYSIS.md)

### 參考資料
- **Rust 參考**: [../rust_engine/USAGE_GUIDE.md](../rust_engine/USAGE_GUIDE.md)
- **依賴清單**: [requirements-global.txt](./requirements-global.txt)

---

## 📞 技術支持

### 快速排查指南

詳細故障排查請參閱: **[全域環境安裝指南 - 故障排查](./GLOBAL_ENVIRONMENT_SETUP.md#-故障排查)** ⭐

1. **BeautifulSoup 導入失敗**
   ```powershell
   # ⚠️ 關鍵依賴！必須安裝
   python -m pip install --force-reinstall beautifulsoup4 lxml
   python -c "from bs4 import BeautifulSoup; print('✅ OK')"
   ```

2. **Playwright 啟動失敗**
   ```bash
   playwright install --with-deps chromium
   ```

3. **導入錯誤**
   ```powershell
   # 設置 PYTHONPATH
   $env:PYTHONPATH="C:\D\fold7\AIVA-git"
   ```

4. **RabbitMQ 連接失敗**
   ```bash
   # 檢查環境變數
   echo $RABBITMQ_URL
   ```

### 日誌級別

```python
# 調試模式
export LOG_LEVEL=DEBUG

# 生產模式
export LOG_LEVEL=INFO
```

### 驗證測試

```powershell
# 快速驗證（5 頁）
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git"
python -c "
import asyncio
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def test():
    request = ScanStartPayload(
        scan_id='quick_test',
        targets=['http://localhost:3000'],
        strategy='deep',
    )
    orchestrator = ScanOrchestrator()
    result = await orchestrator.execute_scan(request)
    print(f'✅ 資產: {len(result.assets)}')

asyncio.run(test())
"
```
