# Python Engine 驗證測試計劃

**日期**: 2025-11-19  
**狀態**: ⚠️ 待驗證  
**參考**: Rust Engine 驗證經驗

---

## 🎯 測試目標

根據 Rust Engine 的驗證經驗,Python Engine 需要驗證:

1. **靜態爬取能力** - 是否能發現表單、API端點
2. **動態渲染處理** - Playwright 是否正常工作
3. **JS分析能力** - 能否提取API端點、敏感資訊
4. **多目標處理** - 並行掃描多個靶場
5. **Phase0結果利用** - 能否接收並利用 Rust 的發現

---

## 📋 待驗證功能清單

### Phase1 核心功能

| 功能 | Rust經驗對照 | 測試方法 | 預期結果 |
|------|-------------|---------|---------|
| **靜態爬取** | Rust: 40端點/靶場 | 爬取Juice Shop首頁 | 發現forms、links |
| **表單發現** | Rust: 基於字典 | 識別登入/註冊表單 | 提取form參數 |
| **API分析** | Rust: 71 findings | 分析main.js等文件 | 發現/api/*端點 |
| **動態渲染** | Rust: 不支援 | Playwright載入SPA | 獲取動態內容 |
| **技術棧識別** | Rust: Angular, jQuery | 分析HTTP頭和內容 | 識別框架/庫 |

### 錯誤處理 (參考 Rust A3 優化)

| 情境 | Rust處理方式 | Python應該如何處理 |
|------|------------|------------------|
| **JS下載失敗** | match語句,記錄錯誤繼續 | try-except,不中斷掃描 |
| **頁面載入超時** | timeout參數 | asyncio.timeout() |
| **無效URL** | 返回Error | 記錄並跳過 |
| **認證失敗** | 記錄為finding | 標記需要認證 |

### 去重功能 (參考 Rust A4 優化)

| 資料類型 | 去重方式 | 實現檢查 |
|---------|---------|---------|
| **表單** | URL + method | 需要驗證 |
| **API端點** | 完整路徑 | 需要驗證 |
| **JS Findings** | HashSet去重 | 需要驗證 |
| **敏感資訊** | 內容hash | 需要驗證 |

---

## 🧪 測試案例設計

### Test 1: 單靶場靜態爬取

```bash
# 目標: Juice Shop (localhost:3000)
# 預期: 發現登入表單、註冊表單、API端點

測試參數:
- URL: http://localhost:3000
- 模式: FAST (不使用動態引擎)
- 深度: 2-3層

預期結果:
✅ 發現表單: login, register, search
✅ 發現端點: /api/Users, /api/Products, /api/BasketItems
✅ 技術棧: Angular, Bootstrap
✅ 執行時間: < 30秒
```

### Test 2: 動態渲染 (SPA)

```bash
# 目標: Juice Shop (Angular SPA)
# 預期: Playwright正確載入,獲取動態路由

測試參數:
- URL: http://localhost:3000
- 模式: DEEP (啟用動態引擎)
- 等待渲染: 3-5秒

預期結果:
✅ Playwright成功啟動
✅ 頁面完整渲染
✅ 獲取動態生成的表單/端點
✅ 捕獲AJAX請求
✅ 執行時間: < 60秒
```

### Test 3: JS文件分析

```bash
# 目標: 分析 Juice Shop 的 main.js, vendor.js
# 預期: 提取API端點、內部域名、敏感註釋

測試參數:
- 文件: main.js, runtime.js, vendor.js
- 分析器: JavaScriptSourceAnalyzer

預期結果:
✅ API端點: 15+ 個 (/api/*)
✅ 內部域名: 2-3個
✅ 敏感註釋: 包含password/secret關鍵字
✅ 去重: 無重複findings
```

### Test 4: 多靶場並行

```bash
# 目標: 同時掃描 3 個 Juice Shop 實例
# 預期: 並行處理,無互相干擾

測試參數:
- URLs: localhost:3000, 3001, 3003
- 模式: FAST
- 並行數: 3

預期結果:
✅ 3個目標都成功掃描
✅ 結果正確分離 (不混淆)
✅ 執行時間: < 40秒 (vs 順序掃描 90秒)
✅ 無資源競爭問題
```

### Test 5: Phase0結果利用

```bash
# 目標: 接收 Rust Phase0 結果,進行 Phase1 掃描
# 預期: 利用已發現的端點,避免重複掃描

測試流程:
1. Rust Phase0 → 發現 40 個端點
2. Python Phase1 → 接收端點列表
3. 優先掃描 high/critical 風險端點
4. 避免重複爬取已知路徑

預期結果:
✅ 正確解析 Phase0 結果
✅ 優先處理高風險端點
✅ 掃描時間減少 30-50%
✅ 無重複 Asset
```

### Test 6: 錯誤處理驗證

```bash
# 目標: 測試各種錯誤情境
# 預期: 優雅處理,不中斷掃描

錯誤情境:
1. JS文件404 → 記錄錯誤,繼續其他文件
2. 頁面載入超時 → 使用已載入內容
3. 無效URL → 跳過並記錄
4. 認證保護頁面 → 標記為需要認證

預期結果:
✅ 所有錯誤都被捕獲
✅ 記錄詳細錯誤信息
✅ 掃描不中斷
✅ 最終報告包含錯誤統計
```

---

## 🔧 實施步驟

### 步驟 1: 檢查依賴和環境

```bash
cd services/scan/engines/python_engine

# 檢查 Playwright 安裝
python -c "import playwright; print('Playwright OK')"

# 檢查其他依賴
python -c "import aiohttp; print('aiohttp OK')"
python -c "import bs4; print('BeautifulSoup OK')"

# 安裝瀏覽器 (如果需要)
playwright install chromium
```

### 步驟 2: 創建驗證腳本

```python
# validate_python_engine.py
import asyncio
from scan_orchestrator import ScanOrchestrator
from services.aiva_common.schemas import ScanStartPayload

async def test_single_target():
    """Test 1: 單靶場靜態爬取"""
    orchestrator = ScanOrchestrator()
    
    request = ScanStartPayload(
        scan_id="test_001",
        target_url="http://localhost:3000",
        strategy="FAST",
        max_depth=2,
    )
    
    result = await orchestrator.execute_scan(request)
    
    print(f"✅ Assets found: {len(result.complete_asset_list)}")
    print(f"✅ Technologies: {result.discovered_technologies}")
    print(f"✅ Execution time: {result.scan_duration_seconds}s")
    
    # 驗證結果
    assert len(result.complete_asset_list) > 0, "No assets found!"
    assert result.scan_status == "completed", "Scan failed!"

# ... 其他測試函數

if __name__ == "__main__":
    asyncio.run(test_single_target())
```

### 步驟 3: 執行測試

```bash
# Test 1: 單靶場
python validate_python_engine.py --test single_target

# Test 2: 動態渲染
python validate_python_engine.py --test dynamic_spa

# Test 3: JS分析
python validate_python_engine.py --test js_analysis

# Test 4: 多靶場
python validate_python_engine.py --test multi_target

# Test 5: Phase0整合
python validate_python_engine.py --test phase0_integration

# Test 6: 錯誤處理
python validate_python_engine.py --test error_handling

# All tests
python validate_python_engine.py --test all
```

### 步驟 4: 收集結果

```bash
# 生成報告
python validate_python_engine.py --generate-report

# 輸出格式:
# - 成功率: X/6 tests passed
# - 性能數據: 平均掃描時間
# - 發現能力: 平均 assets/endpoints 數量
# - 錯誤處理: 錯誤恢復成功率
```

---

## 📊 驗證標準

### 通過標準

| 指標 | 目標值 | Rust對照 |
|------|--------|---------|
| **靜態爬取成功率** | > 95% | Rust: 100% |
| **表單發現率** | > 90% | Rust: 基於字典 |
| **API端點發現** | > 15個/靶場 | Rust: 71 findings |
| **動態渲染成功率** | > 85% | Rust: N/A |
| **多靶場並行** | 3個同時 | Rust: 4個同時 |
| **執行時間 (FAST)** | < 30秒/靶場 | Rust: 178ms |
| **執行時間 (DEEP)** | < 60秒/靶場 | Rust: ~400ms |
| **錯誤恢復率** | 100% | Rust: 100% |
| **去重準確率** | > 95% | Rust: 100% |

### 性能對比 (與 Rust)

| 引擎 | 靜態掃描 | 動態掃描 | 並行數 | 內存 |
|------|---------|---------|-------|------|
| **Rust** | 178ms | N/A | 4+ | ~5MB |
| **Python** | ~5-10秒 | ~20-30秒 | 2-4 | ~50-100MB |

**預期**: Python慢10-100倍,但功能更完整 (支援動態渲染)

---

## 🐛 已知問題和預期修復

### 問題 1: Phase0 結果利用不完善

**現象**: 可能重複掃描 Rust 已發現的端點

**Rust經驗**: Rust直接掃描,無此問題

**修復方案**:
```python
# scan_orchestrator.py
async def execute_phase1(self, request: Phase1StartPayload):
    # 1. 接收 Phase0 結果
    phase0_endpoints = request.discovered_endpoints
    
    # 2. 過濾已掃描的 URL
    urls_to_scan = [url for url in new_urls 
                    if url not in phase0_endpoints]
    
    # 3. 優先掃描高風險端點
    high_risk = [e for e in phase0_endpoints 
                 if e.risk_level == "critical"]
```

### 問題 2: 去重邏輯可能不完整

**現象**: 可能有重複的 Asset

**Rust經驗**: 使用 HashSet 去重,100% 成功

**修復方案**:
```python
# scan_context.py
def add_asset(self, asset: Asset):
    # 生成唯一 key
    key = f"{asset.asset_type}:{asset.url}:{asset.method}"
    
    if key not in self._asset_keys:
        self._asset_keys.add(key)
        self.assets.append(asset)
```

### 問題 3: 錯誤處理可能不夠健壯

**現象**: 某些錯誤可能中斷掃描

**Rust經驗**: match 語句處理所有錯誤,繼續掃描

**修復方案**:
```python
# core_crawling_engine/http_client_hi.py
async def get(self, url: str):
    try:
        response = await self._session.get(url)
        return response
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Timeout: {url}")
        return None  # 繼續掃描
    except aiohttp.ClientError as e:
        logger.warning(f"⚠️ Error fetching {url}: {e}")
        return None  # 繼續掃描
    except Exception as e:
        logger.error(f"⚠️ Unexpected error {url}: {e}")
        return None  # 繼續掃描
```

---

## 📋 驗證檢查清單

### 執行前檢查

- [ ] Juice Shop 運行在 localhost:3000
- [ ] (可選) 多個實例: 3001, 3003
- [ ] Playwright 已安裝: `playwright install`
- [ ] Python依賴完整: `pip install -r requirements.txt`
- [ ] RabbitMQ運行 (如果測試Worker)

### 執行中檢查

- [ ] Test 1: 單靶場靜態爬取 - PASS
- [ ] Test 2: 動態渲染 - PASS
- [ ] Test 3: JS文件分析 - PASS
- [ ] Test 4: 多靶場並行 - PASS
- [ ] Test 5: Phase0結果利用 - PASS
- [ ] Test 6: 錯誤處理 - PASS

### 執行後分析

- [ ] 生成驗證報告
- [ ] 性能數據記錄
- [ ] 與 Rust 對比分析
- [ ] 識別優化機會
- [ ] 更新 README.md

---

## 🚀 下一步

### 驗證完成後

1. **更新文檔**
   - 記錄實際掃描能力
   - 添加使用示例
   - 性能基準數據

2. **處理發現的問題**
   - 修復錯誤處理
   - 完善去重邏輯
   - 優化 Phase0 整合

3. **優化性能** (低優先級)
   - 並行處理優化
   - 內存使用優化
   - 緩存機制

4. **進入 TypeScript 引擎驗證**
   - TypeScript 是最需要處理的 (20% 完成度)
   - 參考 Python 動態引擎經驗
   - 實現 SPA 路由發現

---

## 📞 參考資料

- Rust Engine: `WORKING_STATUS_2025-11-19.md`
- Rust 優化: `OPTIMIZATION_ROADMAP.md`
- Python 架構: `ENGINE_COMPLETION_ANALYSIS.md`
- 掃描流程: `SCAN_FLOW_DIAGRAMS.md`
