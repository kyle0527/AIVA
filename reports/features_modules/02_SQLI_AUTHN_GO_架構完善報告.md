# 🔸 架構完善報告：SQLI + AUTHN_GO 模組

**報告編號**: FEAT-002  
**日期**: 2025年11月6日  
**狀態**: 🔸 中優先級 - 架構完善  
**負責模組**: SQLI (基本完整) + AUTHN_GO (部分實現)

---

## 📊 模組現況分析

### 🗃️ SQLI 模組 - SQL 注入檢測

#### **現況評估**
- **完善度**: 🔸 基本完整 (3/4 組件)
- **程式規模**: 19 檔案, 4,102 行程式碼
- **開發語言**: Python (純架構)
- **組件狀態**: Worker✅ | Detector❌ | Engine✅ | Config✅

#### **現有優勢**
```
function_sqli/
├── ✅ worker.py                    # 重構的高品質 Worker (513行)
├── ✅ config.py                    # 完整配置管理 (178行)
├── ✅ engines/                     # 豐富的檢測引擎群
│   ├── boolean_detection.py       # 布爾盲注檢測
│   ├── time_detection.py          # 時間盲注檢測
│   ├── union_detection.py         # 聯合查詢注入
│   ├── error_detection.py         # 錯誤回顯注入
│   ├── oob_detection.py           # Out-of-Band 檢測
│   └── hackingtool_engine.py      # 外部工具整合
├── ✅ detection_models.py          # 檢測結果模型
├── ✅ backend_db_fingerprinter.py # 資料庫指紋識別
├── ✅ result_binder_publisher.py  # 結果綁定發布器
└── ❌ 缺少統一的 Detector 組件
```

#### **架構特色**
- **策略模式設計**: 實現 `DetectionEngineProtocol` 接口
- **依賴注入**: 解決原始版本複雜度過高問題
- **多引擎支援**: 涵蓋所有主要 SQL 注入類型
- **外部工具整合**: 支援 SQLMap 等知名工具

---

### 🔐 AUTHN_GO 模組 - 認證繞過檢測

#### **現況評估**
- **完善度**: 🔹 部分實現 (2/4 組件)
- **程式規模**: 4 檔案, 1,602 行程式碼
- **開發語言**: Go (純架構)
- **組件狀態**: Worker✅ | Detector❌ | Engine❌ | Config✅

#### **現有結構**
```
function_authn_go/
├── ✅ cmd/worker/main.go           # Go Worker 主程式 (150行)
├── ✅ go.mod                       # 依賴管理 (JWT, Zap, AMQP)
├── ❌ internal/                    # 內部模組 (缺乏整合)
│   ├── brute_force/               # 暴力破解模組
│   ├── token_test/                # 令牌測試模組
│   └── weak_config/               # 弱配置檢測模組
└── ❌ 缺少統一的 Detector 和 Engine
```

#### **Go 語言優勢**
- **高併發能力**: Goroutine 天然適合多會話並行測試
- **JWT 生態**: `github.com/golang-jwt/jwt/v5` 專業令牌處理
- **網路效能**: 高效 HTTP 客戶端處理大量認證請求
- **記憶體效率**: 比 Python 更低的記憶體占用

---

## 🎯 預計改善方向

### 🗃️ SQLI 模組補強策略

#### **缺失組件**: 統一 Detector
**目標**: 實現標準化的 SQL 注入檢測器，整合現有多引擎能力

```python
# 新增 detector.py
class SqliDetector:
    """統一 SQL 注入檢測器"""
    
    def __init__(self):
        self.engines = {
            'boolean': BooleanDetectionEngine(),
            'time': TimeDetectionEngine(), 
            'union': UnionDetectionEngine(),
            'error': ErrorDetectionEngine(),
            'oob': OOBDetectionEngine(),
            'hackingtool': HackingToolDetectionEngine()
        }
        self.db_fingerprinter = BackendDbFingerprinter()
    
    async def detect_sqli(self, target: str, params: dict) -> List[DetectionResult]:
        """智能 SQL 注入檢測
        
        1. 資料庫指紋識別
        2. 根據目標特徵選擇最佳檢測引擎
        3. 並行執行多種檢測技術
        4. 結果去重和置信度評分
        """
        # 指紋識別決定檢測策略
        db_type = await self.db_fingerprinter.identify(target)
        
        # 智能引擎選擇
        selected_engines = self._select_engines(db_type, params)
        
        # 並行檢測
        results = await asyncio.gather(*[
            engine.detect(target, params) 
            for engine in selected_engines
        ])
        
        return self._merge_results(results)
```

#### **智能檢測策略**
- **指紋導向**: 根據資料庫類型選擇最有效的檢測方法
- **並行執行**: 多引擎同時檢測，提升效率和覆蓋率
- **結果融合**: 智能去重，綜合置信度評分

### 🔐 AUTHN_GO 模組架構完善

#### **目標**: 發揮 Go 高併發優勢的認証測試架構

```go
// 新增 detector.go - 統一認證檢測器
package detector

type AuthnDetector struct {
    bruteForcer   *brute_force.BruteForcer
    tokenTester   *token_test.TokenTester
    configChecker *weak_config.ConfigChecker
    
    // Go 併發控制
    semaphore     chan struct{} // 限制併發數
    workerPool    sync.Pool     // Worker 對象池
}

func (d *AuthnDetector) DetectAuthBypass(ctx context.Context, target string) (*DetectionResult, error) {
    // 並行執行三種檢測
    var wg sync.WaitGroup
    resultChan := make(chan *PartialResult, 3)
    
    // 1. 暴力破解檢測 (Goroutine)
    wg.Add(1)
    go func() {
        defer wg.Done()
        result := d.bruteForcer.TestCredentials(ctx, target)
        resultChan <- &PartialResult{Type: "brute_force", Data: result}
    }()
    
    // 2. 令牌測試 (Goroutine)  
    wg.Add(1)
    go func() {
        defer wg.Done()
        result := d.tokenTester.TestTokens(ctx, target)
        resultChan <- &PartialResult{Type: "token_test", Data: result}
    }()
    
    // 3. 弱配置檢測 (Goroutine)
    wg.Add(1)
    go func() {
        defer wg.Done()
        result := d.configChecker.CheckWeakConfig(ctx, target)
        resultChan <- &PartialResult{Type: "weak_config", Data: result}
    }()
    
    // 收集結果
    go func() {
        wg.Wait()
        close(resultChan)
    }()
    
    return d.mergeResults(resultChan), nil
}
```

#### **新增 Engine 抽象層**
```go
// engine.go - 認證測試引擎抽象
type AuthnEngine interface {
    Test(ctx context.Context, target *Target, config *Config) (*TestResult, error)
    GetEngineInfo() *EngineInfo
}

type EngineManager struct {
    engines map[string]AuthnEngine
}

func (em *EngineManager) RegisterEngine(name string, engine AuthnEngine) {
    em.engines[name] = engine
}

func (em *EngineManager) RunTests(ctx context.Context, target *Target) (*CombinedResult, error) {
    // 併發執行所有已註冊的引擎
    // 使用 context 進行超時控制
    // 使用 errgroup 進行錯誤管理
}
```

---

## 💪 能力需求分析

### 🗃️ SQLI 模組技能需求

#### **Python 架構專家** (1人，1-2週)
- **必備技能**:
  - Python 異步程式設計精通
  - 設計模式應用 (策略模式、工廠模式)
  - SQL 注入技術深度理解
  - 現有代碼重構經驗

- **工作內容**:
  - 設計統一 Detector 接口
  - 整合現有 6 個檢測引擎
  - 實現智能檢測策略
  - 優化併發檢測效能

### 🔐 AUTHN_GO 模組技能需求

#### **Go 併發專家** (1人，2-3週)
- **必備技能**:
  - Go 語言專精 (Goroutine、Channel、Context)
  - JWT 和認證協議熟悉
  - 高併發架構設計經驗
  - 網路程式設計和 HTTP 客戶端優化

- **加分技能**:
  - OAuth 2.0 / SAML 協議了解
  - 會話管理和 Cookie 安全
  - 暴力破解防護機制理解

---

## 📋 實現里程碑

### 🎯 第一階段 (1週) - SQLI Detector 實現
- [ ] 分析現有 6 個引擎的接口統一性
- [ ] 設計 SqliDetector 統一接口
- [ ] 實現智能引擎選擇邏輯
- [ ] 實現結果融合和去重機制
- [ ] 單元測試和效能基準測試

### 🎯 第二階段 (2週) - AUTHN_GO 架構完善
- [ ] 設計 AuthnDetector 併發架構
- [ ] 實現 Engine 抽象層和註冊機制
- [ ] 重構現有 3 個內部模組為 Engine
- [ ] 實現併發控制和資源池管理
- [ ] 整合測試和效能調優

### 🎯 第三階段 (1週) - 整合與優化
- [ ] 跨模組功能測試
- [ ] 效能基準對比 (vs 現有實現)
- [ ] 與 AIVA 系統整合測試
- [ ] 文檔更新和使用範例

---

## 🚀 團隊分工建議

### **Team A - SQLI 完善** (1人，1-2週)
- **Python 架構專家**
  - Detector 組件設計實現
  - 現有引擎整合優化
  - 智能檢測策略實現

### **Team B - AUTHN_GO 擴展** (1人，2-3週)  
- **Go 併發專家**
  - 併發架構設計
  - Engine 抽象層實現
  - 效能優化和併發控制

---

## ⚡ 效能提升預期

### **SQLI 模組**
- **檢測覆蓋率**: +15% (智能引擎選擇)
- **誤報率**: -20% (結果融合去重)
- **並行檢測**: 3-5倍效能提升
- **資料庫適配**: 支援 8+ 主流資料庫

### **AUTHN_GO 模組**
- **併發檢測**: 10-50倍效能提升 (vs Python)
- **記憶體效率**: 60% 記憶體節省
- **連接復用**: HTTP/2 多路復用支援
- **會話管理**: 1000+ 並行會話測試

---

## 🔧 技術決策說明

### **為什麼 SQLI 選擇統一 Detector？**
- **現況**: 6 個優秀引擎缺乏統一管理
- **問題**: 重複檢測、結果不一致、配置分散
- **解決**: 統一入口點，智能引擎調度，結果標準化

### **為什麼 AUTHN_GO 強調併發？**
- **認證特性**: 大量重複的 HTTP 請求
- **Go 優勢**: 輕量級 Goroutine，天然併發優勢
- **效能需求**: 暴力破解需要高速請求能力

---

## 📈 成功指標

### **SQLI 模組**
- [ ] 4/4 標準組件完整實現
- [ ] 檢測準確率 > 95% (與 SQLMap 對比)
- [ ] 並行檢測效能提升 3x+
- [ ] 支援 MySQL, PostgreSQL, MSSQL, Oracle 等主流資料庫

### **AUTHN_GO 模組**
- [ ] 4/4 標準組件完整實現  
- [ ] 併發檢測效能提升 10x+ (vs Python 實現)
- [ ] 支援 1000+ 並行認證測試
- [ ] JWT/OAuth/Cookie 全面支援

---

**報告結論**: 這兩個模組具備良好的基礎，通過補全缺失組件和發揮各語言優勢，可快速達到完全實現狀態。建議優先完成，為其他模組提供架構參考。