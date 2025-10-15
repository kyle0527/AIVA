# AIVA 多語言架構後續規劃與行動指南

**制定日期:** 2025-10-14  
**狀態:** ✅ Phase 1 完成,進入實施階段  
**預計完成時間:** 2025年12月底 (10週)

---

## 📊 當前進度總覽

### ✅ 已完成項目 (Week 0)

1. **策略制定**
   - [x] 完整的多語言發展策略文件
   - [x] 各語言職責明確劃分
   - [x] 跨語言整合機制設計

2. **Go 共用模組建立**
   - [x] `aiva_common_go` 基礎架構
   - [x] RabbitMQ 統一客戶端
   - [x] 標準化日誌系統
   - [x] 配置管理
   - [x] Schema 定義
   - [x] 單元測試 (全部通過 ✅)

3. **文件產出**
   - [x] `MULTILANG_STRATEGY.md` - 完整策略 (146KB)
   - [x] `MULTILANG_STRATEGY_SUMMARY.md` - 快速摘要
   - [x] `MULTILANG_IMPLEMENTATION_REPORT.md` - 實施報告
   - [x] `docs/ARCHITECTURE_MULTILANG.md` - 架構圖

4. **工具腳本**
   - [x] `init_go_common.ps1` - 初始化腳本
   - [x] `migrate_sca_service.ps1` - 遷移範例

---

## 🎯 10週實施計劃

### 📅 第1-2週 (2025-10-14 ~ 2025-10-27) - Go 服務統一化

#### 目標
遷移所有 Go 服務使用 `aiva_common_go` 共用模組

#### 任務清單

**Week 1 (10/14-10/20):**
- [ ] **Task 1.1**: 遷移 `function_sca_go`
  - [ ] 更新 `go.mod` 添加依賴
  - [ ] 重構 `main.go` 使用共用模組
  - [ ] 移除重複的 RabbitMQ 程式碼
  - [ ] 執行測試驗證
  - [ ] 更新 Dockerfile
  
- [ ] **Task 1.2**: 遷移 `function_cspm_go`
  - [ ] 同上步驟
  - [ ] 特別注意雲端 API 並發處理
  
- [ ] **Task 1.3**: 建立遷移文件
  - [ ] 記錄遷移過程中的問題
  - [ ] 建立 troubleshooting 指南

**Week 2 (10/21-10/27):**
- [ ] **Task 2.1**: 遷移 `function_authn_go`
- [ ] **Task 2.2**: 遷移 `function_ssrf_go`
- [ ] **Task 2.3**: 建立整合測試
  - [ ] 端到端測試所有 Go 服務
  - [ ] 效能基準測試
  - [ ] 記憶體使用分析

#### 驗收標準
- ✅ 所有 Go 服務使用 `aiva_common_go`
- ✅ 程式碼重複率 < 15%
- ✅ 所有測試通過
- ✅ CI/CD 管道正常運作

#### 預期成果
- 程式碼行數減少 **35-45%**
- 維護成本降低 **40%**
- 新服務開發加速 **50%**

---

### 📅 第3-4週 (2025-10-28 ~ 2025-11-10) - TypeScript 動態掃描增強

#### 目標
強化 Node.js/TypeScript 的動態掃描能力

#### 任務清單

**Week 3 (10/28-11/03):**
- [ ] **Task 3.1**: 實作 `SmartFormFiller` 服務
  ```typescript
  // 智慧表單填充
  - 自動識別欄位類型 (email, password, phone 等)
  - 智慧生成測試資料
  - 支援多種表單框架
  ```

- [ ] **Task 3.2**: 實作 `APIDiscoveryService`
  ```typescript
  // API 端點自動發現
  - 攔截所有 XHR/Fetch 請求
  - 記錄完整的請求/回應
  - 自動生成 API 清單
  ```

- [ ] **Task 3.3**: 確認移除 Python Playwright 殘留
  ```bash
  # 搜尋並移除所有 Python 中的瀏覽器自動化程式碼
  grep -r "playwright\|selenium\|webdriver" services/core/ services/integration/
  ```

**Week 4 (11/04-11/10):**
- [ ] **Task 4.1**: 增強 `InteractionSimulator`
  - [ ] 完整的使用者旅程模擬
  - [ ] DOM 穩定性檢測
  - [ ] 事件觸發優化

- [ ] **Task 4.2**: 建立 `@aiva/common` npm package
  ```typescript
  // TypeScript 共用模組
  - Schema 定義 (對應 Python)
  - RabbitMQ 客戶端
  - 標準化日誌
  ```

- [ ] **Task 4.3**: 效能優化
  - [ ] 瀏覽器資源使用優化
  - [ ] 並發掃描能力
  - [ ] 記憶體洩漏檢測

#### 驗收標準
- ✅ API 發現率從 30% 提升到 **80%+**
- ✅ 表單自動填充成功率 **> 85%**
- ✅ Python 中無 Playwright 相關程式碼
- ✅ `@aiva/common` 可在所有 Node.js 服務中使用

#### 預期成果
- 動態掃描覆蓋率提升 **150%**
- 手動測試工作量減少 **60%**
- SPA 應用測試效率提升 **200%**

---

### 📅 第5-6週 (2025-11-11 ~ 2025-11-24) - Rust SAST 規則引擎優化

#### 目標
將 SAST 規則外部化並大幅擴充規則庫

#### 任務清單

**Week 5 (11/11-11/17):**
- [ ] **Task 5.1**: 實作規則外部化
  ```rust
  // 從 YAML 檔案載入規則
  - 支援 tree-sitter 查詢語法
  - 動態重載機制
  - 規則驗證與測試框架
  ```

- [ ] **Task 5.2**: 建立規則庫結構
  ```
  rules/
  ├── sql_injection/
  │   ├── python.yml
  │   ├── javascript.yml
  │   └── java.yml
  ├── xss/
  ├── command_injection/
  └── ...
  ```

- [ ] **Task 5.3**: 撰寫規則文件
  - [ ] 規則編寫指南
  - [ ] tree-sitter 查詢語法教學
  - [ ] 貢獻規則流程

**Week 6 (11/18-11/24):**
- [ ] **Task 6.1**: 擴充規則庫
  - [ ] SQL Injection: 10條規則
  - [ ] XSS: 8條規則
  - [ ] Command Injection: 6條規則
  - [ ] Path Traversal: 5條規則
  - [ ] 其他漏洞類型: 21條規則
  - [ ] **總計: 50條規則**

- [ ] **Task 6.2**: 效能優化
  - [ ] 並行掃描優化
  - [ ] 記憶體使用優化
  - [ ] 快取機制

- [ ] **Task 6.3**: PyO3 整合 POC
  ```rust
  // 為 Python 提供高效能模組
  #[pyfunction]
  fn fast_sast_scan(code: &str, language: &str) -> Vec<Finding>
  ```

#### 驗收標準
- ✅ 規則可動態載入,無需重新編譯
- ✅ 規則數量從 15 增加到 **50+**
- ✅ 掃描效能提升 **25%+**
- ✅ 誤報率降低 **30%+**

#### 預期成果
- 安全研究員可直接貢獻規則
- 漏洞檢測覆蓋率提升 **230%**
- SAST 引擎業界競爭力顯著提升

---

### 📅 第7週 (2025-11-25 ~ 2025-12-01) - Python 核心優化

#### 目標
深化 Python 核心層的類型安全和效能

#### 任務清單

- [ ] **Task 7.1**: 強化類型檢查
  ```bash
  # 執行嚴格的類型檢查
  mypy services/core services/integration --strict
  # 目標: 類型覆蓋率 > 90%
  ```

- [ ] **Task 7.2**: 優化 FastAPI 依賴注入
  ```python
  # 統一的依賴注入模式
  - 資料庫 session 管理
  - Lifecycle manager 注入
  - AI agent 注入
  - 背景任務管理
  ```

- [ ] **Task 7.3**: 整合 AI 到生命週期管理
  ```python
  # AI 驅動的漏洞分析
  - 自動生成修復建議
  - 根因分析增強
  - 風險評分優化
  ```

- [ ] **Task 7.4**: 抽象化基礎設施程式碼
  ```python
  # aiva_common 增強
  - 統一錯誤處理裝飾器
  - 重試機制
  - 分散式追蹤整合
  ```

#### 驗收標準
- ✅ Python 類型覆蓋率 **> 90%**
- ✅ API 回應時間 **< 200ms** (P95)
- ✅ AI 生成修復建議準確率 **> 80%**

---

### 📅 第8週 (2025-12-02 ~ 2025-12-08) - 跨語言整合測試

#### 目標
建立完善的跨語言整合測試框架

#### 任務清單

- [ ] **Task 8.1**: 端到端測試
  ```python
  # 完整掃描流程測試
  1. Core 發起掃描
  2. Integration 分發任務到各語言 Function
  3. 收集結果
  4. 生命週期管理
  5. 報告生成
  ```

- [ ] **Task 8.2**: 效能測試
  ```
  # 各語言服務效能基準
  - Go 服務吞吐量測試
  - Rust SAST 掃描速度
  - Node.js 瀏覽器並發數
  - Python 分析延遲
  ```

- [ ] **Task 8.3**: 混沌測試
  ```
  # 故障注入測試
  - RabbitMQ 斷線恢復
  - 服務崩潰重啟
  - 資料庫連接失敗
  - 記憶體不足處理
  ```

- [ ] **Task 8.4**: Schema 一致性驗證
  ```python
  # 自動化驗證跨語言 Schema
  - Python Pydantic vs Go struct
  - Python Pydantic vs TypeScript interface
  - 序列化/反序列化測試
  ```

#### 驗收標準
- ✅ 整合測試覆蓋率 **> 70%**
- ✅ 所有核心流程可自動化驗證
- ✅ 故障恢復時間 **< 30秒**
- ✅ Schema 同步準確率 **100%**

---

### 📅 第9-10週 (2025-12-09 ~ 2025-12-22) - Protocol Buffers 評估與遷移規劃

#### 目標
評估並決定是否遷移到 Protocol Buffers

#### 任務清單

**Week 9 (12/09-12/15):**
- [ ] **Task 9.1**: Protocol Buffers 可行性研究
  ```protobuf
  // 定義統一的 Schema
  syntax = "proto3";
  
  message FindingPayload {
    string finding_id = 1;
    string scan_id = 2;
    VulnerabilityType vulnerability_type = 3;
    Severity severity = 4;
    // ...
  }
  ```

- [ ] **Task 9.2**: POC 實作
  - [ ] 建立 `.proto` 檔案
  - [ ] 自動生成 Python/Go/Rust/TypeScript 程式碼
  - [ ] 效能對比測試 (JSON vs Protobuf)
  - [ ] 開發體驗評估

- [ ] **Task 9.3**: 影響分析
  - [ ] 遷移工作量評估
  - [ ] 向後相容性策略
  - [ ] 團隊學習曲線

**Week 10 (12/16-12/22):**
- [ ] **Task 10.1**: 決策會議
  - [ ] 整理評估報告
  - [ ] 成本效益分析
  - [ ] 技術債務評估
  - [ ] 做出遷移決策

- [ ] **Task 10.2**: 如果決定遷移
  - [ ] 制定詳細遷移計劃 (Q1 2026)
  - [ ] 建立遷移工具
  - [ ] 漸進式遷移策略

- [ ] **Task 10.3**: 如果決定不遷移
  - [ ] 建立 Schema 同步自動化工具
  - [ ] 強化手動維護流程
  - [ ] CI/CD 中加入 Schema 驗證

#### 驗收標準
- ✅ 完成詳細評估報告
- ✅ 有明確的遷移/不遷移決策
- ✅ 如遷移,有完整的 Q1 2026 計劃

---

## 📈 成功指標追蹤

### 技術指標

| 指標 | 基準 | Week 2 | Week 4 | Week 6 | Week 10 | 目標 |
|------|------|--------|--------|--------|---------|------|
| Go 程式碼重複率 | 60% | 15% | - | - | < 10% | < 10% |
| 動態掃描 API 發現率 | 30% | - | 75% | - | > 80% | > 80% |
| SAST 規則數量 | 15 | - | - | 45 | 50+ | 50+ |
| Python 類型覆蓋率 | 60% | - | - | - | 90%+ | 90%+ |
| 整合測試覆蓋率 | 40% | - | - | - | 70%+ | 70%+ |

### 業務指標

| 指標 | 預期改善 (Week 10) |
|------|-------------------|
| 新功能開發速度 | +40% |
| 漏洞檢測準確率 | +25% |
| 系統整體吞吐量 | +60% |
| 維護成本 | -30% |
| 誤報率 | -35% |

---

## 🎯 每週工作節奏

### 標準週期

**週一:**
- 晨會: 本週目標對齊
- 審查上週完成項目
- 分配任務

**週二-週四:**
- 專注開發時間
- Daily Standup (15分鐘)
- Pair Programming (需要時)

**週五:**
- 程式碼審查
- 週報撰寫
- Demo 展示 (每2週)
- 回顧會議

---

## 🛠️ 開發工具與流程

### 必要工具

**Python:**
```bash
# 安裝開發依賴
pip install mypy black ruff pytest pytest-cov

# 程式碼格式化
black services/core services/integration

# 類型檢查
mypy services/core --strict

# 單元測試
pytest services/core/tests --cov
```

**Go:**
```bash
# 格式化
go fmt ./...

# 測試
go test ./... -v -cover

# 靜態分析
golangci-lint run
```

**Rust:**
```bash
# 格式化
cargo fmt

# 測試
cargo test

# Clippy 檢查
cargo clippy -- -D warnings
```

**TypeScript:**
```bash
# 格式化
npm run format

# Lint
npm run lint

# 測試
npm test

# 類型檢查
tsc --noEmit
```

---

## 📚 學習資源

### 必讀文件

1. **本專案文件:**
   - `MULTILANG_STRATEGY.md` - 完整策略
   - `ENHANCEMENT_IMPLEMENTATION_REPORT.md` - 生命週期管理
   - `docs/ARCHITECTURE_MULTILANG.md` - 架構圖

2. **外部資源:**
   - [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
   - [tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
   - [Playwright Best Practices](https://playwright.dev/docs/best-practices)
   - [Go Concurrency Patterns](https://go.dev/blog/pipelines)

---

## 🚨 風險與緩解措施

### 高風險項目

**風險1: Go 服務遷移影響生產環境**
- **緩解**: 使用藍綠部署,先在測試環境完整驗證
- **回滾計劃**: 保留舊版本 Docker 映像,可快速回滾

**風險2: Schema 不同步導致通訊失敗**
- **緩解**: CI/CD 中強制執行 Schema 驗證測試
- **監控**: 設定 RabbitMQ 死信佇列監控

**風險3: 團隊學習曲線影響進度**
- **緩解**: 前2週每天安排 1小時技術分享
- **Pairing**: 資深工程師與初級工程師配對

---

## ✅ 快速啟動檢查清單

### Week 1 開始前必須完成

- [ ] 確認所有團隊成員已閱讀策略文件
- [ ] 開發環境設置完成 (Go, Rust, Node.js, Python)
- [ ] Git 分支策略確認
- [ ] CI/CD 管道測試通過
- [ ] 專案管理工具設置 (Jira/GitHub Projects)

### 立即執行

```powershell
# 1. 初始化 Go 共用模組
.\init_go_common.ps1

# 2. 驗證測試全部通過
cd services\function\common\go\aiva_common_go
go test ./... -v

# 3. 建立 Week 1 工作分支
git checkout -b feature/migrate-go-services-week1

# 4. 開始遷移第一個服務
.\migrate_sca_service.ps1
```

---

## 📞 溝通與協作

### 每日 Standup 格式

```
昨天完成:
- [具體任務]

今天計劃:
- [具體任務]

遇到的阻礙:
- [問題描述]
```

### 週報模板

```markdown
## Week X 週報 (YYYY-MM-DD ~ YYYY-MM-DD)

### ✅ 已完成
- [ ] Task X.X: [描述]

### 🚧 進行中
- [ ] Task X.X: [描述] (進度: 60%)

### 📊 指標
- Go 程式碼重複率: XX%
- 測試覆蓋率: XX%

### 🚨 風險與問題
- [描述]

### 📅 下週計劃
- [具體任務]
```

---

## 🎉 里程碑慶祝

### Week 2 里程碑
**🎯 所有 Go 服務完成遷移**
- 團隊聚餐
- 技術分享: Go 共用模組最佳實踐

### Week 6 里程碑
**🎯 SAST 規則庫達到 50 條**
- 內部 Demo 展示
- 發佈技術部落格文章

### Week 10 里程碑
**🎯 多語言架構完全實施**
- 全公司技術分享
- 專案回顧與經驗總結
- Q1 2026 規劃會議

---

## 📝 附錄

### A. 相關文件索引

- [MULTILANG_STRATEGY.md](./MULTILANG_STRATEGY.md) - 完整策略
- [MULTILANG_STRATEGY_SUMMARY.md](./MULTILANG_STRATEGY_SUMMARY.md) - 快速摘要
- [MULTILANG_IMPLEMENTATION_REPORT.md](./MULTILANG_IMPLEMENTATION_REPORT.md) - 實施報告
- [docs/ARCHITECTURE_MULTILANG.md](./docs/ARCHITECTURE_MULTILANG.md) - 架構圖
- [ENHANCEMENT_IMPLEMENTATION_REPORT.md](./ENHANCEMENT_IMPLEMENTATION_REPORT.md) - 生命週期管理

### B. 聯絡人

- **架構師**: [姓名]
- **Python Tech Lead**: [姓名]
- **Go Tech Lead**: [姓名]
- **Rust Tech Lead**: [姓名]
- **TypeScript Tech Lead**: [姓名]

### C. 重要連結

- CI/CD Dashboard: [連結]
- 測試覆蓋率報告: [連結]
- 專案看板: [連結]
- 技術文件 Wiki: [連結]

---

**最後更新:** 2025-10-14  
**下次審查:** 每週五 (週報)  
**版本:** 1.0

---

## 🚀 現在就開始!

```powershell
# 執行以下命令開始 Week 1
.\init_go_common.ps1
cd services\function\common\go\aiva_common_go
go test ./... -v

# 如果測試全部通過,開始遷移第一個服務
.\migrate_sca_service.ps1
```

**記住核心原則:**
> 每種語言做它最擅長的事,透過統一的契約和諧協作 🤝
