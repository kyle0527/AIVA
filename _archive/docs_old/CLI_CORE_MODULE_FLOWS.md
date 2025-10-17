# AIVA 核心模組跨模組流程圖分類

## 📊 現有圖表總覽（14 個架構圖）

### 圖表類型分類

| 類型 | 數量 | 說明 |
|------|------|------|
| **Graph（結構圖）** | 8 個 | 展示模組結構和關係 |
| **Sequence（時序圖）** | 1 個 | 展示跨模組互動順序 |
| **Flowchart（流程圖）** | 5 個 | 展示具體檢測流程 |

---

## 🎯 以核心模組為中心的分類

### 第一類：系統架構圖（Graph）- 8 個

#### A. 整體架構層（3 個）
1. **01_overall_architecture.mmd** - 整體系統架構
   - 展示：Core + Scan + Function + Integration 四大模組
   - 用途：理解全局架構

2. **02_modules_overview.mmd** - 模組概覽
   - 展示：各模組職責和邊界
   - 用途：模組劃分清晰化

3. **13_data_flow.mmd** - 系統資料流向
   - 展示：資料如何在模組間流轉
   - 用途：追蹤資料流

#### B. 模組詳細層（4 個）
4. **03_core_module.mmd** - 核心模組（重點！）
   - 子模組：AI Engine, Analysis, Execution, State, UI
   - 核心職責：任務調度、策略生成、狀態管理

5. **04_scan_module.mmd** - 掃描模組
   - 功能：爬蟲引擎、資產發現、指紋識別
   - 與 Core 關係：接收任務 → 發現資產 → 回報 Core

6. **05_function_module.mmd** - 檢測模組
   - 功能：SQL 注入、XSS、SSRF、IDOR 檢測
   - 與 Core 關係：接收目標 → 執行檢測 → 回報結果

7. **06_integration_module.mmd** - 整合模組
   - 功能：結果彙總、報告生成、風險評估
   - 與 Core 關係：接收結果 → 分析整合 → 生成報告

#### C. 部署架構層（1 個）
8. **14_deployment_architecture.mmd** - Docker 部署架構
   - 展示：容器化部署結構
   - 用途：DevOps 參考

---

### 第二類：時序互動圖（Sequence）- 1 個

9. **11_complete_workflow.mmd** - 完整工作流程（關鍵！）
   - 展示：User → API → Core → Scan → MQ → Function → Integration
   - **核心模組的角色**：
     - 接收請求 → 創建任務
     - 調度掃描 → 分發檢測
     - 收集結果 → 觸發整合
   - 用途：理解跨模組互動順序

---

### 第三類：檢測流程圖（Flowchart）- 5 個

#### A. 漏洞檢測流程（4 個）
10. **07_sqli_flow.mmd** - SQL 注入檢測
    - 5 種引擎：Boolean/Error/Time/Union/OOB
    - 與 Core 關係：接收任務 → 執行檢測 → 回報發現

11. **08_xss_flow.mmd** - XSS 檢測
    - 3 種類型：Reflected/Stored/DOM
    - 與 Core 關係：同上

12. **09_ssrf_flow.mmd** - SSRF 檢測
    - 檢測：內部位址、OAST 帶外
    - 與 Core 關係：同上

13. **10_idor_flow.mmd** - IDOR 檢測
    - 檢測：跨使用者、垂直權限提升
    - 與 Core 關係：同上

#### B. 決策流程（1 個）
14. **12_language_decision.mmd** - 多語言架構決策
    - 展示：為什麼選擇 Python/Go/Rust/TypeScript
    - 用途：架構設計參考

---

## 🔗 以核心模組為中心的跨模組流程

### 核心模組的 6 種對外互動

基於現有圖表，核心模組有以下跨模組流程：

#### 1. **Core → Scan（掃描任務分發）**
```
指令：aiva scan start <url>
流程：Core 接收請求 → 生成掃描任務 → 發送到 Scan 模組
回傳：Scan 發現資產 → 回報給 Core → Core 更新上下文
```

#### 2. **Core → Function（檢測任務分發）**
```
指令：aiva detect <type> <target>
流程：Core 接收目標 → 生成檢測任務 → 發送到 Function 模組
回傳：Function 完成檢測 → 回報結果 → Core 收集整合
子類型：
  - aiva detect sqli
  - aiva detect xss
  - aiva detect ssrf
  - aiva detect idor
```

#### 3. **Core → Integration（整合分析請求）**
```
指令：aiva report generate <scan_id>
流程：Core 觸發整合 → Integration 彙總結果 → 生成報告
回傳：Integration 回報報告 → Core 提供給使用者
```

#### 4. **Core → AI（AI 決策與學習）**
```
指令：aiva ai train / aiva ai strategy
流程：Core 請求 AI 決策 → AI 生成策略 → 回傳給 Core
回傳：AI 策略建議 → Core 調整任務優先級
```

#### 5. **Core → MQ（訊息佇列通訊）**
```
內部通訊：所有模組間透過 MQ 通訊
流程：Core 發布任務到 MQ → 各模組訂閱 → 處理後回報
回傳：結果透過 MQ 回到 Core
```

#### 6. **Core ← → State（狀態管理）**
```
內部管理：維護掃描會話、上下文、進度
流程：持續更新和查詢狀態
```

---

## 🎨 可組合的流程圖方案

### 方案 A：完整端到端流程（推薦）
**組合圖表**：
1. `11_complete_workflow.mmd`（主軸）
2. `03_core_module.mmd`（核心詳細）
3. `04_scan_module.mmd`（掃描詳細）
4. `05_function_module.mmd`（檢測詳細）
5. `06_integration_module.mmd`（整合詳細）

**對應 CLI 指令**：
```bash
aiva workflow run <url>  # 完整工作流程
  ├─ aiva scan start <url>
  ├─ aiva detect all <targets>
  └─ aiva report generate <scan_id>
```

---

### 方案 B：掃描 + 檢測流程
**組合圖表**：
1. `04_scan_module.mmd`（掃描）
2. `07-10_*_flow.mmd`（4 種檢測流程）

**對應 CLI 指令**：
```bash
aiva scan-detect <url>  # 掃描並檢測
  ├─ aiva scan start <url>
  └─ aiva detect auto <scan_id>
```

---

### 方案 C：AI 驅動流程
**組合圖表**：
1. `03_core_module.mmd`（AI Engine 部分）
2. `11_complete_workflow.mmd`（決策點）

**對應 CLI 指令**：
```bash
aiva ai-scan <url>  # AI 智慧掃描
  ├─ Core AI 分析攻擊面
  ├─ 動態調整策略
  └─ 優先檢測高風險點
```

---

### 方案 D：模組通訊流程
**組合圖表**：
1. `13_data_flow.mmd`（資料流）
2. Core Messaging 相關的 15 個詳細圖（從 mermaid_details）

**對應 CLI 指令**：
```bash
aiva message trace <task_id>  # 追蹤訊息流
aiva status modules  # 查看模組狀態
```

---

### 方案 E：檢測專項流程
**組合圖表**：
- `07_sqli_flow.mmd`
- `08_xss_flow.mmd`
- `09_ssrf_flow.mmd`
- `10_idor_flow.mmd`

**對應 CLI 指令**：
```bash
aiva detect sqli <target> --engines all
aiva detect xss <target> --type reflected
aiva detect ssrf <target> --oast
aiva detect idor <target> --users 2
```

---

## 📋 總結：可用的組合類型

| 組合方案 | 圖表數量 | CLI 指令數 | 適用場景 |
|---------|---------|-----------|---------|
| **A. 完整端到端** | 5 個 | 3 個 | 完整掃描工作流 |
| **B. 掃描檢測** | 5 個 | 2 個 | 快速掃描+檢測 |
| **C. AI 驅動** | 2 個 | 1 個 | 智慧化掃描 |
| **D. 模組通訊** | 2+15 個 | 2 個 | 監控與除錯 |
| **E. 檢測專項** | 4 個 | 4 個 | 專項漏洞檢測 |

---

## 🚀 建議實作順序

### 第一階段：基礎流程（已完成 ✅）
- `aiva scan start`
- `aiva detect sqli/xss`
- `aiva report generate`

### 第二階段：組合流程（建議實作）
1. **`aiva workflow run`** - 完整端到端流程
2. **`aiva scan-detect`** - 掃描+檢測組合
3. **`aiva detect auto`** - 自動檢測所有類型

### 第三階段：進階功能
1. **`aiva ai-scan`** - AI 智慧掃描
2. **`aiva message trace`** - 訊息追蹤
3. **`aiva status modules`** - 模組狀態監控

---

**總計可組合流程**：**5 大類，共約 10-15 種組合方式**

您想先實作哪一種組合？我可以立即為您生成對應的 CLI 命令和流程圖！
