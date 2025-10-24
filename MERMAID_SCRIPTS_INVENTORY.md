# AIVA 系統中的 MERMAID 相關腳本清單

## 📋 MERMAID 腳本分類列表

### 1. 🐍 Python Mermaid 工具腳本

#### 核心生成工具
```
📁 tools/features/
├── mermaid_optimizer.py          # 🎯 Mermaid v10+ 圖表優化器 (451行)
```

**功能特性:**
- ✅ 符合 Mermaid.js v10+ 官方語法規範
- ✅ 支援現代主題配置和自定義變數
- ✅ 優化節點和連線樣式，支援 CSS 類
- ✅ 增強可讀性和美觀度，支援 HTML 標籤
- ✅ 支援響應式佈局和高 DPI 顯示
- ✅ 無障礙功能和語意化標籤

#### 轉換和生成工具
```
📁 tools/common/development/
├── py2mermaid.py                 # 🔄 Python AST 轉 Mermaid 流程圖 (514行)
├── generate_mermaid_diagrams.py  # 📊 專案 Mermaid 架構圖生成器 (418行)
└── generate_complete_architecture.py # 🏗️ 完整架構圖生成 (包含 Mermaid 輸出)
```

**py2mermaid.py 功能:**
- 🔍 Python AST 語法樹解析
- 🔄 自動轉換為 Mermaid 流程圖
- 🎨 支援多種節點樣式和連線類型
- ✂️ 智能 ID 清理和驗證

**generate_mermaid_diagrams.py 功能:**
- 🏗️ 多語言架構概覽圖
- 📦 模組關係和依賴圖
- 🔧 技術棧選擇決策圖
- 🌐 部署架構圖

### 2. 📄 Mermaid 圖表文件 (.mmd)

#### 架構圖表目錄
```
📁 _out/architecture_diagrams/ (14個專業架構圖)
├── 01_overall_architecture.mmd      # 🎯 系統整體架構
├── 02_modules_overview.mmd          # 📦 模組概覽圖  
├── 03_core_module.mmd              # ⚙️  核心模組架構
├── 04_scan_module.mmd              # 🔍 掃描模組架構
├── 05_function_module.mmd          # 🔧 功能模組架構
├── 06_integration_module.mmd       # 🔗 整合模組架構
├── 07_sqli_flow.mmd                # 💉 SQL 注入檢測流程
├── 08_xss_flow.mmd                 # 🚨 XSS 檢測流程
├── 09_ssrf_flow.mmd                # 🌐 SSRF 檢測流程
├── 10_idor_flow.mmd                # 🔒 IDOR 檢測流程
├── 11_complete_workflow.mmd        # 📋 完整工作流程
├── 12_language_decision.mmd        # 🔤 語言選擇決策
├── 13_data_flow.mmd                # 📊 數據流程圖
└── 14_deployment_architecture.mmd  # 🚀 部署架構圖
```

### 3. 📚 包含 Mermaid 語法的文檔

#### 主要文檔文件
```
📁 文檔中的 Mermaid 圖表:
├── README.md                       # 主項目說明 (2個 mermaid 區塊)
├── REPOSITORY_STRUCTURE.md         # 倉庫結構圖 (1個 mermaid 區塊)
├── services/scan/README.md         # 掃描模組架構圖 (1個大型架構圖)
└── docs/ARCHITECTURE/COMPLETE_ARCHITECTURE_DIAGRAMS.md # 完整架構文檔 (15+ mermaid 圖表)
```

#### 架構文檔詳情
```
📁 docs/ARCHITECTURE/
└── COMPLETE_ARCHITECTURE_DIAGRAMS.md # 🏗️ 完整架構圖集合
    ├── 系統整體架構
    ├── 核心模組架構 
    ├── 掃描引擎架構
    ├── 功能模組架構
    ├── 整合服務架構
    ├── 數據流程架構
    ├── 部署架構圖
    ├── 漏洞檢測流程 (SQL注入/XSS/SSRF/IDOR)
    ├── 完整工作流程
    ├── 語言選擇決策樹
    └── 技術架構決策
```

## 🎯 主要 Mermaid 工具功能矩陣

| 工具腳本 | 主要功能 | 輸入格式 | 輸出格式 | 代碼行數 |
|---------|---------|---------|---------|---------|
| **mermaid_optimizer.py** | 圖表優化和美化 | Mermaid 語法 | 優化後 Mermaid | 451行 |
| **py2mermaid.py** | Python 代碼轉流程圖 | .py 文件 | Mermaid 流程圖 | 514行 |
| **generate_mermaid_diagrams.py** | 架構圖生成 | 項目結構 | 多種架構圖 | 418行 |

## 🔧 使用方式和範例

### 1. Mermaid 優化器使用
```python
from tools.features.mermaid_optimizer import MermaidOptimizer

# 創建優化器
optimizer = MermaidOptimizer()

# 優化流程圖
optimized_code = optimizer.optimize_flowchart(mermaid_code)

# 添加節點和樣式
optimizer.add_node("node1", "Node Label", shape="rectangle")
optimizer.add_link("node1", "node2", label="connects to")
```

### 2. Python 轉 Mermaid 使用  
```python
from tools.common.development.py2mermaid import py_to_mermaid

# 轉換 Python 文件
mermaid_diagram = py_to_mermaid("services/scan/scan_orchestrator.py")

# 輸出到文件
with open("output.mmd", "w") as f:
    f.write(mermaid_diagram)
```

### 3. 架構圖生成使用
```python
from tools.common.development.generate_mermaid_diagrams import generate_multilang_architecture

# 生成多語言架構圖
arch_diagram = generate_multilang_architecture()

# 生成並保存所有架構圖
python tools/common/development/generate_mermaid_diagrams.py
```

## 🎨 Mermaid 主題和樣式配置

### 支援的主題類型
```python
# 可用主題 (mermaid_optimizer.py)
themes = [
    "default",      # 預設主題
    "dark",         # 深色主題  
    "forest",       # 森林主題
    "neutral",      # 中性主題
    "base"          # 基礎主題
]

# 現代配色方案
primary_colors = "#0F172A"      # Modern Dark Blue
secondary_colors = "#F1F5F9"    # Light Gray  
tertiary_colors = "#ECFDF5"     # Light Green
```

### 自定義樣式類
```css
/* 支援的 CSS 樣式類 */
.primary-node { fill: #0F172A; stroke: #3B82F6; }
.secondary-node { fill: #F1F5F9; stroke: #64748B; }
.tertiary-node { fill: #ECFDF5; stroke: #10B981; }
.warning-node { fill: #FEF3C7; stroke: #D97706; }
.danger-node { fill: #FEE2E2; stroke: #DC2626; }
```

## 📊 圖表類型支援

### 流程圖類型
- ✅ **Flowchart** - 基本流程圖
- ✅ **Sequence** - 時序圖  
- ✅ **Class** - 類圖
- ✅ **State** - 狀態圖
- ✅ **Entity Relationship** - 實體關係圖
- ✅ **User Journey** - 用戶歷程圖
- ✅ **Gantt** - 甘特圖

### 節點形狀支援
```mermaid
graph TB
    A[矩形節點]
    B(圓角矩形)
    C((圓形節點))
    D{菱形決策}
    E[/平行四邊形/]
    F[\\梯形\\]
    G>旗幟形]
```

## 🔍 檔案搜索結果統計

### 檔案類型統計
- **Python 腳本**: 3個主要工具 (1,383 總行數)
- **Mermaid 文件**: 14個 .mmd 架構圖
- **文檔包含**: 20+ 個文檔文件包含 mermaid 語法
- **搜索匹配**: 總共找到 100+ 個相關匹配項

### 分佈位置
- `tools/features/` - 優化工具
- `tools/common/development/` - 開發工具
- `_out/architecture_diagrams/` - 圖表輸出
- `docs/ARCHITECTURE/` - 架構文檔
- `services/scan/` - 掃描模組文檔

## 🚀 快速開始指南

### 環境準備
```bash
# 安裝 Python 依賴
pip install ast pathlib typing dataclasses

# 安裝 Mermaid CLI (可選，用於圖片輸出)
npm install -g @mermaid-js/mermaid-cli

# 或使用線上編輯器
# https://mermaid.live/
```

### 常用命令
```bash
# 生成專案所有架構圖
python tools/common/development/generate_mermaid_diagrams.py

# Python 代碼轉 Mermaid
python tools/common/development/py2mermaid.py services/scan/scan_orchestrator.py

# 優化現有 Mermaid 圖表
python -c "from tools.features.mermaid_optimizer import MermaidOptimizer; print(MermaidOptimizer().optimize_flowchart(open('diagram.mmd').read()))"
```

## 📝 維護和更新

### 版本兼容性
- **Mermaid.js**: v10+ (最新語法標準)
- **Python**: 3.8+ (支援 AST 和 typing)
- **Node.js**: 16+ (Mermaid CLI 需求)

### 定期更新任務
1. 🔄 定期更新架構圖 (當模組結構變化時)
2. 🎨 優化主題和樣式 (跟隨設計規範)
3. 📊 擴充圖表類型支援 (新的 Mermaid 功能)
4. 🔧 改進自動化生成邏輯 (提高準確性)

---

📝 **文檔版本**: v1.0.0  
🔄 **最後更新**: 2025-10-24  
📊 **統計時間**: 2025-10-24 16:30  
👥 **維護者**: AIVA Development Team  

💡 **提示**: 所有 Mermaid 圖表都支援線上預覽，訪問 https://mermaid.live/ 即可即時查看效果！