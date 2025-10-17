# 🎨 AIVA 架構圖使用指南 | Architecture Diagrams Usage Guide

> **專案 Project**: AIVA - AI-Powered Intelligent Vulnerability Analysis Platform
> **版本 Version**: v2.0 (優化版 Optimized)
> **更新時間 Last Updated**: 2025-10-13

---

## 📋 目錄 | Table of Contents

1. [簡介 Introduction](#簡介-introduction)
2. [圖表列表 Diagram List](#圖表列表-diagram-list)
3. [使用方法 Usage](#使用方法-usage)
4. [優化特點 Optimization Features](#優化特點-optimization-features)
5. [匯出指南 Export Guide](#匯出指南-export-guide)
6. [維護與更新 Maintenance](#維護與更新-maintenance)

---

## 🌟 簡介 | Introduction

本目錄包含 AIVA 專案的完整架構圖集，使用 **Mermaid** 語法生成，包含中英文雙語標籤。

### 特色 Features

✨ **14 張專業架構圖**

- 整體系統架構
- 四大模組詳細設計
- 各功能檢測流程
- 資料流與部署架構

🎨 **視覺優化**

- Emoji 圖示增強識別度
- 豐富的顏色方案
- 清晰的層次結構
- 技術細節標註

🌍 **中英雙語**

- 所有標籤同時顯示中英文
- 技術術語使用斜體標註
- 易於國際化團隊理解

⚡ **自動化生成**

- Python 腳本一鍵生成
- 支持批量匯出
- 易於維護更新

---

## 📊 圖表列表 | Diagram List

### 系統架構類 | System Architecture

| # | 圖表名稱 | 檔案 | 說明 |
|---|---------|------|------|
| 01 | 🏗️ 整體系統架構 | `01_overall_architecture.mmd` | 六層架構設計，展示所有主要組件及其交互 |
| 02 | 🔷 四大模組概覽 | `02_modules_overview.mmd` | Core、Scan、Function、Integration 模組關係 |
| 14 | 🐳 部署架構圖 | `14_deployment_architecture.mmd` | Docker/K8s 容器化部署架構 |

### 模組詳細設計 | Module Design

| # | 圖表名稱 | 檔案 | 說明 |
|---|---------|------|------|
| 03 | 🤖 核心引擎模組 | `03_core_module.mmd` | AI 引擎、策略生成、任務管理、狀態管理 |
| 04 | 🔍 掃描引擎模組 | `04_scan_module.mmd` | Python/TypeScript/Rust 三語言掃描器 |
| 05 | ⚡ 檢測功能模組 | `05_function_module.mmd` | 多語言檢測模組架構 |
| 06 | 🔗 整合服務模組 | `06_integration_module.mmd` | 分析、報告、風險評估 |

### 檢測流程圖 | Detection Workflows

| # | 圖表名稱 | 檔案 | 說明 |
|---|---------|------|------|
| 07 | 💉 SQL 注入檢測 | `07_sqli_flow.mmd` | 五引擎檢測流程 (Boolean/Time/Error/Union/OOB) |
| 08 | ⚡ XSS 檢測 | `08_xss_flow.mmd` | Reflected/Stored/DOM XSS 檢測流程 |
| 09 | 🌐 SSRF 檢測 | `09_ssrf_flow.mmd` | 內網探測與 OAST 平台檢測 |
| 10 | 🔒 IDOR 檢測 | `10_idor_flow.mmd` | BFLA/垂直提權/水平越權檢測 |

### 系統流程圖 | System Workflows

| # | 圖表名稱 | 檔案 | 說明 |
|---|---------|------|------|
| 11 | 🔄 完整掃描流程 | `11_complete_workflow.mmd` | 端到端掃描工作流程 (時序圖) |
| 12 | 🎯 語言架構決策 | `12_language_decision.mmd` | 技術選型決策樹 |
| 13 | 💾 資料流程圖 | `13_data_flow.mmd` | 資料在系統中的流轉 |

---

## 🚀 使用方法 | Usage

### 方法 1: VS Code 預覽 (推薦)

1. **安裝 Mermaid 擴展**

   ```
   Extension ID: bierner.markdown-mermaid
   ```

2. **開啟 Markdown 預覽**
   - 在 `.mmd` 檔案上右鍵
   - 選擇 "Open Preview" 或按 `Ctrl+Shift+V`

3. **查看完整文件**
   - 開啟 `INDEX.md` 可查看所有圖表索引
   - 點擊連結跳轉到各個圖表

### 方法 2: 線上預覽

訪問 [Mermaid Live Editor](https://mermaid.live/)：

1. 複製 `.mmd` 檔案內容
2. 貼上到編輯器
3. 即時預覽和編輯

### 方法 3: GitHub/GitLab 直接查看

GitHub 和 GitLab 原生支持 Mermaid，直接查看 Markdown 檔案即可渲染圖表。

---

## ✨ 優化特點 | Optimization Features

### 1. 📱 Emoji 圖示系統

每個模組使用獨特的 Emoji 圖示，增強視覺識別：

| Emoji | 含義 | 使用場景 |
|-------|------|---------|
| 🤖 | AI/機器學習 | 核心引擎、智能分析 |
| 🔍 | 掃描/搜尋 | 掃描引擎 |
| ⚡ | 高性能/檢測 | 檢測模組 |
| 🔗 | 整合/連接 | 整合服務 |
| 💾 | 資料/儲存 | 資料庫 |
| 📨 | 訊息/佇列 | RabbitMQ |
| 🐍 | Python | Python 模組 |
| 🔷 | Go | Go 模組 |
| 🦀 | Rust | Rust 模組 |
| 📘 | TypeScript | TypeScript 模組 |

### 2. 🎨 豐富的顏色方案

根據功能層級使用不同顏色：

```
前端層：藍色 (#E3F2FD) - 清新專業
核心層：黃色 (#FFF9C4) - 醒目重要
掃描層：綠色 (#C8E6C9) - 處理進行中
檢測層：紫色 (#E1BEE7) - 分析判斷
整合層：橙色 (#FFE0B2) - 彙整輸出
資料層：灰色 (#CFD8DC) - 穩定可靠
```

### 3. 📝 技術細節標註

每個節點包含三層資訊：

```mermaid
NODE["🔷 Go Functions<br/>Go 檢測模組<br/><i>AuthN, CSPM, SCA</i>"]
      ↑               ↑                ↑
   Emoji      中英文標籤        技術細節
```

### 4. 🔗 連線標籤

連線上標註資料流向：

```
API -->|HTTP Request| CORE
SCAN -->|Targets| MQ
```

### 5. 🎯 分層設計

使用 `subgraph` 清晰劃分邏輯層：

- 前端層 Frontend Layer
- 核心層 Core Layer
- 掃描層 Scan Layer
- 檢測層 Detection Layer
- 整合層 Integration Layer
- 資料層 Data Layer

---

## 📤 匯出指南 | Export Guide

### 自動匯出 (推薦)

使用提供的 Python 腳本：

```bash
# 匯出 PNG 格式
python tools/generate_complete_architecture.py --export png

# 匯出 SVG 格式 (向量圖，推薦用於文檔)
python tools/generate_complete_architecture.py --export svg

# 匯出 PDF 格式
python tools/generate_complete_architecture.py --export pdf
```

### 手動匯出

#### 1. 安裝 Mermaid CLI

```bash
npm install -g @mermaid-js/mermaid-cli
```

#### 2. 轉換單個檔案

```bash
# PNG 格式
mmdc -i 01_overall_architecture.mmd -o 01_overall_architecture.png

# SVG 格式 (推薦)
mmdc -i 01_overall_architecture.mmd -o 01_overall_architecture.svg

# 自訂背景色
mmdc -i 01_overall_architecture.mmd -o output.png -b transparent

# 高解析度
mmdc -i 01_overall_architecture.mmd -o output.png -w 2048 -H 1536
```

#### 3. 批次轉換

```bash
# Bash 腳本
for file in *.mmd; do
    mmdc -i "$file" -o "${file%.mmd}.png" -b transparent
done

# PowerShell 腳本
Get-ChildItem *.mmd | ForEach-Object {
    mmdc -i $_.Name -o ($_.BaseName + ".png") -b transparent
}
```

### VS Code 擴展匯出

使用 `Markdown PDF` 擴展：

1. 安裝擴展: `yzane.markdown-pdf`
2. 在 Markdown 檔案中按 `F1`
3. 選擇 "Markdown PDF: Export (png/svg/pdf)"

---

## 🔧 維護與更新 | Maintenance

### 更新圖表

#### 方法 1: 修改腳本 (推薦)

編輯 `tools/generate_complete_architecture.py`：

```python
def _generate_overall_architecture(self) -> Path:
    """生成整體系統架構圖"""
    mermaid_code = '''graph TB
        # 在這裡修改 Mermaid 語法
    '''
    # ...
```

然後重新執行：

```bash
python tools/generate_complete_architecture.py
```

#### 方法 2: 直接修改 .mmd 檔案

直接編輯 `_out/architecture_diagrams/*.mmd` 檔案。

**注意**: 直接修改會在下次執行腳本時被覆蓋！

### 添加新圖表

在 `generate_complete_architecture.py` 中添加新方法：

```python
def _generate_new_diagram(self) -> Path:
    """生成新圖表"""
    print("  📊 生成新圖表... | Generating new diagram...")

    mermaid_code = '''graph TB
        # 你的 Mermaid 語法
    '''

    output_file = self.output_dir / "15_new_diagram.mmd"
    self._write_diagram(output_file, mermaid_code,
                      "新圖表 | New Diagram")
    return output_file
```

然後在 `generate_all_diagrams()` 中調用：

```python
def generate_all_diagrams(self) -> List[Path]:
    diagrams = []
    # ... 其他圖表
    diagrams.append(self._generate_new_diagram())
    return diagrams
```

### 版本控制

建議將圖表納入 Git 版本控制：

```bash
# 添加所有圖表
git add _out/architecture_diagrams/

# 提交變更
git commit -m "docs: update architecture diagrams"

# 標註版本
git tag -a diagrams-v2.0 -m "Optimized architecture diagrams"
```

---

## 🎓 Mermaid 語法快速參考 | Syntax Quick Reference

### 圖表類型

```mermaid
graph TB          %% 上到下流程圖
graph LR          %% 左到右流程圖
flowchart TD      %% 增強型流程圖
sequenceDiagram   %% 時序圖
```

### 節點形狀

```mermaid
A[方形節點]
B(圓角節點)
C([體育場型])
D[[子程序]]
E[(資料庫)]
F((圓形))
G{菱形決策}
```

### 連線類型

```mermaid
A --> B           %% 實線箭頭
A -.-> B          %% 虛線箭頭
A ==> B           %% 粗箭頭
A -->|標籤| B      %% 帶標籤連線
```

### 樣式設定

```mermaid
style NODE fill:#FFD54F,stroke:#F57F17,stroke-width:3px
linkStyle 0 stroke:#666,stroke-width:2px
```

---

## 📚 相關資源 | Related Resources

### 官方文檔

- [Mermaid 官方文檔](https://mermaid.js.org/)
- [Mermaid Live Editor](https://mermaid.live/)
- [GitHub Mermaid 支援](https://github.blog/2022-02-14-include-diagrams-markdown-files-mermaid/)

### VS Code 擴展

- **Markdown Preview Mermaid Support**
  `bierner.markdown-mermaid`

- **Mermaid Editor**
  `tomoyukim.vscode-mermaid-editor`

- **Markdown PDF**
  `yzane.markdown-pdf`

### 工具推薦

- **mermaid-cli**: 命令列工具
- **PlantUML**: 替代方案
- **Draw.io**: 手動繪圖工具

---

## 🤝 貢獻 | Contributing

歡迎提交改進建議！

### 改進想法

- [ ] 添加更多檢測模組流程圖
- [ ] 增加錯誤處理流程
- [ ] 添加效能監控圖表
- [ ] 創建 API 文檔圖表
- [ ] 添加安全架構圖

### 提交步驟

1. Fork 專案
2. 創建特性分支: `git checkout -b feature/new-diagram`
3. 修改圖表或腳本
4. 提交變更: `git commit -m "Add: new diagram for XXX"`
5. 推送分支: `git push origin feature/new-diagram`
6. 提交 Pull Request

---

## 📞 支援 | Support

如有問題或建議，請聯繫：

- **GitHub Issues**: [AIVA Issues](https://github.com/kyle0527/AIVA/issues)
- **Email**: <support@aiva-project.com>
- **文檔**: [AIVA Documentation](https://docs.aiva-project.com)

---

## 📄 授權 | License

本專案採用 MIT License。詳見 [LICENSE](../../LICENSE) 檔案。

---

**最後更新 Last Updated**: 2025-10-13
**維護者 Maintainer**: AIVA Development Team
**版本 Version**: v2.0
