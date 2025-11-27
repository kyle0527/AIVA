# AIVA Features 模組 - 功能檢測 ⚡

## 📑 目錄

- [📋 模組概述](#-模組概述)
- [🔧 工具清單](#-工具清單)
  - [圖表優化工具](#圖表優化工具)
  - [內容處理工具](#內容處理工具)
- [🚀 使用工作流](#-使用工作流)
  - [1. Mermaid 圖表優化](#1-mermaid-圖表優化)
  - [2. 樹狀圖清理](#2-樹狀圖清理)
- [⚙️ 配置說明](#-配置說明)
  - [Mermaid 配置選項](#mermaid-配置選項)
  - [現代化特性](#現代化特性)
- [🔗 與其他模組的關係](#-與其他模組的關係)
- [📊 技術規範](#-技術規範)
  - [Mermaid.js 版本支援](#mermaidjs-版本支援)
  - [顏色系統](#顏色系統)
  - [字體系統](#字體系統)
- [📝 維護注意事項](#-維護注意事項)
- [🎨 設計原則](#-設計原則)

---

## 📋 模組概述

Features 模組提供功能增強和優化工具，專注於提升開發體驗和圖表品質。

## 🔧 工具清單

### 圖表優化工具

#### 1. `mermaid_optimizer.py` ⭐
**功能**: 現代化 Mermaid.js v10+ 圖表優化器
- 符合最新 Mermaid.js 官方語法規範
- 支援現代主題配置和自定義主題變數
- 支援 HTML 標籤和 CSS 類
- 提供多種節點形狀和連線樣式
- 支援無障礙功能和響應式佈局

**主要類別**:
- `MermaidTheme`: 現代化主題配置
- `MermaidConfig`: v10+ 配置選項
- `MermaidOptimizer`: 核心優化器

**節點形狀**:
- `rectangle` - 標準矩形
- `rounded` - 圓角矩形
- `stadium` - 體育場形 (Pill)
- `circle` - 圓形
- `rhombus` - 菱形 (決策)
- `hexagon` - 六角形
- 更多現代化形狀...

**連線類型**:
- `arrow` - 實線箭頭
- `dotted` - 虛線箭頭
- `thick` - 粗實線箭頭
- `bidirectional` - 雙向箭頭
- `x_arrow` - X型終止
- `circle_arrow` - 圓型終止

**顏色方案**:
- `python`, `go`, `rust`, `typescript` - 程式語言主題
- `core`, `scan`, `function`, `integration` - 模組主題
- `success`, `warning`, `danger`, `info` - 狀態主題

**使用範例**:
```python
from mermaid_optimizer import MermaidOptimizer

# 建立優化器
optimizer = MermaidOptimizer()

# 創建現代化節點
node = optimizer.create_node(
    "ai-core", "AI 核心", "AI Core Engine", 
    "Bio Neuron Network", icon="🤖"
)

# 創建連線
link = optimizer.create_link("api", "ai-core", "HTTP Request")

# 生成完整圖表
header = optimizer.generate_header("flowchart TD")
```

### 內容處理工具

#### 2. `remove_init_marks.py`
**功能**: 移除 `__init__.py` 檔案的功能標記
- 清理樹狀圖中不必要的標記
- 保持初始化檔案的語意正確性
- 提供移除前後的統計資訊

**使用方式**:
```bash
python tools/features/remove_init_marks.py
```

## 🚀 使用工作流

### 1. Mermaid 圖表優化
```bash
# 在 Python 腳本中使用
python -c "
from tools.features.mermaid_optimizer import MermaidOptimizer
optimizer = MermaidOptimizer()
print(optimizer.generate_header('flowchart TD'))
"
```

### 2. 樹狀圖清理
```bash
# 清理不必要的標記
python tools/features/remove_init_marks.py
```

## ⚙️ 配置說明

### Mermaid 配置選項

**主題配置**:
- `theme`: `"base"`, `"forest"`, `"dark"`, `"neutral"`
- `html_labels`: 支援 HTML 標籤
- `accessibility`: 無障礙功能
- `responsive`: 響應式佈局

**安全配置**:
- `secure`: 啟用安全模式
- `sandbox_mode`: 沙盒模式

**渲染配置**:
- `renderer`: `"svg"`, `"dagre-d3"`
- `flow_curve`: `"basis"`, `"linear"`, `"cardinal"`

### 現代化特性

**v10+ 新功能**:
- CSS 自定義屬性支援
- 改進的主題變數系統
- HTML 標籤支援
- 更好的無障礙功能
- 現代化字體堆疊

**最佳實踐**:
- 使用 kebab-case 節點 ID
- 遵循 WCAG 2.1 AA 顏色對比標準
- 使用語意化的 CSS 類名
- 支援高 DPI 顯示

## 🔗 與其他模組的關係

- **Common**: 提供基礎配置和樣式規範
- **Core**: 接收核心模組的圖表需求
- **Scan**: 處理掃描結果的可視化
- **Integration**: 支援外部工具的圖表生成

## 📊 技術規範

### Mermaid.js 版本支援
- **目標版本**: v10.0+
- **向後兼容**: v9.x (部分功能)
- **語法標準**: 官方 Mermaid.js 規範

### 顏色系統
- **主色系**: 基於現代設計系統
- **對比度**: 符合 WCAG 2.1 AA 標準
- **調色板**: 支援明暗主題切換

### 字體系統
```css
font-family: "Inter", "system-ui", "-apple-system", 
             "BlinkMacSystemFont", "Segoe UI", sans-serif
```

## 📝 維護注意事項

1. **版本更新**: 定期檢查 Mermaid.js 新版本功能
2. **語法驗證**: 確保生成的語法符合官方規範
3. **主題一致性**: 保持與專案整體設計的一致性
4. **無障礙性**: 遵循無障礙設計原則
5. **效能優化**: 優化大型圖表的渲染效能

## 🎨 設計原則

1. **現代化**: 使用最新的 Mermaid.js 功能
2. **一致性**: 統一的視覺語言和樣式規範
3. **可讀性**: 清晰的層次結構和適當的間距
4. **可訪問性**: 支援螢幕閱讀器和鍵盤導航
5. **響應式**: 適應不同螢幕尺寸和解析度

---

*最後更新: 2024-10-19*
*模組版本: Features v2.0*
*Mermaid 支援: v10.0+*