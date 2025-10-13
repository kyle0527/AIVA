#!/bin/bash
# AIVA 專案完整報告生成器 (Bash 版本)
# 整合樹狀圖、統計數據、程式碼分析、多語言支援

PROJECT_ROOT="${1:-/workspaces/AIVA}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/_out}"

echo "🚀 開始生成專案完整報告..."
echo ""

# 要排除的目錄
EXCLUDE_DIRS=(
    ".git"
    "__pycache__"
    ".mypy_cache"
    ".ruff_cache"
    "node_modules"
    ".venv"
    "venv"
    ".pytest_cache"
    ".tox"
    "dist"
    "build"
    ".egg-info"
    ".eggs"
    "htmlcov"
    ".coverage"
    ".hypothesis"
    ".idea"
    ".vscode"
    "target"
    "emoji_backups"
)

# 建立 find 排除參數
FIND_EXCLUDE=""
for dir in "${EXCLUDE_DIRS[@]}"; do
    FIND_EXCLUDE="$FIND_EXCLUDE -path '*/$dir' -prune -o"
done

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# ==================== 1. 收集統計數據 ====================
echo "📊 收集專案統計數據..."
echo ""

# 統計各語言檔案
TOTAL_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -type f -print" | wc -l)
PY_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '*.py' -type f -print" | wc -l)
GO_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '*.go' -type f -print" | wc -l)
RS_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '*.rs' -type f -print" | wc -l)
TS_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '*.ts' -type f -print" | wc -l)
JS_FILES=$(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '*.js' -type f -print" | wc -l)

# 統計程式碼行數
count_lines() {
    local pattern=$1
    local total=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            lines=$(wc -l < "$file" 2>/dev/null || echo 0)
            total=$((total + lines))
        fi
    done < <(eval "find '$PROJECT_ROOT' $FIND_EXCLUDE -name '$pattern' -type f -print")
    echo $total
}

echo "  統計 Python 程式碼..."
PY_LINES=$(count_lines "*.py")
echo "  統計 Go 程式碼..."
GO_LINES=$(count_lines "*.go")
echo "  統計 Rust 程式碼..."
RS_LINES=$(count_lines "*.rs")
echo "  統計 TypeScript 程式碼..."
TS_LINES=$(count_lines "*.ts")
echo "  統計 JavaScript 程式碼..."
JS_LINES=$(count_lines "*.js")

# 計算總計和百分比
TOTAL_CODE_LINES=$((PY_LINES + GO_LINES + RS_LINES + TS_LINES + JS_LINES))
if [ $TOTAL_CODE_LINES -eq 0 ]; then
    TOTAL_CODE_LINES=1  # 避免除以零
fi

# 計算總計和百分比
TOTAL_CODE_LINES=$((PY_LINES + GO_LINES + RS_LINES + TS_LINES + JS_LINES))
if [ $TOTAL_CODE_LINES -eq 0 ]; then
    TOTAL_CODE_LINES=1  # 避免除以零
fi

PY_PCT=$(awk "BEGIN { printf \"%.1f\", $PY_LINES * 100 / $TOTAL_CODE_LINES }")
GO_PCT=$(awk "BEGIN { printf \"%.1f\", $GO_LINES * 100 / $TOTAL_CODE_LINES }")
RS_PCT=$(awk "BEGIN { printf \"%.1f\", $RS_LINES * 100 / $TOTAL_CODE_LINES }")
TS_JS_PCT=$(awk "BEGIN { printf \"%.1f\", ($TS_LINES + $JS_LINES) * 100 / $TOTAL_CODE_LINES }")

PY_AVG=$(if [ $PY_FILES -gt 0 ]; then awk "BEGIN { printf \"%.1f\", $PY_LINES / $PY_FILES }"; else echo "0"; fi)
GO_AVG=$(if [ $GO_FILES -gt 0 ]; then awk "BEGIN { printf \"%.1f\", $GO_LINES / $GO_FILES }"; else echo "0"; fi)
RS_AVG=$(if [ $RS_FILES -gt 0 ]; then awk "BEGIN { printf \"%.1f\", $RS_LINES / $RS_FILES }"; else echo "0"; fi)

echo "✅ 統計完成"
echo ""

# ==================== 2. 生成樹狀圖 ====================
echo "🌳 生成專案樹狀結構..."
echo ""

TREE_FILE="$OUTPUT_DIR/tree_ascii.txt"
if command -v tree &> /dev/null; then
    tree -L 3 -I "$(IFS='|'; echo "${EXCLUDE_DIRS[*]}")" "$PROJECT_ROOT" > "$TREE_FILE"
    echo "✅ 樹狀圖已生成: tree_ascii.txt"
else
    echo "⚠️  tree 命令不存在，跳過樹狀圖生成"
    echo "專案根目錄: $PROJECT_ROOT" > "$TREE_FILE"
fi
echo ""

# ==================== 3. 生成整合報告 ====================
echo "📝 生成整合報告..."
echo ""

REPORT_FILE="$OUTPUT_DIR/PROJECT_REPORT.txt"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

cat > "$REPORT_FILE" << EOF
╔══════════════════════════════════════════════════════════════════════════════╗
║                         AIVA 專案完整分析報告                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

生成時間: $TIMESTAMP
專案路徑: $PROJECT_ROOT

═══════════════════════════════════════════════════════════════════════════════
📊 專案統計摘要
═══════════════════════════════════════════════════════════════════════════════

總文件數量: $TOTAL_FILES
總程式碼行數: $TOTAL_CODE_LINES
程式碼檔案數: $((PY_FILES + GO_FILES + RS_FILES + TS_FILES + JS_FILES))

───────────────────────────────────────────────────────────────────────────────
💻 多語言程式碼統計
───────────────────────────────────────────────────────────────────────────────

🐍 Python
   檔案數: $PY_FILES 個
   程式碼行數: $PY_LINES 行
   平均每個檔案: $PY_AVG 行
   佔比: $PY_PCT%

🔷 Go
   檔案數: $GO_FILES 個
   程式碼行數: $GO_LINES 行
   平均每個檔案: $GO_AVG 行
   佔比: $GO_PCT%

🦀 Rust
   檔案數: $RS_FILES 個
   程式碼行數: $RS_LINES 行
   平均每個檔案: $RS_AVG 行
   佔比: $RS_PCT%

📘 TypeScript/JavaScript
   檔案數: $((TS_FILES + JS_FILES)) 個
   程式碼行數: $((TS_LINES + JS_LINES)) 行
   佔比: $TS_JS_PCT%

───────────────────────────────────────────────────────────────────────────────
📈 專案規模分析
───────────────────────────────────────────────────────────────────────────────

語言分布:
  Python:     $PY_PCT% ████████████████████
  Go:         $GO_PCT%
  Rust:       $RS_PCT%
  TS/JS:      $TS_JS_PCT%

───────────────────────────────────────────────────────────────────────────────
🚫 已排除的目錄類型
───────────────────────────────────────────────────────────────────────────────

EOF

for dir in "${EXCLUDE_DIRS[@]}"; do
    echo "  • $dir" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

═══════════════════════════════════════════════════════════════════════════════
🌳 專案目錄結構
═══════════════════════════════════════════════════════════════════════════════

詳細樹狀結構請參考: tree_ascii.txt

基本結構:
  services/
    ├── aiva_common/     共用模組
    ├── core/            核心引擎
    ├── function/        功能模組 (Python, Go, Rust)
    ├── integration/     整合層
    └── scan/            掃描引擎 (Python, TypeScript)

═══════════════════════════════════════════════════════════════════════════════
🏗️ 專案架構說明
═══════════════════════════════════════════════════════════════════════════════

多語言架構設計:
  • Python: 主要業務邏輯、Web API、核心引擎
  • Go: 高效能模組 (身份驗證、雲端安全、組成分析)
  • Rust: 靜態分析、資訊收集 (記憶體安全、高效能)
  • TypeScript: 動態掃描引擎 (Playwright 瀏覽器自動化)

技術棧詳細資訊:
  • 詳細架構圖請參考: ARCHITECTURE_DIAGRAMS.md
  • 程式碼分析請參考: _out/analysis/ 目錄

═══════════════════════════════════════════════════════════════════════════════
📌 報告說明
═══════════════════════════════════════════════════════════════════════════════

• 本報告整合了專案的檔案統計、程式碼行數分析和目錄結構
• 已自動排除虛擬環境、快取檔案、IDE 配置等非程式碼目錄
• 圖示說明:
  🐍 Python   📜 JavaScript   📘 TypeScript   📝 Markdown
  ⚙️ JSON      🔧 YAML         🗄️ SQL          ⚡ Shell/Batch
  🔷 Go        🦀 Rust         🌐 HTML         🎨 CSS
  📁 目錄      📄 其他檔案

• 多語言架構:
  - Python: 主要業務邏輯、Web API、核心引擎
  - Go: 高效能模組 (身份驗證、雲端安全、組成分析)
  - Rust: 靜態分析、資訊收集 (記憶體安全、高效能)
  - TypeScript: 動態掃描引擎 (Playwright 瀏覽器自動化)

═══════════════════════════════════════════════════════════════════════════════
✨ 報告結束
═══════════════════════════════════════════════════════════════════════════════
EOF

echo "✅ 報告已生成: PROJECT_REPORT.txt"
echo ""

# ==================== 4. 生成 Mermaid 圖表 ====================
echo "📊 生成 Mermaid 架構圖..."
echo ""

python3 "$PROJECT_ROOT/tools/generate_mermaid_diagrams.py"

echo ""

# ==================== 5. 完成 ====================
echo "╔════════════════════════════════════════════════╗"
echo "║          ✨ 報告生成完成！                    ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "📄 報告位置: $REPORT_FILE"
echo "📊 統計資料: 已整合"
echo "🌳 目錄結構: 已整合"
echo "📈 Mermaid 圖表: $OUTPUT_DIR/ARCHITECTURE_DIAGRAMS.md"
echo ""
echo "生成的檔案:"
echo "  • PROJECT_REPORT.txt         - 完整專案報告"
echo "  • ARCHITECTURE_DIAGRAMS.md   - Mermaid 架構圖集"
echo "  • tree_ascii.txt             - 目錄樹狀結構"
echo "  • analysis/                  - 程式碼分析報告"
echo ""
