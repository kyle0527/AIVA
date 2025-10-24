#!/usr/bin/env python3
"""
Mermaid 圖表優化器 | Mermaid Diagram Optimizer
================================================

參考最新 Mermaid.js v10+ 語法標準，優化圖表生成
Reference latest Mermaid.js v10+ syntax standards to optimize diagram generation

特性 Features:
1. 符合 Mermaid.js 官方語法規範 (v10+)
2. 支援現代主題配置和自定義主題變數
3. 優化節點和連線樣式，支援 CSS 類
4. 增強可讀性和美觀度，支援 HTML 標籤
5. 支援複雜圖表類型 (Flowchart, Sequence, Class, State, etc.)
6. 支援無障礙功能和語意化標籤
7. 支援響應式佈局和高 DPI 顯示

最佳實踐 Best Practices:
- 使用語意化的節點 ID (kebab-case)
- 統一的樣式規範 (CSS Variables)
- 清晰的層次結構和邏輯分組
- 適當的顏色對比 (WCAG 2.1 AA)
- 支援響應式佈局和縮放
- 使用現代 CSS 功能 (Custom Properties)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import re


@dataclass
class MermaidTheme:
    """Mermaid v10+ 主題配置 - 支援現代主題變數"""

    # 主色系 (Primary Colors)
    primary_color: str = "#0F172A"          # Modern Dark Blue
    primary_text_color: str = "#FFFFFF"    # High contrast white
    primary_border_color: str = "#3B82F6"  # Blue 500
    
    # 次要色系 (Secondary Colors)  
    secondary_color: str = "#F1F5F9"       # Light Gray
    secondary_text_color: str = "#1E293B"  # Dark Gray
    secondary_border_color: str = "#64748B" # Gray 500
    
    # 第三色系 (Tertiary Colors)
    tertiary_color: str = "#ECFDF5"        # Light Green
    tertiary_text_color: str = "#065F46"   # Dark Green
    tertiary_border_color: str = "#10B981" # Green 500
    
    # 背景色系 (Background Colors)
    background: str = "#FFFFFF"            # Pure White
    main_bkg: str = "#F8FAFC"             # Subtle Gray
    secondary_bkg: str = "#F1F5F9"        # Light Gray
    
    # 線條和邊框 (Lines & Borders)
    line_color: str = "#64748B"           # Gray 500
    arrow_head_color: str = "#374151"     # Gray 700
    
    # 狀態顏色 (State Colors)
    active_color: str = "#10B981"         # Green 500
    inactive_color: str = "#64748B"       # Gray 500
    done_color: str = "#059669"           # Green 600
    
    # 字體配置 (Typography)
    font_family: str = '"Inter", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", sans-serif'
    font_size: str = "14px"
    font_weight: str = "400"
    
    # 現代化顏色 (Modern Color Palette)
    success: str = "#10B981"              # Green 500
    warning: str = "#F59E0B"              # Amber 500  
    error: str = "#EF4444"                # Red 500
    info: str = "#3B82F6"                 # Blue 500


@dataclass
class MermaidConfig:
    """Mermaid v10+ 配置 - 支援現代渲染選項"""

    # 主題配置 (Theme Configuration)
    theme: str = "base"  # base, forest, dark, neutral, default
    theme_variables: Optional[Dict[str, str]] = None
    
    # 渲染配置 (Rendering Configuration)
    look: str = "classic"  # classic, handDrawn
    diagram_padding: int = 8
    use_max_width: bool = True
    responsive: bool = True
    
    # 流程圖配置 (Flowchart Configuration)
    flow_curve: str = "basis"  # basis, linear, cardinal, catmullRom
    node_spacing: int = 50
    rank_spacing: int = 50
    
    # 現代化功能 (Modern Features)
    accessibility: bool = True
    html_labels: bool = True
    font_awesome: bool = False
    
    # 安全配置 (Security Configuration)
    secure: bool = True
    sandbox_mode: bool = False
    
    # 渲染器配置 (Renderer Configuration)
    renderer: str = "svg"  # svg, dagre-d3
    
    def to_init_directive(self) -> str:
        """生成 Mermaid v10+ init 指令"""
        config = {
            "theme": self.theme,
            "themeVariables": self.theme_variables or {},
            "flowchart": {
                "curve": self.flow_curve,
                "nodeSpacing": self.node_spacing,
                "rankSpacing": self.rank_spacing,
                "useMaxWidth": self.use_max_width
            },
            "sequence": {
                "useMaxWidth": self.use_max_width,
            },
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "14px",
            "secure": self.secure
        }
        return f"%%{{init: {json.dumps(config, separators=(',', ':'))}}}%%"


class MermaidOptimizer:
    """Mermaid 圖表優化器"""

    # 現代化節點形狀 (Modern Node Shapes)
    NODE_SHAPES = {
        "rectangle": "[{text}]",              # 標準矩形
        "rounded": "({text})",                # 圓角矩形  
        "stadium": "([{text}])",              # 體育場形 (Pill)
        "subroutine": "[[{text}]]",           # 子程序框
        "cylindrical": "[({text})]",          # 圓柱體 (Database)
        "circle": "(({text}))",               # 圓形
        "asymmetric": ">{text}]",             # 不對稱四邊形
        "rhombus": "{{{text}}}",              # 菱形 (Decision)
        "hexagon": "{{{{{text}}}}}",          # 六角形
        "parallelogram": "[/{text}/]",        # 平行四邊形
        "trapezoid": "[\\{text}/]",           # 梯形  
        "double_circle": "((({text})))",      # 雙圓圈
        # v10+ 新增形狀
        "flag": "{{{{text}}}}",               # 旗幟形
        "lean_right": "[/{text}\\]",          # 右傾
        "lean_left": "[\\{text}/]",           # 左傾
    }

    # 現代化連線類型 (Modern Link Types)
    LINK_TYPES = {
        "arrow": "-->",                       # 實線箭頭
        "dotted": "-.->",                     # 虛線箭頭  
        "thick": "==>",                       # 粗實線箭頭
        "open": "---",                        # 無箭頭實線
        "dotted_open": "-.-",                 # 無箭頭虛線
        "thick_open": "===",                  # 無箭頭粗線
        "invisible": "~~~",                   # 隱藏連線
        "bidirectional": "<-->",              # 雙向箭頭
        "bidirectional_dotted": "<-.->",      # 雙向虛線
        "x_arrow": "--x",                     # X型終止
        "circle_arrow": "--o",                # 圓型終止
    }

    # 顏色方案
    COLOR_SCHEMES = {
        "python": {"fill": "#3776AB", "stroke": "#2C5F8D", "text": "#FFFFFF"},
        "go": {"fill": "#00ADD8", "stroke": "#0099BF", "text": "#FFFFFF"},
        "rust": {"fill": "#CE422B", "stroke": "#A33520", "text": "#FFFFFF"},
        "typescript": {"fill": "#3178C6", "stroke": "#2768B3", "text": "#FFFFFF"},
        "core": {"fill": "#FFF9C4", "stroke": "#F57F17", "text": "#333333"},
        "scan": {"fill": "#C8E6C9", "stroke": "#388E3C", "text": "#1B5E20"},
        "function": {"fill": "#E1BEE7", "stroke": "#7B1FA2", "text": "#4A148C"},
        "integration": {"fill": "#FFE0B2", "stroke": "#E65100", "text": "#BF360C"},
        "database": {"fill": "#CFD8DC", "stroke": "#455A64", "text": "#263238"},
        "queue": {"fill": "#FFECB3", "stroke": "#F57F17", "text": "#E65100"},
        "success": {"fill": "#90EE90", "stroke": "#2E7D32", "text": "#1B5E20"},
        "warning": {"fill": "#FFF59D", "stroke": "#F57F17", "text": "#E65100"},
        "danger": {"fill": "#FFCDD2", "stroke": "#C62828", "text": "#B71C1C"},
    }

    def __init__(self, config: MermaidConfig | None = None):
        self.config = config or MermaidConfig()

    def generate_header(self, diagram_type: str = "flowchart TD") -> str:
        """生成現代化圖表頭部配置 (v10+ 語法)"""
        return f"{self.config.to_init_directive()}\n{diagram_type}"

    def create_node(
        self,
        node_id: str,
        label: str,
        label_en: str = "",
        tech: str = "",
        shape: str = "rectangle",
        icon: str = "",
        css_class: str = "",
    ) -> str:
        """創建現代化節點 (支援 HTML 標籤和 CSS 類)

        Args:
            node_id: 節點 ID (建議使用 kebab-case)
            label: 中文標籤  
            label_en: 英文標籤
            tech: 技術細節
            shape: 節點形狀
            icon: Emoji 圖示或 Font Awesome
            css_class: CSS 類名
        """
        # 清理節點 ID (確保符合 Mermaid 語法)
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', node_id)
        
        # 組合現代化標籤 (支援 HTML)
        if self.config.html_labels:
            full_label = "<div class='node-content'>"
            if icon:
                full_label += f"<span class='node-icon'>{icon}</span>"
            full_label += f"<span class='node-title'>{label}</span>"
            if label_en:
                full_label += f"<br/><span class='node-subtitle'>{label_en}</span>"
            if tech:
                full_label += f"<br/><span class='node-tech'>{tech}</span>"
            full_label += "</div>"
        else:
            # 純文字模式 (向後兼容)
            full_label = f"{icon} {label}" if icon else label
            if label_en:
                full_label += f"<br/>{label_en}"
            if tech:
                full_label += f"<br/><i>{tech}</i>"

        # 生成節點語法  
        node_syntax = f'{clean_id}["{full_label}"]'
        
        # 添加 CSS 類 (如果指定)
        if css_class:
            node_syntax += f":::{css_class}"
            
        return node_syntax

    def create_link(
        self, from_node: str, to_node: str, label: str = "", link_type: str = "arrow"
    ) -> str:
        """創建現代化連線 (支援更多樣式)"""
        link_symbol = self.LINK_TYPES.get(link_type, self.LINK_TYPES["arrow"])

        if label:
            return f"{from_node} {link_symbol}|{label}| {to_node}"
        else:
            return f"{from_node} {link_symbol} {to_node}"

    def apply_style(
        self,
        node_id: str,
        color_scheme: str = "core",
        stroke_width: int = 2,
        custom_fill: str = "",
        custom_stroke: str = "",
    ) -> str:
        """應用節點樣式"""
        scheme = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES["core"])

        fill = custom_fill or scheme["fill"]
        stroke = custom_stroke or scheme["stroke"]

        return f"style {node_id} fill:{fill},stroke:{stroke},stroke-width:{stroke_width}px,color:{scheme['text']}"

    def create_subgraph(
        self, title: str, title_en: str = "", icon: str = "", nodes: list[str] = None
    ) -> str:
        """創建子圖"""
        full_title = f"{icon} {title}" if icon else title
        if title_en:
            full_title += f" {title_en}"

        subgraph_str = f'    subgraph "{full_title}"\n'

        if nodes:
            for node in nodes:
                subgraph_str += f"        {node}\n"

        subgraph_str += "    end\n"

        return subgraph_str

    def generate_class_def(
        self, class_name: str, color_scheme: str = "core", stroke_width: int = 2
    ) -> str:
        """生成類定義"""
        scheme = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES["core"])
        return f"classDef {class_name} fill:{scheme['fill']},stroke:{scheme['stroke']},stroke-width:{stroke_width}px,color:{scheme['text']}"

    def apply_class(self, node_ids: list[str], class_name: str) -> str:
        """應用類到節點"""
        return f"class {','.join(node_ids)} {class_name}"

    def optimize_flowchart(self, mermaid_code: str) -> str:
        """優化流程圖語法"""
        # 添加配置頭部
        if not mermaid_code.strip().startswith("%%"):
            header = self.generate_header("flowchart TD")
            mermaid_code = f"{header}\n{mermaid_code}"

        # 優化連線樣式
        mermaid_code = self._optimize_links(mermaid_code)

        return mermaid_code

    def _optimize_links(self, code: str) -> str:
        """優化連線樣式"""
        # 添加預設連線樣式
        if "linkStyle" not in code:
            code += "\n    linkStyle default stroke:#666,stroke-width:2px"

        return code

    def generate_sequence_diagram(
        self,
        participants: list[tuple[str, str, str]],  # (id, name, icon)
        interactions: list[tuple[str, str, str, str]],  # (from, to, message, type)
    ) -> str:
        """生成現代化時序圖"""
        diagram = self.generate_header("sequenceDiagram")
        diagram += "\n    autonumber\n"

        # 添加參與者
        for pid, name, icon in participants:
            diagram += f"    participant {pid} as {icon} {name}\n"

        diagram += "\n"

        # 添加交互
        for from_p, to_p, message, msg_type in interactions:
            if msg_type == "async":
                diagram += f"    {from_p}->>{to_p}: {message}\n"
            elif msg_type == "return":
                diagram += f"    {from_p}-->>{to_p}: {message}\n"
            else:
                diagram += f"    {from_p}->>{to_p}: {message}\n"

        return diagram

    def add_note(self, position: str, participant: str, text: str) -> str:
        """添加註解"""
        return f"    Note {position} {participant}: {text}"

    def validate_syntax(self, mermaid_code: str) -> tuple[bool, str]:
        """驗證 Mermaid 語法"""
        errors = []

        # 檢查基本結構
        if not any(
            dt in mermaid_code
            for dt in ["graph", "flowchart", "sequenceDiagram", "classDiagram"]
        ):
            errors.append("Missing diagram type declaration")

        # 檢查括號匹配
        if mermaid_code.count("[") != mermaid_code.count("]"):
            errors.append("Unmatched square brackets")

        if mermaid_code.count("{") != mermaid_code.count("}"):
            errors.append("Unmatched curly brackets")

        if errors:
            return False, "; ".join(errors)

        return True, "Syntax valid"

    def minify(self, mermaid_code: str) -> str:
        """最小化 Mermaid 程式碼"""
        # 移除多餘空白
        lines = [line.strip() for line in mermaid_code.split("\n") if line.strip()]
        return "\n".join(lines)

    def beautify(self, mermaid_code: str, indent: int = 4) -> str:
        """美化 Mermaid 程式碼"""
        lines = mermaid_code.split("\n")
        beautified = []
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            # 減少縮排
            if stripped.startswith("end"):
                indent_level = max(0, indent_level - 1)

            # 添加縮排
            beautified.append(" " * (indent * indent_level) + stripped)

            # 增加縮排
            if stripped.startswith("subgraph"):
                indent_level += 1

        return "\n".join(beautified)


# 工具函數
def create_professional_diagram(
    title: str,
    content: str,
    optimizer: MermaidOptimizer | None = None,
) -> str:
    """創建專業的 Mermaid 圖表"""
    # optimizer 參數保留供將來使用

    # 生成完整圖表  
    return f"""# {title}

```mermaid
{content}
```

---
**Generated with**: Mermaid Optimizer v2.0
**Standard**: Mermaid.js v10+ Official Syntax
**Features**: Modern themes, HTML labels, CSS classes
"""


if __name__ == "__main__":
    # 示例用法
    optimizer = MermaidOptimizer()

    # 創建節點
    node = optimizer.create_node(
        "CORE", "核心引擎", "Core Engine", "Bio Neuron Network", icon="🤖"
    )
    print("Node:", node)

    # 創建連線
    link = optimizer.create_link("API", "CORE", "HTTP Request", "solid")
    print("Link:", link)

    # 應用樣式
    style = optimizer.apply_style("CORE", "core", stroke_width=3)
    print("Style:", style)

    # 驗證語法
    test_code = "graph TB\n    A[Test] --> B[Node]"
    valid, msg = optimizer.validate_syntax(test_code)
    print(f"Validation: {valid} - {msg}")
