#!/usr/bin/env python3
"""
Mermaid 圖表優化器 | Mermaid Diagram Optimizer
================================================

參考專業 Mermaid 語法標準，優化圖表生成
Reference professional Mermaid syntax standards to optimize diagram generation

特性 Features:
1. 符合 Mermaid.js 官方語法規範
2. 支援主題配置 (theme configuration)
3. 優化節點和連線樣式
4. 增強可讀性和美觀度
5. 支援複雜圖表類型

最佳實踐 Best Practices:
- 使用語意化的節點 ID
- 統一的樣式規範
- 清晰的層次結構
- 適當的顏色對比
- 支援響應式佈局
"""

from dataclasses import dataclass


@dataclass
class MermaidTheme:
    """Mermaid 主題配置"""

    primary_color: str = "#1976D2"
    secondary_color: str = "#F57F17"
    tertiary_color: str = "#388E3C"
    background_color: str = "#FFFFFF"
    main_bkg: str = "#ECECFF"
    secondary_bkg: str = "#FFFFDE"
    line_color: str = "#666666"
    border_1: str = "#9370DB"
    border_2: str = "#AAAA33"
    arrow_head_color: str = "#333333"
    text_color: str = "#333333"
    font_family: str = "arial, sans-serif"
    font_size: str = "16px"


@dataclass
class MermaidConfig:
    """Mermaid 配置"""

    theme: str = "default"  # default, forest, dark, neutral
    look: str = "classic"  # classic, handDrawn
    diagram_padding: int = 8
    use_max_width: bool = True
    flow_curve: str = "basis"  # basis, linear, cardinal
    node_spacing: int = 50
    rank_spacing: int = 50


class MermaidOptimizer:
    """Mermaid 圖表優化器"""

    # 節點形狀映射
    NODE_SHAPES = {
        "default": "[{text}]",  # 方形
        "round": "({text})",  # 圓角
        "stadium": "([{text}])",  # 體育場
        "subroutine": "[[{text}]]",  # 子程序
        "cylindrical": "[({text})]",  # 圓柱
        "circle": "(({text}))",  # 圓形
        "asymmetric": ">{text}]",  # 不對稱
        "rhombus": "{{{text}}}",  # 菱形
        "hexagon": "{{{{{text}}}}}",  # 六角形
        "parallelogram": "[/{text}/]",  # 平行四邊形
        "trapezoid": "[\\{text}/]",  # 梯形
        "double_circle": "((({text})))",  # 雙圓
    }

    # 連線類型映射
    LINK_TYPES = {
        "solid": "-->",  # 實線箭頭
        "dotted": "-.->",  # 虛線箭頭
        "thick": "==>",  # 粗箭頭
        "invisible": "~~~",  # 隱藏連線
        "open": "---",  # 開放連線
        "bi": "<-->",  # 雙向箭頭
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

    def generate_header(self, diagram_type: str = "graph TB") -> str:
        """生成圖表頭部配置"""
        config_str = f"""%%{{init: {{'theme':'{self.config.theme}', 'themeVariables': {{
  'primaryColor': '#E3F2FD',
  'primaryTextColor': '#1976D2',
  'primaryBorderColor': '#1976D2',
  'lineColor': '#666666',
  'secondaryColor': '#FFF9C4',
  'tertiaryColor': '#C8E6C9',
  'fontFamily': 'arial, sans-serif',
  'fontSize': '14px'
}}}}}}%%"""
        return f"{config_str}\n{diagram_type}"

    def create_node(
        self,
        node_id: str,
        label: str,
        label_en: str = "",
        tech: str = "",
        shape: str = "default",
        icon: str = "",
    ) -> str:
        """創建優化的節點

        Args:
            node_id: 節點 ID
            label: 中文標籤
            label_en: 英文標籤
            tech: 技術細節
            shape: 節點形狀
            icon: Emoji 圖示
        """
        # 組合標籤
        full_label = f"{icon} {label}" if icon else label
        if label_en:
            full_label += f"<br/>{label_en}"
        if tech:
            full_label += f"<br/><i>{tech}</i>"

        # 應用形狀
        shape_template = self.NODE_SHAPES.get(shape, self.NODE_SHAPES["default"])
        node_text = shape_template.format(text=full_label)

        return f'{node_id}["{full_label}"]'

    def create_link(
        self, from_node: str, to_node: str, label: str = "", link_type: str = "solid"
    ) -> str:
        """創建優化的連線"""
        link_symbol = self.LINK_TYPES.get(link_type, self.LINK_TYPES["solid"])

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
        title: str,
        participants: list[tuple[str, str, str]],  # (id, name, icon)
        interactions: list[tuple[str, str, str, str]],  # (from, to, message, type)
    ) -> str:
        """生成時序圖"""
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
    diagram_type: str,
    title: str,
    content: str,
    optimizer: MermaidOptimizer | None = None,
) -> str:
    """創建專業的 Mermaid 圖表"""
    if optimizer is None:
        optimizer = MermaidOptimizer()

    # 生成完整圖表
    diagram = f"""# {title}

```mermaid
{content}
```

---
**Generated with**: Mermaid Optimizer v1.0
**Standard**: Mermaid.js Official Syntax
"""

    return diagram


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
