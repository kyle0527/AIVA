"""
Attack Path Visualizer - 攻擊路徑視覺化

將攻擊路徑匯出為 Mermaid 圖表或 Cytoscape JSON 格式
"""

import json

from .engine import AttackPath


class AttackPathVisualizer:
    """攻擊路徑視覺化器"""

    @staticmethod
    def to_mermaid(paths: list[AttackPath], title: str = "Attack Paths") -> str:
        """
        轉換為 Mermaid 流程圖

        Args:
            paths: 攻擊路徑列表
            title: 圖表標題

        Returns:
            Mermaid 語法字串
        """
        lines = [
            "```mermaid",
            "graph TD",
            f"    title[{title}]",
            "    style title fill:#f9f,stroke:#333,stroke-width:4px",
            "",
        ]

        node_ids = set()
        edges = []

        for _, path in enumerate(paths):  # path_idx 未使用,改為 _
            for i, node in enumerate(path.nodes):
                node_id = node.get("id", f"node_{i}")
                node_type = list(node.get("labels", ["Unknown"]))[0] if "labels" in node else "Unknown"

                # 生成節點 ID（移除特殊字元）
                safe_id = node_id.replace("-", "_").replace(".", "_")

                if safe_id not in node_ids:
                    # 根據節點類型設定樣式
                    shape = AttackPathVisualizer._get_node_shape(node_type)
                    lines.append(f"    {safe_id}{shape}")
                    node_ids.add(safe_id)

                # 建立邊
                if i > 0:
                    prev_node = path.nodes[i - 1]
                    prev_id = prev_node.get("id", f"node_{i-1}").replace("-", "_").replace(".", "_")

                    if i - 1 < len(path.edges):
                        edge = path.edges[i - 1]
                        edge_type = edge.get("type", "UNKNOWN")
                        edge_label = f"|{edge_type}|"
                    else:
                        edge_label = ""

                    edges.append(f"    {prev_id} -->{edge_label} {safe_id}")

        # 加入所有邊
        lines.extend(edges)

        # 加入樣式
        lines.extend(
            [
                "",
                "    classDef attacker fill:#f44,stroke:#f00,stroke-width:2px,color:#fff",
                "    classDef vulnerability fill:#fa0,stroke:#f80,stroke-width:2px",
                "    classDef database fill:#44f,stroke:#00f,stroke-width:2px,color:#fff",
                "    classDef critical fill:#f00,stroke:#a00,stroke-width:3px,color:#fff",
                "",
                "    class external_attacker attacker",
            ]
        )

        lines.append("```")

        return "\n".join(lines)

    @staticmethod
    def _get_node_shape(node_type: str) -> str:
        """取得 Mermaid 節點形狀"""
        shapes = {
            "Attacker": "[(External Attacker)]",
            "Asset": "[Asset]",
            "Vulnerability": "{{Vulnerability}}",
            "Database": "[(Database)]",
            "APIEndpoint": "[API Endpoint]",
            "InternalNetwork": "[(Internal Network)]",
            "Credential": "{Credential}",
        }
        return shapes.get(node_type, "[Unknown]")

    @staticmethod
    def to_cytoscape_json(paths: list[AttackPath]) -> str:
        """
        轉換為 Cytoscape JSON 格式（用於互動式視覺化）

        Args:
            paths: 攻擊路徑列表

        Returns:
            JSON 字串
        """
        elements = {"nodes": [], "edges": []}
        node_ids = set()

        for path_idx, path in enumerate(paths):
            for i, node in enumerate(path.nodes):
                node_id = node.get("id", f"node_{i}")

                if node_id not in node_ids:
                    node_data = {
                        "id": node_id,
                        "label": node.get("name", node.get("url", node_id)),
                        "type": list(node.get("labels", ["Unknown"]))[0] if "labels" in node else "Unknown",
                        "properties": node,
                    }
                    elements["nodes"].append({"data": node_data})
                    node_ids.add(node_id)

                if i > 0:
                    prev_node = path.nodes[i - 1]
                    prev_id = prev_node.get("id", f"node_{i-1}")

                    edge_data = {
                        "id": f"edge_{path_idx}_{i}",
                        "source": prev_id,
                        "target": node_id,
                        "label": path.edges[i - 1].get("type", "UNKNOWN") if i - 1 < len(path.edges) else "",
                        "risk": path.edges[i - 1].get("risk", 0) if i - 1 < len(path.edges) else 0,
                    }
                    elements["edges"].append({"data": edge_data})

        return json.dumps(elements, indent=2)

    @staticmethod
    def to_html(paths: list[AttackPath], output_file: str = "attack_paths.html") -> None:
        """
        生成 HTML 互動式視覺化頁面

        Args:
            paths: 攻擊路徑列表
            output_file: 輸出檔案路徑
        """
        cytoscape_data = AttackPathVisualizer.to_cytoscape_json(paths)

        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AIVA Attack Path Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        #cy {{
            width: 100%;
            height: 800px;
            border: 2px solid #333;
            background-color: white;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .info {{
            background-color: white;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>[SEARCH] AIVA Attack Path Visualization</h1>
    <div class="info">
        <p><strong>Found {len(paths)} attack paths</strong></p>
        <p>節點: 紅色=攻擊者, 橙色=漏洞, 藍色=資料庫, 綠色=資產</p>
    </div>
    <div id="cy"></div>

    <script>
        const elements = {cytoscape_data};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'background-color': '#4CAF50',
                        'color': '#fff',
                        'text-outline-color': '#333',
                        'text-outline-width': 2,
                        'font-size': '12px',
                        'width': 80,
                        'height': 80
                    }}
                }},
                {{
                    selector: 'node[type="Attacker"]',
                    style: {{
                        'background-color': '#f44336',
                        'shape': 'triangle'
                    }}
                }},
                {{
                    selector: 'node[type="Vulnerability"]',
                    style: {{
                        'background-color': '#FF9800',
                        'shape': 'diamond'
                    }}
                }},
                {{
                    selector: 'node[type="Database"]',
                    style: {{
                        'background-color': '#2196F3',
                        'shape': 'barrel'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'label': 'data(label)',
                        'width': 3,
                        'line-color': '#9E9E9E',
                        'target-arrow-color': '#9E9E9E',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'font-size': '10px'
                    }}
                }}
            ],
            layout: {{
                name: 'breadthfirst',
                directed: true,
                padding: 10,
                spacingFactor: 1.5
            }}
        }});

        // 點擊節點顯示詳細資訊
        cy.on('tap', 'node', function(evt){{
            const node = evt.target;
            const data = node.data();
            alert('Node: ' + data.label + '\\nType: ' + data.type);
        }});
    </script>
</body>
</html>
        """

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_template)

        print(f"HTML visualization saved to {output_file}")


# 使用範例
if __name__ == "__main__":
    from .engine import AttackPathEngine

    engine = AttackPathEngine()
    try:
        engine.initialize_graph()
        paths = engine.find_attack_paths()

        # 生成 Mermaid 圖
        mermaid_code = AttackPathVisualizer.to_mermaid(paths)
        print(mermaid_code)

        # 生成 HTML
        AttackPathVisualizer.to_html(paths)

    finally:
        engine.close()
