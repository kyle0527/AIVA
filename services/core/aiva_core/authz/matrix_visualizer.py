"""
Matrix Visualizer - 權限矩陣視覺化

生成權限矩陣的 HTML 互動視覺化和圖表。
"""

from datetime import datetime
from pathlib import Path

from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import structlog

from .permission_matrix import AccessDecision, PermissionMatrix

logger = structlog.get_logger(__name__)


class MatrixVisualizer:
    """
    權限矩陣視覺化器

    提供多種視覺化方式展示權限矩陣。
    """

    def __init__(self, permission_matrix: PermissionMatrix):
        """
        初始化視覺化器

        Args:
            permission_matrix: 權限矩陣實例
        """
        self.matrix = permission_matrix
        logger.info("matrix_visualizer_initialized")

    def generate_heatmap(self, permission_type: str | None = None) -> go.Figure:
        """
        生成權限熱力圖

        Args:
            permission_type: 指定權限類型，None 表示所有權限

        Returns:
            Plotly 圖表對象
        """
        df = self.matrix.to_dataframe()

        if df.empty:
            logger.warning("cannot_generate_heatmap_empty_matrix")
            return go.Figure()

        if permission_type:
            df = df[df["permission"] == permission_type]

        # 創建樞紐表
        pivot_data = []
        roles = sorted(df["role"].unique())
        resources = sorted(df["resource"].unique())

        for resource in resources:
            row = []
            for role in roles:
                mask = (df["role"] == role) & (df["resource"] == resource)
                if mask.any():
                    decision = df[mask]["decision"].iloc[0]
                    # ALLOW=2, CONDITIONAL=1, DENY=0
                    if decision == AccessDecision.ALLOW:
                        row.append(2)
                    elif decision == AccessDecision.CONDITIONAL:
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            pivot_data.append(row)

        # 創建熱力圖
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data,
            x=roles,
            y=resources,
            colorscale=[
                [0, "rgb(220,220,220)"],      # DENY - 灰色
                [0.5, "rgb(255,200,100)"],    # CONDITIONAL - 橙色
                [1, "rgb(100,200,100)"],      # ALLOW - 綠色
            ],
            text=[[
                f"{df[(df['role']==r) & (df['resource']==res)]['decision'].iloc[0]}"
                if not df[(df['role']==r) & (df['resource']==res)].empty else "N/A"
                for r in roles
            ] for res in resources],
            texttemplate="%{text}",
            hovertemplate="Role: %{x}<br>Resource: %{y}<br>Decision: %{text}<extra></extra>",
        ))

        fig.update_layout(
            title=f"Permission Matrix Heatmap{f' - {permission_type}' if permission_type else ''}",
            xaxis_title="Roles",
            yaxis_title="Resources",
            width=800,
            height=600,
        )

        logger.info("heatmap_generated", permission_type=permission_type)
        return fig

    def generate_coverage_chart(self) -> go.Figure:
        """
        生成權限覆蓋率圖表

        Returns:
            Plotly 圖表對象
        """
        analysis = self.matrix.analyze_coverage()

        # 創建子圖
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Decision Distribution", "Coverage"),
            specs=[[{"type": "pie"}, {"type": "indicator"}]],
        )

        # 決策分布餅圖
        stats = analysis["decision_statistics"]
        fig.add_trace(
            go.Pie(
                labels=list(stats.keys()),
                values=list(stats.values()),
                marker={"colors": ["#66bb6a", "#ef5350", "#ffa726"]},
            ),
            row=1,
            col=1,
        )

        # 覆蓋率儀表板
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=analysis["coverage_percentage"],
                title={"text": "Coverage %"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#42a5f5"},
                    "steps": [
                        {"range": [0, 30], "color": "#ffcdd2"},
                        {"range": [30, 70], "color": "#fff9c4"},
                        {"range": [70, 100], "color": "#c8e6c9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Permission Matrix Coverage Analysis",
            showlegend=True,
            height=400,
        )

        logger.info("coverage_chart_generated")
        return fig

    def generate_role_comparison_chart(self) -> go.Figure:
        """
        生成角色權限比較圖表

        Returns:
            Plotly 圖表對象
        """
        df = self.matrix.to_dataframe()

        if df.empty:
            logger.warning("cannot_generate_comparison_empty_matrix")
            return go.Figure()

        # 計算每個角色的權限數量
        role_stats = {}
        for role in self.matrix.roles:
            perms = self.matrix.get_role_permissions(role)
            role_stats[role] = {
                "total": len(perms),
                "allow": sum(1 for p in perms if p["decision"] == AccessDecision.ALLOW),
                "deny": sum(1 for p in perms if p["decision"] == AccessDecision.DENY),
                "conditional": sum(1 for p in perms if p["decision"] == AccessDecision.CONDITIONAL),
            }

        roles = list(role_stats.keys())

        # 創建堆疊柱狀圖
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Allow",
            x=roles,
            y=[stats["allow"] for stats in role_stats.values()],
            marker_color="#66bb6a",
        ))

        fig.add_trace(go.Bar(
            name="Conditional",
            x=roles,
            y=[stats["conditional"] for stats in role_stats.values()],
            marker_color="#ffa726",
        ))

        fig.add_trace(go.Bar(
            name="Deny",
            x=roles,
            y=[stats["deny"] for stats in role_stats.values()],
            marker_color="#ef5350",
        ))

        fig.update_layout(
            title="Role Permission Comparison",
            xaxis_title="Roles",
            yaxis_title="Permission Count",
            barmode="stack",
            height=500,
        )

        logger.info("role_comparison_chart_generated")
        return fig

    def generate_html_report(self, output_path: str | Path) -> None:
        """
        生成 HTML 報告

        Args:
            output_path: 輸出文件路徑
        """
        output_path = Path(output_path)

        # 生成圖表
        heatmap = self.generate_heatmap()
        coverage = self.generate_coverage_chart()
        comparison = self.generate_role_comparison_chart()

        # 獲取分析數據
        analysis = self.matrix.analyze_coverage()
        over_privileged = self.matrix.find_over_privileged_roles()

        # HTML 模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Permission Matrix Report</title>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1976d2;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 10px;
        }
        h2 {
            color: #424242;
            margin-top: 30px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 5px;
        }
        .chart {
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
        }
        .alert {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .alert-title {
            font-weight: bold;
            color: #856404;
            margin-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            color: #757575;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Permission Matrix Report</h1>
        <p>Generated: {{ timestamp }}</p>

        <h2>📊 Overview Statistics</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ analysis.total_roles }}</div>
                <div class="stat-label">Total Roles</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis.total_resources }}</div>
                <div class="stat-label">Total Resources</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ analysis.total_permissions }}</div>
                <div class="stat-label">Total Permissions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ "%.1f"|format(analysis.coverage_percentage) }}%</div>
                <div class="stat-label">Coverage</div>
            </div>
        </div>

        {% if over_privileged %}
        <div class="alert">
            <div class="alert-title">⚠️ Over-Privileged Roles Detected</div>
            <p>The following roles have significantly more permissions than average:</p>
            <table>
                <thead>
                    <tr>
                        <th>Role</th>
                        <th>Permission Count</th>
                        <th>Excess Permissions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for role in over_privileged %}
                    <tr>
                        <td>{{ role.role }}</td>
                        <td>{{ role.permission_count }}</td>
                        <td>{{ "%.0f"|format(role.excess_permissions) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <h2>📈 Visualizations</h2>

        <div class="chart">
            <h3>Permission Heatmap</h3>
            <div id="heatmap"></div>
        </div>

        <div class="chart">
            <h3>Coverage Analysis</h3>
            <div id="coverage"></div>
        </div>

        <div class="chart">
            <h3>Role Comparison</h3>
            <div id="comparison"></div>
        </div>

        <div class="footer">
            <p>AIVA Permission Matrix Visualizer v1.0.0</p>
            <p>© 2025 AIVA Platform</p>
        </div>
    </div>

    <script>
        Plotly.newPlot('heatmap', {{ heatmap_json }});
        Plotly.newPlot('coverage', {{ coverage_json }});
        Plotly.newPlot('comparison', {{ comparison_json }});
    </script>
</body>
</html>
        """

        template = Template(html_template)
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis=analysis,
            over_privileged=over_privileged,
            heatmap_json=heatmap.to_json(),
            coverage_json=coverage.to_json(),
            comparison_json=comparison.to_json(),
        )

        output_path.write_text(html_content, encoding="utf-8")
        logger.info("html_report_generated", output_path=str(output_path))

    def export_to_csv(self, output_path: str | Path) -> None:
        """
        匯出為 CSV

        Args:
            output_path: 輸出文件路徑
        """
        df = self.matrix.to_dataframe()
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("matrix_exported_to_csv", output_path=str(output_path))


def main():
    """測試範例"""
    from .permission_matrix import PermissionMatrix

    # 創建測試矩陣
    matrix = PermissionMatrix()

    # 添加測試數據
    roles = ["admin", "editor", "viewer", "guest"]
    resources = ["articles", "comments", "users", "settings"]

    for role in roles:
        matrix.add_role(role)

    for resource in resources:
        matrix.add_resource(resource)

    # 授予權限
    matrix.grant_permission("admin", "articles", "read", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "articles", "write", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "articles", "delete", AccessDecision.ALLOW)

    matrix.grant_permission("editor", "articles", "read", AccessDecision.ALLOW)
    matrix.grant_permission("editor", "articles", "write", AccessDecision.ALLOW)
    matrix.grant_permission("editor", "comments", "read", AccessDecision.ALLOW)
    matrix.grant_permission("editor", "comments", "delete", AccessDecision.ALLOW)

    matrix.grant_permission("viewer", "articles", "read", AccessDecision.ALLOW)
    matrix.grant_permission("viewer", "comments", "read", AccessDecision.ALLOW)

    matrix.grant_permission("guest", "articles", "read", AccessDecision.ALLOW)

    # 創建視覺化
    visualizer = MatrixVisualizer(matrix)

    # 生成 HTML 報告
    visualizer.generate_html_report("permission_matrix_report.html")
    print("✅ HTML report generated: permission_matrix_report.html")

    # 匯出 CSV
    visualizer.export_to_csv("permission_matrix.csv")
    print("✅ CSV exported: permission_matrix.csv")


if __name__ == "__main__":
    main()
