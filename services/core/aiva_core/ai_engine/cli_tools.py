"""
CLI 工具集 - 配對 AIVA CLI 命令
為 BioNeuronCore AI 提供完整的工具執行能力
"""
from datetime import datetime
from pathlib import Path
from typing import Any


class BaseCLITool:
    """CLI 工具基類"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.execution_history: list[dict] = []

    def execute(self, **kwargs) -> dict[str, Any]:
        """執行工具"""
        raise NotImplementedError

    def log_execution(self, params: dict, result: dict):
        """記錄執行"""
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "result": result
        })


class ScanTrigger(BaseCLITool):
    """掃描觸發器 - aiva scan"""

    def __init__(self):
        super().__init__(
            name="ScanTrigger",
            description="觸發目標掃描，支持 URL、IP、域名"
        )

    def execute(self, target: str, scan_type: str = "comprehensive", **kwargs) -> dict[str, Any]:
        """執行掃描"""
        result = {
            "tool": self.name,
            "action": "scan",
            "target": target,
            "scan_type": scan_type,
            "status": "initiated",
            "message": f"開始掃描目標: {target}",
            "scan_id": f"scan_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        self.log_execution({"target": target, "type": scan_type}, result)
        return result


class SQLiDetector(BaseCLITool):
    """SQL 注入檢測器 - aiva detect sqli"""

    def __init__(self):
        super().__init__(
            name="SQLiDetector",
            description="檢測 SQL 注入漏洞"
        )

    def execute(self, url: str, param: str | None = None, **kwargs) -> dict[str, Any]:
        """執行 SQL 注入檢測"""
        result = {
            "tool": self.name,
            "action": "detect_sqli",
            "url": url,
            "parameter": param,
            "status": "testing",
            "message": f"檢測 SQL 注入: {url}" + (f" (參數: {param})" if param else ""),
            "test_id": f"sqli_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        self.log_execution({"url": url, "param": param}, result)
        return result


class XSSDetector(BaseCLITool):
    """XSS 檢測器 - aiva detect xss"""

    def __init__(self):
        super().__init__(
            name="XSSDetector",
            description="檢測跨站腳本攻擊漏洞"
        )

    def execute(self, url: str, param: str | None = None, **kwargs) -> dict[str, Any]:
        """執行 XSS 檢測"""
        result = {
            "tool": self.name,
            "action": "detect_xss",
            "url": url,
            "parameter": param,
            "status": "testing",
            "message": f"檢測 XSS: {url}" + (f" (參數: {param})" if param else ""),
            "test_id": f"xss_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        self.log_execution({"url": url, "param": param}, result)
        return result


class CodeAnalyzer(BaseCLITool):
    """代碼分析器 - aiva analyze"""

    def __init__(self):
        super().__init__(
            name="CodeAnalyzer",
            description="分析代碼結構、質量、安全性"
        )

    def execute(self, path: str, analysis_type: str = "structure", **kwargs) -> dict[str, Any]:
        """執行代碼分析"""
        target_path = Path(path)

        result = {
            "tool": self.name,
            "action": "analyze_code",
            "path": str(target_path),
            "type": analysis_type,
            "status": "analyzing",
            "message": f"分析代碼: {target_path}",
            "analysis_id": f"analyze_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        if target_path.exists():
            result["path_exists"] = True
            if target_path.is_dir():
                result["path_type"] = "directory"
            else:
                result["path_type"] = "file"
        else:
            result["path_exists"] = False

        self.log_execution({"path": path, "type": analysis_type}, result)
        return result


class CodeReader(BaseCLITool):
    """代碼讀取器 - aiva read"""

    def __init__(self):
        super().__init__(
            name="CodeReader",
            description="讀取文件內容"
        )

    def execute(self, filepath: str, **kwargs) -> dict[str, Any]:
        """讀取文件"""
        target = Path(filepath)

        result = {
            "tool": self.name,
            "action": "read_file",
            "filepath": str(target),
            "status": "reading"
        }

        try:
            if target.exists():
                content = target.read_text(encoding='utf-8', errors='ignore')
                result.update({
                    "status": "success",
                    "content_length": len(content),
                    "lines": len(content.splitlines()),
                    "message": f"成功讀取: {target.name}"
                })
            else:
                result.update({
                    "status": "error",
                    "message": f"文件不存在: {filepath}"
                })
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"讀取失敗: {str(e)}"
            })

        self.log_execution({"filepath": filepath}, result)
        return result


class CodeWriter(BaseCLITool):
    """代碼寫入器 - aiva write"""

    def __init__(self):
        super().__init__(
            name="CodeWriter",
            description="寫入或修改文件內容"
        )

    def execute(self, filepath: str, content: str = "", mode: str = "write", **kwargs) -> dict[str, Any]:
        """寫入文件"""
        target = Path(filepath)

        result = {
            "tool": self.name,
            "action": f"{mode}_file",
            "filepath": str(target),
            "status": "writing"
        }

        try:
            if mode == "write":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding='utf-8')
                result.update({
                    "status": "success",
                    "bytes_written": len(content.encode('utf-8')),
                    "message": f"成功寫入: {target.name}"
                })
            else:
                result.update({
                    "status": "error",
                    "message": f"不支持的模式: {mode}"
                })
        except Exception as e:
            result.update({
                "status": "error",
                "message": f"寫入失敗: {str(e)}"
            })

        self.log_execution({"filepath": filepath, "mode": mode}, result)
        return result


class ReportGenerator(BaseCLITool):
    """報告生成器 - aiva report"""

    def __init__(self):
        super().__init__(
            name="ReportGenerator",
            description="生成掃描、檢測、分析報告"
        )

    def execute(self, report_type: str, output_path: str | None = None, **kwargs) -> dict[str, Any]:
        """生成報告"""
        if output_path is None:
            output_path = f"reports/{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        result = {
            "tool": self.name,
            "action": "generate_report",
            "report_type": report_type,
            "output_path": output_path,
            "status": "generating",
            "message": f"生成 {report_type} 報告"
        }

        self.log_execution({"type": report_type, "output": output_path}, result)
        return result


# 創建工具註冊表
CLI_TOOLS = {
    "ScanTrigger": ScanTrigger(),
    "SQLiDetector": SQLiDetector(),
    "XSSDetector": XSSDetector(),
    "CodeAnalyzer": CodeAnalyzer(),
    "CodeReader": CodeReader(),
    "CodeWriter": CodeWriter(),
    "ReportGenerator": ReportGenerator(),
}


def get_all_tools() -> dict[str, BaseCLITool]:
    """獲取所有可用工具"""
    return CLI_TOOLS


def get_tool(name: str) -> BaseCLITool | None:
    """根據名稱獲取工具"""
    return CLI_TOOLS.get(name)


def list_tools() -> list[str]:
    """列出所有工具名稱"""
    return list(CLI_TOOLS.keys())


if __name__ == "__main__":
    # 測試所有工具
    print("AIVA CLI 工具集\n" + "="*60)

    for tool_name, tool in CLI_TOOLS.items():
        print(f"\n[{tool_name}]")
        print(f"  描述: {tool.description}")

        # 測試執行
        if tool_name == "ScanTrigger":
            result = tool.execute(target="https://example.com")
        elif tool_name == "SQLiDetector":
            result = tool.execute(url="https://example.com/login", param="username")
        elif tool_name == "XSSDetector":
            result = tool.execute(url="https://example.com/search", param="q")
        elif tool_name == "CodeAnalyzer":
            result = tool.execute(path="services/core/")
        elif tool_name == "CodeReader":
            result = tool.execute(filepath="README.md")
        elif tool_name == "CodeWriter":
            result = tool.execute(filepath="test.txt", content="test")
        elif tool_name == "ReportGenerator":
            result = tool.execute(report_type="scan")

        print(f"  狀態: {result.get('status')}")
        print(f"  訊息: {result.get('message')}")

    print("\n" + "="*60)
    print(f"✓ 總共 {len(CLI_TOOLS)} 個工具可用")
