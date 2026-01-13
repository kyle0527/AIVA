"""
HackingTool XSS 工具配置模組
支援多語言 XSS 檢測工具的統一配置管理
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import asyncio


# 通用正則表達式常數
PAYLOAD_PATTERN = r'"payload":\s*".*?"'
PAYLOAD_TEXT_PATTERN = r'Payload:\s*(.+)'
VULNERABILITY_PATTERN = r'"vulnerability":\s*"XSS"'
VULNERABLE_PATTERN = r'"vulnerable":\s*true'
SEVERITY_PATTERN = r'"severity":\s*"(high|medium|low)"'
XSS_TYPE_PATTERN = r'"type":\s*"xss"'
CONFIDENCE_PATTERN = r'"confidence":\s*\d+'


@dataclass
class XSSToolConfig:
    """XSS 工具配置資料結構"""
    name: str
    language: str
    priority: int
    timeout: int
    install_commands: List[str]
    run_pattern: str
    result_patterns: List[str]
    project_url: str
    description: str
    supported_modes: List[str]
    requirements: List[str]
    binary_path: Optional[str] = None
    config_path: Optional[str] = None


class HackingToolXSSConfig:
    """HackingTool XSS 工具整合配置管理器"""
    
    def __init__(self):
        self.tools: Dict[str, XSSToolConfig] = self._initialize_tools()
        self.priority_order = self._calculate_priority_order()
    
    def _initialize_tools(self) -> Dict[str, XSSToolConfig]:
        """初始化所有 XSS 工具配置"""
        return {
            # Go 語言工具 - 最高優先級
            "dalfox": XSSToolConfig(
                name="Dalfox",
                language="go",
                priority=1,
                timeout=300,
                install_commands=[
                    "go install github.com/hahwul/dalfox/v2@latest"
                ],
                run_pattern="dalfox url {target} --format json --output {output_file}",
                result_patterns=[
                    VULNERABILITY_PATTERN,
                    r'"poc":\s*".*?"',
                    SEVERITY_PATTERN
                ],
                project_url="https://github.com/hahwul/dalfox",
                description="Parameter Analysis and XSS Scanning tool based on golang",
                supported_modes=["url", "sxss", "pipe", "file"],
                requirements=["go >= 1.19"],
                binary_path="$GOPATH/bin/dalfox"
            ),
            
            # Ruby 語言工具 - 高優先級
            "xspear": XSSToolConfig(
                name="XSpear",
                language="ruby", 
                priority=2,
                timeout=180,
                install_commands=[
                    "gem install XSpear"
                ],
                run_pattern="XSpear -u {target} --json-report {output_file}",
                result_patterns=[
                    XSS_TYPE_PATTERN,
                    PAYLOAD_PATTERN,
                    CONFIDENCE_PATTERN
                ],
                project_url="https://github.com/hahwul/XSpear",
                description="Powerful XSS Scanner and Parameter Analysis tool",
                supported_modes=["single", "batch", "crawling"],
                requirements=["ruby >= 2.7", "gem"]
            ),
            
            # Python 工具 - 中等優先級
            "xss_payload_generator": XSSToolConfig(
                name="XSSPayloadGenerator",
                language="python",
                priority=3,
                timeout=60,
                install_commands=[
                    "git clone https://github.com/capture0x/XSS-LOADER.git"
                ],
                run_pattern="python3 xss-loader.py -t {target} -o {output_file}",
                result_patterns=[
                    PAYLOAD_TEXT_PATTERN,
                    r'Status:\s*(vulnerable|safe)',
                    r'Response:\s*(.+)'
                ],
                project_url="https://github.com/capture0x/XSS-LOADER",
                description="XSS payload generator and testing tool",
                supported_modes=["generate", "test", "validate"],
                requirements=["python3", "requests", "beautifulsoup4"]
            ),
            
            "xss_finder": XSSToolConfig(
                name="XSSFinder", 
                language="python",
                priority=4,
                timeout=120,
                install_commands=[
                    "git clone https://github.com/Damian89/xssfinder.git"
                ],
                run_pattern="python3 xssfinder.py -u {target} --json {output_file}",
                result_patterns=[
                    VULNERABLE_PATTERN,
                    PAYLOAD_PATTERN,
                    r'"parameter":\s*".*?"'
                ],
                project_url="https://github.com/Damian89/xssfinder",
                description="Tool to find Cross Site Scripting vulnerabilities",
                supported_modes=["single", "list", "crawl"],
                requirements=["python3", "selenium", "requests"]
            ),
            
            "xss_freak": XSSToolConfig(
                name="XSSFreak",
                language="python", 
                priority=5,
                timeout=90,
                install_commands=[
                    "git clone https://github.com/PR0PH3CY33/XSSFreak.git"
                ],
                run_pattern="python3 xssfreak.py -u {target} -o {output_file}",
                result_patterns=[
                    r'XSS Found:\s*(.+)',
                    PAYLOAD_TEXT_PATTERN, 
                    r'Parameter:\s*(.+)'
                ],
                project_url="https://github.com/PR0PH3CY33/XSSFreak",
                description="XSS Scanner with multiple detection methods",
                supported_modes=["fast", "deep", "custom"],
                requirements=["python3", "requests", "lxml"]
            ),
            
            "xss_con": XSSToolConfig(
                name="XSSCon",
                language="python",
                priority=6,
                timeout=150,
                install_commands=[
                    "git clone https://github.com/menkrep1337/XSSCon.git"
                ],
                run_pattern="python3 xsscon.py -u {target}",
                result_patterns=[
                    r'Vulnerable:\s*(.+)',
                    PAYLOAD_TEXT_PATTERN,
                    r'Method:\s*(GET|POST)'
                ],
                project_url="https://github.com/menkrep1337/XSSCon",
                description="XSS Scanner and Payload Generator",
                supported_modes=["scan", "generate", "test"],
                requirements=["python3", "requests"]
            ),
            
            # Rust 工具 - 實驗性
            "rvuln": XSSToolConfig(
                name="RVuln",
                language="rust",
                priority=7,
                timeout=240,
                install_commands=[
                    "git clone https://github.com/iinc0gnit0/RVuln.git",
                    "cd RVuln && cargo build --release"
                ],
                run_pattern="./target/release/RVuln -u {target} --output json",
                result_patterns=[
                    r'"vulnerability_type":\s*"xss"',
                    SEVERITY_PATTERN,
                    PAYLOAD_PATTERN
                ],
                project_url="https://github.com/iinc0gnit0/RVuln",
                description="Multi-threaded Web Vulnerability Scanner in Rust",
                supported_modes=["fast", "comprehensive", "stealth"],
                requirements=["rust", "cargo", "libssl-dev"]
            ),
            
            # Advanced Python 工具
            "xss_strike": XSSToolConfig(
                name="XSStrike",
                language="python",
                priority=8,
                timeout=200,
                install_commands=[
                    "git clone https://github.com/UltimateHackers/XSStrike.git",
                    "cd XSStrike && pip install -r requirements.txt"
                ],
                run_pattern="python3 xsstrike.py -u {target} --json --output {output_file}",
                result_patterns=[
                    r'"vulnerability":\s*true',
                    PAYLOAD_PATTERN,
                    r'"context":\s*".*?"'
                ],
                project_url="https://github.com/UltimateHackers/XSStrike",
                description="Advanced XSS Detection Suite with ML capabilities",
                supported_modes=["crawl", "single", "blind"],
                requirements=["python3", "requests", "fuzzywuzzy"]
            )
        }
    
    def _calculate_priority_order(self) -> List[str]:
        """計算工具執行優先順序"""
        sorted_tools = sorted(
            self.tools.items(),
            key=lambda x: x[1].priority
        )
        return [tool_name for tool_name, _ in sorted_tools]
    
    def get_tool_config(self, tool_name: str) -> Optional[XSSToolConfig]:
        """獲取特定工具配置"""
        return self.tools.get(tool_name)
    
    def get_tools_by_language(self, language: str) -> List[XSSToolConfig]:
        """按語言獲取工具列表"""
        return [
            tool for tool in self.tools.values()
            if tool.language.lower() == language.lower()
        ]
    
    def get_high_priority_tools(self, limit: int = 3) -> List[XSSToolConfig]:
        """獲取高優先級工具"""
        return [
            self.tools[tool_name] for tool_name in self.priority_order[:limit]
        ]
    
    def get_tools_by_mode(self, mode: str) -> List[XSSToolConfig]:
        """按支援模式獲取工具"""
        return [
            tool for tool in self.tools.values()
            if mode in tool.supported_modes
        ]
    
    def validate_tool_requirements(self, tool_name: str) -> Dict[str, Any]:
        """驗證工具依賴需求"""
        tool = self.get_tool_config(tool_name)
        if not tool:
            return {"valid": False, "error": "Tool not found"}
        
        validation_result = {
            "valid": True,
            "tool_name": tool_name,
            "language": tool.language,
            "requirements": tool.requirements,
            "install_commands": tool.install_commands,
            "checks": []
        }
        
        # 這裡可以添加實際的依賴檢查邏輯
        # 例如檢查 Go, Ruby, Python, Rust 環境
        
        return validation_result
    
    def export_config(self, file_path: str) -> bool:
        """匯出配置到 JSON 檔案"""
        try:
            config_data = {}
            for name, tool in self.tools.items():
                config_data[name] = {
                    "name": tool.name,
                    "language": tool.language,
                    "priority": tool.priority,
                    "timeout": tool.timeout,
                    "install_commands": tool.install_commands,
                    "run_pattern": tool.run_pattern,
                    "result_patterns": tool.result_patterns,
                    "project_url": tool.project_url,
                    "description": tool.description,
                    "supported_modes": tool.supported_modes,
                    "requirements": tool.requirements,
                    "binary_path": tool.binary_path,
                    "config_path": tool.config_path
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export config: {e}")
            return False
    
    def get_execution_plan(self, target_url: str, mode: str = "comprehensive") -> List[Dict[str, Any]]:
        """生成執行計劃"""
        if mode == "fast":
            tools = self.get_high_priority_tools(2)
        elif mode == "comprehensive":
            tools = [self.tools[name] for name in self.priority_order]
        else:
            tools = self.get_tools_by_mode(mode)
        
        execution_plan = []
        for tool in tools:
            plan_item = {
                "tool_name": tool.name,
                "language": tool.language,
                "priority": tool.priority,
                "timeout": tool.timeout,
                "command_template": tool.run_pattern,
                "target": target_url,
                "expected_patterns": tool.result_patterns
            }
            execution_plan.append(plan_item)
        
        return execution_plan


# 全域配置實例
xss_config = HackingToolXSSConfig()


def get_xss_tools_config() -> HackingToolXSSConfig:
    """獲取 XSS 工具配置管理器實例"""
    return xss_config


# 配置常數
XSS_SCAN_MODES = {
    "FAST": "fast",           # 快速掃描，使用高優先級工具
    "COMPREHENSIVE": "comprehensive",  # 全面掃描，使用所有工具
    "SINGLE_URL": "single",   # 單一 URL 掃描
    "CRAWLING": "crawling",   # 爬蟲模式
    "DEEP": "deep",          # 深度掃描
    "STEALTH": "stealth"     # 隱蔽掃描
}

XSS_OUTPUT_FORMATS = {
    "JSON": "json",
    "XML": "xml", 
    "TEXT": "text",
    "HTML": "html"
}

# 工具語言支援
SUPPORTED_LANGUAGES = {
    "GO": "go",
    "RUBY": "ruby", 
    "PYTHON": "python",
    "RUST": "rust"
}