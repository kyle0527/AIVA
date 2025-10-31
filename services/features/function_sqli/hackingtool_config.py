"""
HackingTool SQL 注入工具配置
整合 HackingTool 的 SQL 注入工具到 AIVA function_sqli 模組中

這個配置檔案定義了如何將 HackingTool 的 SQL 工具
整合到 AIVA 的統一檢測引擎中。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import subprocess
import shutil
from pathlib import Path

from services.aiva_common.enums import ProgrammingLanguage, Severity, Confidence
from services.integration.capability.models import CapabilityRecord, CapabilityType, CapabilityStatus


class SQLToolType(Enum):
    """SQL 工具類型枚舉"""
    AUTOMATED_SCANNER = "automated_scanner"      # 自動化掃描器 (如 sqlmap)
    NOSQL_SCANNER = "nosql_scanner"             # NoSQL 掃描器  
    LIGHTWEIGHT_SCANNER = "lightweight_scanner" # 輕量級掃描器
    BLIND_SCANNER = "blind_scanner"             # 盲注掃描器
    MASS_AUDIT = "mass_audit"                   # 批量審計工具
    CUSTOM_EXPLOIT = "custom_exploit"           # 自定義漏洞利用


@dataclass
class HackingToolSQLConfig:
    """HackingTool SQL 工具配置"""
    
    # 基本資訊
    name: str
    title: str
    description: str
    tool_type: SQLToolType
    project_url: str
    
    # 安裝配置
    install_commands: List[str] = field(default_factory=list)
    run_commands: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # 檢測配置
    supports_get: bool = True
    supports_post: bool = True
    supports_headers: bool = False
    supports_cookies: bool = False
    supports_blind: bool = False
    supports_time_based: bool = False
    
    # 整合配置
    priority: int = 5  # 1(最高) - 10(最低)
    timeout_seconds: int = 300
    max_retries: int = 3
    enable_by_default: bool = True
    
    # 輸出解析配置
    output_format: str = "text"  # text, json, xml
    result_patterns: Dict[str, str] = field(default_factory=dict)
    confidence_mapping: Dict[str, float] = field(default_factory=dict)


# HackingTool SQL 工具配置定義
HACKINGTOOL_SQL_CONFIGS = {
    "sqlmap": HackingToolSQLConfig(
        name="sqlmap",
        title="Sqlmap - 自動化 SQL 注入檢測工具",
        description="自動化 SQL 注入漏洞檢測和利用工具，支援多種數據庫類型和注入技術",
        tool_type=SQLToolType.AUTOMATED_SCANNER,
        project_url="https://github.com/sqlmapproject/sqlmap",
        install_commands=[
            "git clone --depth 1 https://github.com/sqlmapproject/sqlmap.git sqlmap-dev"
        ],
        run_commands=[
            "cd sqlmap-dev && python3 sqlmap.py --batch --banner --url='{target}'"
        ],
        dependencies=["python3", "git"],
        supports_get=True,
        supports_post=True,
        supports_headers=True,
        supports_cookies=True,
        supports_blind=True,
        supports_time_based=True,
        priority=1,  # 最高優先級
        timeout_seconds=600,
        result_patterns={
            "vulnerable": r"Parameter: .* \(.*\) is vulnerable",
            "injectable": r"injectable",
            "database": r"back-end DBMS: (.+)",
            "payload": r"Payload: (.+)"
        },
        confidence_mapping={
            "vulnerable": 0.9,
            "injectable": 0.8,
            "possible": 0.6
        }
    ),
    
    "nosqlmap": HackingToolSQLConfig(
        name="nosqlmap",
        title="NoSQLMap - NoSQL 注入檢測工具",
        description="專門針對 NoSQL 數據庫的注入檢測和利用工具，支援 MongoDB 等",
        tool_type=SQLToolType.NOSQL_SCANNER,
        project_url="https://github.com/codingo/NoSQLMap",
        install_commands=[
            "git clone https://github.com/codingo/NoSQLMap.git",
            "cd NoSQLMap && python setup.py install"
        ],
        run_commands=[
            "cd NoSQLMap && python NoSQLMap.py -u '{target}'"
        ],
        dependencies=["python3", "git", "mongodb-clients"],
        supports_get=True,
        supports_post=True,
        supports_headers=False,
        supports_cookies=False,
        supports_blind=True,
        priority=2,
        timeout_seconds=300,
        result_patterns={
            "vulnerable": r"Vulnerable to .* injection",
            "nosql_injection": r"NoSQL injection detected",
            "payload": r"Payload: (.+)"
        }
    ),
    
    "dsss": HackingToolSQLConfig(
        name="dsss",
        title="DSSS - 輕量級 SQL 注入掃描器",
        description="輕量級 SQL 注入漏洞掃描器，支援 GET 和 POST 參數檢測",
        tool_type=SQLToolType.LIGHTWEIGHT_SCANNER,
        project_url="https://github.com/stamparm/DSSS",
        install_commands=[
            "git clone https://github.com/stamparm/DSSS.git"
        ],
        run_commands=[
            "cd DSSS && python3 dsss.py -u '{target}'"
        ],
        dependencies=["python3", "git"],
        supports_get=True,
        supports_post=True,
        priority=3,
        timeout_seconds=180,
        result_patterns={
            "vulnerable": r"vulnerable.*SQL injection",
            "parameter": r"parameter '(.*)' appears to be vulnerable",
            "dbms": r"DBMS: (.+)"
        }
    ),
    
    "blisqy": HackingToolSQLConfig(
        name="blisqy",
        title="Blisqy - 時間盲注檢測工具",
        description="專門檢測 HTTP 頭部時間盲注 SQL 注入漏洞的工具",
        tool_type=SQLToolType.BLIND_SCANNER,
        project_url="https://github.com/JohnTroony/Blisqy",
        install_commands=[
            "git clone https://github.com/JohnTroony/Blisqy.git"
        ],
        run_commands=[
            "cd Blisqy && python blisqy.py -u '{target}'"
        ],
        dependencies=["python3", "git"],
        supports_get=False,
        supports_post=False,
        supports_headers=True,
        supports_cookies=False,
        supports_blind=True,
        supports_time_based=True,
        priority=4,
        timeout_seconds=300,
        result_patterns={
            "time_based": r"Time-based.*injection",
            "blind_sqli": r"Blind SQL injection detected",
            "vulnerable_header": r"Header '(.*)' vulnerable"
        }
    ),
    
    "leviathan": HackingToolSQLConfig(
        name="leviathan",
        title="Leviathan - 大規模審計工具包",
        description="大規模審計工具包，具備服務發現、暴力破解和 SQL 注入檢測功能",
        tool_type=SQLToolType.MASS_AUDIT,
        project_url="https://github.com/leviathan-framework/leviathan",
        install_commands=[
            "git clone https://github.com/leviathan-framework/leviathan.git",
            "cd leviathan && pip install -r requirements.txt"
        ],
        run_commands=[
            "cd leviathan && python leviathan.py --target '{target}' --sql-injection"
        ],
        dependencies=["python3", "git", "pip"],
        supports_get=True,
        supports_post=True,
        supports_headers=True,
        priority=5,
        timeout_seconds=900,  # 更長超時，因為是批量工具
        enable_by_default=False,  # 預設不啟用，因為需要 API 金鑰
        result_patterns={
            "sql_vuln": r"SQL injection vulnerability found",
            "service_info": r"Service: (.+)",
            "endpoint": r"Endpoint: (.+)"
        }
    ),
    
    "sqlscan": HackingToolSQLConfig(
        name="sqlscan",
        title="SQLScan - 快速 SQL 注入掃描器",
        description="基於 PHP 的快速 Web SQL 注入點掃描器",
        tool_type=SQLToolType.LIGHTWEIGHT_SCANNER,
        project_url="https://github.com/Cvar1984/sqlscan",
        install_commands=[
            "apt install php php-bz2 php-curl php-mbstring curl",
            "curl https://raw.githubusercontent.com/Cvar1984/sqlscan/dev/build/main.phar --output /usr/local/bin/sqlscan",
            "chmod +x /usr/local/bin/sqlscan"
        ],
        run_commands=[
            "sqlscan '{target}'"
        ],
        dependencies=["php", "curl"],
        supports_get=True,
        supports_post=True,
        priority=6,
        timeout_seconds=120,
        result_patterns={
            "injection_point": r"Injection point found",
            "vulnerable_param": r"Parameter '(.*)' vulnerable",
            "error_based": r"Error-based injection"
        }
    )
}


class HackingToolSQLIntegrator:
    """HackingTool SQL 工具整合器"""
    
    def __init__(self):
        self.configs = HACKINGTOOL_SQL_CONFIGS
        self.installed_tools: Dict[str, bool] = {}
    
    def check_tool_availability(self, tool_name: str) -> bool:
        """檢查工具是否可用"""
        if tool_name not in self.configs:
            return False
        
        config = self.configs[tool_name]
        
        # 檢查依賴
        for dep in config.dependencies:
            if not shutil.which(dep):
                return False
        
        # 檢查安裝路徑
        install_path = Path(f"./{tool_name}")
        if not install_path.exists():
            return False
        
        return True
    
    def get_available_tools(self) -> List[str]:
        """獲取可用的工具列表"""
        available = []
        for tool_name in self.configs:
            if self.check_tool_availability(tool_name):
                available.append(tool_name)
        return available
    
    def get_enabled_tools(self) -> List[str]:
        """獲取啟用的工具列表"""
        enabled = []
        for tool_name, config in self.configs.items():
            if config.enable_by_default and self.check_tool_availability(tool_name):
                enabled.append(tool_name)
        return enabled
    
    def get_tools_by_type(self, tool_type: SQLToolType) -> List[str]:
        """根據類型獲取工具"""
        return [
            name for name, config in self.configs.items()
            if config.tool_type == tool_type
        ]
    
    def get_tools_by_capability(self, capability: str) -> List[str]:
        """根據能力獲取工具"""
        capability_map = {
            "blind": "supports_blind",
            "time_based": "supports_time_based", 
            "headers": "supports_headers",
            "cookies": "supports_cookies",
            "get": "supports_get",
            "post": "supports_post"
        }
        
        if capability not in capability_map:
            return []
        
        attr_name = capability_map[capability]
        return [
            name for name, config in self.configs.items()
            if getattr(config, attr_name, False)
        ]
    
    def generate_capability_records(self) -> List[CapabilityRecord]:
        """生成 AIVA CapabilityRecord"""
        records = []
        
        for tool_name, config in self.configs.items():
            # 構建執行入口點
            if config.run_commands:
                entrypoint = config.run_commands[0]
            else:
                entrypoint = f"echo 'No run command for {tool_name}'"
            
            # 創建能力記錄
            record = CapabilityRecord(
                id=f"hackingtool_sql_{tool_name}",
                name=config.title,
                description=config.description,
                type=CapabilityType.SECURITY_SCANNER,
                language=ProgrammingLanguage.PYTHON,
                entrypoint=entrypoint,
                status=CapabilityStatus.ACTIVE if config.enable_by_default else CapabilityStatus.INACTIVE,
                dependencies=config.dependencies,
                metadata={
                    "tool_type": config.tool_type.value,
                    "project_url": config.project_url,
                    "priority": config.priority,
                    "timeout": config.timeout_seconds,
                    "supports": {
                        "get": config.supports_get,
                        "post": config.supports_post,
                        "headers": config.supports_headers,
                        "cookies": config.supports_cookies,
                        "blind": config.supports_blind,
                        "time_based": config.supports_time_based
                    },
                    "patterns": config.result_patterns,
                    "confidence_mapping": config.confidence_mapping
                }
            )
            
            records.append(record)
        
        return records
    
    def install_tool(self, tool_name: str) -> bool:
        """安裝指定工具"""
        if tool_name not in self.configs:
            return False
        
        config = self.configs[tool_name]
        
        try:
            for cmd in config.install_commands:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds
                )
                
                if result.returncode != 0:
                    print(f"安裝命令失敗: {cmd}")
                    print(f"錯誤: {result.stderr}")
                    return False
            
            self.installed_tools[tool_name] = True
            return True
            
        except subprocess.TimeoutExpired:
            print(f"安裝 {tool_name} 超時")
            return False
        except Exception as e:
            print(f"安裝 {tool_name} 時發生錯誤: {e}")
            return False
    
    def run_tool(self, tool_name: str, target: str, **kwargs) -> Dict[str, Any]:
        """執行指定工具"""
        if tool_name not in self.configs:
            return {"success": False, "error": "Tool not found"}
        
        config = self.configs[tool_name]
        
        if not config.run_commands:
            return {"success": False, "error": "No run commands defined"}
        
        try:
            # 格式化命令
            cmd = config.run_commands[0].format(target=target, **kwargs)
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": cmd
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timeout after {config.timeout_seconds}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# 全域整合器實例
sql_integrator = HackingToolSQLIntegrator()