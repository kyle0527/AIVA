#!/usr/bin/env python3
"""
HackingTool Adapter for AIVA Integration
======================================

將 HackingTool 專案的工具格式轉換為 AIVA 的 CapabilityRecord 格式，
實現自動映射機制，支持跨語言工具整合。

功能特色:
- 🔄 自動格式轉換：HackingTool → CapabilityRecord
- 🧠 智能語言識別：根據安裝命令自動檢測程式語言
- 🏷️ 標籤生成：基於工具類型自動生成分類標籤
- 📋 依賴解析：自動解析和管理工具依賴關係
- 🔗 URL 整合：整合專案 URL 到 AIVA 生態系統
"""

import re
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from aiva_common.models import (
    CapabilityRecord,
    CapabilityType,
    CapabilityStatus,
    ProgrammingLanguage,
    InputParameter,
    OutputParameter
)
from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id

logger = get_logger(__name__)


@dataclass
class HackingToolDefinition:
    """HackingTool 原始定義"""
    title: str
    description: str
    install_commands: List[str]
    run_commands: List[str]
    uninstall_commands: Optional[List[str]] = None
    project_url: str = ""
    options: Optional[List[tuple]] = None
    category: str = ""
    
    def __post_init__(self):
        self.install_commands = self.install_commands or []
        self.run_commands = self.run_commands or []
        self.uninstall_commands = self.uninstall_commands or []
        self.options = self.options or []


class AIVAToolAdapter:
    """AIVA 工具適配器 - 將 HackingTool 轉換為 AIVA CapabilityRecord"""
    
    def __init__(self):
        """初始化適配器"""
        self.language_patterns = {
            ProgrammingLanguage.PYTHON: [
                r'python3?\s+', r'pip3?\s+(install|uninstall)',
                r'\.py$', r'requirements\.txt'
            ],
            ProgrammingLanguage.GO: [
                r'go\s+(install|build|run)', r'golang',
                r'~/go/bin/', r'\.go$'
            ],
            ProgrammingLanguage.RUST: [
                r'cargo\s+(build|install)', r'rustc',
                r'\.rs$', r'Cargo\.toml'
            ],
            ProgrammingLanguage.TYPESCRIPT: [
                r'npm\s+(install|run)', r'node\s+', r'yarn\s+',
                r'\.ts$', r'\.js$', r'package\.json'
            ],
            ProgrammingLanguage.RUBY: [
                r'gem\s+install', r'ruby\s+', r'bundle\s+',
                r'\.rb$', r'Gemfile'
            ],
            ProgrammingLanguage.PHP: [
                r'php\s+', r'composer\s+', r'\.php$'
            ]
        }
        
        self.category_mapping = {
            'sql': CapabilityType.VULNERABILITY_SCANNER,
            'xss': CapabilityType.VULNERABILITY_SCANNER,
            'information_gathering': CapabilityType.RECONNAISSANCE,
            'payload_creator': CapabilityType.EXPLOIT_GENERATOR,
            'hash_crack': CapabilityType.POST_EXPLOITATION,
            'forensic': CapabilityType.FORENSICS,
            'ddos': CapabilityType.ATTACK_SIMULATION,
            'wireless': CapabilityType.NETWORK_SCANNER,
            'steganography': CapabilityType.CRYPTOGRAPHY,
            'webattack': CapabilityType.VULNERABILITY_SCANNER,
            'reverse_engineering': CapabilityType.REVERSE_ENGINEERING
        }

    def detect_language(self, tool_def: HackingToolDefinition) -> ProgrammingLanguage:
        """檢測工具的主要程式語言"""
        all_commands = (
            tool_def.install_commands + 
            tool_def.run_commands + 
            tool_def.uninstall_commands
        )
        
        language_scores = dict.fromkeys(self.language_patterns.keys(), 0)
        
        for command in all_commands:
            if not command:
                continue
                
            for language, patterns in self.language_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, command, re.IGNORECASE):
                        language_scores[language] += 1
        
        # 返回分數最高的語言，預設為 Python
        max_score = max(language_scores.values())
        if max_score == 0:
            return ProgrammingLanguage.PYTHON
            
        return max(language_scores, key=language_scores.get)

    def generate_capability_id(self, tool_def: HackingToolDefinition) -> str:
        """生成 AIVA 能力 ID"""
        # 清理標題，生成合法的 ID
        clean_title = re.sub(r'\W', '_', tool_def.title.lower())
        clean_title = re.sub(r'_+', '_', clean_title).strip('_')
        
        category_prefix = tool_def.category or 'security'
        return f"{category_prefix}.{clean_title}"

    def extract_entrypoint(self, tool_def: HackingToolDefinition) -> str:
        """提取工具的執行入口點"""
        if not tool_def.run_commands:
            return ""
        
        # 取第一個執行命令作為主要入口點
        main_command = tool_def.run_commands[0]
        
        # 清理命令，提取核心部分
        if ';' in main_command:
            # 如果是複合命令（如 "cd dir;python script.py"），取最後一部分
            main_command = main_command.split(';')[-1].strip()
        
        return main_command

    def generate_tags(self, tool_def: HackingToolDefinition) -> List[str]:
        """基於工具信息生成標籤"""
        tags = []
        
        # 基於類別生成標籤
        if tool_def.category:
            tags.append(tool_def.category)
        
        # 基於標題和描述生成標籤
        title_lower = tool_def.title.lower()
        desc_lower = tool_def.description.lower() if tool_def.description else ""
        
        keyword_tags = {
            'scanner': ['scan', 'scanner', 'detect'],
            'injection': ['injection', 'sqli', 'sql'],
            'xss': ['xss', 'cross-site'],
            'payload': ['payload', 'exploit', 'shellcode'],
            'recon': ['reconnaissance', 'information', 'gathering'],
            'hash': ['hash', 'crack', 'password'],
            'network': ['network', 'port', 'nmap'],
            'web': ['web', 'http', 'website']
        }
        
        for tag, keywords in keyword_tags.items():
            if any(keyword in title_lower or keyword in desc_lower for keyword in keywords):
                tags.append(tag)
        
        # 去重並返回
        return list(set(tags))

    def parse_dependencies(self, tool_def: HackingToolDefinition) -> List[str]:
        """解析工具依賴"""
        dependencies = []
        
        for command in tool_def.install_commands:
            if not command:
                continue
                
            # 解析不同類型的依賴安裝命令
            if 'apt' in command and 'install' in command:
                # Ubuntu/Debian 包管理
                packages = re.findall(r'install\s+([^;&\n]+)', command)
                if packages:
                    dependencies.extend(packages[0].split())
            
            elif 'pip' in command and 'install' in command:
                # Python 包管理
                if 'requirements.txt' in command:
                    dependencies.append('requirements.txt')
                else:
                    packages = re.findall(r'install\s+([^;&\n]+)', command)
                    if packages:
                        dependencies.extend(packages[0].split())
            
            elif 'git clone' in command:
                # Git 倉庫依賴
                repo_urls = re.findall(r'git clone\s+([^\s&;\n]+)', command)
                dependencies.extend(repo_urls)
        
        return list(set(dependencies))

    def create_input_parameters(self, tool_def: HackingToolDefinition) -> List[InputParameter]:
        """創建輸入參數定義"""
        parameters = []
        
        # 基本目標參數（大多數安全工具都需要）
        parameters.append(InputParameter(
            name="target",
            type="string",
            required=True,
            description="掃描目標（URL、IP地址或域名）",
            default_value=None,
            validation_pattern=r"^https?://.*|^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}.*|^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}.*$"
        ))
        
        # 基於工具類型添加特定參數
        if 'sql' in tool_def.title.lower() or 'sqli' in tool_def.category:
            parameters.append(InputParameter(
                name="parameters",
                type="array",
                required=False,
                description="要測試的URL參數列表",
                default_value=[]
            ))
            
        if 'xss' in tool_def.title.lower() or 'xss' in tool_def.category:
            parameters.append(InputParameter(
                name="payloads",
                type="array", 
                required=False,
                description="自定義XSS測試載荷",
                default_value=[]
            ))
        
        # 通用參數
        parameters.extend([
            InputParameter(
                name="timeout",
                type="integer",
                required=False,
                description="掃描超時時間（秒）",
                default_value=300,
                validation_min=1,
                validation_max=3600
            ),
            InputParameter(
                name="output_format",
                type="string",
                required=False,
                description="輸出格式",
                default_value="json",
                validation_choices=["json", "xml", "txt"]
            )
        ])
        
        return parameters

    def create_output_parameters(self) -> List[OutputParameter]:
        """創建輸出參數定義"""
        return [
            OutputParameter(
                name="findings",
                type="array",
                description="發現的安全問題列表",
                sample_value=[{
                    "vulnerability_type": "SQL Injection",
                    "severity": "HIGH",
                    "confidence": "CONFIRMED",
                    "description": "SQL注入漏洞",
                    "payload": "' OR 1=1 --",
                    "evidence": "Database error revealed"
                }]
            ),
            OutputParameter(
                name="scan_summary",
                type="object",
                description="掃描摘要信息",
                sample_value={
                    "total_tests": 100,
                    "vulnerabilities_found": 3,
                    "scan_duration": 120,
                    "status": "completed"
                }
            ),
            OutputParameter(
                name="raw_output",
                type="string",
                description="工具原始輸出",
                sample_value="Scanning completed successfully..."
            )
        ]

    def convert_to_capability_record(self, tool_def: HackingToolDefinition) -> CapabilityRecord:
        """將 HackingTool 定義轉換為 AIVA CapabilityRecord"""
        
        logger.info(f"正在轉換工具: {tool_def.title}")
        
        # 檢測程式語言
        language = self.detect_language(tool_def)
        
        # 生成 ID 和入口點
        capability_id = self.generate_capability_id(tool_def)
        entrypoint = self.extract_entrypoint(tool_def)
        
        # 確定能力類型
        capability_type = self.category_mapping.get(
            tool_def.category, 
            CapabilityType.SECURITY_TOOL
        )
        
        # 創建 CapabilityRecord
        capability = CapabilityRecord(
            id=capability_id,
            name=tool_def.title,
            description=tool_def.description or f"Security tool: {tool_def.title}",
            module=tool_def.title.lower().replace(' ', '_'),
            language=language,
            capability_type=capability_type,
            entrypoint=entrypoint,
            dependencies=self.parse_dependencies(tool_def),
            inputs=self.create_input_parameters(tool_def),
            outputs=self.create_output_parameters(),
            tags=self.generate_tags(tool_def),
            status=CapabilityStatus.DISCOVERED,
            version="1.0.0",
            author="HackingTool Integration",
            homepage=tool_def.project_url,
            priority=50,  # 中等優先級
            install_commands=tool_def.install_commands,
            uninstall_commands=tool_def.uninstall_commands,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={
                "original_tool": "HackingTool",
                "run_commands": tool_def.run_commands,
                "options": [str(opt) for opt in tool_def.options],
                "category": tool_def.category
            }
        )
        
        logger.info(f"✅ 轉換完成: {capability_id} ({language.value})")
        return capability

    def batch_convert_tools(self, tools: List[HackingToolDefinition]) -> List[CapabilityRecord]:
        """批量轉換工具列表"""
        capabilities = []
        
        logger.info(f"開始批量轉換 {len(tools)} 個工具")
        
        for tool_def in tools:
            try:
                capability = self.convert_to_capability_record(tool_def)
                capabilities.append(capability)
            except Exception as e:
                logger.error(f"轉換工具 {tool_def.title} 失敗: {e}")
                continue
        
        logger.info(f"✅ 批量轉換完成: {len(capabilities)}/{len(tools)} 個工具成功轉換")
        return capabilities


# 預定義的工具映射配置
HACKINGTOOL_CATEGORIES = {
    "sql_tools": "sql",
    "xss_attack": "xss", 
    "information_gathering_tools": "information_gathering",
    "payload_creator": "payload_creator",
    "hash_crack": "hash_crack",
    "forensic_tools": "forensic",
    "ddos": "ddos",
    "wireless_attack_tools": "wireless",
    "steganography": "steganography",
    "webattack": "webattack",
    "reverse_engineering": "reverse_engineering"
}


def create_adapter_factory() -> AIVAToolAdapter:
    """創建工具適配器工廠實例"""
    adapter = AIVAToolAdapter()
    logger.info("AIVA 工具適配器已初始化")
    return adapter


# 使用示例和測試
if __name__ == "__main__":
    # 測試適配器功能
    test_tool = HackingToolDefinition(
        title="SQLMap Test Tool",
        description="Automatic SQL injection detection and exploitation tool",
        install_commands=["sudo git clone https://github.com/sqlmapproject/sqlmap.git"],
        run_commands=["cd sqlmap;python3 sqlmap.py --wizard"],
        project_url="https://github.com/sqlmapproject/sqlmap",
        category="sql"
    )
    
    adapter = create_adapter_factory()
    capability = adapter.convert_to_capability_record(test_tool)
    
    print("✅ 測試轉換成功:")
    print(f"   ID: {capability.id}")
    print(f"   語言: {capability.language.value}")
    print(f"   類型: {capability.capability_type.value}")
    print(f"   標籤: {capability.tags}")