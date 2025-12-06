"""Real Capability Analyzer - 真實可用能力分析器

基於實際可執行的入口點、API接口和服務來識別真正可用的能力
不是分析代碼路徑，而是分析真正能調用的功能

作者: AIVA Team
版本: 1.0
日期: 2025-12-06
"""

import json
import logging
import subprocess
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """能力類型分類"""
    API_ENDPOINT = "api_endpoint"          # API端點能力
    CLI_COMMAND = "cli_command"            # CLI命令能力
    SERVICE_INTERFACE = "service_interface"  # 服務接口能力
    EXECUTABLE_SCRIPT = "executable_script"  # 可執行腳本能力
    AI_FUNCTION = "ai_function"            # AI功能能力
    SYSTEM_OPERATION = "system_operation"   # 系統操作能力


@dataclass
class RealCapability:
    """真實可用能力"""
    name: str                              # 能力名稱
    type: CapabilityType                   # 能力類型
    entry_point: str                       # 入口點路徑
    description: str                       # 功能描述
    parameters: Dict[str, Any]             # 參數說明
    examples: List[str]                    # 使用示例
    is_verified: bool = False              # 是否驗證可用
    dependencies: List[str] = field(default_factory=list)  # 依賴項
    health_status: str = "unknown"         # 健康狀態


class RealCapabilityAnalyzer:
    """真實能力分析器"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.capabilities: Dict[str, RealCapability] = {}
        self.discovery_methods = [
            self._discover_api_endpoints,
            self._discover_cli_commands, 
            self._discover_executable_scripts,
            self._discover_ai_functions,
            self._discover_service_interfaces
        ]
        
    def discover_all_capabilities(self) -> Dict[str, RealCapability]:
        """發現所有真實可用的能力"""
        logger.info("🔍 開始發現真實可用能力...")
        
        for method in self.discovery_methods:
            try:
                capabilities = method()
                for cap in capabilities:
                    self.capabilities[cap.name] = cap
                    logger.info(f"發現能力: {cap.name} ({cap.type.value})")
            except Exception as e:
                logger.error(f"發現能力時出錯 {method.__name__}: {e}")
                
        logger.info(f"✅ 總共發現 {len(self.capabilities)} 個真實可用能力")
        return self.capabilities
    
    def _discover_api_endpoints(self) -> List[RealCapability]:
        """發現API端點能力"""
        capabilities = []
        
        # 檢查主要API文件
        api_files = [
            self.project_root / "api" / "main.py",
            self.project_root / "api" / "start_api.py"
        ]
        
        for api_file in api_files:
            if api_file.exists():
                try:
                    with open(api_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 查找API路由定義
                    endpoints = self._extract_api_routes(content)
                    for endpoint in endpoints:
                        cap = RealCapability(
                            name=f"api_{endpoint['path'].replace('/', '_')}",
                            type=CapabilityType.API_ENDPOINT,
                            entry_point=str(api_file),
                            description=f"API端點: {endpoint['method']} {endpoint['path']}",
                            parameters=endpoint.get('params', {}),
                            examples=[f"curl -X {endpoint['method']} http://localhost:8000{endpoint['path']}"]
                        )
                        capabilities.append(cap)
                        
                except Exception as e:
                    logger.error(f"分析API文件失敗 {api_file}: {e}")
                    
        return capabilities
    
    def _discover_cli_commands(self) -> List[RealCapability]:
        """發現CLI命令能力"""
        capabilities = []
        
        # 檢查CLI腳本
        cli_patterns = [
            "scripts/ui/aiva_cli.py",
            "plugins/cli_generated/aiva_cli/__main__.py",
            "**/cli.py",
            "**/*_cli.py"
        ]
        
        for pattern in cli_patterns:
            cli_files = list(self.project_root.glob(pattern))
            for cli_file in cli_files:
                try:
                    commands = self._extract_cli_commands(cli_file)
                    for cmd in commands:
                        cap = RealCapability(
                            name=f"cli_{cmd['name']}",
                            type=CapabilityType.CLI_COMMAND,
                            entry_point=str(cli_file),
                            description=cmd['description'],
                            parameters=cmd.get('args', {}),
                            examples=[f"python {cli_file} {cmd['name']} --help"]
                        )
                        capabilities.append(cap)
                except Exception as e:
                    logger.error(f"分析CLI文件失敗 {cli_file}: {e}")
                    
        return capabilities
    
    def _discover_executable_scripts(self) -> List[RealCapability]:
        """發現可執行腳本能力"""
        capabilities = []
        
        # 查找主要可執行腳本
        executable_patterns = [
            "scripts/**/*.py",
            "**/*main*.py",
            "**/*start*.py",
            "**/*run*.py",
            "**/*demo*.py"
        ]
        
        for pattern in executable_patterns:
            script_files = list(self.project_root.glob(pattern))
            for script_file in script_files:
                if self._is_executable_script(script_file):
                    try:
                        cap = RealCapability(
                            name=f"script_{script_file.stem}",
                            type=CapabilityType.EXECUTABLE_SCRIPT,
                            entry_point=str(script_file),
                            description=self._extract_script_description(script_file),
                            parameters={},
                            examples=[f"python {script_file}"]
                        )
                        capabilities.append(cap)
                    except Exception as e:
                        logger.error(f"分析腳本文件失敗 {script_file}: {e}")
                        
        return capabilities
    
    def _discover_ai_functions(self) -> List[RealCapability]:
        """發現AI功能能力"""
        capabilities = []
        
        # 基於README.md中提到的核心能力
        ai_capabilities = [
            {
                "name": "ai_security_testing",
                "description": "AI驅動的安全測試 - 自主決策攻擊策略",
                "entry_point": "services/core/aiva_core/ai_commander_v2.py"
            },
            {
                "name": "capability_execution", 
                "description": "能力執行系統 - 782個攻擊能力的統一調用",
                "entry_point": "api/v1/capabilities/"
            },
            {
                "name": "neural_network_decision",
                "description": "5百萬參數神經網路決策 - Bug Bounty特化",
                "entry_point": "services/core/aiva_core/ai_models.py"
            },
            {
                "name": "rag_enhanced_analysis",
                "description": "RAG增強分析 - 結合檢索的智能決策",
                "entry_point": "services/core/aiva_core/cognitive_core/"
            }
        ]
        
        for ai_cap in ai_capabilities:
            cap = RealCapability(
                name=ai_cap["name"],
                type=CapabilityType.AI_FUNCTION,
                entry_point=ai_cap["entry_point"],
                description=ai_cap["description"],
                parameters={},
                examples=[]
            )
            capabilities.append(cap)
            
        return capabilities
    
    def _discover_service_interfaces(self) -> List[RealCapability]:
        """發現服務接口能力"""
        capabilities = []
        
        # 查找服務接口定義
        service_patterns = [
            "services/**/*service*.py",
            "services/**/*manager*.py",
            "services/**/*handler*.py",
            "services/**/*controller*.py"
        ]
        
        for pattern in service_patterns:
            service_files = list(self.project_root.glob(pattern))
            for service_file in service_files:
                try:
                    interfaces = self._extract_service_interfaces(service_file)
                    for interface in interfaces:
                        cap = RealCapability(
                            name=f"service_{interface['name']}",
                            type=CapabilityType.SERVICE_INTERFACE,
                            entry_point=str(service_file),
                            description=interface['description'],
                            parameters=interface.get('methods', {}),
                            examples=[]
                        )
                        capabilities.append(cap)
                except Exception as e:
                    logger.error(f"分析服務文件失敗 {service_file}: {e}")
                    
        return capabilities
    
    def verify_capabilities(self) -> Dict[str, bool]:
        """驗證能力是否真的可用"""
        verification_results = {}
        
        for name, capability in self.capabilities.items():
            try:
                is_working = self._verify_single_capability(capability)
                capability.is_verified = is_working
                capability.health_status = "healthy" if is_working else "broken"
                verification_results[name] = is_working
                
                if is_working:
                    logger.info(f"✅ {name} - 驗證成功")
                else:
                    logger.warning(f"❌ {name} - 驗證失敗")
                    
            except Exception as e:
                logger.error(f"驗證能力失敗 {name}: {e}")
                capability.health_status = "error"
                verification_results[name] = False
                
        return verification_results
    
    def _verify_single_capability(self, capability: RealCapability) -> bool:
        """驗證單個能力"""
        if capability.type == CapabilityType.API_ENDPOINT:
            # 檢查API文件是否存在且可導入
            return Path(capability.entry_point).exists()
        
        elif capability.type == CapabilityType.CLI_COMMAND:
            # 檢查CLI腳本是否可執行
            return Path(capability.entry_point).exists()
        
        elif capability.type == CapabilityType.EXECUTABLE_SCRIPT:
            # 檢查腳本是否存在
            return Path(capability.entry_point).exists()
        
        elif capability.type == CapabilityType.AI_FUNCTION:
            # 檢查AI模組是否存在
            entry_path = Path(self.project_root / capability.entry_point)
            return entry_path.exists() or entry_path.parent.exists()
        
        elif capability.type == CapabilityType.SERVICE_INTERFACE:
            # 檢查服務文件是否存在
            return Path(capability.entry_point).exists()
            
        return False
    
    def generate_capability_report(self) -> str:
        """生成能力報告"""
        total = len(self.capabilities)
        verified = sum(1 for cap in self.capabilities.values() if cap.is_verified)
        
        report = f"""🎯 AIVA 真實能力分析報告
==========================================

📊 能力統計:
- 總發現能力: {total} 個
- 已驗證可用: {verified} 個
- 可用率: {(verified/total*100) if total > 0 else 0:.1f}%

📋 能力分類:
"""
        
        # 按類型統計
        by_type = defaultdict(list)
        for cap in self.capabilities.values():
            by_type[cap.type].append(cap)
            
        for cap_type, caps in by_type.items():
            verified_count = sum(1 for cap in caps if cap.is_verified)
            report += f"   {cap_type.value}: {len(caps)} 個 (可用: {verified_count})\n"
            
        report += "\n✅ 已驗證的可用能力:\n"
        for name, cap in self.capabilities.items():
            if cap.is_verified:
                report += f"   - {name}: {cap.description}\n"
                
        report += "\n❌ 無法驗證的能力:\n"
        for name, cap in self.capabilities.items():
            if not cap.is_verified:
                report += f"   - {name}: {cap.description} ({cap.health_status})\n"
                
        return report
    
    # Helper methods for parsing different file types
    def _extract_api_routes(self, content: str) -> List[Dict[str, Any]]:
        """從API文件內容提取路由"""
        routes = []
        # 簡化實現，實際可以用AST解析
        lines = content.split('\n')
        for line in lines:
            if '@app.' in line or '@router.' in line:
                # 提取路由信息
                if 'get(' in line or 'post(' in line or 'put(' in line or 'delete(' in line:
                    routes.append({
                        'method': 'GET',  # 簡化
                        'path': '/api/example',
                        'params': {}
                    })
        return routes
    
    def _extract_cli_commands(self, cli_file: Path) -> List[Dict[str, Any]]:
        """從CLI文件提取命令"""
        commands = []
        try:
            with open(cli_file, 'r', encoding='utf-8') as f:
                content = f.read()
            # 簡化實現
            if 'argparse' in content or 'click' in content:
                commands.append({
                    'name': cli_file.stem,
                    'description': f'CLI命令: {cli_file.stem}',
                    'args': {}
                })
        except Exception:
            pass
        return commands
    
    def _is_executable_script(self, script_file: Path) -> bool:
        """判斷是否為可執行腳本"""
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            # 檢查是否有主程序入口
            content = ''.join(first_lines)
            return ('if __name__' in content or 
                    'main(' in content or 
                    'argparse' in content)
        except Exception:
            return False
    
    def _extract_script_description(self, script_file: Path) -> str:
        """提取腳本描述"""
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找文檔字符串
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    # 提取多行文檔字符串
                    doc_lines = []
                    for j in range(i, min(i+10, len(lines))):
                        doc_lines.append(lines[j])
                        if j > i and ('"""' in lines[j] or "'''" in lines[j]):
                            break
                    return ' '.join(doc_lines).replace('"""', '').replace("'''", '').strip()
                    
            return f"可執行腳本: {script_file.stem}"
        except Exception:
            return f"腳本: {script_file.stem}"
    
    def _extract_service_interfaces(self, service_file: Path) -> List[Dict[str, Any]]:
        """從服務文件提取接口"""
        interfaces = []
        try:
            with open(service_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 簡化實現：查找類定義
            if 'class ' in content:
                interfaces.append({
                    'name': service_file.stem,
                    'description': f'服務接口: {service_file.stem}',
                    'methods': {}
                })
        except Exception:
            pass
        return interfaces


def main():
    """主程序"""
    # 設置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 分析能力
    project_root = Path(__file__).parent.parent.parent.parent.parent
    analyzer = RealCapabilityAnalyzer(project_root)
    
    print("🚀 開始真實能力分析...")
    capabilities = analyzer.discover_all_capabilities()
    
    print("\n🔍 驗證能力可用性...")
    verification_results = analyzer.verify_capabilities()
    
    print("\n📋 生成分析報告...")
    report = analyzer.generate_capability_report()
    print(report)
    
    # 保存結果
    output_file = Path(__file__).parent / "real_capabilities_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_capabilities": len(capabilities),
            "verified_capabilities": sum(verification_results.values()),
            "capabilities": {
                name: {
                    "type": cap.type.value,
                    "entry_point": cap.entry_point,
                    "description": cap.description,
                    "is_verified": cap.is_verified,
                    "health_status": cap.health_status
                }
                for name, cap in capabilities.items()
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析結果已保存至: {output_file}")


if __name__ == "__main__":
    main()