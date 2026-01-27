"""
SystemSelfExplorer - 系統自我探索器

內部閉環的核心組件，負責：
1. 掃描系統現有能力 (從 latest_classification.json 的 286 flows)
2. 解析能力映射 (flow → 攻擊類型)
3. 檢查引擎健康狀態
4. 提供能力查詢接口

實現日期: 2026-01-20
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SystemCapability:
    """系統能力定義"""
    capability_type: str  # 能力類型: sqli, xss, port_scan, etc.
    flow_ids: list[int]   # 對應的 flow IDs
    module_path: str      # 模組路徑
    status: str = "available"  # available, unavailable, error
    description: str = ""
    confidence: float = 1.0  # 可信度 0-1


@dataclass
class EngineInfo:
    """引擎信息"""
    engine_name: str
    language: str
    status: str
    capabilities: list[str] = field(default_factory=list)
    flow_ids: list[int] = field(default_factory=list)


class SystemSelfExplorer:
    """系統自我探索器
    
    通過分析 latest_classification.json 來了解系統能力
    
    使用示例:
        explorer = SystemSelfExplorer()
        await explorer.initialize()
        
        # 查詢可用攻擊
        attacks = await explorer.get_available_attacks()
        print(attacks)  # {"sqli": {...}, "xss": {...}}
        
        # 查詢可用引擎
        engines = await explorer.get_available_engines()
    """
    
    # 能力關鍵字映射 (用於從 flow path 判斷能力類型)
    CAPABILITY_KEYWORDS = {
        "sqli": ["sqli", "sql_injection", "sql injection"],
        "xss": ["xss", "cross_site_scripting"],
        "csrf": ["csrf", "cross_site_request"],
        "ssrf": ["ssrf", "server_side_request"],
        "rce": ["rce", "remote_code_execution", "command_injection"],
        "lfi": ["lfi", "local_file_inclusion"],
        "rfi": ["rfi", "remote_file_inclusion"],
        "xxe": ["xxe", "xml_external_entity"],
        "deserial": ["deserial", "deserialization"],
        "auth_bypass": ["auth", "authentication", "bypass"],
        "port_scan": ["port", "scan", "nmap"],
        "web_crawl": ["crawl", "spider", "scrape"],
        "fuzzing": ["fuzz", "fuzzing"],
        "business_logic": ["business", "logic"],
        "unified_attack": ["unified_attack", "attack_coordinator"],
    }
    
    # 引擎關鍵字映射
    ENGINE_KEYWORDS = {
        "rust_scanner": ["rust", "rust_engine"],
        "python_passive": ["python", "passive"],
        "go_engine": ["go", "golang"],
        "typescript_engine": ["typescript", "ts", "node"],
    }
    
    def __init__(self, classification_file: Optional[Path] = None):
        """初始化探索器
        
        Args:
            classification_file: latest_classification.json 路徑，None 則自動查找
        """
        self._initialized = False
        self._flows_data: dict = {}
        self._capabilities: dict[str, SystemCapability] = {}
        self._engines: dict[str, EngineInfo] = {}
        
        # 自動查找 classification 文件
        if classification_file is None:
            # 方案 1: 從當前文件位置查找
            current_dir = Path(__file__).parent
            local_file = current_dir / "latest_classification.json"
            
            # 方案 2: 從 integration data 查找
            integration_file = Path(__file__).parents[4] / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
            
            if local_file.exists():
                classification_file = local_file
            elif integration_file.exists():
                classification_file = integration_file
            else:
                raise FileNotFoundError(
                    f"找不到 latest_classification.json\n"
                    f"嘗試位置:\n"
                    f"  1. {local_file}\n"
                    f"  2. {integration_file}"
                )
        
        self.classification_file = Path(classification_file)
        logger.info(f"SystemSelfExplorer 使用數據文件: {self.classification_file}")
    
    async def initialize(self) -> None:
        """初始化探索器，加載並解析 flows 數據"""
        logger.info("🔍 正在初始化 SystemSelfExplorer...")
        
        # 1. 加載 flows 數據
        await self._load_flows_data()
        logger.info(f"   ✓ 已加載 {len(self._flows_data.get('flows', []))} 個 flows")
        
        # 2. 解析能力映射
        await self._parse_capabilities()
        logger.info(f"   ✓ 已識別 {len(self._capabilities)} 種能力")
        
        # 3. 解析引擎信息
        await self._parse_engines()
        logger.info(f"   ✓ 已識別 {len(self._engines)} 個引擎")
        
        self._initialized = True
        logger.info("✅ SystemSelfExplorer 初始化完成")
    
    async def _load_flows_data(self) -> None:
        """加載 latest_classification.json"""
        try:
            with open(self.classification_file, 'r', encoding='utf-8') as f:
                self._flows_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {self.classification_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析錯誤: {e}")
    
    async def _parse_capabilities(self) -> None:
        """解析 flows 並識別能力類型"""
        flows = self._flows_data.get('flows', [])
        
        for capability_type, keywords in self.CAPABILITY_KEYWORDS.items():
            matching_flows = []
            
            for flow in flows:
                flow_id = flow.get('id')
                flow_path = flow.get('path', [])
                flow_path_str = ' -> '.join(flow_path).lower()
                
                # 檢查關鍵字匹配
                if any(keyword in flow_path_str for keyword in keywords):
                    matching_flows.append({
                        'id': flow_id,
                        'path': flow_path,
                        'module': flow.get('primary_module', 'unknown')
                    })
            
            if matching_flows:
                self._capabilities[capability_type] = SystemCapability(
                    capability_type=capability_type,
                    flow_ids=[f['id'] for f in matching_flows],
                    module_path=matching_flows[0]['module'],
                    description=f"{len(matching_flows)} flows available",
                    confidence=1.0 if len(matching_flows) > 1 else 0.8
                )
    
    async def _parse_engines(self) -> None:
        """解析引擎信息 (基於 flow 的 module)"""
        flows = self._flows_data.get('flows', [])
        
        # 從 flows 中識別不同的引擎/語言
        engine_flows = {
            "python": [],
            "rust": [],
            "go": [],
            "typescript": []
        }
        
        for flow in flows:
            flow_id = flow.get('id')
            module = flow.get('primary_module', '').lower()
            
            if 'rust' in module:
                engine_flows["rust"].append(flow_id)
            elif 'go' in module or 'golang' in module:
                engine_flows["go"].append(flow_id)
            elif 'typescript' in module or 'node' in module:
                engine_flows["typescript"].append(flow_id)
            else:
                # 默認為 Python
                engine_flows["python"].append(flow_id)
        
        # 創建引擎信息
        for engine_name, flow_ids in engine_flows.items():
            if flow_ids:
                self._engines[f"{engine_name}_engine"] = EngineInfo(
                    engine_name=f"{engine_name.capitalize()} Engine",
                    language=engine_name,
                    status="available",
                    flow_ids=flow_ids,
                    capabilities=self._get_capabilities_for_flows(flow_ids)
                )
    
    def _get_capabilities_for_flows(self, flow_ids: list[int]) -> list[str]:
        """根據 flow IDs 獲取對應的能力類型"""
        capabilities = []
        for cap_type, cap_info in self._capabilities.items():
            if any(fid in flow_ids for fid in cap_info.flow_ids):
                capabilities.append(cap_type)
        return capabilities
    
    # ==================== 公共查詢接口 ====================
    
    async def get_available_attacks(self) -> dict[str, SystemCapability]:
        """獲取所有可用的攻擊能力
        
        Returns:
            {
                "sqli": SystemCapability(...),
                "xss": SystemCapability(...),
                ...
            }
        """
        if not self._initialized:
            await self.initialize()
        
        # 過濾出攻擊相關能力
        attack_types = ["sqli", "xss", "csrf", "ssrf", "rce", "lfi", "rfi", 
                       "xxe", "deserial", "auth_bypass", "unified_attack"]
        
        return {
            cap_type: cap_info 
            for cap_type, cap_info in self._capabilities.items()
            if cap_type in attack_types
        }
    
    async def get_available_engines(self) -> dict[str, EngineInfo]:
        """獲取所有可用的引擎"""
        if not self._initialized:
            await self.initialize()
        
        return self._engines.copy()
    
    async def get_capability_by_type(self, capability_type: str) -> Optional[SystemCapability]:
        """根據類型獲取能力信息
        
        Args:
            capability_type: 能力類型，如 "sqli", "xss"
            
        Returns:
            SystemCapability 或 None
        """
        if not self._initialized:
            await self.initialize()
        
        return self._capabilities.get(capability_type.lower())
    
    async def check_capability_available(self, capability_type: str) -> bool:
        """檢查指定能力是否可用"""
        capability = await self.get_capability_by_type(capability_type)
        return capability is not None and capability.status == "available"
    
    async def get_all_capabilities(self) -> dict[str, SystemCapability]:
        """獲取所有能力"""
        if not self._initialized:
            await self.initialize()
        
        return self._capabilities.copy()
    
    async def get_system_summary(self) -> dict:
        """獲取系統能力摘要
        
        Returns:
            {
                "total_flows": 286,
                "total_capabilities": 15,
                "total_engines": 4,
                "attack_capabilities": ["sqli", "xss", ...],
                "scan_capabilities": ["port_scan", "web_crawl"],
                "broken_modules": []
            }
        """
        if not self._initialized:
            await self.initialize()
        
        attacks = await self.get_available_attacks()
        
        return {
            "total_flows": len(self._flows_data.get('flows', [])),
            "total_capabilities": len(self._capabilities),
            "total_engines": len(self._engines),
            "attack_capabilities": list(attacks.keys()),
            "scan_capabilities": [
                cap for cap in self._capabilities.keys() 
                if cap not in attacks
            ],
            "capability_details": {
                cap_type: {
                    "flow_count": len(cap.flow_ids),
                    "status": cap.status,
                    "confidence": cap.confidence
                }
                for cap_type, cap in self._capabilities.items()
            }
        }


# ==================== 便利函數 ====================

async def quick_explore() -> dict:
    """快速探索系統能力 (便利函數)
    
    Returns:
        系統能力摘要
    """
    explorer = SystemSelfExplorer()
    await explorer.initialize()
    return await explorer.get_system_summary()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=" * 60)
        print("SystemSelfExplorer 測試".center(60))
        print("=" * 60)
        
        # 創建探索器
        explorer = SystemSelfExplorer()
        await explorer.initialize()
        
        # 獲取能力摘要
        summary = await explorer.get_system_summary()
        print("\n📊 系統能力摘要:")
        print(f"   總 Flows: {summary['total_flows']}")
        print(f"   總能力數: {summary['total_capabilities']}")
        print(f"   總引擎數: {summary['total_engines']}")
        
        # 顯示攻擊能力
        print("\n🎯 可用攻擊能力:")
        for cap in summary['attack_capabilities']:
            details = summary['capability_details'][cap]
            print(f"   • {cap:20s} - {details['flow_count']} flows (信心度: {details['confidence']:.2f})")
        
        # 顯示引擎
        print("\n⚙️  可用引擎:")
        engines = await explorer.get_available_engines()
        for eng_id, eng_info in engines.items():
            print(f"   • {eng_info.engine_name:20s} - {len(eng_info.flow_ids)} flows")
        
        print("\n" + "=" * 60)
    
    asyncio.run(main())
