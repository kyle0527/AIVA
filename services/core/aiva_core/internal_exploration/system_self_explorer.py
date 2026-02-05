"""
SystemSelfExplorer - 系統自我探索器

負責探索並暴露系統內部能力信息，包括：
1. 可用攻擊能力及其 flow IDs
2. 可用引擎 (Python, Rust, Go)
3. 系統模組和組件信息

通過讀取 external_classification.json 來獲取系統能力清單。

實現日期: 2026-02-02
符合規範: aiva_common README.md (Pydantic v2, 類型提示, 開箱即用)
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
from dataclasses import dataclass, field

from aiva_common.utils import get_logger

logger = get_logger(__name__)

# 常量定義
_NOT_INITIALIZED_ERROR = "SystemSelfExplorer 未初始化，請先調用 initialize()"


@dataclass
class AttackCapability:
    """攻擊能力信息
    
    Attributes:
        capability_type: 能力類型 (如 xss, sqli)
        module_name: 所屬模組名稱
        flow_ids: 相關的 flow IDs
        language: 實現語言 (Python, Go, Rust)
        flow_count: flow 數量
        description: 能力描述
    """
    capability_type: str
    module_name: str
    flow_ids: List[int] = field(default_factory=list)
    language: str = "Python"
    flow_count: int = 0
    description: str = ""


@dataclass
class EngineInfo:
    """引擎信息
    
    Attributes:
        engine_name: 引擎名稱
        language: 實現語言
        module_count: 相關模組數量
        capabilities: 支援的能力列表
        description: 引擎描述
    """
    engine_name: str
    language: str
    module_count: int = 0
    capabilities: List[str] = field(default_factory=list)
    description: str = ""


class SystemSelfExplorer:
    """系統自我探索器
    
    掃描並解析系統內部的攻擊能力、引擎和模組信息。
    數據來源於 external_classification.json。
    
    使用示例:
        explorer = SystemSelfExplorer()
        await explorer.initialize()
        
        attacks = await explorer.get_available_attacks()
        print(attacks.keys())  # ["xss", "sqli", "ssrf", ...]
        
        engines = await explorer.get_available_engines()
        print(engines.keys())  # ["python_engine", "rust_engine", ...]
    """
    
    def __init__(self, classification_path: Optional[Path] = None):
        """初始化系統探索器
        
        Args:
            classification_path: external_classification.json 路徑
                                如果為 None，使用默認位置
        """
        self._initialized = False
        self._classification_path = classification_path
        self._classification_data: Dict = {}
        self._attacks_cache: Optional[Dict[str, AttackCapability]] = None
        self._engines_cache: Optional[Dict[str, EngineInfo]] = None
    
    def initialize(self) -> None:
        """初始化探索器，加載分類數據"""
        logger.info("🔍 正在初始化 SystemSelfExplorer...")
        
        # 確定分類文件路徑
        if self._classification_path is None:
            # 默認路徑：向上 4 層到 AIVA-git，再到 services/integration/data/internal_exploration
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent
            self._classification_path = (
                project_root / "services" / "integration" / "data" / 
                "internal_exploration" / "external_classification.json"
            )
        
        # 驗證文件存在
        if not self._classification_path.exists():
            error_msg = f"找不到分類文件: {self._classification_path}"
            logger.error(f"   ✗ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # 加載分類數據
        try:
            with open(self._classification_path, "r", encoding="utf-8") as f:
                self._classification_data = json.load(f)
            
            logger.info(f"   ✓ 分類文件已加載: {self._classification_path.name}")
            
            # 記錄統計信息
            metadata = self._classification_data.get("metadata", {})
            total_flows = metadata.get("total_flows", 0)
            total_modules = metadata.get("total_modules", 0)
            languages = metadata.get("languages", [])
            
            logger.info(f"   ✓ 總 Flows: {total_flows}")
            logger.info(f"   ✓ 總模組: {total_modules}")
            logger.info(f"   ✓ 支援語言: {', '.join(languages)}")
            
            self._initialized = True
            logger.info("✅ SystemSelfExplorer 初始化完成")
            
        except json.JSONDecodeError as e:
            error_msg = f"分類文件 JSON 格式錯誤: {e}"
            logger.error(f"   ✗ {error_msg}")
            raise ValueError(error_msg) from e
        except Exception as e:
            logger.error(f"   ✗ 加載分類數據失敗: {e}")
            raise
    
    def get_available_attacks(self) -> Dict[str, AttackCapability]:
        """獲取所有可用的攻擊能力
        
        Returns:
            Dict[capability_type, AttackCapability]: 攻擊能力字典
            
            示例:
            {
                "xss": AttackCapability(
                    capability_type="xss",
                    module_name="function_xss",
                    flow_ids=[11, 12, 13, ...],
                    language="Python",
                    flow_count=97
                ),
                "sqli": AttackCapability(...)
            }
        
        Raises:
            RuntimeError: 如果探索器未初始化
        """
        if not self._initialized:
            raise RuntimeError(_NOT_INITIALIZED_ERROR)
        
        # 使用緩存避免重複解析
        if self._attacks_cache is not None:
            return self._attacks_cache
        
        logger.debug("🔍 正在解析攻擊能力...")
        attacks: Dict[str, AttackCapability] = {}
        
        modules = self._classification_data.get("modules", {})
        
        for module_name, module_data in modules.items():
            # 提取模組信息
            language = module_data.get("language", "Python")
            module_info = module_data.get("info", {})
            module_desc = module_info.get("name", "")
            
            # 跳過引擎模組（不是攻擊能力）
            if "engine" in module_name.lower():
                continue
            
            # 收集該模組的所有 flow IDs
            flows = module_data.get("flows", [])
            flow_ids = [flow["id"] for flow in flows if "id" in flow]
            
            # 映射能力類型（從模組名稱提取）
            # 例如: function_xss -> xss, function_sqli -> sqli
            capability_type = module_name.replace("function_", "").replace("_", "")
            
            # 創建攻擊能力對象
            capability = AttackCapability(
                capability_type=capability_type,
                module_name=module_name,
                flow_ids=flow_ids,
                language=language,
                flow_count=len(flow_ids),
                description=module_desc
            )
            
            attacks[capability_type] = capability
            logger.debug(f"   ✓ {capability_type}: {len(flow_ids)} flows ({language})")
        
        self._attacks_cache = attacks
        logger.debug(f"✅ 解析完成: {len(attacks)} 種攻擊能力")
        return attacks
    
    def get_available_engines(self) -> Dict[str, EngineInfo]:
        """獲取所有可用的引擎
        
        Returns:
            Dict[engine_name, EngineInfo]: 引擎信息字典
            
            示例:
            {
                "python_engine": EngineInfo(
                    engine_name="python_engine",
                    language="Python",
                    module_count=24,
                    capabilities=["xss", "sqli", ...]
                ),
                "rust_engine": EngineInfo(...)
            }
        
        Raises:
            RuntimeError: 如果探索器未初始化
        """
        if not self._initialized:
            raise RuntimeError(_NOT_INITIALIZED_ERROR)
        
        # 使用緩存避免重複解析
        if self._engines_cache is not None:
            return self._engines_cache
        
        logger.debug("🔍 正在解析引擎信息...")
        engines: Dict[str, EngineInfo] = {}
        
        modules = self._classification_data.get("modules", {})
        
        for module_name, module_data in modules.items():
            # 只處理引擎模組
            if "engine" not in module_name.lower():
                continue
            
            language = module_data.get("language", "Unknown")
            module_info = module_data.get("info", {})
            module_desc = module_info.get("name", "")
            flows = module_data.get("flows", [])
            
            # 提取該引擎支援的能力
            capabilities = []
            for flow in flows:
                sub_module = flow.get("sub_module", "")
                if sub_module and sub_module not in capabilities:
                    capabilities.append(sub_module)
            
            # 創建引擎信息對象
            engine = EngineInfo(
                engine_name=module_name,
                language=language,
                module_count=len(flows),
                capabilities=capabilities,
                description=module_desc
            )
            
            engines[module_name] = engine
            logger.debug(f"   ✓ {module_name}: {language} ({len(flows)} flows)")
        
        self._engines_cache = engines
        logger.debug(f"✅ 解析完成: {len(engines)} 個引擎")
        return engines
    
    def get_capability_details(self, capability_type: str) -> Optional[AttackCapability]:
        """獲取特定能力的詳細信息
        
        Args:
            capability_type: 能力類型 (如 "xss", "sqli")
        
        Returns:
            AttackCapability 對象，如果不存在返回 None
        """
        attacks = self.get_available_attacks()
        return attacks.get(capability_type)
    
    def get_engine_details(self, engine_name: str) -> Optional[EngineInfo]:
        """獲取特定引擎的詳細信息
        
        Args:
            engine_name: 引擎名稱 (如 "python_engine")
        
        Returns:
            EngineInfo 對象，如果不存在返回 None
        """
        engines = self.get_available_engines()
        return engines.get(engine_name)
    
    def get_classification_metadata(self) -> Dict:
        """獲取分類數據的元信息
        
        Returns:
            元數據字典，包含生成時間、版本、統計等
        """
        if not self._initialized:
            raise RuntimeError(_NOT_INITIALIZED_ERROR)
        
        return self._classification_data.get("metadata", {})


# ==================== 便利函數 ====================

def quick_explore() -> Dict:
    """快速探索系統能力 (便利函數)
    
    Returns:
        {
            "attacks": Dict[str, AttackCapability],
            "engines": Dict[str, EngineInfo],
            "metadata": Dict
        }
    """
    explorer = SystemSelfExplorer()
    explorer.initialize()
    
    return {
        "attacks": explorer.get_available_attacks(),
        "engines": explorer.get_available_engines(),
        "metadata": explorer.get_classification_metadata()
    }


if __name__ == "__main__":
    def main():
        print("=" * 60)
        print("SystemSelfExplorer 測試".center(60))
        print("=" * 60)
        
        # 創建探索器
        explorer = SystemSelfExplorer()
        explorer.initialize()
        
        # 測試 1: 獲取攻擊能力
        print("\n【測試 1】獲取攻擊能力")
        attacks = explorer.get_available_attacks()
        print(f"總攻擊能力: {len(attacks)}")
        
        # 顯示前 5 個
        for i, (cap_type, cap_info) in enumerate(list(attacks.items())[:5], 1):
            print(f"{i}. {cap_type}:")
            print(f"   - 模組: {cap_info.module_name}")
            print(f"   - Flows: {cap_info.flow_count} 個")
            print(f"   - 語言: {cap_info.language}")
        
        print("...")
        
        # 測試 2: 獲取引擎
        print("\n【測試 2】獲取引擎")
        engines = explorer.get_available_engines()
        print(f"總引擎數: {len(engines)}")
        
        for engine_name, engine_info in engines.items():
            print(f"• {engine_name}:")
            print(f"   - 語言: {engine_info.language}")
            print(f"   - Flows: {engine_info.module_count}")
        
        # 測試 3: 獲取特定能力詳情
        print("\n【測試 3】獲取 XSS 能力詳情")
        xss_details = explorer.get_capability_details("xss")
        if xss_details:
            print(f"能力類型: {xss_details.capability_type}")
            print(f"所屬模組: {xss_details.module_name}")
            print(f"Flow IDs 數量: {len(xss_details.flow_ids)}")
            print(f"前 10 個 Flow IDs: {xss_details.flow_ids[:10]}")
        
        # 測試 4: 獲取元數據
        print("\n【測試 4】獲取元數據")
        metadata = explorer.get_classification_metadata()
        print(f"生成時間: {metadata.get('generated_at')}")
        print(f"版本: {metadata.get('version')}")
        print(f"總 Flows: {metadata.get('total_flows')}")
        print(f"支援語言: {metadata.get('languages')}")
        
        print("\n" + "=" * 60)
    
    main()
