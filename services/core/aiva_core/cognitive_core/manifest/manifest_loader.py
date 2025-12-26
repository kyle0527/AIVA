"""Manifest Loader - Manifest JSON 加載器

從文件系統加載和驗證 Tool Manifest JSON。

遵循 aiva_common v2.0 規範:
✅ 使用統一日誌 (get_logger)
✅ 使用 Pydantic 模型驗證 (ToolManifest from aiva_common)
✅ 統一錯誤處理
✅ 完整類型註解

Design:
- 加載單個/全部 Manifest JSON
- Pydantic 自動驗證
- 錯誤處理和日誌記錄
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.capability import ToolManifest
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = get_logger(__name__)


class ManifestLoadError(AIVAError):
    """Manifest 加載錯誤"""
    
    def __init__(self, message: str, file_path: Path | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM
        )
        self.file_path = file_path


class ManifestLoader:
    """Manifest JSON 加載器
    
    職責:
    1. 從指定目錄加載所有 .json 文件
    2. 使用 ToolManifest 驗證結構
    3. 提供按 flow_id 查詢接口
    4. 錯誤處理和日誌記錄
    
    Usage:
        loader = ManifestLoader(Path("core_capabilities/manifests/capabilities"))
        manifests = loader.load_all()
        manifest = loader.get_by_flow_id(0)
    """
    
    def __init__(self, manifests_dir: Path):
        """初始化加載器
        
        Args:
            manifests_dir: Manifest JSON 文件目錄
        """
        self.manifests_dir = Path(manifests_dir)
        self._manifests: Dict[int, ToolManifest] = {}  # flow_id -> ToolManifest
        self._loaded = False
        
        if not self.manifests_dir.exists():
            logger.warning(
                f"Manifest directory does not exist: {self.manifests_dir}"
            )
    
    def load_all(self, reload: bool = False) -> Dict[int, ToolManifest]:
        """加載所有 Manifest 文件
        
        Args:
            reload: 是否強制重新加載
            
        Returns:
            {flow_id: ToolManifest} 字典
            
        Raises:
            ManifestLoadError: 加載或驗證失敗
        """
        if self._loaded and not reload:
            logger.debug("Manifests already loaded, returning cached results")
            return self._manifests
        
        if not self.manifests_dir.exists():
            raise ManifestLoadError(
                f"Manifest directory not found: {self.manifests_dir}"
            )
        
        self._manifests.clear()
        json_files = list(self.manifests_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No manifest JSON files found in {self.manifests_dir}")
            return self._manifests
        
        logger.info(f"🔄 Loading {len(json_files)} manifest files from {self.manifests_dir}")
        
        errors = []
        for json_file in json_files:
            try:
                manifest = self._load_single(json_file)
                flow_id = manifest.meta.flow_id
                
                if flow_id in self._manifests:
                    logger.warning(
                        f"⚠️ Duplicate flow_id={flow_id} in {json_file.name}, "
                        f"overwriting previous manifest"
                    )
                
                self._manifests[flow_id] = manifest
                logger.debug(f"✅ Loaded manifest: flow_id={flow_id}, name={manifest.meta.tool_name}")
                
            except Exception as e:
                error_msg = f"Failed to load {json_file.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        self._loaded = True
        
        if errors and not self._manifests:
            # 全部失敗
            raise ManifestLoadError(
                f"Failed to load any manifests. Errors:\n" + "\n".join(errors)
            )
        
        logger.info(f"✅ Successfully loaded {len(self._manifests)} manifests")
        if errors:
            logger.warning(f"⚠️ {len(errors)} files failed to load")
        
        return self._manifests
    
    def _load_single(self, json_file: Path) -> ToolManifest:
        """加載單個 Manifest 文件
        
        Args:
            json_file: JSON 文件路徑
            
        Returns:
            ToolManifest 實例
            
        Raises:
            ManifestLoadError: 加載或驗證失敗
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Pydantic 自動驗證
            manifest = ToolManifest(**data)
            return manifest
            
        except json.JSONDecodeError as e:
            raise ManifestLoadError(
                f"Invalid JSON in {json_file.name}: {str(e)}",
                file_path=json_file
            )
        except Exception as e:
            raise ManifestLoadError(
                f"Validation failed for {json_file.name}: {str(e)}",
                file_path=json_file
            )
    
    def get_by_flow_id(self, flow_id: int) -> Optional[ToolManifest]:
        """根據 flow_id 獲取 Manifest
        
        Args:
            flow_id: 流程 ID (0-839)
            
        Returns:
            ToolManifest 或 None
        """
        if not self._loaded:
            self.load_all()
        
        return self._manifests.get(flow_id)
    
    def list_all_flow_ids(self) -> List[int]:
        """列出所有已加載的 flow_id
        
        Returns:
            flow_id 列表 (已排序)
        """
        if not self._loaded:
            self.load_all()
        
        return sorted(self._manifests.keys())
    
    def filter_by_tags(self, required: List[str]) -> List[ToolManifest]:
        """根據必需標籤過濾 Manifest (簡化版 Hard Mask)
        
        Args:
            required: 必需標籤列表
            
        Returns:
            匹配的 Manifest 列表
        """
        if not self._loaded:
            self.load_all()
        
        results = []
        for manifest in self._manifests.values():
            manifest_tags = set(manifest.ai_cognitive.tags.required)
            if manifest_tags.issuperset(required):
                results.append(manifest)
        
        return results
    
    def get_stats(self) -> dict:
        """獲取加載統計
        
        Returns:
            統計信息字典
        """
        if not self._loaded:
            self.load_all()
        
        # 按語言分組
        lang_distribution = {}
        # 按模組分組
        module_distribution = {}
        # 按風險等級分組
        risk_distribution = {}
        
        for manifest in self._manifests.values():
            # 語言統計
            lang = manifest.meta.language.value if hasattr(manifest.meta.language, 'value') else str(manifest.meta.language)
            lang_distribution[lang] = lang_distribution.get(lang, 0) + 1
            
            # 模組統計
            module = manifest.meta.module or "unknown"
            module_distribution[module] = module_distribution.get(module, 0) + 1
            
            # 風險統計
            risk = manifest.ai_cognitive.risk_level
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        return {
            "total_manifests": len(self._manifests),
            "flow_id_range": (min(self._manifests.keys()), max(self._manifests.keys())) if self._manifests else (None, None),
            "language_distribution": lang_distribution,
            "module_distribution": module_distribution,
            "risk_distribution": risk_distribution
        }
