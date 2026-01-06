"""Manifest Loader - Manifest JSON 加載器

從文件系統加載和驗證 Tool Manifest JSON。

架構更新 (2026-01-04):
- 改用 integration 的 MinimalManifest（統一數據來源）
- internal_exploration 產出 → integration 提供 Schema → core 載入使用

遵循 aiva_common v2.0 規範:
✅ 使用統一日誌 (get_logger)
✅ 使用 MinimalManifest 驗證結構 (from integration)
✅ 統一錯誤處理
✅ 完整類型註解

Design:
- 加載單個/全部 Manifest JSON
- Pydantic 自動驗證
- 錯誤處理和日誌記錄
"""

import json
from pathlib import Path
from typing import List, Optional

from aiva_common.utils.logging import get_logger
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

# 統一使用 integration 的 MinimalManifest
from services.integration.capability.minimal_manifest import MinimalManifest

logger = get_logger(__name__)


class ManifestLoadError(AIVAError):
    """Manifest 加載錯誤"""
    
    def __init__(self, message: str, file_path: Path | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.MEDIUM
        )
        self.file_path = file_path


class ManifestLoader:
    """能力清單 JSON 加載器
    
    職責:
    1. 從指定目錄加載所有 .json 文件
    2. 使用 MinimalManifest 驗證結構
    3. 提供按 capability_id 查詢接口
    4. 錯誤處理和日誌記錄
    
    Usage:
        loader = ManifestLoader(Path("core_capabilities/manifests/capabilities"))
        manifests = loader.load_all()
        manifest = loader.get_by_id("xss.scan.web")
    """
    
    def __init__(self, manifests_dir: Path):
        """初始化加載器
        
        Args:
            manifests_dir: Manifest JSON 文件目錄
        """
        self.manifests_dir = Path(manifests_dir)
        self._manifests: dict[str, MinimalManifest] = {}  # capability_id -> MinimalManifest
        self._loaded = False
        
        if not self.manifests_dir.exists():
            logger.warning(
                f"Manifest directory does not exist: {self.manifests_dir}"
            )
    
    def load_all(self, reload: bool = False) -> dict[str, MinimalManifest]:
        """加載所有能力清單文件
        
        Args:
            reload: 是否強制重新加載
            
        Returns:
            {capability_id: MinimalManifest} 字典
            
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
                capability_id = manifest.id  # 從根層級獲取 id 字段
                
                if capability_id in self._manifests:
                    logger.warning(
                        f"⚠️ Duplicate capability_id={capability_id} in {json_file.name}, "
                        f"overwriting previous manifest"
                    )
                
                self._manifests[capability_id] = manifest
                logger.debug(f"✅ Loaded manifest: capability_id={capability_id}, name={manifest.name}")
                
            except Exception as e:
                error_msg = f"Failed to load {json_file.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        self._loaded = True
        
        if errors and not self._manifests:
            # 全部失敗
            raise ManifestLoadError(
                "Failed to load any manifests. Errors:\n" + "\n".join(errors)
            )
        
        logger.info(f"✅ Successfully loaded {len(self._manifests)} manifests")
        if errors:
            logger.warning(f"⚠️ {len(errors)} files failed to load")
        
        return self._manifests
    
    def _load_single(self, json_file: Path) -> MinimalManifest:
        """加載單個 Manifest 文件
        
        Args:
            json_file: JSON 文件路徑
            
        Returns:
            MinimalManifest 實例
            
        Raises:
            ManifestLoadError: 加載或驗證失敗
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Pydantic 自動驗證
            manifest = MinimalManifest(**data)
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
    
    def get_by_id(self, capability_id: str) -> MinimalManifest | None:
        """根據 capability_id 獲取能力清單
        
        Args:
            capability_id: 能力 ID（例：xss.scan.web）
            
        Returns:
            MinimalManifest 或 None
        """
        if not self._loaded:
            self.load_all()
        
        return self._manifests.get(capability_id)
    
    def list_all_ids(self) -> list[str]:
        """列出所有已加載的 capability_id
        
        Returns:
            capability_id 列表 (已排序)
        """
        if not self._loaded:
            self.load_all()
        
        return sorted(self._manifests.keys())
    
    def filter_by_tags(self, required: list[str]) -> list[MinimalManifest]:
        """根據必需標籤過濾能力
        
        Args:
            required: 必需標籤列表
            
        Returns:
            匹配的 MinimalManifest 列表
        """
        if not self._loaded:
            self.load_all()
        
        results = []
        for manifest in self._manifests.values():
            # MinimalManifest.tags 是 List[str]
            manifest_tags = set(manifest.tags)
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
        # 按標籤分組
        tag_distribution = {}
        
        for manifest in self._manifests.values():
            # 語言統計 - 從 tags 中推斷
            lang = "python"  # 預設語言
            for tag in manifest.tags:
                if tag in ["python", "rust", "go", "typescript", "javascript"]:
                    lang = tag
                    break
            lang_distribution[lang] = lang_distribution.get(lang, 0) + 1
            
            # 標籤統計
            for tag in manifest.tags:
                tag_distribution[tag] = tag_distribution.get(tag, 0) + 1
        
        return {
            "total_manifests": len(self._manifests),
            "manifest_ids": list(self._manifests.keys()),
            "language_distribution": lang_distribution,
            "tag_distribution": tag_distribution
        }
