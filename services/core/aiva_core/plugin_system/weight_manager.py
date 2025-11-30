#!/usr/bin/env python3
"""
AI 模型權重管理器

負責 AI 模型權重的版本管理、完整性驗證和存儲。
設計參考 HuggingFace Model Hub 和 TensorFlow Serving。

核心功能:
1. 權重註冊和版本控制 (語義化版本)
2. SHA256 完整性驗證
3. 本地存儲管理
4. 雲端同步 (可選)
5. latest 符號鏈接管理

使用方式:
    weight_manager = WeightManager(Path("data/weights"))
    
    # 註冊權重
    weight_manager.register_weights(
        module_id="bio_neuron",
        version="v1.0.0",
        weight_path=Path("/path/to/model.safetensors"),
        metadata={"parameters": 5000000}
    )
    
    # 獲取權重
    weight_path = weight_manager.get_weights("bio_neuron", "latest")
    
    # 驗證完整性
    is_valid = weight_manager.verify_weights("bio_neuron", "v1.0.0")
"""

import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


class WeightManager:
    """AI 模型權重管理器
    
    功能：
    1. 本地權重存儲和版本管理
    2. 權重完整性驗證 (SHA256)
    3. 語義化版本控制
    4. 雲端同步 (可選)
    
    目錄結構:
        weights_dir/
        ├── bio_neuron/
        │   ├── v1.0.0/
        │   │   ├── model.safetensors
        │   │   ├── config.json
        │   │   └── metadata.yaml
        │   ├── v1.1.0/
        │   └── latest -> v1.1.0/
        ├── embeddings/
        └── registry.json
    """
    
    def __init__(self, weights_dir: Path):
        """初始化權重管理器
        
        Args:
            weights_dir: 權重存儲根目錄
        """
        self.weights_dir = weights_dir
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = weights_dir / "registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"WeightManager initialized: {weights_dir}")
    
    def register_weights(
        self,
        module_id: str,
        version: str,
        weight_path: Path,
        metadata: Optional[Dict] = None
    ) -> bool:
        """註冊模組權重
        
        執行流程:
        1. 計算 SHA256 校驗和
        2. 複製權重到版本目錄
        3. 生成配置文件
        4. 生成元數據 YAML
        5. 更新註冊表
        6. 更新 latest 符號鏈接
        
        Args:
            module_id: 模組 ID (如 "bio_neuron")
            version: 版本號 (如 "v1.0.0")
            weight_path: 權重文件路徑
            metadata: 額外元數據 (作者、訓練時間等)
        
        Returns:
            註冊是否成功
        """
        try:
            if not weight_path.exists():
                raise FileNotFoundError(f"Weight file not found: {weight_path}")
            
            logger.info(f"Registering weights: {module_id} v{version}")
            
            # 1. 計算 SHA256 校驗和
            checksum = self._calculate_checksum(weight_path)
            logger.info(f"Weight checksum: {checksum[:16]}...")
            
            # 2. 創建版本目錄
            target_dir = self.weights_dir / module_id / version
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. 複製權重文件
            target_file = target_dir / weight_path.name
            shutil.copy2(weight_path, target_file)
            logger.info(f"Weight file copied to: {target_file}")
            
            # 4. 生成配置文件
            config = {
                "module_id": module_id,
                "version": version,
                "weight_file": weight_path.name,
                "checksum": checksum,
                "size_mb": weight_path.stat().st_size / (1024 * 1024),
                "registered_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            config_file = target_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 5. 生成元數據 YAML
            metadata_content = {
                "module": module_id,
                "version": version,
                "checksum": checksum,
                "description": metadata.get("description", "") if metadata else "",
                "author": metadata.get("author", "AIVA Team") if metadata else "AIVA Team",
                "training_date": metadata.get("training_date", "") if metadata else "",
                "architecture": metadata.get("architecture", "") if metadata else "",
                "parameters": metadata.get("parameters", 0) if metadata else 0,
                "metrics": metadata.get("metrics", {}) if metadata else {},
            }
            
            metadata_file = target_dir / "metadata.yaml"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                yaml.dump(metadata_content, f, default_flow_style=False, allow_unicode=True)
            
            # 6. 更新註冊表
            if module_id not in self.registry:
                self.registry[module_id] = {"versions": []}
            
            version_info = {
                "version": version,
                "path": str(target_file),
                "checksum": checksum,
                "size_mb": round(config["size_mb"], 2),
                "registered_at": config["registered_at"]
            }
            
            # 檢查是否已存在此版本
            existing_versions = [v["version"] for v in self.registry[module_id]["versions"]]
            if version in existing_versions:
                # 更新現有版本
                for i, v in enumerate(self.registry[module_id]["versions"]):
                    if v["version"] == version:
                        self.registry[module_id]["versions"][i] = version_info
                        break
            else:
                # 添加新版本
                self.registry[module_id]["versions"].append(version_info)
            
            self._save_registry()
            
            # 7. 更新 latest 符號鏈接
            self._update_latest_link(module_id, version)
            
            logger.info(f"✅ Weights registered successfully: {module_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register weights: {e}", exc_info=True)
            return False
    
    def get_weights(
        self,
        module_id: str,
        version: str = "latest"
    ) -> Optional[Path]:
        """獲取模組權重路徑
        
        Args:
            module_id: 模組 ID
            version: 版本號 (default: "latest")
        
        Returns:
            權重文件路徑，如果不存在則返回 None
        """
        try:
            weight_dir = self.weights_dir / module_id / version
            
            if not weight_dir.exists():
                logger.warning(f"Weight directory not found: {weight_dir}")
                return None
            
            # 查找權重文件 (支持多種格式)
            for ext in ['.safetensors', '.pt', '.pth', '.onnx', '.h5', '.bin']:
                weight_files = list(weight_dir.glob(f"*{ext}"))
                if weight_files:
                    logger.info(f"Found weight file: {weight_files[0]}")
                    return weight_files[0]
            
            logger.warning(f"No weight file found in {weight_dir}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting weights: {e}")
            return None
    
    def verify_weights(self, module_id: str, version: str = "latest") -> bool:
        """驗證權重完整性
        
        通過比較當前文件的 SHA256 校驗和與註冊時的校驗和
        
        Args:
            module_id: 模組 ID
            version: 版本號
        
        Returns:
            驗證是否通過
        """
        try:
            weight_path = self.get_weights(module_id, version)
            if not weight_path:
                return False
            
            # 計算當前校驗和
            current_checksum = self._calculate_checksum(weight_path)
            
            # 讀取註冊時的校驗和
            config_file = weight_path.parent / "config.json"
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return False
            
            with open(config_file, encoding='utf-8') as f:
                config = json.load(f)
                registered_checksum = config["checksum"]
            
            if current_checksum == registered_checksum:
                logger.info(f"✅ Weight integrity verified: {module_id} v{version}")
                return True
            else:
                logger.error(f"❌ Weight integrity check failed: {module_id} v{version}")
                logger.error(f"Expected: {registered_checksum[:16]}...")
                logger.error(f"Got: {current_checksum[:16]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying weights: {e}", exc_info=True)
            return False
    
    def list_weights(self, module_id: Optional[str] = None) -> List[Dict]:
        """列出所有權重
        
        Args:
            module_id: 如果指定，只列出該模組的權重
        
        Returns:
            權重信息列表
        """
        weights = []
        
        if module_id:
            # 只列出指定模組
            if module_id in self.registry:
                for version_info in self.registry[module_id]["versions"]:
                    weights.append({
                        "module_id": module_id,
                        **version_info
                    })
        else:
            # 列出所有模組
            for mod_id, mod_data in self.registry.items():
                for version_info in mod_data["versions"]:
                    weights.append({
                        "module_id": mod_id,
                        **version_info
                    })
        
        return weights
    
    def delete_weights(self, module_id: str, version: str) -> bool:
        """刪除指定版本的權重
        
        Args:
            module_id: 模組 ID
            version: 版本號 (不能是 "latest")
        
        Returns:
            刪除是否成功
        """
        if version == "latest":
            logger.error("Cannot delete 'latest', specify exact version")
            return False
        
        try:
            weight_dir = self.weights_dir / module_id / version
            
            if not weight_dir.exists():
                logger.warning(f"Weight directory not found: {weight_dir}")
                return False
            
            # 刪除目錄
            shutil.rmtree(weight_dir)
            
            # 更新註冊表
            if module_id in self.registry:
                self.registry[module_id]["versions"] = [
                    v for v in self.registry[module_id]["versions"]
                    if v["version"] != version
                ]
                self._save_registry()
            
            logger.info(f"✅ Deleted weights: {module_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete weights: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """計算文件 SHA256 校驗和
        
        使用分塊讀取，支持大文件
        
        Args:
            file_path: 文件路徑
        
        Returns:
            SHA256 十六進制字符串
        """
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _load_registry(self) -> Dict:
        """載入權重註冊表
        
        Returns:
            註冊表字典
        """
        if self.registry_file.exists():
            try:
                with open(self.registry_file, encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """保存權重註冊表"""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _update_latest_link(self, module_id: str, version: str):
        """更新 latest 符號鏈接
        
        Args:
            module_id: 模組 ID
            version: 版本號
        """
        try:
            latest_link = self.weights_dir / module_id / "latest"
            
            # 移除舊鏈接
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # 創建新鏈接
            latest_link.symlink_to(version, target_is_directory=True)
            logger.info(f"Updated latest link: {module_id}/latest -> {version}")
            
        except Exception as e:
            logger.warning(f"Failed to update latest link: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取權重管理器統計信息
        
        Returns:
            統計信息字典
        """
        total_size_mb = 0.0
        total_versions = 0
        
        for mod_id, mod_data in self.registry.items():
            for version_info in mod_data["versions"]:
                total_size_mb += version_info.get("size_mb", 0)
                total_versions += 1
        
        return {
            "total_modules": len(self.registry),
            "total_versions": total_versions,
            "total_size_mb": round(total_size_mb, 2),
            "modules": list(self.registry.keys()),
        }
