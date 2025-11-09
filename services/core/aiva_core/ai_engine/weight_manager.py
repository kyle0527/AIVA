#!/usr/bin/env python3
"""
真實AI權重管理系統

基於PyTorch官方最佳實踐實現的權重管理系統，包含：
- 自動載入/儲存機制
- 檔案完整性檢查  
- 權重版本管理
- 錯誤處理和容錯
- 安全的序列化/反序列化
"""

import torch
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

@dataclass
class WeightMetadata:
    """權重檔案元數據"""
    version: str
    created_at: str
    model_architecture: Dict[str, Any]
    total_parameters: int
    file_hash: str
    pytorch_version: str
    device_type: str
    
class AIWeightManager:
    """
    AI權重管理器
    
    基於PyTorch官方序列化最佳實踐實現的權重管理系統。
    支援安全載入、版本控制、完整性檢查和自動備份。
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path] = "weights",
                 backup_enabled: bool = True,
                 max_backups: int = 5,
                 use_weights_only: bool = True):
        """
        初始化權重管理器
        
        Args:
            base_dir: 權重檔案基礎目錄
            backup_enabled: 是否啟用自動備份
            max_backups: 最大備份數量
            use_weights_only: 是否使用weights_only=True安全模式
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.backup_enabled = backup_enabled
        self.max_backups = max_backups
        self.use_weights_only = use_weights_only
        
        # 創建子目錄
        self.weights_dir = self.base_dir / "models"
        self.backups_dir = self.base_dir / "backups" 
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.weights_dir, self.backups_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"權重管理器初始化: {self.base_dir}")
    
    def save_model_weights(self, 
                          model: torch.nn.Module,
                          model_name: str,
                          version: str = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, WeightMetadata]:
        """
        保存模型權重 (基於PyTorch最佳實踐)
        
        Args:
            model: 要保存的模型
            model_name: 模型名稱
            version: 版本號 (自動生成如果未提供)
            metadata: 額外元數據
            
        Returns:
            (檔案路徑, 元數據對象)
        """
        try:
            if version is None:
                version = f"v{int(time.time())}"
            
            # 生成檔案名
            filename = f"{model_name}_{version}.pth"
            filepath = self.weights_dir / filename
            
            # 備份現有檔案
            if filepath.exists() and self.backup_enabled:
                self._create_backup(filepath)
            
            # 準備儲存數據 (PyTorch推薦格式)
            save_data = {
                'model_state_dict': model.state_dict(),
                'model_architecture': {
                    'class_name': model.__class__.__name__,
                    'total_parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                },
                'pytorch_version': torch.__version__,
                'timestamp': time.time(),
                'version': version,
                'metadata': metadata or {}
            }
            
            # 保存模型 (使用最新的PyTorch格式)
            torch.save(save_data, filepath)
            
            # 計算檔案哈希
            file_hash = self._calculate_file_hash(filepath)
            
            # 創建元數據
            weight_metadata = WeightMetadata(
                version=version,
                created_at=datetime.now().isoformat(),
                model_architecture=save_data['model_architecture'],
                total_parameters=save_data['model_architecture']['total_parameters'],
                file_hash=file_hash,
                pytorch_version=torch.__version__,
                device_type=str(next(model.parameters()).device)
            )
            
            # 保存元數據
            self._save_metadata(model_name, version, weight_metadata)
            
            file_size = filepath.stat().st_size
            logger.info(f"權重保存成功: {filepath} ({file_size/1024/1024:.1f} MB)")
            logger.info(f"參數數量: {weight_metadata.total_parameters:,}")
            
            return str(filepath), weight_metadata
            
        except Exception as e:
            logger.error(f"權重保存失敗: {e}")
            raise
    
    def load_model_weights(self, 
                          model: torch.nn.Module,
                          model_name: str,
                          version: str = "latest",
                          device: Optional[Union[str, torch.device]] = None) -> WeightMetadata:
        """
        載入模型權重 (安全模式)
        
        Args:
            model: 要載入權重的模型
            model_name: 模型名稱  
            version: 版本號 ("latest"為最新版本)
            device: 目標設備
            
        Returns:
            權重元數據
        """
        try:
            # 找到權重檔案
            filepath = self._find_weight_file(model_name, version)
            if not filepath.exists():
                raise FileNotFoundError(f"權重檔案不存在: {filepath}")
            
            # 載入元數據並驗證
            metadata = self._load_and_verify_metadata(model_name, version, filepath)
            
            # 設定設備
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 載入權重 (安全模式)
            logger.info(f"載入權重: {filepath}")
            
            if self.use_weights_only:
                # 使用weights_only=True安全模式 (PyTorch 2.6+推薦)
                checkpoint = torch.load(filepath, map_location=device, weights_only=True)
            else:
                # 傳統模式 (向後相容)
                checkpoint = torch.load(filepath, map_location=device)
            
            # 驗證模型架構相容性
            self._verify_model_compatibility(model, checkpoint.get('model_architecture', {}))
            
            # 載入狀態字典
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 向後相容：直接載入狀態字典
                model.load_state_dict(checkpoint)
            
            file_size = filepath.stat().st_size
            logger.info(f"權重載入成功: {filepath} ({file_size/1024/1024:.1f} MB)")
            logger.info(f"參數數量: {metadata.total_parameters:,}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"權重載入失敗: {e}")
            raise
    
    def list_available_weights(self, model_name: Optional[str] = None) -> Dict[str, list]:
        """列出可用的權重檔案"""
        result = {}
        
        try:
            if model_name:
                # 列出特定模型的版本
                pattern = f"{model_name}_*.pth"
                files = list(self.weights_dir.glob(pattern))
                versions = []
                
                for file in files:
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        version = '_'.join(parts[1:])
                        metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                meta = json.load(f)
                                versions.append({
                                    'version': version,
                                    'created_at': meta.get('created_at'),
                                    'file_size_mb': file.stat().st_size / 1024 / 1024,
                                    'parameters': meta.get('total_parameters', 0)
                                })
                
                result[model_name] = sorted(versions, key=lambda x: x['created_at'], reverse=True)
            else:
                # 列出所有模型
                for file in self.weights_dir.glob("*.pth"):
                    parts = file.stem.split('_')
                    if len(parts) >= 2:
                        name = parts[0]
                        if name not in result:
                            result[name] = []
                        
                        version = '_'.join(parts[1:])
                        result[name].append({
                            'version': version,
                            'file_size_mb': file.stat().st_size / 1024 / 1024
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"列出權重檔案失敗: {e}")
            return {}
    
    def delete_weights(self, model_name: str, version: str) -> bool:
        """刪除特定版本的權重"""
        try:
            filepath = self._find_weight_file(model_name, version)
            metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
            
            if filepath.exists():
                # 創建備份
                if self.backup_enabled:
                    self._create_backup(filepath)
                filepath.unlink()
                
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"權重刪除成功: {model_name} {version}")
            return True
            
        except Exception as e:
            logger.error(f"權重刪除失敗: {e}")
            return False
    
    def _find_weight_file(self, model_name: str, version: str) -> Path:
        """找到權重檔案路徑"""
        if version == "latest":
            # 找到最新版本
            pattern = f"{model_name}_*.pth"
            files = list(self.weights_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"未找到模型權重: {model_name}")
            
            # 按修改時間排序，返回最新的
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return files[0]
        else:
            # 特定版本
            filename = f"{model_name}_{version}.pth"
            return self.weights_dir / filename
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """計算檔案SHA256哈希"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_metadata(self, model_name: str, version: str, metadata: WeightMetadata) -> None:
        """保存元數據"""
        metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
        
        metadata_dict = {
            'version': metadata.version,
            'created_at': metadata.created_at,
            'model_architecture': metadata.model_architecture,
            'total_parameters': metadata.total_parameters,
            'file_hash': metadata.file_hash,
            'pytorch_version': metadata.pytorch_version,
            'device_type': metadata.device_type
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_and_verify_metadata(self, model_name: str, version: str, filepath: Path) -> WeightMetadata:
        """載入並驗證元數據"""
        if version == "latest":
            # 從檔案名提取版本
            version = filepath.stem.split('_', 1)[1]
        
        metadata_path = self.metadata_dir / f"{model_name}_{version}.json"
        
        if not metadata_path.exists():
            # 生成基本元數據
            logger.warning(f"元數據檔案不存在，生成基本資訊: {metadata_path}")
            file_hash = self._calculate_file_hash(filepath)
            return WeightMetadata(
                version=version,
                created_at=datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                model_architecture={},
                total_parameters=0,
                file_hash=file_hash,
                pytorch_version="unknown",
                device_type="unknown"
            )
        
        with open(metadata_path, 'r') as f:
            meta_dict = json.load(f)
        
        metadata = WeightMetadata(**meta_dict)
        
        # 驗證檔案完整性
        current_hash = self._calculate_file_hash(filepath)
        if metadata.file_hash != current_hash:
            logger.warning(f"檔案哈希不匹配，可能已損壞: {filepath}")
        
        return metadata
    
    def _verify_model_compatibility(self, model: torch.nn.Module, arch_info: Dict[str, Any]) -> None:
        """驗證模型相容性"""
        if not arch_info:
            return
        
        current_params = sum(p.numel() for p in model.parameters())
        saved_params = arch_info.get('total_parameters', 0)
        
        if saved_params > 0 and abs(current_params - saved_params) > 0:
            logger.warning(f"參數數量不匹配: 當前={current_params:,}, 儲存={saved_params:,}")
    
    def _create_backup(self, filepath: Path) -> None:
        """創建備份檔案"""
        if not self.backup_enabled:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
            backup_path = self.backups_dir / backup_name
            
            shutil.copy2(filepath, backup_path)
            logger.info(f"備份創建: {backup_path}")
            
            # 清理舊備份
            self._cleanup_old_backups(filepath.stem)
            
        except Exception as e:
            logger.warning(f"備份創建失敗: {e}")
    
    def _cleanup_old_backups(self, model_stem: str) -> None:
        """清理舊的備份檔案"""
        try:
            pattern = f"{model_stem}_backup_*.pth"
            backups = list(self.backups_dir.glob(pattern))
            
            if len(backups) > self.max_backups:
                # 按修改時間排序
                backups.sort(key=lambda f: f.stat().st_mtime)
                
                # 刪除最舊的備份
                for backup in backups[:-self.max_backups]:
                    backup.unlink()
                    logger.info(f"舊備份已刪除: {backup}")
                    
        except Exception as e:
            logger.warning(f"清理舊備份失敗: {e}")

# 全局權重管理器實例
_global_weight_manager: Optional[AIWeightManager] = None

def get_weight_manager() -> AIWeightManager:
    """獲取全局權重管理器實例"""
    global _global_weight_manager
    if _global_weight_manager is None:
        _global_weight_manager = AIWeightManager()
    return _global_weight_manager

def initialize_weight_manager(base_dir: str = "weights", **kwargs) -> AIWeightManager:
    """初始化全局權重管理器"""
    global _global_weight_manager
    _global_weight_manager = AIWeightManager(base_dir, **kwargs)
    return _global_weight_manager