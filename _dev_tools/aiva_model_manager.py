#!/usr/bin/env python3
"""
AIVA 模型管理器原型 - 立即可用的權重整合方案
基於分析報告的實際實現，可直接整合到現有系統

作者: GitHub Copilot
目標: 提供生產級的 AI 模型載入和管理功能
"""

import os
import torch
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """模型類型枚舉"""
    STRUCTURED = "structured_model"  # 包含元數據的完整模型
    RAW_WEIGHTS = "raw_weights"     # 純權重文件
    UNKNOWN = "unknown"

class LoadStatus(Enum):
    """載入狀態"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    file_path: str
    model_type: ModelType
    total_params: int
    layers: int
    file_size_mb: float
    architecture: Dict[str, Any]
    timestamp: Optional[str] = None
    load_time_ms: Optional[float] = None
    status: LoadStatus = LoadStatus.NOT_LOADED
    error_message: Optional[str] = None

class AIVAModelManager:
    """AIVA AI 模型管理器
    
    功能：
    - 自動發現和載入 PyTorch 權重文件
    - 支援多模型並存和切換
    - 提供健康監控和錯誤處理
    - 與現有 real_ai_core.py 整合
    """
    
    def __init__(self, models_dir: str = "."):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ModelInfo] = {}
        self.active_model: Optional[str] = None
        self.model_data: Dict[str, Any] = {}
        
        # 配置文件
        self.config = {
            "primary_model": "aiva_real_ai_core",
            "backup_models": ["aiva_real_weights", "aiva_5M_weights"],
            "auto_fallback": True,
            "health_check_interval": 30
        }
        
        # 已知的模型文件
        self.known_models = {
            "aiva_real_ai_core": "aiva_real_ai_core.pth",
            "aiva_real_weights": "aiva_real_weights.pth",
            "aiva_5M_weights": "aiva_5M_weights.pth"
        }
    
    async def initialize(self) -> bool:
        """初始化模型管理器"""
        logger.info("🚀 初始化 AIVA AI 模型管理器...")
        
        try:
            # 發現可用模型
            await self.discover_models()
            
            # 載入主要模型
            success = await self.load_primary_model()
            
            if success:
                logger.info("✅ AIVA AI 模型管理器初始化成功!")
                return True
            else:
                logger.error("❌ 模型管理器初始化失敗")
                return False
                
        except Exception as e:
            logger.error(f"❌ 模型管理器初始化異常: {e}")
            return False
    
    async def discover_models(self) -> List[ModelInfo]:
        """發現可用的模型文件"""
        logger.info("🔍 正在發現可用的模型文件...")
        
        discovered_models = []
        
        for model_name, filename in self.known_models.items():
            file_path = self.models_dir / filename
            
            if file_path.exists():
                try:
                    model_info = await self._analyze_model_file(model_name, str(file_path))
                    self.models[model_name] = model_info
                    discovered_models.append(model_info)
                    logger.info(f"✅ 發現模型: {model_name} ({model_info.file_size_mb:.1f}MB)")
                    
                except Exception as e:
                    logger.error(f"❌ 分析模型 {model_name} 失敗: {e}")
            else:
                logger.warning(f"⚠️ 模型文件不存在: {filename}")
        
        logger.info(f"🎯 總計發現 {len(discovered_models)} 個可用模型")
        return discovered_models
    
    async def _analyze_model_file(self, model_name: str, file_path: str) -> ModelInfo:
        """分析模型文件"""
        import time
        
        start_time = time.time()
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # 載入 PyTorch 模型
        data = torch.load(file_path, map_location='cpu')
        load_time_ms = (time.time() - start_time) * 1000
        
        # 分析模型結構
        if isinstance(data, dict) and 'model_state_dict' in data:
            # 結構化模型 (推薦格式)
            state_dict = data['model_state_dict']
            architecture = data.get('architecture', {})
            total_params = data.get('total_params', 0)
            timestamp = data.get('timestamp')
            model_type = ModelType.STRUCTURED
            
        elif isinstance(data, dict):
            # 純權重字典
            state_dict = data
            total_params = sum(tensor.numel() for tensor in state_dict.values() 
                             if isinstance(tensor, torch.Tensor))
            architecture = self._infer_architecture_from_weights(state_dict)
            timestamp = None
            model_type = ModelType.RAW_WEIGHTS
            
        else:
            # 未知格式
            raise ValueError(f"不支援的模型格式: {type(data)}")
        
        layers = len(state_dict)
        
        return ModelInfo(
            name=model_name,
            file_path=file_path,
            model_type=model_type,
            total_params=total_params,
            layers=layers,
            file_size_mb=file_size_mb,
            architecture=architecture,
            timestamp=timestamp,
            load_time_ms=load_time_ms
        )
    
    def _infer_architecture_from_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """從權重推斷架構信息"""
        layer_shapes = []
        total_params = 0
        
        for name, tensor in state_dict.items():
            layer_shapes.append({
                "name": name,
                "shape": list(tensor.shape),
                "params": tensor.numel()
            })
            total_params += tensor.numel()
        
        # 嘗試推斷網路結構
        weight_layers = [info for info in layer_shapes if 'weight' in info['name']]
        
        if len(weight_layers) >= 2:
            input_size = weight_layers[0]['shape'][0] if len(weight_layers[0]['shape']) >= 2 else None
            output_size = weight_layers[-1]['shape'][1] if len(weight_layers[-1]['shape']) >= 2 else None
            hidden_sizes = [layer['shape'][1] for layer in weight_layers[:-1] 
                          if len(layer['shape']) >= 2]
        else:
            input_size = output_size = None
            hidden_sizes = []
        
        return {
            "type": "inferred",
            "input_size": input_size,
            "output_size": output_size,
            "hidden_sizes": hidden_sizes,
            "total_params": total_params,
            "layer_details": layer_shapes[:5]  # 只保留前5層的詳細信息
        }
    
    async def load_primary_model(self) -> bool:
        """載入主要模型"""
        primary_name = self.config["primary_model"]
        
        if primary_name in self.models:
            return await self.load_model(primary_name)
        else:
            logger.warning(f"⚠️ 主要模型 {primary_name} 不可用，嘗試載入備用模型...")
            return await self.load_fallback_model()
    
    async def load_fallback_model(self) -> bool:
        """載入備用模型"""
        for backup_name in self.config["backup_models"]:
            if backup_name in self.models:
                logger.info(f"🔄 嘗試載入備用模型: {backup_name}")
                if await self.load_model(backup_name):
                    return True
        
        logger.error("❌ 所有備用模型都無法載入")
        return False
    
    async def load_model(self, model_name: str) -> bool:
        """載入指定模型"""
        if model_name not in self.models:
            logger.error(f"❌ 模型 {model_name} 不存在")
            return False
        
        model_info = self.models[model_name]
        model_info.status = LoadStatus.LOADING
        
        try:
            logger.info(f"📥 正在載入模型: {model_name}")
            
            # 載入模型數據
            data = torch.load(model_info.file_path, map_location='cpu')
            
            # 提取權重
            if model_info.model_type == ModelType.STRUCTURED:
                weights = data['model_state_dict']
            else:
                weights = data
            
            # 存儲模型數據
            self.model_data[model_name] = {
                "weights": weights,
                "raw_data": data,
                "info": model_info
            }
            
            # 更新狀態
            model_info.status = LoadStatus.LOADED
            self.active_model = model_name
            
            logger.info(f"✅ 模型 {model_name} 載入成功!")
            logger.info(f"   參數量: {model_info.total_params:,}")
            logger.info(f"   模型類型: {model_info.model_type.value}")
            
            return True
            
        except Exception as e:
            model_info.status = LoadStatus.ERROR
            model_info.error_message = str(e)
            logger.error(f"❌ 模型 {model_name} 載入失敗: {e}")
            return False
    
    def get_active_model_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """獲取當前活躍模型的權重"""
        if not self.active_model or self.active_model not in self.model_data:
            return None
        
        return self.model_data[self.active_model]["weights"]
    
    def get_active_model_info(self) -> Optional[ModelInfo]:
        """獲取當前活躍模型的信息"""
        if not self.active_model or self.active_model not in self.models:
            return None
        
        return self.models[self.active_model]
    
    def get_model_status(self) -> Dict[str, Any]:
        """獲取模型管理器狀態"""
        return {
            "active_model": self.active_model,
            "total_models": len(self.models),
            "loaded_models": len(self.model_data),
            "model_list": {
                name: {
                    "status": info.status.value,
                    "params": info.total_params,
                    "size_mb": info.file_size_mb,
                    "type": info.model_type.value
                }
                for name, info in self.models.items()
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def switch_model(self, model_name: str) -> bool:
        """切換到指定模型"""
        if model_name == self.active_model:
            logger.info(f"ℹ️ 模型 {model_name} 已經是活躍模型")
            return True
        
        if model_name not in self.models:
            logger.error(f"❌ 模型 {model_name} 不存在")
            return False
        
        # 如果模型未載入，先載入
        if model_name not in self.model_data:
            if not await self.load_model(model_name):
                return False
        
        # 切換活躍模型
        old_model = self.active_model
        self.active_model = model_name
        
        logger.info(f"🔄 模型切換: {old_model} → {model_name}")
        return True
    
    def export_model_info(self, output_file: str = "model_status.json") -> bool:
        """導出模型信息到 JSON 文件"""
        try:
            status = self.get_model_status()
            
            # 添加詳細的模型信息
            detailed_info = {
                "manager_status": status,
                "detailed_models": {}
            }
            
            for name, info in self.models.items():
                detailed_info["detailed_models"][name] = asdict(info)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_info, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"📄 模型信息已導出到: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 導出模型信息失敗: {e}")
            return False

async def demo_model_manager():
    """示範模型管理器的使用"""
    print("=" * 70)
    print("🧠 AIVA 模型管理器原型示範 🧠")
    print("=" * 70)
    
    # 初始化模型管理器
    manager = AIVAModelManager()
    
    print("\n📦 第一步: 初始化管理器")
    print("-" * 40)
    success = await manager.initialize()
    
    if not success:
        print("❌ 初始化失敗，請檢查權重文件是否存在")
        return
    
    # 顯示模型狀態
    print("\n📊 第二步: 模型狀態概覽")
    print("-" * 40)
    status = manager.get_model_status()
    
    print(f"🎯 活躍模型: {status['active_model']}")
    print(f"📈 總計模型: {status['total_models']}")
    print(f"💾 已載入模型: {status['loaded_models']}")
    print()
    
    for name, info in status['model_list'].items():
        status_emoji = "✅" if info['status'] == 'loaded' else "⏳"
        print(f"{status_emoji} {name}:")
        print(f"   狀態: {info['status']}")
        print(f"   參數: {info['params']:,}")
        print(f"   大小: {info['size_mb']:.1f} MB")
        print(f"   類型: {info['type']}")
        print()
    
    # 測試模型切換
    print("\n🔄 第三步: 測試模型切換")
    print("-" * 40)
    
    available_models = [name for name in manager.models.keys() if name != manager.active_model]
    
    if available_models:
        test_model = available_models[0]
        print(f"🎯 切換到模型: {test_model}")
        
        switch_success = await manager.switch_model(test_model)
        if switch_success:
            print("✅ 模型切換成功!")
            
            # 獲取新模型的權重
            weights = manager.get_active_model_weights()
            if weights:
                print(f"📊 新模型權重層數: {len(weights)}")
                print(f"📊 前3層: {list(weights.keys())[:3]}")
        else:
            print("❌ 模型切換失敗")
    
    # 測試權重訪問
    print("\n🔍 第四步: 測試權重訪問")
    print("-" * 40)
    
    active_info = manager.get_active_model_info()
    if active_info:
        print(f"📁 當前模型: {active_info.name}")
        print(f"⚙️ 架構: {active_info.architecture.get('type', 'unknown')}")
        print(f"📊 參數: {active_info.total_params:,}")
        
        weights = manager.get_active_model_weights()
        if weights:
            print(f"🔢 權重鍵: {len(weights)} 個")
            print(f"🎯 示例層: {list(weights.keys())[:2]}")
    
    # 導出狀態
    print("\n📄 第五步: 導出模型信息")
    print("-" * 40)
    
    export_success = manager.export_model_info("aiva_model_status.json")
    if export_success:
        print("✅ 模型信息已導出到 aiva_model_status.json")
    
    print("\n🎉 模型管理器示範完成!")
    print("\n📋 下一步整合建議:")
    print("  1. 將此管理器整合到 real_ai_core.py")
    print("  2. 修改 RealAIDecisionEngine 使用真實權重")
    print("  3. 添加到 services/core/ai/ 模組系統")
    print("  4. 建立 A/B 測試框架")
    print("=" * 70)

if __name__ == "__main__":
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 運行示範
    asyncio.run(demo_model_manager())