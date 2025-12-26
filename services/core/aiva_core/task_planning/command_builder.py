"""Command Builder - AI 決策到 CLI 命令生成器

將 AI 的抽象決策 (flow_id + intensity + context) 轉換為可執行的 CLI 命令。

遵循「這是一個非常關鍵的架構轉折點.md」規範:
✅ 參數分離 (A類: context, B類: intensity mapping)
✅ 映射策略支持 (linear/inverse/exponential/logarithmic)
✅ 使用 ToolManifest 定義
✅ 遵循 aiva_common v2.0 規範

Architecture:
    AI → flow_id + intensity + context → CommandBuilder → CLI command

Example:
    builder = CommandBuilder(manifests)
    cmd = builder.build_command(
        flow_id=0,
        context_data={"query": "SQL injection"},
        ai_intensity=0.5
    )
    # Output: "python -m aiva_core.cognitive_core.internal_loop_connector query --text 'SQL injection' --top-k 10"
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.capability import (
    ToolManifest,
    TunableParameter,
    ParameterMappingStrategy
)
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = get_logger(__name__)


class CommandBuildError(AIVAError):
    """命令生成錯誤"""
    
    def __init__(self, message: str, flow_id: int | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM
        )
        self.flow_id = flow_id


class CommandBuilder:
    """命令構建器
    
    職責:
    1. 從 Manifest 獲取命令模板
    2. 填充 A類參數 (從 context_data)
    3. 計算 B類參數 (從 ai_intensity 映射)
    4. 生成最終可執行命令
    
    Mapping Strategies:
    - LINEAR: value = min + intensity * (max - min)
        例: intensity=0.5, range=[1,10] → 5.5
    - INVERSE: value = max - intensity * (max - min)
        例: intensity=0.9, range=[1,10] → 1.9 (低強度反而值高)
    - EXPONENTIAL: value = min * (max/min) ^ intensity
        例: intensity=0.5, range=[1,100] → 10 (幾何增長)
    - LOGARITHMIC: value = min + (max-min) * log(1 + intensity*9) / log(10)
        例: intensity=0.5, range=[1,100] → ~50 (對數增長)
    
    Usage:
        builder = CommandBuilder(manifests)
        cmd = builder.build_command(flow_id=0, context_data={...}, ai_intensity=0.8)
        # Execute: subprocess.run(cmd, shell=True)
    """
    
    def __init__(self, manifests: Dict[int, ToolManifest]):
        """初始化命令構建器
        
        Args:
            manifests: {flow_id: ToolManifest} 字典
        """
        self.manifests = manifests
        logger.info(f"✅ CommandBuilder initialized with {len(manifests)} manifests")
    
    def build_command(
        self,
        flow_id: int,
        context_data: Dict[str, Any],
        ai_intensity: float = 0.5,
        dry_run: bool = False
    ) -> str:
        """構建 CLI 命令
        
        Args:
            flow_id: 流程 ID (0-839)
            context_data: 上下文數據 (提供 A類參數)
            ai_intensity: AI 強度等級 (0.0-1.0)
            dry_run: 是否為乾運行 (添加 --dry-run 標誌)
            
        Returns:
            完整的 CLI 命令字符串
            
        Raises:
            CommandBuildError: 命令生成失敗
        """
        # 1. 獲取 Manifest
        manifest = self.manifests.get(flow_id)
        if not manifest:
            raise CommandBuildError(
                f"Manifest not found for flow_id={flow_id}",
                flow_id=flow_id
            )
        
        # 2. 驗證 intensity 範圍
        if not (0.0 <= ai_intensity <= 1.0):
            logger.warning(
                f"⚠️ Invalid ai_intensity={ai_intensity}, clamping to [0.0, 1.0]"
            )
            ai_intensity = max(0.0, min(1.0, ai_intensity))
        
        # 3. 構建參數字典
        params = {}
        
        # 3.1 填充 A類參數 (從 context_data)
        for param_name, context_key in manifest.parameters.context_mapping.items():
            if context_key not in context_data:
                raise CommandBuildError(
                    f"Missing required context key '{context_key}' for parameter '{param_name}'",
                    flow_id=flow_id
                )
            params[param_name] = context_data[context_key]
        
        # 3.2 計算 B類參數 (從 ai_intensity)
        for param_name, tunable in manifest.parameters.tunable.items():
            value = self._map_intensity_to_value(
                intensity=ai_intensity,
                tunable=tunable,
                param_name=param_name
            )
            params[param_name] = value
        
        # 4. 生成命令
        try:
            command = manifest.execution.command_template.format(**params)
        except KeyError as e:
            raise CommandBuildError(
                f"Missing parameter in template: {str(e)}",
                flow_id=flow_id
            )
        
        # 5. 添加乾運行標誌
        if dry_run:
            command += " --dry-run"
        
        logger.info(
            f"✅ Built command for flow_id={flow_id}, "
            f"intensity={ai_intensity:.2f}: {command}"
        )
        
        return command
    
    def _map_intensity_to_value(
        self,
        intensity: float,
        tunable: TunableParameter,
        param_name: str
    ) -> Any:
        """將 AI 強度映射為參數值
        
        Args:
            intensity: AI 強度 (0.0-1.0)
            tunable: 參數定義
            param_name: 參數名稱 (用於日誌)
            
        Returns:
            映射後的參數值
        """
        param_type = tunable.type.lower()
        strategy = tunable.mapping_strategy
        
        # Boolean 類型特殊處理
        if param_type == "boolean":
            return intensity >= 0.5  # threshold=0.5
        
        # String 類型直接返回默認值
        if param_type == "string":
            return tunable.default
        
        # 數值類型需要 min/max
        if tunable.min is None or tunable.max is None:
            logger.warning(
                f"⚠️ Parameter '{param_name}' missing min/max, using default={tunable.default}"
            )
            return tunable.default
        
        min_val = float(tunable.min)
        max_val = float(tunable.max)
        
        # 根據策略計算
        if strategy == ParameterMappingStrategy.LINEAR:
            value = min_val + intensity * (max_val - min_val)
        
        elif strategy == ParameterMappingStrategy.INVERSE:
            value = max_val - intensity * (max_val - min_val)
        
        elif strategy == ParameterMappingStrategy.EXPONENTIAL:
            if min_val <= 0:
                logger.warning(
                    f"⚠️ EXPONENTIAL strategy requires min>0, falling back to LINEAR"
                )
                value = min_val + intensity * (max_val - min_val)
            else:
                value = min_val * math.pow(max_val / min_val, intensity)
        
        elif strategy == ParameterMappingStrategy.LOGARITHMIC:
            # log(1 + intensity*9) / log(10) 範圍 [0, 1]
            log_factor = math.log10(1 + intensity * 9)
            value = min_val + (max_val - min_val) * log_factor
        
        else:
            logger.warning(
                f"⚠️ Unknown strategy '{strategy}', falling back to LINEAR"
            )
            value = min_val + intensity * (max_val - min_val)
        
        # 類型轉換
        if param_type == "integer":
            value = int(round(value))
        elif param_type == "float":
            value = float(value)
        
        logger.debug(
            f"Mapped '{param_name}': intensity={intensity:.2f} "
            f"→ {value} (strategy={strategy}, range=[{min_val}, {max_val}])"
        )
        
        return value
    
    def preview_parameters(
        self,
        flow_id: int,
        context_data: Dict[str, Any],
        intensity_samples: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ) -> Dict[str, Any]:
        """預覽不同強度下的參數值 (用於調試)
        
        Args:
            flow_id: 流程 ID
            context_data: 上下文數據
            intensity_samples: 強度採樣點
            
        Returns:
            {
                "manifest": {...},
                "context_params": {...},
                "tunable_params": {
                    "param_name": [
                        {"intensity": 0.0, "value": ...},
                        {"intensity": 0.25, "value": ...},
                        ...
                    ]
                }
            }
        """
        manifest = self.manifests.get(flow_id)
        if not manifest:
            raise CommandBuildError(
                f"Manifest not found for flow_id={flow_id}",
                flow_id=flow_id
            )
        
        # A類參數
        context_params = {}
        for param_name, context_key in manifest.parameters.context_mapping.items():
            context_params[param_name] = context_data.get(context_key, "<missing>")
        
        # B類參數 (多強度採樣)
        tunable_params = {}
        for param_name, tunable in manifest.parameters.tunable.items():
            samples = []
            for intensity in intensity_samples:
                value = self._map_intensity_to_value(intensity, tunable, param_name)
                samples.append({"intensity": intensity, "value": value})
            tunable_params[param_name] = samples
        
        return {
            "manifest": {
                "flow_id": flow_id,
                "tool_name": manifest.meta.tool_name,
                "command_template": manifest.execution.command_template
            },
            "context_params": context_params,
            "tunable_params": tunable_params
        }
