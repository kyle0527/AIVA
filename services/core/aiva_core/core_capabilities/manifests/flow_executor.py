"""Flow Executor - 執行 Manifest 定義的函數調用鏈

基於 840 flows，依序調用每個節點的 Python 函數，而非啟動完整腳本。
"""
import importlib
from typing import Any, Dict
from services.aiva_common.utils import get_logger
from services.core.aiva_core.cognitive_core.manifest.manifest_loader import ManifestLoader
from services.core.aiva_core.task_planning.command_builder import CommandBuilder

logger = get_logger(__name__)


class FlowExecutor:
    """Flow 執行器 - 基於 Manifest 執行函數調用鏈"""
    
    def __init__(self):
        self.loader = ManifestLoader()
        self.builder = CommandBuilder()
        self.loader.load_all()
    
    def execute_flow(self, flow_id: int, context_data: Dict[str, Any], ai_intensity: float) -> Dict[str, Any]:
        """執行指定的 Flow
        
        Args:
            flow_id: Flow ID
            context_data: 上下文數據 (A-class 參數)
            ai_intensity: AI 強度 (0.0-1.0, 影響 B-class 參數)
        
        Returns:
            執行結果字典
        """
        manifest = self.loader.get_by_flow_id(flow_id)
        if not manifest:
            raise ValueError(f"找不到 Flow {flow_id} 的 Manifest")
        
        logger.info(f"執行 Flow {flow_id}: {manifest.meta.tool_name}")
        
        # 根據 execution 類型決定執行方式
        execution_type = getattr(manifest.execution, 'type', 'command')
        
        if execution_type == 'python_function':
            return self._execute_python_function(manifest, context_data, ai_intensity)
        elif execution_type == 'api_call':
            return self._execute_api_call(manifest, context_data, ai_intensity)
        else:
            # 默認：命令行執行（向後兼容）
            return self._execute_command(manifest, context_data, ai_intensity)
    
    def _execute_python_function(self, manifest, context_data: Dict, ai_intensity: float) -> Dict[str, Any]:
        """執行 Python 函數調用
        
        這是 840 flows 的核心模式：依序調用函數，不啟動腳本
        """
        exec_config = manifest.execution
        
        # 動態導入模組
        module_path = exec_config.command_template.get('module')
        class_name = exec_config.command_template.get('class')
        method_name = exec_config.command_template.get('method')
        
        logger.info(f"導入: {module_path}.{class_name}.{method_name}")
        
        try:
            # 導入模組
            module = importlib.import_module(module_path)
            
            # 獲取類
            cls = getattr(module, class_name)
            
            # 實例化（如果需要）
            if exec_config.command_template.get('instantiate', True):
                instance = cls()
                method = getattr(instance, method_name)
            else:
                method = getattr(cls, method_name)
            
            # 準備參數
            params = self._prepare_function_parameters(
                manifest, context_data, ai_intensity
            )
            
            logger.info(f"調用函數，參數: {params}")
            
            # 執行函數
            result = method(**params)
            
            return {
                'status': 'success',
                'flow_id': manifest.meta.flow_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"函數執行失敗: {e}", exc_info=True)
            return {
                'status': 'error',
                'flow_id': manifest.meta.flow_id,
                'error': str(e)
            }
    
    def _execute_api_call(self, manifest, context_data: Dict, ai_intensity: float) -> Dict[str, Any]:
        """執行 API 調用（適用於 FastAPI 端點）"""
        import httpx
        
        exec_config = manifest.execution
        url = self.builder.build_command(manifest.meta.flow_id, context_data, ai_intensity)
        
        logger.info(f"API 調用: {url}")
        
        try:
            response = httpx.get(url, timeout=exec_config.timeout_seconds)
            response.raise_for_status()
            
            return {
                'status': 'success',
                'flow_id': manifest.meta.flow_id,
                'result': response.json()
            }
        except Exception as e:
            logger.error(f"API 調用失敗: {e}", exc_info=True)
            return {
                'status': 'error',
                'flow_id': manifest.meta.flow_id,
                'error': str(e)
            }
    
    def _execute_command(self, manifest, context_data: Dict, ai_intensity: float) -> Dict[str, Any]:
        """執行命令行（向後兼容，較少使用）"""
        import subprocess
        
        command = self.builder.build_command(
            manifest.meta.flow_id, context_data, ai_intensity
        )
        
        logger.info(f"執行命令: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=manifest.execution.timeout_seconds,
                cwd=manifest.execution.working_directory
            )
            
            return {
                'status': 'success' if result.returncode == 0 else 'error',
                'flow_id': manifest.meta.flow_id,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except Exception as e:
            logger.error(f"命令執行失敗: {e}", exc_info=True)
            return {
                'status': 'error',
                'flow_id': manifest.meta.flow_id,
                'error': str(e)
            }
    
    def _prepare_function_parameters(
        self, manifest, context_data: Dict, ai_intensity: float
    ) -> Dict[str, Any]:
        """準備函數參數（A-class + B-class）"""
        params = {}
        
        # A-class: 從上下文映射
        for param_name, context_key in manifest.parameters.context_mapping.items():
            if context_key in context_data:
                params[param_name] = context_data[context_key]
        
        # B-class: 根據 intensity 計算
        for param_name, config in manifest.parameters.tunable.items():
            value = self.builder._map_intensity_to_value(
                ai_intensity, config, param_name
            )
            params[param_name] = value
        
        return params
