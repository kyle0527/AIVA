import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class CapabilityMatcher:
    """能力匹配器 - 負責將 AI 意圖轉換為可執行的 Flow ID
    
    主要功能:
    1. 加載 external_classification.json (優先)
    2. 提供 match_intent(description) 方法
    3. 生成標準 CLI 指令
    """
    
    def __init__(self, classification_path: str | None = None):
        if classification_path:
            self.classification_path = Path(classification_path)
        else:
            # 默認路徑: services/integration/data/internal_exploration/external_classification.json
            base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            self.classification_path = base_dir / "services/integration/data/internal_exploration/external_classification.json"
            
        self.capabilities: dict[str, Any] = {}
        self.flows: list[dict[str, Any]] = []
        self._load_capabilities()

    def _load_capabilities(self):
        """加載能力數據"""
        try:
            if not self.classification_path.exists():
                logger.warning(f"Classification file not found at: {self.classification_path}")
                return

            with open(self.classification_path, encoding='utf-8') as f:
                data = json.load(f)
                
            self.capabilities = data
            
            # 展平所有 Flows 以便搜索
            if "modules" in data:
                for module_name, module_data in data["modules"].items():
                    if "flows" in module_data:
                        self.flows.extend(module_data["flows"])
                        
            logger.info(f"Loaded {len(self.flows)} flows from {self.classification_path}")
            
        except Exception as e:
            logger.error(f"Failed to load capabilities: {e}", exc_info=True)

    def match_intent(self, description: str, top_k: int = 1) -> list[dict[str, Any]]:
        """根據描述匹配最佳 Flow
        
        Args:
            description: 意圖描述 (e.g. "perform xss scan")
            top_k: 返回前 K 個結果
            
        Returns:
            匹配的 Flow 列表
        """
        if not self.flows:
            return []
            
        description = description.lower()
        candidates = []
        
        for flow in self.flows:
            score = 0
            flow_desc = flow.get("module_description", "").lower()
            flow_name = flow.get("name", "").lower()
            module_name = flow.get("module", "").lower()
            use_case = flow.get("use_case", "").lower()
            
            # 1. 關鍵字匹配
            keywords = description.split()
            for kw in keywords:
                if kw in flow_desc: score += 10
                if kw in flow_name: score += 15
                if kw in module_name: score += 5
                if kw in use_case: score += 8
                
            # 2. 類型匹配 (e.g., "scan", "xss")
            if "xss" in description and "xss" in flow_name: score += 20
            if "sql" in description and ("sql" in flow_name or "sqli" in flow_name): score += 20
            
            if score > 0:
                candidates.append((score, flow))
                
        # 排序並返回
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:top_k]]

    def format_command(self, flow: dict[str, Any], target: str) -> str:
        """格式化為標準 CLI 指令"""
        flow_id = flow.get("id")
        # 修正: 使用統一入口 aiva_cli.py
        # 假設在專案根目錄執行
        return f"python -m services.core.aiva_core.core_capabilities.cli.aiva_cli --flow {flow_id} --target {target}"

    def get_flow_by_id(self, flow_id: int) -> dict[str, Any] | None:
        """根據 ID 獲取 Flow"""
        for flow in self.flows:
            if flow.get("id") == flow_id:
                return flow
        return None
