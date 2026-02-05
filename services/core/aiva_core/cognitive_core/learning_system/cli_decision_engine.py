"""
CLI Decision Engine - 基於 RAG 的 CLI 命令決策引擎

直接使用 external_classification.json 進行攻擊能力檢索和調用決策。
不需要額外的 JSONL 轉換，直接讀取 AST 分析結果。

使用流程:
    1. AI 分析掃描結果，識別漏洞類型
    2. CLIDecisionEngine 檢索對應的攻擊 flows
    3. 根據上下文調整優先級
    4. 執行選定的攻擊流程

Architecture:
    - 數據源: services/integration/data/internal_exploration/external_classification.json
    - 檢索: 基於模組類型、使用場景的語義搜索
    - 執行: 通過 FlowExecutor 動態調用
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class AttackCapability(str, Enum):
    """攻擊能力類型"""
    XSS = "function_xss"
    SQLI = "function_sqli"
    SSRF = "function_ssrf"
    SCANNER = "function_web_scanner"
    POSTEX = "function_postex"
    FILE_UPLOAD = "function_file_upload"
    XXE = "function_xxe"


@dataclass
class AttackFlow:
    """攻擊流程資訊"""
    flow_id: int
    module: str
    path: List[str]
    file_name: str
    use_case: str
    is_operable: bool
    operability_reason: str
    language: str = "python"
    
    @property
    def entry_point(self) -> str:
        """獲取入口點（調用路徑的第一個元素）"""
        return self.path[0] if self.path else ""
    
    @property
    def full_path(self) -> str:
        """獲取完整調用路徑"""
        return " -> ".join(self.path)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "flow_id": self.flow_id,
            "module": self.module,
            "entry_point": self.entry_point,
            "full_path": self.full_path,
            "file_name": self.file_name,
            "use_case": self.use_case,
            "is_operable": self.is_operable,
            "operability_reason": self.operability_reason,
            "language": self.language,
        }


class CLIDecisionEngine:
    """CLI 命令決策引擎
    
    基於 external_classification.json 進行攻擊能力檢索。
    AI 可以根據掃描結果，智能選擇合適的攻擊流程。
    """
    
    def __init__(self, classification_path: Optional[str] = None):
        """初始化決策引擎
        
        Args:
            classification_path: external_classification.json 路徑，
                                 默認為 services/integration/data/internal_exploration/external_classification.json
        """
        if classification_path is None:
            # 自動定位專案根目錄
            current_dir = Path(__file__).resolve()
            project_root = current_dir
            while project_root.parent != project_root:
                # 尋找包含 services 目錄的根目錄
                if (project_root / "services" / "integration").exists():
                    break
                project_root = project_root.parent
            
            # 如果在 services/core 內執行，向上找到真正的根目錄
            if "services" in str(current_dir):
                parts = current_dir.parts
                if "services" in parts:
                    idx = parts.index("services")
                    project_root = Path(*parts[:idx])
            
            classification_path = str(
                project_root / "services" / "integration" / "data" / 
                "internal_exploration" / "external_classification.json"
            )
        
        self.classification_path = classification_path
        self.flows_by_module: Dict[str, List[AttackFlow]] = {}
        self.all_flows: List[AttackFlow] = []
        
        self._load_classification()
        logger.info(f"✅ CLI Decision Engine 已初始化，載入 {len(self.all_flows)} 個攻擊 flows")
    
    def _load_classification(self) -> None:
        """載入 external_classification.json"""
        try:
            with open(self.classification_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modules = data.get('modules', {})
            for module_name, module_data in modules.items():
                flows = []
                for flow_data in module_data.get('flows', []):
                    # 處理可能缺失的欄位
                    flow = AttackFlow(
                        flow_id=flow_data.get('id', 0),
                        module=module_name,
                        path=flow_data.get('path', []),
                        file_name=flow_data.get('script_name', flow_data.get('file_name', 'unknown')),
                        use_case=flow_data.get('use_case', flow_data.get('module_description', '')),
                        is_operable=flow_data.get('is_operable', False),
                        operability_reason=flow_data.get('operability_reason', ''),
                        language=flow_data.get('language', 'python').lower()
                    )
                    flows.append(flow)
                    self.all_flows.append(flow)
                
                self.flows_by_module[module_name] = flows
            
            logger.info(f"載入模組: {list(self.flows_by_module.keys())}")
            for module, flows in self.flows_by_module.items():
                operable_count = sum(1 for f in flows if f.is_operable)
                logger.info(f"  - {module}: {len(flows)} flows ({operable_count} 可操作)")
        
        except Exception as e:
            logger.error(f"❌ 載入 classification 失敗: {e}")
            raise
    
    def search_flows(
        self, 
        capability: Optional[AttackCapability] = None,
        keywords: Optional[List[str]] = None,
        only_operable: bool = True,
        limit: int = 10
    ) -> List[AttackFlow]:
        """檢索攻擊流程
        
        Args:
            capability: 攻擊能力類型（如 AttackCapability.XSS）
            keywords: 關鍵字列表（在 use_case 中搜索）
            only_operable: 是否只返回可操作的 flows
            limit: 最大返回數量
        
        Returns:
            匹配的 AttackFlow 列表
        """
        # 1. 根據 capability 篩選
        if capability:
            candidates = self.flows_by_module.get(capability.value, [])
        else:
            candidates = self.all_flows
        
        # 2. 只保留可操作的
        if only_operable:
            candidates = [f for f in candidates if f.is_operable]
        
        # 3. 關鍵字搜索
        if keywords:
            keyword_lower = [k.lower() for k in keywords]
            matched = []
            for flow in candidates:
                use_case_lower = flow.use_case.lower()
                if any(kw in use_case_lower for kw in keyword_lower):
                    matched.append(flow)
            candidates = matched
        
        # 4. 限制數量
        return candidates[:limit]
    
    def get_flow_by_id(self, flow_id: int) -> Optional[AttackFlow]:
        """根據 flow_id 獲取流程"""
        for flow in self.all_flows:
            if flow.flow_id == flow_id:
                return flow
        return None
    
    def recommend_flows(
        self, 
        scan_context: Dict[str, Any],
        top_k: int = 5
    ) -> List[AttackFlow]:
        """根據掃描上下文推薦攻擊流程
        
        Args:
            scan_context: 掃描上下文，包含:
                - vulnerability_type: 漏洞類型 (e.g., "XSS", "SQLi")
                - target_info: 目標資訊 (e.g., {"parameter": "id", "method": "GET"})
                - scan_results: 掃描結果摘要
            top_k: 返回前 K 個推薦
        
        Returns:
            推薦的 AttackFlow 列表
        """
        vuln_type = scan_context.get('vulnerability_type', '').upper()
        
        # 映射漏洞類型到攻擊能力
        capability_map = {
            'XSS': AttackCapability.XSS,
            'SQLI': AttackCapability.SQLI,
            'SQL INJECTION': AttackCapability.SQLI,
            'SSRF': AttackCapability.SSRF,
            'XXE': AttackCapability.XXE,
        }
        
        capability = capability_map.get(vuln_type)
        
        # 從 scan_context 提取關鍵字
        keywords = []
        if 'target_info' in scan_context:
            target = scan_context['target_info']
            if 'parameter' in target:
                keywords.append(target['parameter'])
            if 'method' in target:
                keywords.append(target['method'])
        
        # 檢索
        flows = self.search_flows(
            capability=capability,
            keywords=keywords if keywords else None,
            only_operable=True,
            limit=top_k
        )
        
        logger.info(f"推薦 {len(flows)} 個攻擊流程 (類型: {vuln_type}, 關鍵字: {keywords})")
        return flows
    
    def get_module_statistics(self) -> Dict[str, Dict[str, int]]:
        """獲取模組統計資訊"""
        stats = {}
        for module, flows in self.flows_by_module.items():
            stats[module] = {
                "total": len(flows),
                "operable": sum(1 for f in flows if f.is_operable),
                "non_operable": sum(1 for f in flows if not f.is_operable),
            }
        return stats


# ==================== 使用範例 ====================

def example_usage():
    """使用範例"""
    # 1. 初始化引擎
    engine = CLIDecisionEngine()
    
    # 2. 查看模組統計
    stats = engine.get_module_statistics()
    print("模組統計:")
    for module, stat in stats.items():
        print(f"  {module}: {stat['operable']}/{stat['total']} 可操作")
    
    # 3. 檢索 XSS 攻擊流程
    xss_flows = engine.search_flows(
        capability=AttackCapability.XSS,
        keywords=["反射", "檢測"],
        only_operable=True,
        limit=5
    )
    print(f"\n找到 {len(xss_flows)} 個 XSS 攻擊流程:")
    for flow in xss_flows:
        print(f"  - {flow.entry_point} ({flow.use_case})")
    
    # 4. 根據掃描結果推薦
    scan_context = {
        "vulnerability_type": "XSS",
        "target_info": {
            "url": "https://example.com/search",
            "parameter": "q",
            "method": "GET"
        }
    }
    recommended = engine.recommend_flows(scan_context, top_k=3)
    print(f"\n推薦的攻擊流程:")
    for flow in recommended:
        print(f"  - Flow {flow.flow_id}: {flow.full_path}")
        print(f"    使用場景: {flow.use_case}")


if __name__ == "__main__":
    example_usage()
