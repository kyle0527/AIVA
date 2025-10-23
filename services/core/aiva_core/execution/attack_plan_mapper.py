# services/core/aiva_core/execution/attack_plan_mapper.py
"""
AI 攻擊計畫映射器

負責將 AI 生成的抽象攻擊計畫 (來自 BioNeuron) 轉換為
一系列具體的可執行的 FunctionTaskPayload。
"""

import logging
from typing import Dict, Any, List, Optional

# 假設 Schema 已可正確導入
try:
    from services.aiva_common.schemas.generated.tasks import FunctionTaskPayload, FunctionTaskTarget, FunctionTaskContext
    from services.aiva_common.schemas.generated.messaging import AivaMessage  # AI 決策消息
    from services.aiva_common.enums.modules import AivaModule
    # 可能需要導入其他 Schema 或工具
    IMPORT_SUCCESS = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import schemas in attack_plan_mapper: {e}")
    IMPORT_SUCCESS = False
    # Define dummy classes if import fails to allow file loading
    class FunctionTaskPayload: pass
    class FunctionTaskTarget: pass
    class FunctionTaskContext: pass
    class AivaMessage: pass
    class AivaModule: pass


logger = logging.getLogger(__name__)

class AttackPlanMapper:
    """
    將 AI 決策映射到具體執行任務。
    """

    def __init__(self):
        if not IMPORT_SUCCESS:
            logger.error("AttackPlanMapper initialized with failed schema imports!")
        logger.info("AttackPlanMapper initialized.")
        # 可能需要初始化資源，例如工具註冊表、目標信息等

    async def map_decision_to_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        將單個 AI 決策步驟轉換為一個或多個 FunctionTaskPayload。

        Args:
            ai_decision: AI 生成的決策對象。
            scan_context: 當前的掃描上下文信息 (例如目標 URL、先前結果等)。

        Returns:
            一個包含具體任務 Payload 的列表。
        """
        tasks: List[FunctionTaskPayload] = []
        logger.info(f"Mapping AI decision: {ai_decision}")

        # --- 核心映射邏輯 ---
        # 1. 解析 ai_decision 中的意圖、目標、使用的工具/技術。
        # 2. 根據意圖選擇合適的 Aiva 功能模組 (Function Module)。
        # 3. 從 scan_context 和 ai_decision 中提取目標信息 (URL, params, headers...)。
        # 4. 構建 FunctionTaskTarget。
        # 5. 構建 FunctionTaskContext (可能包含先前步驟的結果)。
        # 6. 構建 FunctionTaskPayload，指定 module_name、target、context、策略等。
        # 7. 如果一個決策需要多個步驟，則生成多個 Task。

        # 示例：基本映射邏輯
        try:
            # 假設 ai_decision 包含工具選擇和參數
            if hasattr(ai_decision, 'payload') and ai_decision.payload:
                decision_data = ai_decision.payload
                
                # 根據決策類型進行映射
                if decision_data.get('action_type') == 'vulnerability_scan':
                    target_info = decision_data.get('target', {})
                    if target_info.get('url'):
                        # 創建掃描任務
                        target = FunctionTaskTarget(
                            url=target_info.get('url'),
                            method=target_info.get('method', 'GET'),
                            params=target_info.get('params'),
                            data=target_info.get('data'),
                            headers=scan_context.get('base_headers', {})
                        )
                        
                        context = FunctionTaskContext(
                            session_id=scan_context.get('session_id'),
                            parent_task_id=scan_context.get('parent_task_id'),
                            scan_config=scan_context.get('scan_config', {})
                        )
                        
                        # 根據漏洞類型選擇模組
                        vuln_type = decision_data.get('vulnerability_type', 'general')
                        module_name = self._map_vulnerability_to_module(vuln_type)
                        
                        payload = FunctionTaskPayload(
                            task_id=f"task_{ai_decision.header.message_id}_{vuln_type}",
                            module_name=module_name,
                            target=target,
                            context=context,
                            strategy=decision_data.get('strategy', 'NORMAL')
                        )
                        tasks.append(payload)
                        logger.info(f"Generated {vuln_type} task for decision {ai_decision.header.message_id}")
                        
                elif decision_data.get('action_type') == 'information_gathering':
                    # 信息收集任務映射
                    tasks.extend(await self._create_info_gathering_tasks(ai_decision, scan_context))
                    
                elif decision_data.get('action_type') == 'exploitation':
                    # 漏洞利用任務映射
                    tasks.extend(await self._create_exploitation_tasks(ai_decision, scan_context))

        except Exception as e:
            logger.error(f"Error mapping decision {getattr(ai_decision, 'header', {}).get('message_id', 'unknown')}: {e}", exc_info=True)

        if not tasks:
            logger.warning(f"No tasks generated for AI decision: {ai_decision}. Decision might be informational or mapping logic is missing.")

        return tasks

    def _map_vulnerability_to_module(self, vuln_type: str) -> str:
        """將漏洞類型映射到對應的模組"""
        mapping = {
            'sqli': 'FUNC_SQLI',
            'xss': 'FUNC_XSS', 
            'ssrf': 'FUNC_SSRF',
            'client_auth_bypass': 'FUNC_CLIENT_AUTH_BYPASS',
            'rce': 'FUNC_RCE',
            'lfi': 'FUNC_LFI',
            'general': 'FUNC_GENERAL_SCAN'
        }
        return mapping.get(vuln_type, 'FUNC_GENERAL_SCAN')

    async def _create_info_gathering_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """創建信息收集任務"""
        tasks = []
        # TODO: 實現信息收集任務創建邏輯
        logger.debug("Creating information gathering tasks")
        return tasks

    async def _create_exploitation_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """創建漏洞利用任務"""
        tasks = []
        # TODO: 實現漏洞利用任務創建邏輯
        logger.debug("Creating exploitation tasks")
        return tasks

    async def map_entire_plan(self, attack_plan: List[AivaMessage], initial_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        將整個 AI 攻擊計畫 (一系列決策) 映射為任務列表。

        Args:
            attack_plan: AI 生成的包含多個決策步驟的計畫。
            initial_context: 初始掃描上下文。

        Returns:
            一個包含所有計畫步驟對應任務的列表。
        """
        all_tasks: List[FunctionTaskPayload] = []
        current_context = initial_context.copy()

        logger.info(f"Mapping entire attack plan with {len(attack_plan)} steps.")

        for i, decision in enumerate(attack_plan):
            logger.debug(f"Mapping step {i+1}/{len(attack_plan)}")
            step_tasks = await self.map_decision_to_tasks(decision, current_context)
            all_tasks.extend(step_tasks)
            
            # 更新上下文以便後續步驟使用
            if step_tasks:
                current_context['previous_task_ids'] = [task.task_id for task in step_tasks]
                current_context['step_number'] = i + 1

        logger.info(f"Generated {len(all_tasks)} tasks from the attack plan.")
        return all_tasks