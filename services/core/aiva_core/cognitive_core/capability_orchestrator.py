"""
能力編排器 (Capability Orchestrator) - AI 決策引擎核心

基於內部閉環分析結果,實現智能能力編排和自我優化

職責:
1. 查詢 RAG 知識庫,了解可用能力
2. 根據任務需求,智能選擇和組合能力
3. 生成 AICommand 序列
4. 監控執行效果,實現自我優化

數據來源:
- RAG 知識庫: 包含所有能力的詳細信息 (由 InternalLoopConnector 同步)
- CapabilityRegistry: 能力註冊表 (實時狀態)
- 歷史執行記錄: 用於學習和優化

架構流程:
    User Input → Orchestrator.plan() → Query RAG → Select Capabilities 
    → Generate AICommands → Execute → Monitor → Learn → Optimize
"""

import asyncio
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from aiva_common.schemas import (
    AICommand,
    AICommandResult,
    CommandType,
    CommandStatus,
)
from aiva_common.utils import get_logger
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

from .internal_loop_connector import InternalLoopConnector
from .rag.knowledge_base import KnowledgeBase
from ..core_capabilities.capability_registry import get_capability_registry

logger = get_logger(__name__)


class TaskRequirement(BaseModel):
    """任務需求定義"""
    task_id: str = Field(..., description="任務唯一ID")
    task_type: str = Field(..., description="任務類型: scan, attack, analysis, comprehensive")
    target: str = Field(..., description="目標 URL 或 IP")
    objectives: List[str] = Field(default_factory=list, description="任務目標列表")
    constraints: dict[str, Any] = Field(default_factory=dict, description="約束條件")
    priority: int = Field(default=5, ge=1, le=10, description="優先級 1-10")


class CapabilityPlan(BaseModel):
    """能力執行計劃"""
    plan_id: str = Field(..., description="計劃唯一ID")
    task_id: str = Field(..., description="關聯的任務ID")
    selected_capabilities: List[dict[str, Any]] = Field(default_factory=list, description="選中的能力列表")
    execution_sequence: List[str] = Field(default_factory=list, description="執行順序 (capability_id)")
    estimated_duration: int = Field(default=0, description="預計耗時(秒)")
    risk_level: str = Field(default="medium", description="風險等級: low, medium, high")
    commands: List[AICommand] = Field(default_factory=list, description="生成的命令列表")
    reasoning: str = Field(default="", description="AI 決策理由")


class ExecutionResult(BaseModel):
    """執行結果"""
    plan_id: str = Field(..., description="計劃ID")
    success: bool = Field(default=False, description="是否成功")
    completed_commands: List[str] = Field(default_factory=list, description="已完成的命令ID")
    failed_commands: List[str] = Field(default_factory=list, description="失敗的命令ID")
    results: Dict[str, AICommandResult] = Field(default_factory=dict, description="執行結果映射")
    total_duration: float = Field(default=0.0, description="總耗時(秒)")
    issues_found: List[dict[str, Any]] = Field(default_factory=list, description="發現的問題列表")


class CapabilityOrchestrator:
    """能力編排器
    
    這是 AI 決策的核心組件,負責:
    1. 理解任務需求
    2. 查詢 RAG 了解可用能力
    3. 智能選擇和組合能力
    4. 生成可執行的命令序列
    5. 監控執行並自我優化
    
    使用示例:
        orchestrator = CapabilityOrchestrator()
        
        # 定義任務需求
        requirement = TaskRequirement(
            task_id="task_001",
            task_type="comprehensive_scan",
            target="https://example.com",
            objectives=["find_vulnerabilities", "test_xss", "test_sqli"]
        )
        
        # 生成執行計劃
        plan = await orchestrator.plan(requirement)
        
        # 執行計劃
        result = await orchestrator.execute(plan)
        
        # 學習優化
        orchestrator.learn_from_execution(plan, result)
    """
    
    def __init__(
        self,
        rag_kb: Optional[KnowledgeBase] = None,
        internal_connector: Optional[InternalLoopConnector] = None
    ):
        """初始化能力編排器
        
        Args:
            rag_kb: RAG 知識庫實例
            internal_connector: 內部閉環連接器
        """
        self.rag_kb = rag_kb
        self.internal_connector = internal_connector or InternalLoopConnector(rag_knowledge_base=rag_kb)
        self.capability_registry = get_capability_registry()
        
        # 執行歷史記錄 (用於學習優化)
        self.execution_history: List[Tuple[CapabilityPlan, ExecutionResult]] = []
        
        # 能力性能統計 (用於優化選擇)
        self.capability_performance: dict[str, dict[str, Any]] = {}
        
        logger.info("✅ CapabilityOrchestrator initialized")
    
    async def plan(self, requirement: TaskRequirement) -> CapabilityPlan:
        """生成能力執行計劃
        
        這是核心決策方法,根據任務需求智能選擇能力並生成執行計劃
        
        決策流程:
        1. 分析任務需求和目標
        2. 查詢 RAG 找到相關能力
        3. 過濾可用且健康的能力
        4. 根據性能歷史排序能力
        5. 生成執行序列
        6. 轉換為 AICommand
        
        Args:
            requirement: 任務需求
            
        Returns:
            能力執行計劃
        """
        plan_id = f"plan_{requirement.task_id}_{uuid4().hex[:8]}"
        logger.info(f"🎯 Planning for task: {requirement.task_id}")
        logger.info(f"   Type: {requirement.task_type}, Target: {requirement.target}")
        logger.info(f"   Objectives: {requirement.objectives}")
        
        try:
            # Step 1: 查詢 RAG 找到相關能力
            relevant_capabilities = await self._query_relevant_capabilities(requirement)
            logger.info(f"   Found {len(relevant_capabilities)} relevant capabilities")
            
            # Step 2: 過濾健康且可用的能力
            available_capabilities = self._filter_available_capabilities(relevant_capabilities)
            logger.info(f"   {len(available_capabilities)} capabilities available")
            
            if not available_capabilities:
                raise AIVAError(
                    error_type=ErrorType.VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"No available capabilities for task: {requirement.task_type}, objectives: {requirement.objectives}"
                )
            
            # Step 3: 選擇最佳能力組合
            selected_capabilities = self._select_best_capabilities(
                available_capabilities, 
                requirement
            )
            logger.info(f"   Selected {len(selected_capabilities)} capabilities")
            
            # Step 4: 生成執行序列
            execution_sequence = self._generate_execution_sequence(
                selected_capabilities,
                requirement
            )
            logger.info(f"   Execution sequence: {execution_sequence}")
            
            # Step 5: 轉換為 AICommand
            commands = self._capabilities_to_commands(
                selected_capabilities,
                execution_sequence,
                requirement
            )
            logger.info(f"   Generated {len(commands)} commands")
            
            # Step 6: 評估風險和耗時
            estimated_duration = sum(c.timeout for c in commands)
            risk_level = self._assess_risk_level(selected_capabilities, requirement)
            
            # Step 7: 生成決策理由
            reasoning = self._generate_reasoning(
                requirement,
                selected_capabilities,
                execution_sequence
            )
            
            plan = CapabilityPlan(
                plan_id=plan_id,
                task_id=requirement.task_id,
                selected_capabilities=selected_capabilities,
                execution_sequence=execution_sequence,
                estimated_duration=estimated_duration,
                risk_level=risk_level,
                commands=commands,
                reasoning=reasoning
            )
            
            logger.info(f"✅ Plan generated: {plan_id}")
            logger.info(f"   Duration: {estimated_duration}s, Risk: {risk_level}")
            logger.info(f"   Reasoning: {reasoning[:100]}...")
            
            return plan
            
        except Exception as e:
            logger.error(f"❌ Planning failed: {e}", exc_info=True)
            raise
    
    async def _query_relevant_capabilities(
        self, 
        requirement: TaskRequirement,
        aiva_module_filter: Optional[str] = None,
        entry_point_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """查詢相關能力（使用 InternalLoopConnector.query_capabilities）
        
        ✅ 架構修復 (2025-12-16):
        改用 InternalLoopConnector.query_capabilities() 方法，
        該方法專門用於查詢 RAG 中的 840 個能力。
        
        使用 RAG 語義搜索找到與任務需求相關的能力
        
        Args:
            requirement: 任務需求
            aiva_module_filter: 六大模組過濾 (可選)
            entry_point_filter: 入口點過濾 (可選)
            
        Returns:
            相關能力列表
        """
        # 構建查詢語句
        query_parts = [
            f"Task: {requirement.task_type}",
            f"Objectives: {', '.join(requirement.objectives)}",
        ]
        
        # 根據任務類型添加特定查詢
        if requirement.task_type == "scan":
            query_parts.append("scanning and detection capabilities")
        elif requirement.task_type == "attack":
            query_parts.append("vulnerability testing and exploitation capabilities")
        elif requirement.task_type == "comprehensive":
            query_parts.append("all relevant security testing capabilities")
        
        query = " | ".join(query_parts)
        logger.info(f"   🔍 Querying capabilities from RAG: {query[:100]}...")
        
        # ✅ 使用 InternalLoopConnector.query_capabilities（正確的能力查詢方法）
        if self.internal_connector:
            try:
                # 構建過濾條件
                filters = {}
                if aiva_module_filter:
                    filters['module'] = aiva_module_filter
                if entry_point_filter:
                    filters['entry_point'] = entry_point_filter
                
                # 查詢 RAG（使用正確的方法）
                rag_result = await asyncio.to_thread(
                    self.internal_connector.query_capabilities,
                    query=query,
                    filters=filters if filters else None,
                    top_k=20  # 獲取更多候選
                )
                
                logger.info(f"   ✅ RAG query returned {len(rag_result.results)} capabilities")
                
                # 轉換 RAG 結果為能力字典列表
                capabilities = []
                for result in rag_result.results:
                    # result 是 RAG 返回的文檔，metadata 包含能力信息
                    cap_dict = {
                        "metadata": result.get("metadata", {}),
                        "text": result.get("text", ""),
                        "relevance_score": result.get("score", 0.0)
                    }
                    capabilities.append(cap_dict)
                
                return capabilities
                
            except Exception as e:
                logger.error(f"   ❌ RAG query failed: {e}", exc_info=True)
                logger.warning("   Falling back to registry search")
                return self._fallback_capability_search(requirement)
        else:
            logger.warning("   ⚠️ InternalLoopConnector not available, using fallback")
            return self._fallback_capability_search(requirement)
    
    def _filter_by_aiva_module(
        self,
        capabilities: List[Dict[str, Any]],
        aiva_module_filter: Optional[str] = None,
        entry_point_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """✅ 新增：按六大模組和入口點過濾能力
        
        Args:
            capabilities: 能力列表
            aiva_module_filter: 六大模組過濾
            entry_point_filter: 入口點過濾
            
        Returns:
            過濾後的能力列表
        """
        if not aiva_module_filter and not entry_point_filter:
            return capabilities
        
        filtered = []
        for cap in capabilities:
            meta = cap.get("metadata", {})
            
            # 檢查模組過濾
            if aiva_module_filter:
                cap_module = meta.get("aiva_module", "")
                if cap_module != aiva_module_filter:
                    continue
            
            # 檢查入口點過濾
            if entry_point_filter:
                cap_entry = meta.get("entry_point", "")
                if cap_entry != entry_point_filter:
                    continue
            
            filtered.append(cap)
        
        logger.debug(f"   Filtered by module/entry_point: {len(capabilities)} -> {len(filtered)}")
        return filtered
    
    def group_capabilities_by_aiva_module(
        self,
        capabilities: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """✅ 新增：按六大模組分組能力
        
        用於展示和分析能力分佈
        
        Args:
            capabilities: 能力列表
            
        Returns:
            按模組分組的能力字典
        """
        grouped = {}
        
        for cap in capabilities:
            meta = cap.get("metadata", {})
            aiva_module = meta.get("aiva_module", "unclassified")
            
            if aiva_module not in grouped:
                grouped[aiva_module] = []
            
            grouped[aiva_module].append(cap)
        
        return grouped
    
    def _fallback_capability_search(
        self,
        requirement: TaskRequirement
    ) -> List[Dict[str, Any]]:
        """降級方案: 直接從 CapabilityRegistry 搜索
        
        當 RAG 不可用時使用。遵循 aiva_common 原則：
        - 不使用 mock 數據
        - 不返回假的成功結果
        - 如果真的查不到，返回空列表，讓錯誤暴露
        """
        logger.warning("   ⚠️ Using fallback capability search (RAG unavailable)")
        capabilities = []
        
        try:
            for objective in requirement.objectives:
                # 搜索包含目標關鍵字的能力
                matched = self.capability_registry.search_capabilities(objective)
                for cap in matched:
                    capabilities.append({
                        "metadata": {
                            "capability_id": cap.id,
                            "capability_name": cap.name,
                            "module": cap.module,
                            "language": cap.language,
                            "description": cap.description,
                            "aiva_module": getattr(cap, "aiva_module", None),
                            "sub_module": getattr(cap, "sub_module", None),
                            "entry_point": getattr(cap, "entry_point", None),
                        },
                        "text": cap.description,
                        "relevance_score": 0.5  # 降級方案的相關性分數較低
                    })
            
            logger.info(f"   Fallback search found {len(capabilities)} capabilities")
            
        except Exception as e:
            logger.error(f"   ❌ Fallback search failed: {e}")
            # 遵循「有錯就報錯」原則：不返回假數據
            return []
        
        return capabilities
    
    def _filter_available_capabilities(
        self,
        capabilities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """過濾可用且健康的能力
        
        檢查能力的健康狀態、可用性,過濾掉不可用的能力
        
        Args:
            capabilities: 候選能力列表
            
        Returns:
            可用能力列表
        """
        available = []
        
        for cap in capabilities:
            meta = cap.get("metadata", {})
            cap_id = meta.get("capability_id")
            
            # 檢查健康分數
            health_score = meta.get("health_score", 1.0)
            availability = meta.get("availability", 1.0)
            
            # 過濾條件
            if health_score >= 0.5 and availability >= 0.7:
                available.append(cap)
            else:
                logger.debug(f"   Filtered out {cap_id}: health={health_score}, availability={availability}")
        
        return available
    
    def _select_best_capabilities(
        self,
        capabilities: List[Dict[str, Any]],
        requirement: TaskRequirement
    ) -> List[Dict[str, Any]]:
        """選擇最佳能力組合
        
        根據歷史性能、健康狀態、任務匹配度選擇最佳能力
        
        Args:
            capabilities: 可用能力列表
            requirement: 任務需求
            
        Returns:
            選中的能力列表
        """
        # 為每個能力計算評分
        scored_capabilities = []
        
        for cap in capabilities:
            score = self._calculate_capability_score(cap, requirement)
            scored_capabilities.append((score, cap))
        
        # 按分數排序
        scored_capabilities.sort(key=lambda x: x[0], reverse=True)
        
        # 選擇 top N
        max_capabilities = requirement.constraints.get("max_capabilities", 5)
        selected = [cap for _, cap in scored_capabilities[:max_capabilities]]
        
        return selected
    
    def _calculate_capability_score(
        self,
        capability: Dict[str, Any],
        requirement: TaskRequirement
    ) -> float:
        """計算能力評分
        
        綜合考慮:
        1. 任務匹配度 (40%)
        2. 健康狀態 (30%)
        3. 歷史性能 (30%)
        """
        meta = capability.get("metadata", {})
        cap_id = meta.get("capability_id")
        
        # 1. 任務匹配度 (基於目標關鍵字)
        match_score = 0.0
        cap_name = meta.get("capability_name", "").lower()
        cap_desc = meta.get("description", "").lower()
        
        for objective in requirement.objectives:
            obj_lower = objective.lower()
            if obj_lower in cap_name:
                match_score += 1.0
            elif obj_lower in cap_desc:
                match_score += 0.5
        
        match_score = min(match_score / len(requirement.objectives), 1.0)
        
        # 2. 健康狀態
        health_score = meta.get("health_score", 1.0)
        availability = meta.get("availability", 1.0)
        health = (health_score + availability) / 2
        
        # 3. 歷史性能
        perf = self.capability_performance.get(cap_id, {})
        success_rate = perf.get("success_rate", 0.8)  # 默認 0.8
        avg_latency = perf.get("avg_latency_ms", 1000)
        
        # 延遲懲罰 (超過 5 秒開始懲罰)
        latency_penalty = max(0, (avg_latency - 5000) / 5000)
        perf_score = success_rate * (1 - min(latency_penalty, 0.5))
        
        # 綜合評分
        total_score = (
            match_score * 0.4 +
            health * 0.3 +
            perf_score * 0.3
        )
        
        logger.debug(f"   Score for {cap_id}: {total_score:.2f} "
                    f"(match={match_score:.2f}, health={health:.2f}, perf={perf_score:.2f})")
        
        return total_score
    
    def _generate_execution_sequence(
        self,
        capabilities: List[Dict[str, Any]],
        requirement: TaskRequirement
    ) -> List[str]:
        """生成執行序列
        
        根據能力間的依賴關係和任務類型決定執行順序
        
        Args:
            capabilities: 選中的能力列表
            requirement: 任務需求
            
        Returns:
            能力 ID 執行序列
        """
        cap_ids = [cap["metadata"]["capability_id"] for cap in capabilities]
        
        # 根據任務類型決定執行順序
        if requirement.task_type == "scan":
            # 掃描任務: Phase 0 → Phase 1 → 深度掃描
            return self._order_scan_sequence(cap_ids, capabilities)
        
        elif requirement.task_type == "attack":
            # 攻擊任務: 按風險等級排序 (低風險先執行)
            return self._order_attack_sequence(cap_ids, capabilities)
        
        elif requirement.task_type == "comprehensive":
            # 綜合任務: 掃描 → 分析 → 攻擊
            return self._order_comprehensive_sequence(cap_ids, capabilities)
        
        else:
            # 默認: 保持原順序
            return cap_ids
    
    def _order_scan_sequence(
        self,
        cap_ids: List[str],
        capabilities: List[Dict[str, Any]]
    ) -> List[str]:
        """排序掃描能力執行順序
        
        Phase 0 (快速偵察) → Phase 1 (深度掃描) → 專項掃描
        """
        phase0_caps = []
        phase1_caps = []
        other_caps = []
        
        for cap_id, cap in zip(cap_ids, capabilities):
            name = cap["metadata"].get("capability_name", "").lower()
            
            if "phase0" in name or "reconnaissance" in name:
                phase0_caps.append(cap_id)
            elif "phase1" in name or "deep_scan" in name:
                phase1_caps.append(cap_id)
            else:
                other_caps.append(cap_id)
        
        return phase0_caps + phase1_caps + other_caps
    
    def _order_attack_sequence(
        self,
        cap_ids: List[str],
        capabilities: List[Dict[str, Any]]
    ) -> List[str]:
        """排序攻擊能力執行順序
        
        低風險 → 中風險 → 高風險
        """
        # 簡單的風險評估規則
        risk_levels = {
            "xss": 1,
            "sqli": 2,
            "ssrf": 2,
            "rce": 3,
            "auth": 2,
        }
        
        scored = []
        for cap_id, cap in zip(cap_ids, capabilities):
            name = cap["metadata"].get("capability_name", "").lower()
            risk = 1
            
            for keyword, level in risk_levels.items():
                if keyword in name:
                    risk = level
                    break
            
            scored.append((risk, cap_id))
        
        scored.sort(key=lambda x: x[0])
        return [cap_id for _, cap_id in scored]
    
    def _order_comprehensive_sequence(
        self,
        cap_ids: List[str],
        capabilities: List[Dict[str, Any]]
    ) -> List[str]:
        """排序綜合任務執行順序
        
        掃描 → 分析 → 攻擊
        """
        scan_caps = []
        analysis_caps = []
        attack_caps = []
        other_caps = []
        
        for cap_id, cap in zip(cap_ids, capabilities):
            category = cap["metadata"].get("category", "").lower()
            
            if "scan" in category:
                scan_caps.append(cap_id)
            elif "analysis" in category:
                analysis_caps.append(cap_id)
            elif "attack" in category:
                attack_caps.append(cap_id)
            else:
                other_caps.append(cap_id)
        
        return scan_caps + analysis_caps + attack_caps + other_caps
    
    def _capabilities_to_commands(
        self,
        capabilities: List[Dict[str, Any]],
        execution_sequence: List[str],
        requirement: TaskRequirement
    ) -> List[AICommand]:
        """將能力轉換為 AICommand
        
        這是連接內部分析和實際執行的關鍵步驟
        
        Args:
            capabilities: 選中的能力列表
            execution_sequence: 執行順序
            requirement: 任務需求
            
        Returns:
            AICommand 列表
        """
        commands = []
        cap_map = {cap["metadata"]["capability_id"]: cap for cap in capabilities}
        
        for cap_id in execution_sequence:
            cap = cap_map.get(cap_id)
            if not cap:
                continue
            
            meta = cap["metadata"]
            
            # 生成命令
            command = self._build_command_from_capability(meta, requirement)
            commands.append(command)
        
        return commands
    
    def _build_command_from_capability(
        self,
        capability_meta: Dict[str, Any],
        requirement: TaskRequirement
    ) -> AICommand:
        """從能力元數據構建 AICommand
        
        根據能力類型映射到對應的 CommandType
        
        Args:
            capability_meta: 能力元數據（包含名稱、模組等信息）
            requirement: 任務需求
        """
        cap_name = capability_meta.get("capability_name", "")
        module = capability_meta.get("module", "")
        
        # 映射到 CommandType
        command_type = self._map_capability_to_command_type(cap_name, module)
        
        # 構建 payload
        payload = self._build_command_payload(capability_meta, requirement)
        
        # 創建 AICommand
        # 將 int priority 轉換為 CommandPriority 枚舉
        from services.aiva_common.schemas.commands import CommandPriority
        priority_map = {1: CommandPriority.LOW, 2: CommandPriority.NORMAL, 3: CommandPriority.HIGH, 4: CommandPriority.HIGH}
        priority_enum = priority_map.get(requirement.priority, CommandPriority.NORMAL)
        
        command = AICommand(
            command_id=f"cmd_{requirement.task_id}_{uuid4().hex[:8]}",
            command_type=command_type,
            target_module=self._get_target_module(module),
            payload=payload,
            timeout=requirement.constraints.get("timeout", 300),
            priority=priority_enum,
            trace_id=requirement.task_id,
            session_id=requirement.task_id,
            parent_command_id=None,
            callback_url=None
        )
        
        return command
    
    def _map_capability_to_command_type(
        self,
        capability_name: str,
        module: str
    ) -> CommandType:
        """映射能力到命令類型
        
        基於能力名稱和模組推斷 CommandType
        """
        name_lower = capability_name.lower()
        
        # Scan 模組
        if "scan" in module.lower():
            if "phase0" in name_lower:
                return CommandType.SCAN_PHASE0
            elif "phase1" in name_lower:
                return CommandType.SCAN_PHASE1
            else:
                return CommandType.SCAN_COMPREHENSIVE
        
        # Features 模組
        if "xss" in name_lower:
            return CommandType.FEATURE_XSS_TEST
        if "sql" in name_lower or "sqli" in name_lower:
            return CommandType.FEATURE_SQLI_TEST
        if "ssrf" in name_lower:
            return CommandType.FEATURE_SSRF_TEST
        if "idor" in name_lower:
            return CommandType.FEATURE_IDOR_TEST
        
        # 默認
        return CommandType.SCAN_COMPREHENSIVE
    
    def _build_command_payload(
        self,
        capability_meta: Dict[str, Any],
        requirement: TaskRequirement
    ) -> Dict[str, Any]:
        """構建命令 payload
        
        根據能力的參數定義和任務需求構建 payload
        
        Args:
            capability_meta: 能力元數據
            requirement: 任務需求
        """
        payload = {
            "target": requirement.target,
            "task_id": requirement.task_id,
            "capability_id": capability_meta.get("capability_id", ""),
        }
        
        # 添加約束條件
        if requirement.constraints:
            payload.update(requirement.constraints)
        
        return payload
    
    def _get_target_module(self, module_path: str) -> str:
        """從模組路徑提取目標模組名稱"""
        if "scan" in module_path.lower():
            return "scan"
        elif "features" in module_path.lower():
            return "features"
        elif "integration" in module_path.lower():
            return "integration"
        else:
            return "core"
    
    def _assess_risk_level(
        self,
        _capabilities: List[Dict[str, Any]],
        requirement: TaskRequirement
    ) -> str:
        """評估風險等級
        
        Args:
            _capabilities: 能力列表（保留用於未來擴展）
            requirement: 任務需求
        """
        # 簡單規則: 掃描低風險,攻擊中/高風險
        if requirement.task_type == "scan":
            return "low"
        elif requirement.task_type == "attack":
            return "high"
        else:
            return "medium"
    
    def _generate_reasoning(
        self,
        requirement: TaskRequirement,
        capabilities: List[Dict[str, Any]],
        _sequence: List[str]
    ) -> str:
        """生成決策理由
        
        解釋 AI 為什麼選擇這些能力和執行順序
        
        Args:
            requirement: 任務需求
            capabilities: 選中的能力列表
            _sequence: 執行序列（保留用於未來擴展）
        """
        reasoning_parts = [
            f"Task: {requirement.task_type} on {requirement.target}",
            f"Objectives: {', '.join(requirement.objectives)}",
            f"Selected {len(capabilities)} capabilities based on:",
            "  1. Task relevance and keyword matching",
            "  2. Capability health and availability",
            "  3. Historical performance metrics",
            f"Execution order optimized for {requirement.task_type} workflow",
        ]
        
        return "\n".join(reasoning_parts)
    
    async def execute(self, plan: CapabilityPlan) -> ExecutionResult:
        """執行計劃
        
        這裡應該調用 AICommandCenter 執行命令
        
        Args:
            plan: 執行計劃
            
        Returns:
            執行結果
        """
        logger.info(f"🚀 Executing plan: {plan.plan_id}")
        
        from aiva_common.command_center import AICommandCenter
        command_center = AICommandCenter()
        
        start_time = datetime.now(UTC)
        completed = []
        failed = []
        results = {}
        
        try:
            for command in plan.commands:
                logger.info(f"   Executing: {command.command_id} [{command.command_type.value}]")
                
                try:
                    result = await command_center.execute(command)
                    results[command.command_id] = result
                    
                    if result.status == CommandStatus.COMPLETED:
                        completed.append(command.command_id)
                    else:
                        failed.append(command.command_id)
                        logger.warning(f"   Command failed: {result.error}")
                
                except Exception as e:
                    logger.error(f"   Command execution error: {e}")
                    failed.append(command.command_id)
            
            duration = (datetime.now(UTC) - start_time).total_seconds()
            
            execution_result = ExecutionResult(
                plan_id=plan.plan_id,
                success=len(failed) == 0,
                completed_commands=completed,
                failed_commands=failed,
                results=results,
                total_duration=duration,
                issues_found=self._extract_issues(results)
            )
            
            logger.info(f"✅ Execution completed: {len(completed)}/{len(plan.commands)} succeeded")
            
            # 記錄到歷史
            self.execution_history.append((plan, execution_result))
            
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}", exc_info=True)
            raise
    
    def _extract_issues(self, results: Dict[str, AICommandResult]) -> List[Dict[str, Any]]:
        """從執行結果中提取發現的問題"""
        issues = []
        
        for cmd_id, result in results.items():
            if result.status == CommandStatus.COMPLETED:
                # 從 result.result 提取漏洞信息
                findings = result.result.get("findings", [])
                issues.extend(findings)
        
        return issues
    
    def learn_from_execution(
        self,
        plan: CapabilityPlan,
        result: ExecutionResult
    ):
        """從執行結果學習優化
        
        更新能力性能統計,用於未來的決策優化
        
        Args:
            plan: 執行計劃
            result: 執行結果
        """
        logger.info(f"📚 Learning from execution: {plan.plan_id}")
        
        for cap in plan.selected_capabilities:
            cap_id = cap["metadata"]["capability_id"]
            
            # 初始化統計
            if cap_id not in self.capability_performance:
                self.capability_performance[cap_id] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "total_latency_ms": 0,
                    "avg_latency_ms": 0,
                    "success_rate": 0.8,
                }
            
            stats = self.capability_performance[cap_id]
            
            # 更新統計
            stats["total_executions"] += 1
            
            # 檢查這個能力是否成功
            cap_name = cap["metadata"].get("capability_name", "")
            was_successful = any(
                cap_name in cmd_id 
                for cmd_id in result.completed_commands
            )
            
            if was_successful:
                stats["successful_executions"] += 1
            else:
                stats["failed_executions"] += 1
            
            # 更新成功率
            stats["success_rate"] = (
                stats["successful_executions"] / stats["total_executions"]
            )
            
            # 更新延遲 (簡化版)
            if result.total_duration:
                avg_per_cmd = (result.total_duration * 1000) / len(plan.commands)
                stats["total_latency_ms"] += avg_per_cmd
                stats["avg_latency_ms"] = (
                    stats["total_latency_ms"] / stats["total_executions"]
                )
            
            logger.debug(f"   Updated {cap_id}: "
                        f"success_rate={stats['success_rate']:.2f}, "
                        f"avg_latency={stats['avg_latency_ms']:.0f}ms")
        
        logger.info("✅ Learning completed, performance stats updated")


# ==================== 輔助功能 ====================

async def quick_plan_and_execute(
    task_type: str,
    target: str,
    objectives: List[str],
    **kwargs
) -> Tuple[CapabilityPlan, ExecutionResult]:
    """快速計劃和執行
    
    便捷函數,用於快速測試和演示
    
    Args:
        task_type: 任務類型
        target: 目標
        objectives: 目標列表
        **kwargs: 其他參數
        
    Returns:
        (計劃, 執行結果)
    """
    orchestrator = CapabilityOrchestrator()
    
    requirement = TaskRequirement(
        task_id=f"task_{uuid4().hex[:8]}",
        task_type=task_type,
        target=target,
        objectives=objectives,
        constraints=kwargs
    )
    
    plan = await orchestrator.plan(requirement)
    result = await orchestrator.execute(plan)
    
    orchestrator.learn_from_execution(plan, result)
    
    return plan, result


# ==================== 導出 ====================

__all__ = [
    "CapabilityOrchestrator",
    "TaskRequirement",
    "CapabilityPlan",
    "ExecutionResult",
    "quick_plan_and_execute",
]
