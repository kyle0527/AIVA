"""
BioNeuron Core - 生物啟發式神經網路決策核心
來源: BioNeuronAI/src/bioneuronai/scalable_architecture.py
來源: BioNeuronAI/src/bioneuronai/improved_core.py

這個模組實現了一個可擴展的生物啟發式神經網路 (500萬參數規模)
用於 AI 代理的決策核心,包含 RAG 功能和抗幻覺機制

新增功能：
- 攻擊計畫執行器 (Planner)
- 任務執行追蹤 (Tracer)
- AST 與 Trace 對比分析
- 經驗學習機制
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# --- 核心神經網路元件 ---


class BiologicalSpikingLayer:
    """模擬生物尖峰神經元行為的神經層 (整合 v2 改進 + 優化版本)."""

    def __init__(self, input_size: int, output_size: int) -> None:
        """初始化尖峰神經層.

        Args:
            input_size: 輸入維度
            output_size: 輸出維度
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.threshold = 1.0  # 尖峰閾值
        self.refractory_period = 0.05  # 優化: 減少不反應期，提升響應速度
        self.last_spike_time = np.zeros(output_size) - self.refractory_period
        self.params = input_size * output_size
        
        # 新增: 自適應閾值機制
        self.adaptive_threshold = True
        self.threshold_decay = 0.98
        self.min_threshold = 0.3

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向傳播,產生尖峰訊號 (使用 v2 改進的向量化操作).

        Args:
            x: 輸入訊號

        Returns:
            尖峰輸出 (0 或 1)
        """
        current_time = time.time()
        potential = np.dot(x, self.weights)
        
        # 自適應閾值調整
        if self.adaptive_threshold:
            self.threshold = max(self.min_threshold, self.threshold * self.threshold_decay)
        
        can_spike = (current_time - self.last_spike_time) > self.refractory_period
        spikes = (potential > self.threshold) & can_spike

        self.last_spike_time[spikes] = current_time
        return spikes.astype(np.float32)
    
    def forward_batch(self, x_batch: np.ndarray) -> np.ndarray:
        """批次處理優化，提升並發能力.
        
        Args:
            x_batch: 批次輸入 (batch_size, input_size)
            
        Returns:
            批次尖峰輸出 (batch_size, output_size)
        """
        current_time = time.time()
        potentials = np.dot(x_batch, self.weights)
        
        # 自適應閾值調整
        if self.adaptive_threshold:
            self.threshold = max(self.min_threshold, self.threshold * self.threshold_decay)
            
        # 向量化尖峰檢測
        can_spike = (current_time - self.last_spike_time) > self.refractory_period
        spikes = (potentials > self.threshold) & can_spike
        
        # 更新尖峰時間 (只更新有尖峰的神經元)
        spike_mask = np.any(spikes, axis=0)
        self.last_spike_time[spike_mask] = current_time
            
        return spikes.astype(np.float32)


class AntiHallucinationModule:
    """抗幻覺模組,用於評估決策的信心度 (整合 v2 改進 + 多層驗證)."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        """初始化抗幻覺模組.

        Args:
            confidence_threshold: 信心度閾值
        """
        self.confidence_threshold = confidence_threshold
        # 新增: 多層驗證機制
        self.validation_layers = 3
        self.consensus_threshold = 0.6
        self.validation_history = []

    def check_confidence(
        self, decision_potential: np.ndarray, threshold: float = None
    ) -> tuple[bool, float]:
        """檢查決策的信心度是否足夠 (基礎版本，保持向後相容).

        Args:
            decision_potential: 決策潛力向量
            threshold: 信心度閾值 (可選，使用實例預設值)

        Returns:
            (是否有信心, 信心度分數)
        """
        if threshold is None:
            threshold = self.confidence_threshold
            
        confidence = float(np.max(decision_potential))
        is_confident = confidence >= threshold
        if not is_confident:
            logger.warning(
                f"[Anti-Hallucination] 決策信心度不足 "
                f"({confidence:.2f} < {threshold})，建議請求確認。"
            )
        return is_confident, confidence

    def multi_layer_validation(self, decision_potential: np.ndarray) -> tuple[bool, float, dict]:
        """多層次信心度驗證 (新增優化功能).
        
        Args:
            decision_potential: 決策潛力向量
            
        Returns:
            (是否通過驗證, 最終信心度, 詳細分析)
        """
        validations = []
        
        # 第一層: 基本信心度
        basic_confidence = float(np.max(decision_potential))
        validations.append(basic_confidence)
        
        # 第二層: 穩定性檢查 (標準差相對於平均值)
        mean_val = np.mean(decision_potential)
        std_val = np.std(decision_potential)
        stability = 1.0 - (std_val / max(mean_val, 0.001))  # 避免除零
        stability = max(0.0, min(1.0, stability))  # 限制在 [0,1] 範圍
        validations.append(stability)
        
        # 第三層: 一致性檢查 (高信心選項比例)
        high_confidence_ratio = float(np.sum(decision_potential > 0.5) / len(decision_potential))
        validations.append(high_confidence_ratio)
        
        # 綜合評估
        final_confidence = np.mean(validations)
        consensus_reached = sum(v > self.consensus_threshold for v in validations) >= 2
        
        # 記錄驗證歷史
        validation_record = {
            'basic_confidence': basic_confidence,
            'stability': stability,
            'consistency': high_confidence_ratio,
            'final_confidence': final_confidence,
            'consensus_reached': consensus_reached,
            'timestamp': time.time()
        }
        self.validation_history.append(validation_record)
        
        # 保持歷史記錄在合理範圍內
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-50:]
        
        is_valid = consensus_reached and (final_confidence >= self.confidence_threshold)
        
        return is_valid, final_confidence, validation_record

    def check(self, decision_logits: np.ndarray) -> tuple[bool, float]:
        """檢查決策的信心度 (v2 相容方法).

        Args:
            decision_logits: 決策邏輯值

        Returns:
            (是否通過檢查, 信心度分數)
        """
        return self.check_confidence(decision_logits)
    
    def get_validation_stats(self) -> dict:
        """獲取驗證統計信息.
        
        Returns:
            驗證統計數據
        """
        if not self.validation_history:
            return {'total_validations': 0}
            
        recent_validations = self.validation_history[-20:]  # 最近 20 次
        
        return {
            'total_validations': len(self.validation_history),
            'recent_average_confidence': np.mean([v['final_confidence'] for v in recent_validations]),
            'recent_consensus_rate': np.mean([v['consensus_reached'] for v in recent_validations]),
            'recent_stability_score': np.mean([v['stability'] for v in recent_validations])
        }


class ScalableBioNet:
    """
    可擴展的生物啟發式神經網路 - 500萬參數規模.

    這是 AI 代理的「決策核心」
    """

    def __init__(self, input_size: int, num_tools: int) -> None:
        """初始化決策網路.

        Args:
            input_size: 輸入向量大小
            num_tools: 可用工具數量
        """
        # EXTRA_LARGE (5M 參數) 配置
        # 這是為了達到約 500 萬參數目標的設計
        self.hidden_size_1 = 2048
        self.hidden_size_2 = 1024

        # 層定義
        self.fc1 = np.random.randn(input_size, self.hidden_size_1)
        self.spiking1 = BiologicalSpikingLayer(self.hidden_size_1, self.hidden_size_2)
        self.fc2 = np.random.randn(self.hidden_size_2, num_tools)

        # 參數計算
        self.params_fc1 = input_size * self.hidden_size_1
        self.params_spiking1 = self.spiking1.params
        self.params_fc2 = self.hidden_size_2 * num_tools
        self.total_params = self.params_fc1 + self.params_spiking1 + self.params_fc2

        logger.info("--- ScalableBioNet (決策核心) 初始化 ---")
        logger.info(f"  - FC1 參數: {self.params_fc1:,}")
        logger.info(f"  - Spiking1 參數: {self.params_spiking1:,}")
        logger.info(f"  - FC2 參數: {self.params_fc2:,}")
        logger.info(f"  - 總參數約: {self.total_params / 1_000_000:.2f}M")
        logger.info("-" * 41)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向傳播,產生決策潛力.

        Args:
            x: 輸入向量

        Returns:
            決策機率分布
        """
        x = np.tanh(x @ self.fc1)
        x = self.spiking1.forward(x)
        decision_potential = x @ self.fc2
        return self._softmax(decision_potential)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax 激活函數.

        Args:
            x: 輸入向量

        Returns:
            機率分布
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class BioNeuronRAGAgent:
    """
    具備 RAG 功能的 BioNeuron AI 代理.

    結合檢索增強生成 (RAG) 與生物啟發式決策核心，
    並整合攻擊計畫執行、追蹤記錄和經驗學習能力。
    """

    def __init__(
        self,
        codebase_path: str,
        enable_planner: bool = True,
        enable_tracer: bool = True,
        enable_experience: bool = True,
        database_url: str = "sqlite:///./aiva_experience.db",
    ) -> None:
        """初始化 RAG 代理.

        Args:
            codebase_path: 程式碼庫路徑
            enable_planner: 是否啟用計畫執行器
            enable_tracer: 是否啟用執行追蹤
            enable_experience: 是否啟用經驗學習
            database_url: 經驗資料庫連接字串
        """
        logger.info("正在初始化 BioNeuronRAGAgent...")

        # 基本配置
        self.codebase_path = codebase_path
        self.enable_planner = enable_planner
        self.enable_tracer = enable_tracer
        self.enable_experience = enable_experience

        # 注意: 這裡需要實現 KnowledgeBase, Tool, CodeReader, CodeWriter
        # 目前暫時使用 mock 實作
        self.tools: list[dict[str, str]] = [
            {"name": "CodeReader"},
            {"name": "CodeWriter"},
        ]
        self.tool_map: dict[str, dict[str, str]] = {
            tool["name"]: tool for tool in self.tools
        }

        # RAG 檢索的上下文會被嵌入,需要一個固定的向量大小
        # 假設嵌入向量大小 + 任務向量大小 = 1024
        self.input_vector_size = 1024

        self.decision_core = ScalableBioNet(self.input_vector_size, len(self.tools))
        self.anti_hallucination = AntiHallucinationModule()
        self.history: list[dict[str, Any]] = []

        # 新增：攻擊計畫執行器
        if enable_planner:
            from ..planner.orchestrator import AttackOrchestrator

            self.orchestrator = AttackOrchestrator()
            logger.info("✓ Planner/Orchestrator enabled")
        else:
            self.orchestrator = None

        # 新增：執行監控與追蹤
        if enable_tracer:
            from ..execution_tracer.execution_monitor import ExecutionMonitor
            from ..execution_tracer.task_executor import TaskExecutor

            self.execution_monitor = ExecutionMonitor()
            self.task_executor = TaskExecutor(self.execution_monitor)
            logger.info("✓ Execution Tracer enabled")
        else:
            self.execution_monitor = None
            self.task_executor = None

        # 新增：經驗資料庫和對比分析
        if enable_experience:
            try:
                # 延遲導入以避免循環依賴
                from pathlib import Path
                import sys

                # 添加 integration 路徑
                integration_path = (
                    Path(__file__).parent.parent.parent.parent.parent / "integration"
                )
                if str(integration_path) not in sys.path:
                    sys.path.insert(0, str(integration_path))

                from aiva_integration.reception.experience_repository import (
                    ExperienceRepository,
                )

                from ..analysis.ast_trace_comparator import ASTTraceComparator
                from .training.model_updater import ModelUpdater

                self.experience_repo = ExperienceRepository(database_url)
                self.comparator = ASTTraceComparator()
                self.model_updater = ModelUpdater(
                    self.decision_core, self.experience_repo
                )
                logger.info(f"✓ Experience learning enabled (DB: {database_url})")
            except Exception as e:
                logger.warning(f"Failed to enable experience learning: {e}")
                self.experience_repo = None
                self.comparator = None
                self.model_updater = None
        else:
            self.experience_repo = None
            self.comparator = None
            self.model_updater = None

        logger.info("BioNeuronRAGAgent 初始化完成 ✓")

    def _create_input_vector(self, task: str, context: str) -> np.ndarray:
        """將任務和上下文轉換為輸入向量.

        Args:
            task: 任務描述
            context: 上下文資訊

        Returns:
            輸入向量
        """
        # 這是一個簡化的嵌入過程
        # 在實際應用中,會使用真正的 embedding model (如 SentenceTransformer)
        task_hash = np.array([ord(c) for c in task], dtype=np.float32)
        context_hash = np.array([ord(c) for c in context], dtype=np.float32)

        task_vec = np.pad(
            task_hash,
            (0, self.input_vector_size // 2 - len(task_hash)),
            "constant",
        )
        context_vec = np.pad(
            context_hash,
            (0, self.input_vector_size // 2 - len(context_hash)),
            "constant",
        )

        combined_vec = np.concatenate([task_vec, context_vec])
        return combined_vec / np.linalg.norm(combined_vec)  # 正規化

    def invoke(self, task: str) -> dict[str, Any]:
        """
        執行一個任務,包含完整的 RAG 流程.

        Args:
            task: 任務描述

        Returns:
            執行結果字典
        """
        logger.info(f"\n--- 開始新任務: {task} ---")

        # 1. RAG - 檢索 (Retrieve)
        logger.info("1. [檢索] 正在從知識庫中搜尋相關上下文...")
        # TODO: 實作實際的知識庫檢索
        context_str = ""

        # 2. RAG - 增強 (Augment)
        logger.info("2. [增強] 正在結合任務與上下文...")
        input_vector = self._create_input_vector(task, context_str)

        # 3. 決策 (Decision Making)
        logger.info("3. [決策] BioNeuronAI 決策核心正在思考...")
        decision_potential = self.decision_core.forward(input_vector)

        # 4. 可靠性檢查 (Anti-Hallucination)
        is_confident, confidence = self.anti_hallucination.check_confidence(
            decision_potential
        )
        if not is_confident:
            return {
                "status": "uncertain",
                "message": (
                    f"我對下一步操作的信心度 ({confidence:.2f}) 不足，"
                    "需要您提供更多資訊或確認。"
                ),
                "confidence": confidence,
            }

        # 5. 選擇工具並執行
        chosen_tool_index = int(np.argmax(decision_potential))
        chosen_tool = self.tools[chosen_tool_index]
        tool_confidence = float(decision_potential[chosen_tool_index])
        logger.info(
            f"4. [執行] 選擇工具: '{chosen_tool['name']}' "
            f"(信心度: {tool_confidence:.2f})"
        )

        logger.info("5. [完成] 任務步驟執行完畢。")

        response = {
            "status": "success",
            "tool_used": chosen_tool["name"],
            "confidence": tool_confidence,
            "result": "執行成功 (Mock)",
        }
        self.history.append(response)
        return response

    def get_knowledge_stats(self) -> dict[str, int]:
        """獲取知識庫統計信息"""
        stats = {
            "total_chunks": 1279,  # 模擬知識庫塊數
            "total_keywords": 997,  # 模擬關鍵詞數
            "indexed_files": 156,  # 模擬已索引檔案數
            "ai_tools": len(self.tools),  # AI 工具數量
        }

        # 如果啟用經驗學習，添加經驗統計
        if self.experience_repo:
            try:
                exp_stats = self.experience_repo.get_statistics()
                stats["total_experiences"] = exp_stats["total_experiences"]
                stats["avg_experience_score"] = exp_stats["average_score"]
            except Exception as e:
                logger.warning(f"Failed to get experience stats: {e}")

        return stats

    async def execute_attack_plan(
        self, attack_plan_ast: dict[str, Any]
    ) -> dict[str, Any]:
        """執行攻擊計畫（新增方法）

        Args:
            attack_plan_ast: 攻擊計畫 AST 字典

        Returns:
            執行結果，包含 trace 和比較指標
        """
        if not self.orchestrator:
            return {
                "status": "error",
                "message": "Planner not enabled",
            }

        logger.info("開始執行攻擊計畫...")

        try:
            # 1. 創建執行計畫
            execution_plan = self.orchestrator.create_execution_plan(attack_plan_ast)

            # 2. 開始監控
            if self.execution_monitor:
                trace = self.execution_monitor.start_monitoring(execution_plan)
                trace_session_id = trace.trace_session_id
            else:
                trace_session_id = "mock_trace"

            # 3. 執行任務
            executed_tasks = []
            while not self.orchestrator.is_plan_complete(execution_plan):
                # 獲取可執行任務
                next_tasks = self.orchestrator.get_next_executable_tasks(execution_plan)

                if not next_tasks:
                    break

                # 執行每個任務
                for task, tool_decision in next_tasks:
                    if self.task_executor:
                        result = await self.task_executor.execute_task(
                            task, tool_decision, trace_session_id
                        )

                        # 更新任務狀態
                        from ..planner.task_converter import TaskStatus

                        status = (
                            TaskStatus.SUCCESS if result.success else TaskStatus.FAILED
                        )
                        self.orchestrator.update_task_status(
                            execution_plan,
                            task.task_id,
                            status,
                            result.output,
                            result.error,
                        )

                        executed_tasks.append(
                            {
                                "task_id": task.task_id,
                                "success": result.success,
                                "output": result.output,
                            }
                        )

            # 4. 結束監控
            if self.execution_monitor:
                final_trace = self.execution_monitor.finalize_monitoring(
                    trace_session_id
                )
            else:
                final_trace = None

            # 5. 對比分析
            if self.comparator and final_trace:
                metrics = self.comparator.compare(execution_plan.graph, final_trace)
                feedback = self.comparator.generate_feedback(metrics)

                # 6. 保存經驗
                if self.experience_repo:
                    self.experience_repo.save_experience(
                        plan_id=execution_plan.plan_id,
                        attack_type=execution_plan.metadata.get(
                            "attack_type", "unknown"
                        ),
                        ast_graph=attack_plan_ast,
                        execution_trace=final_trace.to_dict(),
                        metrics=metrics.to_dict(),
                        feedback=feedback,
                        target_info=execution_plan.graph.metadata.get("target", {}),
                    )

                return {
                    "status": "success",
                    "plan_id": execution_plan.plan_id,
                    "executed_tasks": executed_tasks,
                    "trace_session_id": trace_session_id,
                    "metrics": metrics.to_dict(),
                    "feedback": feedback,
                    "plan_complete": True,
                }
            else:
                return {
                    "status": "success",
                    "plan_id": execution_plan.plan_id,
                    "executed_tasks": executed_tasks,
                    "plan_complete": True,
                }

        except Exception as e:
            logger.error(f"Attack plan execution failed: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def train_from_experiences(
        self, min_score: float = 0.6, max_samples: int = 1000
    ) -> dict[str, Any]:
        """從經驗庫訓練模型（新增方法）

        Args:
            min_score: 最低分數閾值
            max_samples: 最大樣本數

        Returns:
            訓練結果
        """
        if not self.model_updater:
            return {
                "status": "error",
                "message": "Experience learning not enabled",
            }

        logger.info("開始從經驗庫訓練模型...")

        try:
            result = self.model_updater.update_from_recent_experiences(
                min_score=min_score, max_samples=max_samples
            )
            return result
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}
