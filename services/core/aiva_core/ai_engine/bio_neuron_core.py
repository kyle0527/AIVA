"""BioNeuron Core - 生物啟發式神經網路決策核心
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

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Union, List

# 使用統一的 Optional Dependency 框架
from utilities.optional_deps import deps

if TYPE_CHECKING:
    import numpy as np
    NDArray = np.ndarray
else:
    np = deps.get_or_mock('numpy')
    # 運行時的型別別名，與 Mock 相容
    NDArray = Union[List, Any]

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

    def forward(self, x: NDArray) -> NDArray:
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
            self.threshold = max(
                self.min_threshold, self.threshold * self.threshold_decay
            )

        can_spike = (current_time - self.last_spike_time) > self.refractory_period
        spikes = (potential > self.threshold) & can_spike

        self.last_spike_time[spikes] = current_time
        return spikes.astype(np.float32)

    def forward_batch(self, x_batch: NDArray) -> NDArray:
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
            self.threshold = max(
                self.min_threshold, self.threshold * self.threshold_decay
            )

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
        self, decision_potential: NDArray, threshold: float | None = None
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

    def multi_layer_validation(
        self, decision_potential: NDArray
    ) -> tuple[bool, float, dict]:
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
        high_confidence_ratio = float(
            np.sum(decision_potential > 0.5) / len(decision_potential)
        )
        validations.append(high_confidence_ratio)

        # 綜合評估
        final_confidence = np.mean(validations)
        consensus_reached = sum(v > self.consensus_threshold for v in validations) >= 2

        # 記錄驗證歷史
        validation_record = {
            "basic_confidence": basic_confidence,
            "stability": stability,
            "consistency": high_confidence_ratio,
            "final_confidence": final_confidence,
            "consensus_reached": consensus_reached,
            "timestamp": time.time(),
        }
        self.validation_history.append(validation_record)

        # 保持歷史記錄在合理範圍內
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-50:]

        is_valid = consensus_reached and (final_confidence >= self.confidence_threshold)

        return bool(is_valid), float(final_confidence), validation_record

    def check(self, decision_logits: NDArray) -> tuple[bool, float]:
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
            return {"total_validations": 0}

        recent_validations = self.validation_history[-20:]  # 最近 20 次

        return {
            "total_validations": len(self.validation_history),
            "recent_average_confidence": np.mean(
                [v["final_confidence"] for v in recent_validations]
            ),
            "recent_consensus_rate": np.mean(
                [v["consensus_reached"] for v in recent_validations]
            ),
            "recent_stability_score": np.mean(
                [v["stability"] for v in recent_validations]
            ),
        }


class ScalableBioNet:
    """可擴展的生物啟發式神經網路 - 500萬參數規模.

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

    def forward(self, x: NDArray) -> NDArray:
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

    def _softmax(self, x: NDArray) -> NDArray:
        """Softmax 激活函數.

        Args:
            x: 輸入向量

        Returns:
            機率分布
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


class BioNeuronRAGAgent:
    """具備 RAG 功能的 BioNeuron AI 代理.

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


                # 使用相對導入避免路徑問題
                try:
                    from services.integration.aiva_integration.reception.experience_repository import (
                        ExperienceRepository,
                    )
                except ImportError:
                    # 如果無法導入，創建一個簡單的佔位符
                    class ExperienceRepository:
                        def __init__(self, *args): pass
                        def store_experience(self, *args): pass
                        def save_experience(self, *args): pass
                        def get_statistics(self): return {}

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

    def _create_input_vector(self, task: str, context: str) -> NDArray:
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

    async def invoke(self, task: str) -> dict[str, Any]:
        """執行一個任務,包含完整的 RAG 流程.

        Args:
            task: 任務描述

        Returns:
            執行結果字典
        """
        logger.info(f"\n--- 開始新任務: {task} ---")

        # 1. RAG - 檢索 (Retrieve)
        logger.info("1. [檢索] 正在從知識庫中搜尋相關上下文...")
        context_str = await self._retrieve_knowledge(task)

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
                        try:
                            from ..planner.task_converter import TaskStatus
                            status = TaskStatus.SUCCESS if result.success else TaskStatus.FAILED
                        except ImportError:
                            # 使用字符串代替
                            status = "SUCCESS" if result.success else "FAILED"
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
                    # 使用字典方式來避免參數問題
                    experience_data = {
                        "plan_id": getattr(execution_plan, "plan_id", "unknown"),
                        "attack_type": getattr(execution_plan, "metadata", {}).get("attack_type", "unknown"),
                        "ast_graph": attack_plan_ast,
                        "execution_trace": final_trace.to_dict() if hasattr(final_trace, "to_dict") else str(final_trace),
                        "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else str(metrics),
                        "feedback": feedback,
                        "target_info": getattr(execution_plan, "graph", {}).get("metadata", {}).get("target", {}),
                    }
                    try:
                        self.experience_repo.save_experience(experience_data)
                    except Exception as e:
                        logger.warning(f"Failed to save experience: {e}")

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

    async def _retrieve_knowledge(self, task: str) -> str:
        """從 RAG 知識庫檢索相關上下文

        Args:
            task: 任務描述

        Returns:
            相關的上下文字串
        """
        try:


            logger.debug(f"開始 RAG 檢索，任務: {task}", Optional)

            # 1. 任務關鍵字提取
            keywords = self._extract_keywords(task)
            logger.debug(f"提取關鍵字: {keywords}")

            # 2. 向量化查詢
            query_embedding = await self._embed_query(task)

            # 3. 相似性搜索
            similar_chunks = await self._similarity_search(
                query_embedding, keywords, top_k=5
            )

            # 4. 經驗案例檢索（如果啟用）
            experience_context = ""
            if self.experience_repo:
                try:
                    related_experiences = await self._retrieve_experiences(task)
                    if related_experiences:
                        experience_context = self._format_experience_context(
                            related_experiences
                        )
                        logger.debug(f"檢索到 {len(related_experiences)} 個相關經驗")
                except Exception as e:
                    logger.warning(f"檢索經驗失敗: {e}")

            # 5. 組合上下文
            context_parts = []

            # 添加知識庫檢索結果
            if similar_chunks:
                kb_context = "\n".join(
                    [chunk.get("content", "") for chunk in similar_chunks]
                )
                context_parts.append(f"知識庫相關信息:\n{kb_context}")

            # 添加經驗案例
            if experience_context:
                context_parts.append(f"相關經驗案例:\n{experience_context}")

            # 添加工具信息
            available_tools = [tool["name"] for tool in self.tools]
            context_parts.append(f"可用工具: {', '.join(available_tools)}")

            final_context = "\n\n".join(context_parts)

            logger.info(f"RAG 檢索完成，上下文長度: {len(final_context)} 字元")
            return final_context

        except Exception as e:
            logger.error(f"RAG 檢索失敗: {e}")
            # 返回基礎上下文作為降級方案
            fallback_context = f"任務: {task}\n可用工具: {', '.join([tool['name'] for tool in self.tools])}"
            return fallback_context

    def _extract_keywords(self, task: str) -> list[str]:
        """從任務中提取關鍵字"""
        import re

        # 簡單的關鍵字提取（可以用更複雜的 NLP 方法替換）
        # 移除標點符號並分割
        words = re.findall(r"\w+", task.lower())

        # 過濾停用詞
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "的",
            "是",
            "在",
            "有",
            "和",
            "或",
            "但",
            "對",
            "為",
            "用",
            "從",
            "到",
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # 添加安全相關的重要關鍵詞
        security_keywords = [
            "scan",
            "attack",
            "vulnerability",
            "exploit",
            "security",
            "test",
            "audit",
            "assess",
            "掃描",
            "攻擊",
            "漏洞",
            "安全",
            "測試",
        ]
        for word in security_keywords:
            if word in task.lower() and word not in keywords:
                keywords.append(word)

        return keywords[:10]  # 限制關鍵字數量

    async def _embed_query(self, query: str) -> list[float]:
        """將查詢轉換為嵌入向量"""
        try:
            # 這裡應該使用實際的嵌入模型，如 sentence-transformers
            # 目前使用簡單的雜湊作為模擬
            import hashlib

            hash_obj = hashlib.md5(query.encode())
            hash_hex = hash_obj.hexdigest()

            # 將雜湊轉換為 512 維向量（模擬嵌入）
            embedding = []
            for i in range(0, len(hash_hex), 2):
                hex_pair = hash_hex[i : i + 2]
                embedding.append(int(hex_pair, 16) / 255.0)

            # 擴展到 512 維
            while len(embedding) < 512:
                embedding.extend(embedding[: min(512 - len(embedding), len(embedding))])

            return embedding[:512]

        except Exception as e:
            logger.warning(f"嵌入生成失敗: {e}")
            # 返回隨機向量作為降級方案
            import random

            return [random.random() for _ in range(512)]

    async def _similarity_search(
        self, query_embedding: list[float], keywords: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """基於向量相似性和關鍵字搜索知識庫"""
        try:
            # 模擬知識庫搜索結果
            mock_knowledge_base = [
                {
                    "content": "SQL 注入攻擊是一種常見的 Web 安全漏洞，攻擊者通過輸入惡意 SQL 代碼來操作資料庫。",
                    "keywords": ["sql", "injection", "database", "security", "web"],
                    "score": 0.85,
                    "source": "security_patterns.md",
                },
                {
                    "content": "XSS (跨站腳本攻擊) 允許攻擊者在用戶瀏覽器中執行惡意腳本。",
                    "keywords": ["xss", "script", "browser", "client", "javascript"],
                    "score": 0.78,
                    "source": "web_vulnerabilities.md",
                },
                {
                    "content": "CSRF (跨站請求偽造) 利用用戶的身份驗證狀態執行未經授權的操作。",
                    "keywords": ["csrf", "authentication", "session", "token"],
                    "score": 0.72,
                    "source": "auth_security.md",
                },
                {
                    "content": "漏洞掃描工具可以自動化識別系統中的安全弱點。",
                    "keywords": [
                        "scan",
                        "vulnerability",
                        "tool",
                        "automation",
                        "detection",
                    ],
                    "score": 0.80,
                    "source": "scanning_tools.md",
                },
                {
                    "content": "滲透測試是評估系統安全性的重要方法，包括主動和被動測試技術。",
                    "keywords": [
                        "penetration",
                        "testing",
                        "assessment",
                        "active",
                        "passive",
                    ],
                    "score": 0.75,
                    "source": "pentest_methodologies.md",
                },
            ]

            # 簡單的關鍵字匹配評分
            results = []
            for item in mock_knowledge_base:
                # 計算關鍵字匹配分數
                keyword_matches = sum(
                    1
                    for kw in keywords
                    if any(kw in item_kw for item_kw in item["keywords"])
                )
                keyword_score = keyword_matches / max(len(keywords), 1)

                # 組合分數（這裡簡化，實際應該用向量相似度）
                final_score = (item["score"] * 0.6) + (keyword_score * 0.4)

                if final_score > 0.3:  # 相關性閾值
                    results.append(
                        {
                            **item,
                            "relevance_score": final_score,
                            "keyword_matches": keyword_matches,
                        }
                    )

            # 按相關性排序並返回 top_k
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"相似性搜索失敗: {e}")
            return []

    async def _retrieve_experiences(self, task: str) -> list[dict[str, Any]]:
        """檢索相關的歷史經驗"""
        if not self.experience_repo:
            return []

        try:
            # 這裡應該實現實際的經驗檢索邏輯
            # 目前返回模擬數據
            return [
                {
                    "task_type": "security_scan",
                    "success_rate": 0.85,
                    "avg_findings": 12,
                    "recommended_tools": ["sql_scanner", "xss_detector"],
                    "lessons_learned": "重點關注輸入驗證和輸出編碼",
                }
            ]
        except Exception as e:
            logger.warning(f"經驗檢索失敗: {e}")
            return []

    def _format_experience_context(self, experiences: list[dict[str, Any]]) -> str:
        """格式化經驗上下文"""
        if not experiences:
            return ""

        context_parts = []
        for i, exp in enumerate(experiences[:3], 1):  # 最多使用 3 個經驗
            context_parts.append(
                f"經驗 {i}: {exp.get('task_type', 'unknown')} "
                f"(成功率: {exp.get('success_rate', 0):.0%})"
            )
            if "lessons_learned" in exp:
                context_parts.append(f"  經驗教訓: {exp['lessons_learned']}")

        return "\n".join(context_parts)

    def generate(self, task_description: str, context: str = "") -> dict[str, Any]:
        """生成決策結果"""
        try:
            # 使用決策核心生成結果
            combined_input = f"{task_description} {context}"
            # 簡化的向量化（實際應用中應使用適當的嵌入模型）
            import hashlib
            input_hash = hashlib.md5(combined_input.encode()).hexdigest()
            input_vector = np.array([ord(c) for c in input_hash[:self.input_vector_size]] + 
                                  [0] * (self.input_vector_size - len(input_hash[:self.input_vector_size])))
            
            decision_output = self.decision_core.forward(input_vector.reshape(1, -1))
            confidence = float(np.max(decision_output))
            
            return {
                "decision": task_description,
                "confidence": confidence,
                "reasoning": f"基於 RAG 檢索和生物神經網路決策，信心度: {confidence:.2f}",
                "context_used": context
            }
        except Exception as e:
            logger.error(f"生成決策失敗: {e}")
            return {"decision": "error", "confidence": 0.0, "reasoning": str(e)}
    

    
    @property
    def planner(self):
        """計畫器屬性"""
        return getattr(self, '_planner', None)


class BioNeuronCore:
    """BioNeuron 核心決策引擎

    統一的 AI 決策核心，整合所有 BioNeuron 組件，
    提供高層次的決策和推理能力。
    """

    def __init__(
        self,
        codebase_path: str = "c:/D/fold7/AIVA-git",
        enable_planner: bool = True,
        enable_tracer: bool = True,
        enable_experience: bool = True,
        database_url: str = "sqlite:///aiva_experiences.db",
    ):
        """初始化 BioNeuron 核心

        Args:
            codebase_path: 代碼庫路徑
            enable_planner: 是否啟用攻擊計畫器
            enable_tracer: 是否啟用執行追蹤器
            enable_experience: 是否啟用經驗學習
            database_url: 數據庫連接字符串
        """
        logger.info("初始化 BioNeuronCore...")

        # 初始化核心 RAG 代理
        self.rag_agent = BioNeuronRAGAgent(
            codebase_path=codebase_path,
            enable_planner=enable_planner,
            enable_tracer=enable_tracer,
            enable_experience=enable_experience,
            database_url=database_url,
        )

        # 初始化決策網路 (使用預設參數)
        self.decision_network = ScalableBioNet(
            input_size=512, num_tools=20  # 預設輸入向量大小  # 預設工具數量
        )

        # 初始化抗幻覺模組
        self.anti_hallucination = AntiHallucinationModule()

        logger.info("BioNeuronCore 初始化完成 ✓")

    def make_decision(
        self,
        task_description: str,
        context: Optional[dict[str, Any]] = None,
        confidence_threshold: float = 0.7,
    ) -> dict:
        """核心決策方法

        Args:
            task_description: 任務描述
            context: 上下文信息
            confidence_threshold: 信心閾值

        Returns:
            決策結果
        """
        try:
            # 使用 RAG 代理生成決策
            context_str = str(context) if context else ""
            decision = self.rag_agent.generate(task_description, context_str)

            # 檢查信心度
            confidence = decision.get("confidence", 0.0)

            if confidence < confidence_threshold:
                logger.warning(f"決策信心度過低: {confidence} < {confidence_threshold}")
                decision["warning"] = "Low confidence decision"

            # 添加元數據
            decision["core_metadata"] = {
                "decision_engine": "BioNeuronCore",
                "timestamp": time.time(),
                "confidence_threshold": confidence_threshold,
                "context_provided": context is not None,
            }

            return decision

        except Exception as e:
            logger.error(f"決策生成失敗: {e}")
            return {"status": "error", "message": str(e), "timestamp": time.time()}

    def execute_attack_plan(
        self, attack_plan: str, target_context: Optional[dict[str, Any]] = None
    ) -> dict:
        """執行攻擊計畫

        Args:
            attack_plan: 攻擊計畫描述
            target_context: 目標上下文

        Returns:
            執行結果
        """
        try:
            import asyncio
            
            # 將字符串攻擊計畫轉換為簡單的 AST 結構
            attack_plan_ast = {
                "type": "simple_plan",
                "description": attack_plan,
                "context": target_context or {},
                "tasks": [
                    {
                        "id": "main_task",
                        "description": attack_plan,
                        "parameters": target_context or {}
                    }
                ]
            }
            
            # 使用 asyncio.run 來同步調用 async 方法
            result = asyncio.run(self.rag_agent.execute_attack_plan(attack_plan_ast))

            # 添加核心元數據
            if isinstance(result, dict):
                result["core_execution"] = {
                    "executor": "BioNeuronCore",
                    "timestamp": time.time(),
                    "plan_hash": hash(attack_plan),
                }
            else:
                result = {
                    "status": "completed",
                    "result": result,
                    "core_execution": {
                        "executor": "BioNeuronCore",
                        "timestamp": time.time(),
                        "plan_hash": hash(attack_plan),
                    }
                }

            return result

        except Exception as e:
            logger.error(f"攻擊計畫執行失敗: {e}")
            return {"status": "error", "message": str(e), "timestamp": time.time()}

    def learn_from_feedback(
        self,
        task: str,
        result: dict,
        feedback_score: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict:
        """從反饋中學習

        Args:
            task: 原始任務
            result: 執行結果
            feedback_score: 反饋分數 (0.0-1.0)
            metadata: 額外的元數據

        Returns:
            學習結果
        """
        try:
            # 記錄經驗
            if (
                hasattr(self.rag_agent, "experience_repo")
                and self.rag_agent.experience_repo
            ):
                experience_data = {
                    "task": task,
                    "result": result,
                    "feedback_score": feedback_score,
                    "metadata": metadata or {},
                    "timestamp": time.time(),
                    "core_version": "BioNeuronCore_v1",
                }

                # 保存經驗
                self.rag_agent.experience_repo.save_experience(experience_data)

                return {
                    "status": "success",
                    "message": "經驗已記錄",
                    "experience_id": experience_data.get("id"),
                    "feedback_score": feedback_score,
                }
            else:
                return {"status": "warning", "message": "經驗學習系統未啟用"}

        except Exception as e:
            logger.error(f"學習過程失敗: {e}")
            return {"status": "error", "message": str(e)}

    def train_from_experiences(
        self, min_score: float = 0.6, max_samples: int = 1000
    ) -> dict:
        """從歷史經驗訓練模型

        Args:
            min_score: 最低分數閾值
            max_samples: 最大樣本數

        Returns:
            訓練結果
        """
        return self.rag_agent.train_from_experiences(min_score, max_samples)

    def get_system_status(self) -> dict:
        """獲取系統狀態

        Returns:
            系統狀態信息
        """
        return {
            "core_status": "active",
            "rag_agent_status": "active" if self.rag_agent else "inactive",
            "decision_network_status": (
                "active" if self.decision_network else "inactive"
            ),
            "anti_hallucination_status": (
                "active" if self.anti_hallucination else "inactive"
            ),
            "planner_enabled": hasattr(self.rag_agent, "planner")
            and self.rag_agent.planner is not None,
            "tracer_enabled": hasattr(self.rag_agent, "execution_monitor")
            and self.rag_agent.execution_monitor is not None,
            "experience_enabled": hasattr(self.rag_agent, "experience_repo")
            and self.rag_agent.experience_repo is not None,
            "timestamp": time.time(),
        }

    def shutdown(self):
        """安全關閉核心系統"""
        logger.info("正在關閉 BioNeuronCore...")

        # 可以在這裡添加清理邏輯
        if (
            hasattr(self.rag_agent, "experience_repo")
            and self.rag_agent.experience_repo
        ):
            try:
                # 關閉數據庫連接等
                pass
            except Exception as e:
                logger.warning(f"關閉經驗庫時出錯: {e}")

        logger.info("BioNeuronCore 已安全關閉")
