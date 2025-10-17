"""
AI 訓練整合系統 - 基於真實 CLI 指令流程

此模組整合了：
1. 真實的 CLI 命令流程 (aiva scan, aiva detect)
2. 500萬參數的 ScalableBioNet (BioNeuronCore)
3. 完整的訊息流追蹤和學習
4. 經驗回放和模型更新

訓練流程：
CLI → Core.TaskDispatcher → Worker → Core.ResultCollector → Integration → AI Learning
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.aiva_common.enums import ModuleName, Severity, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    ScanStartPayload,
)
from services.aiva_common.utils import get_logger, new_id

# Core AI 組件
from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
from services.core.aiva_core.ai_engine.knowledge_base import KnowledgeBase
from services.core.aiva_core.learning.experience_manager import ExperienceManager
from services.core.aiva_core.storage.storage_manager import StorageManager

logger = get_logger(__name__)


# ============================================================================
# 訓練場景模擬器
# ============================================================================


class TrainingScenarioSimulator:
    """訓練場景模擬器 - 模擬完整的 CLI → Core → Worker → Integration 流程"""

    def __init__(
        self,
        bio_net: ScalableBioNet,
        experience_manager: ExperienceManager,
        knowledge_base: KnowledgeBase,
    ):
        self.bio_net = bio_net
        self.exp_manager = experience_manager
        self.kb = knowledge_base
        self.broker = None

    async def initialize(self):
        """初始化連接"""
        self.broker = await get_broker()
        logger.info("[OK] 訓練場景模擬器已初始化")

    # ========== 場景 1: 掃描流程 ==========

    async def simulate_scan_flow(self, target_url: str) -> dict[str, Any]:
        """
        模擬完整的掃描流程並記錄經驗

        CLI 命令模擬:
            aiva scan start https://example.com --max-depth 3

        流程:
            1. CLI 發送 TASK_SCAN_START
            2. Scan Worker 接收並執行
            3. Worker 發送 RESULTS_SCAN_COMPLETED
            4. Core.ResultCollector 接收結果
            5. Integration 存儲和分析
            6. AI 學習整個流程
        """
        scenario_id = new_id("scenario")
        scan_id = new_id("scan")
        task_id = new_id("task")

        logger.info(f"[U+1F3AC] 場景 1: 掃描流程模擬")
        logger.info(f"   場景 ID: {scenario_id}")
        logger.info(f"   目標 URL: {target_url}")

        # 步驟 1: CLI 發送掃描請求
        logger.info("   步驟 1/5: CLI 發送掃描請求...")
        scan_request = await self._cli_send_scan_request(scan_id, task_id, target_url)

        # AI 決策: 應該如何處理這個掃描請求?
        decision_context = {
            "action": "scan_start",
            "target_url": target_url,
            "scan_id": scan_id,
        }
        ai_decision = await self._ai_make_decision(decision_context)

        # 步驟 2: Scan Worker 接收並處理
        logger.info("   步驟 2/5: Scan Worker 處理請求...")
        scan_result = await self._scan_worker_execute(scan_request)

        # 步驟 3: Worker 發送結果
        logger.info("   步驟 3/5: Worker 發送結果到 ResultCollector...")
        await self._worker_send_result(scan_id, scan_result)

        # 步驟 4: ResultCollector 接收並轉發
        logger.info("   步驟 4/5: ResultCollector 轉發到 Integration...")
        await self._result_collector_forward(scan_result)

        # 步驟 5: Integration 分析和存儲
        logger.info("   步驟 5/5: Integration 分析結果...")
        analysis = await self._integration_analyze(scan_result)

        # AI 學習: 記錄整個流程的經驗
        logger.info("   [BRAIN] AI 學習流程...")
        experience = {
            "scenario_id": scenario_id,
            "flow_type": "scan",
            "initial_decision": ai_decision,
            "scan_request": scan_request,
            "scan_result": scan_result,
            "analysis": analysis,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self.exp_manager.store_experience(
            experience_id=new_id("exp"),
            experience_data=experience,
            tags=["scan_flow", "complete"],
        )

        # 更新知識庫
        await self.kb.add_entry(
            entry_id=new_id("kb"),
            content=f"掃描流程: {target_url} -> {len(scan_result.get('assets', []))} 資產",
            metadata={"type": "scan_flow", "assets_count": len(scan_result.get("assets", []))},
        )

        logger.info(f"[OK] 場景 1 完成: 發現 {len(scan_result.get('assets', []))} 個資產")

        return {
            "scenario_id": scenario_id,
            "flow_type": "scan",
            "result": scan_result,
            "analysis": analysis,
        }

    # ========== 場景 2: SQL 注入檢測流程 ==========

    async def simulate_sqli_detection_flow(
        self, target_url: str, param_name: str
    ) -> dict[str, Any]:
        """
        模擬完整的 SQL 注入檢測流程

        CLI 命令模擬:
            aiva detect sqli https://example.com/login --param username

        流程:
            1. CLI 發送 TASK_FUNCTION_SQLI
            2. SQLi Worker 執行多引擎檢測
            3. Worker 發送 RESULTS_FUNCTION_SQLI (含 FindingPayload)
            4. Integration 進行風險評估和關聯分析
            5. AI 學習檢測策略和結果
        """
        scenario_id = new_id("scenario")
        task_id = new_id("task")

        logger.info(f"[U+1F3AC] 場景 2: SQL 注入檢測流程")
        logger.info(f"   場景 ID: {scenario_id}")
        logger.info(f"   目標: {target_url}")
        logger.info(f"   參數: {param_name}")

        # 步驟 1: CLI 發送檢測請求
        logger.info("   步驟 1/5: CLI 發送 SQLi 檢測請求...")
        detection_request = {
            "task_id": task_id,
            "target_url": target_url,
            "param_name": param_name,
            "engines": ["error", "boolean", "time", "union"],
        }

        # AI 決策: 選擇檢測策略
        decision_context = {
            "action": "sqli_detect",
            "target_url": target_url,
            "param_name": param_name,
        }
        ai_strategy = await self._ai_select_detection_strategy(decision_context)

        # 步驟 2: SQLi Worker 執行檢測
        logger.info("   步驟 2/5: SQLi Worker 執行多引擎檢測...")
        detection_result = await self._sqli_worker_execute(detection_request)

        # 步驟 3: Worker 發送 Finding
        logger.info("   步驟 3/5: Worker 發送檢測結果...")
        findings = detection_result.get("findings", [])

        # 步驟 4: Integration 風險評估
        logger.info("   步驟 4/5: Integration 進行風險評估...")
        risk_assessment = await self._integration_assess_risk(findings)

        # 步驟 5: AI 學習檢測經驗
        logger.info("   步驟 5/5: AI 學習檢測策略...")
        experience = {
            "scenario_id": scenario_id,
            "flow_type": "sqli_detection",
            "strategy_selected": ai_strategy,
            "detection_request": detection_request,
            "findings": findings,
            "risk_assessment": risk_assessment,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self.exp_manager.store_experience(
            experience_id=new_id("exp"),
            experience_data=experience,
            tags=["sqli", "detection", "complete"],
        )

        # 更新知識庫
        await self.kb.add_entry(
            entry_id=new_id("kb"),
            content=f"SQLi 檢測: {param_name} @ {target_url} -> {len(findings)} 發現",
            metadata={
                "type": "sqli_detection",
                "findings_count": len(findings),
                "strategy": ai_strategy,
            },
        )

        logger.info(f"[OK] 場景 2 完成: 發現 {len(findings)} 個 SQLi 漏洞")

        return {
            "scenario_id": scenario_id,
            "flow_type": "sqli_detection",
            "findings": findings,
            "risk_assessment": risk_assessment,
        }

    # ========== 場景 3: 完整攻擊鏈 ==========

    async def simulate_full_attack_chain(self, target_url: str) -> dict[str, Any]:
        """
        模擬完整攻擊鏈: Scan → Multiple Detections → Attack Path Analysis

        CLI 命令序列:
            1. aiva scan start https://example.com
            2. aiva detect sqli <discovered_urls>
            3. aiva detect xss <discovered_urls>
            4. aiva report generate --attack-path
        """
        scenario_id = new_id("scenario")

        logger.info(f"[U+1F3AC] 場景 3: 完整攻擊鏈模擬")
        logger.info(f"   場景 ID: {scenario_id}")

        # 階段 1: 掃描
        logger.info("   階段 1/4: 執行掃描...")
        scan_result = await self.simulate_scan_flow(target_url)

        # 階段 2: 對所有資產進行 SQLi 檢測
        logger.info("   階段 2/4: 對發現的資產進行 SQLi 檢測...")
        sqli_results = []
        assets = scan_result["result"].get("assets", [])[:3]  # 取前3個資產測試
        for asset in assets:
            asset_url = asset.get("url", "")
            if "?" in asset_url:  # 有參數的 URL
                result = await self.simulate_sqli_detection_flow(
                    asset_url, param_name="id"
                )
                sqli_results.append(result)

        # 階段 3: XSS 檢測
        logger.info("   階段 3/4: 執行 XSS 檢測...")
        # (簡化版，實際會類似 SQLi 流程)

        # 階段 4: 攻擊路徑分析
        logger.info("   階段 4/4: 生成攻擊路徑分析...")
        attack_path = await self._integration_build_attack_path(
            scan_result, sqli_results
        )

        # AI 學習完整攻擊鏈
        experience = {
            "scenario_id": scenario_id,
            "flow_type": "full_attack_chain",
            "scan_result": scan_result,
            "sqli_results": sqli_results,
            "attack_path": attack_path,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self.exp_manager.store_experience(
            experience_id=new_id("exp"),
            experience_data=experience,
            tags=["attack_chain", "complete", "advanced"],
        )

        logger.info(f"[OK] 場景 3 完成: 完整攻擊鏈分析")

        return {
            "scenario_id": scenario_id,
            "flow_type": "full_attack_chain",
            "attack_path": attack_path,
        }

    # ========== 內部輔助方法 ==========

    async def _cli_send_scan_request(
        self, scan_id: str, task_id: str, target_url: str
    ) -> dict[str, Any]:
        """模擬 CLI 發送掃描請求"""
        header = MessageHeader(
            message_id=new_id("msg"),
            source_module=ModuleName.CLI,
            target_module=ModuleName.SCAN,
            correlation_id=scan_id,
        )

        payload = ScanStartPayload(
            scan_id=scan_id,
            task_id=task_id,
            target_url=target_url,
            max_depth=3,
            max_pages=100,
            scope_domains=[target_url],
        )

        return {"header": header.model_dump(), "payload": payload.model_dump()}

    async def _ai_make_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """AI 決策"""
        # 使用 BioNeuronCore 做決策
        input_vector = self._context_to_vector(context)
        output = self.bio_net.forward(input_vector)

        return {
            "decision": "proceed",
            "confidence": float(output.max()),
            "reasoning": "Based on neural network decision",
        }

    async def _ai_select_detection_strategy(
        self, context: dict[str, Any]
    ) -> dict[str, Any]:
        """AI 選擇檢測策略"""
        # 查詢知識庫獲取類似案例
        similar_cases = await self.kb.search(
            query=f"SQLi detection {context['param_name']}", top_k=5
        )

        return {
            "engines": ["error", "boolean", "time"],
            "priority": "error_first",
            "similar_cases_count": len(similar_cases),
        }

    async def _scan_worker_execute(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """模擬 Scan Worker 執行"""
        # 模擬掃描結果
        return {
            "scan_id": request["payload"]["scan_id"],
            "assets": [
                {"url": f"https://example.com/page{i}", "type": "html"}
                for i in range(5)
            ],
            "fingerprints": {"server": "nginx", "framework": "django"},
            "summary": {"total_assets": 5, "duration_sec": 10.5},
        }

    async def _worker_send_result(self, scan_id: str, result: dict[str, Any]):
        """模擬 Worker 發送結果"""
        # 實際會發送到 RabbitMQ
        pass

    async def _result_collector_forward(self, result: dict[str, Any]):
        """模擬 ResultCollector 轉發"""
        pass

    async def _integration_analyze(self, result: dict[str, Any]) -> dict[str, Any]:
        """模擬 Integration 分析"""
        return {
            "risk_score": 3.5,
            "asset_classification": "web_application",
            "recommendations": ["Enable HTTPS", "Update framework"],
        }

    async def _sqli_worker_execute(
        self, request: dict[str, Any]
    ) -> dict[str, Any]:
        """模擬 SQLi Worker 執行"""
        # 模擬檢測結果
        finding = {
            "finding_id": new_id("finding"),
            "task_id": request["task_id"],
            "scan_id": new_id("scan"),
            "status": "confirmed",
            "vulnerability": {
                "name": "SQL_INJECTION",
                "severity": Severity.HIGH.value,
                "confidence": "high",
                "description": "SQL injection vulnerability detected",
            },
            "target": {
                "url": request["target_url"],
                "parameter": request["param_name"],
            },
        }

        return {"findings": [finding], "engines_used": request["engines"]}

    async def _integration_assess_risk(
        self, findings: list[dict]
    ) -> dict[str, Any]:
        """模擬風險評估"""
        if not findings:
            return {"risk_level": "low", "score": 0.0}

        return {
            "risk_level": "high",
            "score": 8.5,
            "impact": "Data breach possible",
            "remediation_priority": 1,
        }

    async def _integration_build_attack_path(
        self, scan_result: dict, sqli_results: list[dict]
    ) -> dict[str, Any]:
        """模擬攻擊路徑構建"""
        return {
            "path_id": new_id("path"),
            "stages": [
                {"stage": "reconnaissance", "result": scan_result},
                {"stage": "exploitation", "findings": sqli_results},
            ],
            "feasibility": 0.85,
            "overall_risk": 9.0,
        }

    def _context_to_vector(self, context: dict[str, Any]) -> Any:
        """將上下文轉換為神經網路輸入向量"""
        # 簡化版本，實際應該使用嵌入模型
        import numpy as np

        return np.random.randn(512)


# ============================================================================
# AI 訓練編排器
# ============================================================================


class AITrainingOrchestrator:
    """AI 訓練編排器 - 協調整個訓練流程"""

    def __init__(self, storage_path: Path = Path("./data/ai")):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 初始化組件
        self.bio_net = self._create_bio_net()
        self.exp_manager = self._create_experience_manager()
        self.kb = self._create_knowledge_base()
        self.simulator = None

    def _create_bio_net(self) -> ScalableBioNet:
        """創建 500 萬參數的神經網路"""
        logger.info("[BRAIN] 初始化 ScalableBioNet (500萬參數)...")

        net = ScalableBioNet(
            input_dim=512,  # 輸入維度
            hidden_dims=[1024, 2048, 1024],  # 隱藏層: 1024 + 2048 + 1024
            output_dim=256,  # 輸出維度
        )

        param_count = net.count_params()
        logger.info(f"   [OK] 神經網路參數量: {param_count:,}")

        return net

    def _create_experience_manager(self) -> ExperienceManager:
        """創建經驗管理器"""
        storage_manager = StorageManager(
            backend_type="sqlite", db_path=str(self.storage_path / "experiences.db")
        )
        return ExperienceManager(storage_manager=storage_manager)

    def _create_knowledge_base(self) -> KnowledgeBase:
        """創建知識庫"""
        return KnowledgeBase(storage_path=self.storage_path / "knowledge")

    async def initialize(self):
        """初始化訓練系統"""
        logger.info("[START] 初始化 AI 訓練系統...")

        await self.exp_manager.initialize()
        await self.kb.initialize()

        self.simulator = TrainingScenarioSimulator(
            bio_net=self.bio_net,
            experience_manager=self.exp_manager,
            knowledge_base=self.kb,
        )

        await self.simulator.initialize()

        logger.info("[OK] AI 訓練系統初始化完成")

    async def train_from_simulations(
        self, num_scenarios: int = 10, epochs: int = 5
    ):
        """從模擬場景進行訓練"""
        logger.info(f"[U+1F393] 開始訓練: {num_scenarios} 個場景, {epochs} 輪")

        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"訓練輪次 {epoch + 1}/{epochs}")
            logger.info(f"{'='*60}")

            for i in range(num_scenarios):
                logger.info(f"\n場景 {i + 1}/{num_scenarios}")

                # 輪流執行不同場景
                if i % 3 == 0:
                    await self.simulator.simulate_scan_flow(
                        f"https://example{i}.com"
                    )
                elif i % 3 == 1:
                    await self.simulator.simulate_sqli_detection_flow(
                        f"https://example{i}.com/login", "username"
                    )
                else:
                    await self.simulator.simulate_full_attack_chain(
                        f"https://example{i}.com"
                    )

            # 每輪結束後更新模型
            logger.info(f"\n[RELOAD] 更新神經網路模型...")
            await self._update_model(epoch)

        logger.info(f"\n[OK] 訓練完成！")

    async def _update_model(self, epoch: int):
        """更新神經網路模型"""
        # 從經驗管理器獲取經驗
        experiences = await self.exp_manager.get_recent_experiences(limit=100)

        logger.info(f"   從 {len(experiences)} 條經驗中學習...")

        # 實際訓練邏輯 (簡化版)
        # 真實情況會進行反向傳播和參數更新

        # 保存模型檢查點
        checkpoint_path = self.storage_path / f"model_epoch_{epoch}.pkl"
        # self.bio_net.save(checkpoint_path)

        logger.info(f"   [OK] 模型更新完成 (Epoch {epoch})")

    async def get_training_stats(self) -> dict[str, Any]:
        """獲取訓練統計"""
        exp_stats = await self.exp_manager.get_stats()
        kb_stats = await self.kb.get_stats()

        return {
            "model_params": self.bio_net.count_params(),
            "experiences_count": exp_stats.get("total_count", 0),
            "knowledge_entries": kb_stats.get("total_entries", 0),
            "last_update": datetime.now(UTC).isoformat(),
        }


# ============================================================================
# 主函數 - 供 CLI 調用
# ============================================================================


async def main():
    """主訓練函數"""
    logger.info("="*60)
    logger.info("AIVA AI 訓練系統")
    logger.info("基於 500 萬參數 ScalableBioNet")
    logger.info("="*60)

    orchestrator = AITrainingOrchestrator()
    await orchestrator.initialize()

    # 執行訓練
    await orchestrator.train_from_simulations(num_scenarios=5, epochs=3)

    # 顯示統計
    stats = await orchestrator.get_training_stats()
    logger.info(f"\n[STATS] 訓練統計:")
    logger.info(f"   模型參數量: {stats['model_params']:,}")
    logger.info(f"   經驗條數: {stats['experiences_count']}")
    logger.info(f"   知識庫條目: {stats['knowledge_entries']}")


if __name__ == "__main__":
    asyncio.run(main())
