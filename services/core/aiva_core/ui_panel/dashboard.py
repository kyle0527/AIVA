"""Dashboard - AIVA 主控制面板
提供 Web UI 來管理掃描、AI 代理、漏洞檢測等功能
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Dashboard:
    """AIVA 控制面板主類別."""

    def __init__(self, mode: str = "hybrid") -> None:
        """初始化控制面板.

        Args:
            mode: 運作模式
                - "ui": 只使用 UI 介面
                - "ai": 只使用 AI 代理
                - "hybrid": 同時使用 UI 和 AI (預設)
        """
        self.mode = mode
        self.ai_agent: Any = None
        self.ai_commander: Any = None
        self.scan_tasks: list[dict[str, Any]] = []
        self.detection_results: list[dict[str, Any]] = []

        logger.info(f"\n{'='*60}")
        logger.info("   AIVA 控制面板初始化")
        logger.info(f"{'='*60}")
        logger.info(f"運作模式: {self._get_mode_display()}")

        if mode in ("ai", "hybrid"):
            self._init_ai_agent()
            self._init_ai_commander()

        logger.info(f"{'='*60}\n")

    def _get_mode_display(self) -> str:
        """獲取模式的顯示名稱."""
        mode_map = {
            "ui": "僅 UI 介面",
            "ai": "僅 AI 代理",
            "hybrid": "UI + AI 混合模式",
        }
        return mode_map.get(self.mode, "未知模式")

    def _init_ai_agent(self) -> None:
        """初始化 AI 代理."""
        try:
            from ..cognitive_core.neural import BioNeuronRAGAgent

            logger.info("\n[AI] 正在初始化 BioNeuronRAGAgent...")
            self.ai_agent = BioNeuronRAGAgent(codebase_path="c:/D/E/AIVA/AIVA-main")
            logger.info("[AI] AI 代理初始化成功")
        except Exception as e:
            logger.error(f"[AI] AI 代理初始化失敗: {e}")
            if self.mode == "ai":
                raise
            logger.warning("[AI] 將以純 UI 模式運作")
            self.mode = "ui"

    def _init_ai_commander(self) -> None:
        """初始化 AI Commander - 真正執行攻擊的組件."""
        try:
            from ..task_planning.ai_commander import AICommander

            logger.info("\n[AI Commander] 正在初始化 AICommander...")
            self.ai_commander = AICommander()
            logger.info("[AI Commander] AICommander 初始化成功")
        except Exception as e:
            logger.warning(f"[AI Commander] AICommander 初始化失敗: {e}")
            self.ai_commander = None
            logger.info("[AI Commander] 將使用 AttackExecutor 作為 fallback")

    def create_scan_task(
        self,
        target_url: str,
        scan_type: str = "full",
        use_ai: bool | None = None,
    ) -> dict[str, Any]:
        """建立掃描任務.

        Args:
            target_url: 目標 URL
            scan_type: 掃描類型 (full/quick/custom)
            use_ai: 是否使用 AI (None 則根據 mode 自動決定)

        Returns:
            任務資訊
        """
        # 決定是否使用 AI
        if use_ai is None:
            use_ai = self.mode in ("ai", "hybrid")

        task_id = f"scan_{hash(target_url) % 100000}"

        if use_ai and self.ai_commander:
            logger.info(f"\n{'='*60}")
            logger.info(f"[AI 自主攻擊] 啟動針對 {target_url} 的全面攻擊")
            logger.info(f"{'='*60}")
            
            # 使用 AICommander 執行真正的攻擊
            try:
                from ..task_planning.ai_commander import AITaskType
                
                # 創建攻擊任務上下文
                context = {
                    "target": target_url,
                    "scan_type": scan_type,
                    "objective": f"對 {target_url} 執行全面安全測試",
                    "constraints": {
                        "timeout": 300,  # 5分鐘超時
                        "max_depth": 3,
                        "stealth_mode": False
                    }
                }
                
                # 在新的事件循環中執行異步任務
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # 執行攻擊計畫生成
                    logger.info("[攻擊階段 1] AI 正在生成攻擊計畫...")
                    plan_result = loop.run_until_complete(
                        self.ai_commander.execute_command(
                            AITaskType.ATTACK_PLANNING,
                            context
                        )
                    )
                    
                    # 執行漏洞檢測
                    logger.info("[攻擊階段 2] AI 正在執行漏洞檢測...")
                    detection_result = loop.run_until_complete(
                        self.ai_commander.execute_command(
                            AITaskType.VULNERABILITY_DETECTION,
                            {**context, "attack_plan": plan_result}
                        )
                    )
                    
                    logger.info(f"[攻擊完成] ✅ 已完成對 {target_url} 的攻擊")
                    
                    task = {
                        "task_id": task_id,
                        "target": target_url,
                        "scan_type": scan_type,
                        "status": "completed",
                        "created_by": "ai_commander",
                        "attack_plan": plan_result,
                        "detection_result": detection_result,
                        "ai_result": {
                            "plan": plan_result,
                            "detection": detection_result
                        },
                    }
                finally:
                    loop.close()
                
            except Exception as e:
                logger.error(f"[AI 攻擊失敗] ❌ {e}", exc_info=True)
                task = {
                    "task_id": task_id,
                    "target": target_url,
                    "scan_type": scan_type,
                    "status": "failed",
                    "created_by": "ai_commander",
                    "error": str(e),
                }
        
        elif use_ai:
            # Fallback: 直接使用 AttackExecutor（當 AICommander 不可用時）
            logger.info("\n" + "="*60)
            logger.info("[直接攻擊模式] 使用 AttackExecutor 執行攻擊")
            logger.info("目標: " + target_url)
            logger.info("="*60)
            
            try:
                from ..core_capabilities.attack.attack_executor import AttackExecutor, ExecutionMode
                
                executor = AttackExecutor(
                    mode=ExecutionMode.TESTING,
                    max_concurrent=3,
                    timeout=300,
                    safety_enabled=True
                )
                
                attack_plan = {
                    "plan_id": f"plan_{task_id}",
                    "target_url": target_url,
                    "scan_type": scan_type,
                    "steps": [
                        {"name": "偵查階段", "type": "reconnaissance", "description": "收集目標信息", "critical": False},
                        {"name": "XSS 檢測", "type": "xss_scan", "description": "檢測跨站腳本漏洞", "critical": False},
                        {"name": "SQL 注入檢測", "type": "sqli_scan", "description": "檢測 SQL 注入漏洞", "critical": False},
                        {"name": "業務邏輯測試", "type": "bizlogic_test", "description": "測試業務邏輯漏洞", "critical": False}
                    ]
                }
                
                attack_target = {"target_url": target_url, "scan_type": scan_type}
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    logger.info("[執行中] 正在執行攻擊計劃...")
                    execution_result = loop.run_until_complete(
                        executor.execute_plan(attack_plan, attack_target)
                    )
                    logger.info("[完成] ✅ 攻擊執行完成")
                    
                    task = {
                        "task_id": task_id,
                        "target": target_url,
                        "scan_type": scan_type,
                        "status": "completed",
                        "created_by": "attack_executor",
                        "execution_result": execution_result,
                        "findings": execution_result.get("findings", []),
                        "metrics": execution_result.get("metrics", {}),
                    }
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"[攻擊失敗] ❌ {e}", exc_info=True)
                task = {
                    "task_id": task_id,
                    "target": target_url,
                    "scan_type": scan_type,
                    "status": "failed",
                    "created_by": "attack_executor",
                    "error": str(e),
                }
                
        elif self.ai_agent:
            logger.info("\n[AI] 使用 AI 代理建立掃描任務...")
            result = self.ai_agent.invoke(
                query=f"對 {target_url} 執行 {scan_type} 掃描",
                target_url=target_url,
                scan_type=scan_type,
            )
            task = {
                "task_id": task_id,
                "target": target_url,
                "scan_type": scan_type,
                "status": "pending",
                "created_by": "ai",
                "ai_result": result,
            }
        else:
            logger.info("\n[UI] 使用 UI 模式建立掃描任務...")
            task = {
                "task_id": task_id,
                "target": target_url,
                "scan_type": scan_type,
                "status": "pending",
                "created_by": "ui",
            }

        self.scan_tasks.append(task)
        logger.info(f"掃描任務已建立: {task_id}")
        return task

    def detect_vulnerability(
        self,
        vuln_type: str,
        target: str,
        use_ai: bool | None = None,
    ) -> dict[str, Any]:
        """執行漏洞檢測.

        Args:
            vuln_type: 漏洞類型 (xss/sqli/ssrf/idor)
            target: 目標
            use_ai: 是否使用 AI

        Returns:
            檢測結果
        """
        if use_ai is None:
            use_ai = self.mode in ("ai", "hybrid")

        if use_ai and self.ai_agent:
            logger.info(f"\n[AI] 使用 AI 代理執行 {vuln_type.upper()} 檢測...")
            result = self.ai_agent.invoke(
                query=f"對 {target} 執行 {vuln_type} 漏洞檢測",
                vuln_type=vuln_type,
                target=target,
            )
            detection = {
                "vuln_type": vuln_type,
                "target": target,
                "status": "completed",
                "method": "ai",
                "result": result,
            }
        else:
            logger.info(f"\n[UI] 使用 UI 模式執行 {vuln_type.upper()} 檢測...")
            detection = {
                "vuln_type": vuln_type,
                "target": target,
                "status": "pending",
                "method": "ui",
                "findings": [],
            }

        self.detection_results.append(detection)
        return detection

    def read_code(self, file_path: str, use_ai: bool | None = None) -> dict[str, Any]:
        """讀取程式碼檔案.

        Args:
            file_path: 檔案路徑
            use_ai: 是否使用 AI

        Returns:
            檔案內容
        """
        if use_ai is None:
            use_ai = self.mode in ("ai", "hybrid")

        if use_ai and self.ai_agent:
            logger.info(f"\n[AI] 使用 AI 代理讀取檔案: {file_path}")
            result = self.ai_agent.invoke(
                query=f"讀取檔案 {file_path}",
                path=file_path,
            )
            return result
        else:
            logger.info(f"\n[UI] 使用 UI 模式讀取檔案: {file_path}")
            from pathlib import Path

            try:
                content = Path(f"c:/D/E/AIVA/AIVA-main/{file_path}").read_text(
                    encoding="utf-8"
                )
                return {
                    "status": "success",
                    "path": file_path,
                    "content": content,
                    "method": "ui",
                }
            except Exception as e:
                return {"status": "error", "path": file_path, "error": str(e)}

    def analyze_code(
        self, file_path: str, use_ai: bool | None = None
    ) -> dict[str, Any]:
        """分析程式碼.

        Args:
            file_path: 檔案路徑
            use_ai: 是否使用 AI

        Returns:
            分析結果
        """
        if use_ai is None:
            use_ai = self.mode in ("ai", "hybrid")

        if use_ai and self.ai_agent:
            logger.info(f"\n[AI] 使用 AI 代理分析程式碼: {file_path}")
            result = self.ai_agent.invoke(
                query=f"分析程式碼檔案 {file_path} 的結構",
                path=file_path,
            )
            return result
        else:
            logger.info(f"\n[UI] 使用 UI 模式分析程式碼: {file_path}")
            # 簡單的手動分析
            from pathlib import Path

            try:
                content = Path(f"c:/D/E/AIVA/AIVA-main/{file_path}").read_text(
                    encoding="utf-8"
                )
                lines = content.splitlines()
                return {
                    "status": "success",
                    "path": file_path,
                    "total_lines": len(lines),
                    "imports": sum(
                        1 for line in lines if line.strip().startswith("import")
                    ),
                    "functions": sum(
                        1 for line in lines if line.strip().startswith("def ")
                    ),
                    "classes": sum(
                        1 for line in lines if line.strip().startswith("class ")
                    ),
                    "method": "ui",
                }
            except Exception as e:
                return {"status": "error", "path": file_path, "error": str(e)}

    def get_tasks(self) -> list[dict[str, Any]]:
        """獲取所有掃描任務."""
        return self.scan_tasks

    def get_detections(self) -> list[dict[str, Any]]:
        """獲取所有檢測結果."""
        return self.detection_results

    def get_ai_history(self) -> list[dict[str, Any]]:
        """獲取 AI 代理的執行歷史."""
        if self.ai_agent:
            return self.ai_agent.get_history()
        return []

    def get_stats(self) -> dict[str, Any]:
        """獲取統計資訊."""
        stats = {
            "mode": self.mode,
            "mode_display": self._get_mode_display(),
            "total_tasks": len(self.scan_tasks),
            "total_detections": len(self.detection_results),
            "ai_enabled": self.ai_agent is not None,
        }

        if self.ai_agent:
            ai_stats = self.ai_agent.get_knowledge_stats()
            stats.update(
                {
                    "ai_chunks": ai_stats.get("total_chunks", 0),
                    "ai_keywords": ai_stats.get("total_keywords", 0),
                    "ai_history_count": len(self.ai_agent.get_history()),
                }
            )

        return stats
