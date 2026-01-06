from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class StrategyAdjuster:
    """動態策略調整器

    基於回饋結果、指紋識別、WAF檢測等資訊進行策略調整，
    實現自適應測試策略優化。
    """

    def __init__(self) -> None:
        self._learning_data: dict[str, list[dict[str, Any]]] = {}
        self._waf_patterns: dict[str, list[str]] = {}
        self._success_patterns: dict[str, list[dict[str, Any]]] = {}

    def adjust(self, plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """動態調整測試策略

        Args:
            plan: 基礎測試計劃
            context: 會話上下文資訊

        Returns:
            調整後的測試策略
        """
        adjusted_plan = plan.copy()

        # 1. WAF適應調整
        if context.get("waf_detected"):
            adjusted_plan = self._adjust_for_waf(adjusted_plan, context)

        # 2. 基於歷史成功率調整
        adjusted_plan = self._adjust_based_on_success_rate(adjusted_plan, context)

        # 3. 基於目標技術棧調整
        if context.get("fingerprints"):
            adjusted_plan = self._adjust_for_tech_stack(
                adjusted_plan, context["fingerprints"]
            )

        # 4. 基於已發現漏洞調整優先級
        if context.get("findings_count", 0) > 0:
            adjusted_plan = self._adjust_for_findings(adjusted_plan, context)

        logger.info(f"Strategy adjusted for scan {context.get('scan_id')}")
        return adjusted_plan

    def learn_from_result(self, feedback_data: dict[str, Any]) -> None:
        """從測試結果中學習，更新策略知識庫

        Args:
            feedback_data: 回饋數據
        """
        scan_id = feedback_data.get("scan_id")
        module = feedback_data.get("module")
        success = feedback_data.get("success", False)

        if not scan_id or not module:
            return

        # 記錄學習數據
        if scan_id not in self._learning_data:
            self._learning_data[scan_id] = []

        self._learning_data[scan_id].append(feedback_data)

        # 更新成功模式
        if success and module:
            if module not in self._success_patterns:
                self._success_patterns[module] = []
            self._success_patterns[module].append(feedback_data)

        logger.info(
            f"Learned from {module} result: {'success' if success else 'failure'}"
        )

    def _adjust_for_waf(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """WAF適應調整"""
        waf_vendor = context.get("fingerprints", {}).get("waf_vendor")

        if waf_vendor:
            logger.info(f"Adjusting strategy for WAF: {waf_vendor}")

            # 降低攻擊頻率
            if "timing" in plan:
                plan["timing"]["delay_between_requests"] = max(
                    plan["timing"].get("delay_between_requests", 1), 2
                )
            else:
                plan["timing"] = {"delay_between_requests": 2}

            # 選擇WAF繞過payload
            for task in plan.get("tasks", []):
                if task.get("type") in ["xss", "sqli"]:
                    task["use_evasion"] = True
                    task["payload_type"] = "waf_bypass"

        return plan

    def _adjust_based_on_success_rate(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """基於成功率調整"""
        scan_id = str(context.get("scan_id", ""))
        learning_data = self._learning_data.get(scan_id, [])

        if not learning_data:
            return plan

        # 計算各模組成功率
        module_stats = {}
        for data in learning_data:
            module = data.get("module")
            success = data.get("success", False)

            if module not in module_stats:
                module_stats[module] = {"total": 0, "success": 0}
            module_stats[module]["total"] += 1
            if success:
                module_stats[module]["success"] += 1

        # 調整任務優先級
        for task in plan.get("tasks", []):
            task_type = task.get("type")
            if task_type in module_stats:
                success_rate = (
                    module_stats[task_type]["success"]
                    / module_stats[task_type]["total"]
                )
                if success_rate > 0.7:
                    task["priority"] = min(task.get("priority", 5) + 1, 10)
                elif success_rate < 0.3:
                    task["priority"] = max(task.get("priority", 5) - 1, 1)

        return plan

    def _adjust_for_tech_stack(
        self, plan: dict[str, Any], fingerprints: dict[str, Any]
    ) -> dict[str, Any]:
        """基於技術棧調整"""
        web_server = fingerprints.get("web_server", {}).get("name", "").lower()
        framework = fingerprints.get("framework", {}).get("name", "").lower()

        # 針對特定技術棧的調整
        if "nginx" in web_server:
            # Nginx特定調整
            for task in plan.get("tasks", []):
                if task.get("type") == "sqli":
                    task["check_nginx_vars"] = True

        if "django" in framework:
            # Django CSRF保護
            for task in plan.get("tasks", []):
                if task.get("type") == "csrf":
                    task["check_csrf_token"] = True

        if "php" in fingerprints.get("language", {}).get("name", "").lower():
            # PHP特定漏洞
            for task in plan.get("tasks", []):
                if task.get("type") == "lfi":
                    task["php_wrappers"] = True

        return plan

    def _adjust_for_findings(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """基於已發現漏洞調整"""
        findings_count = context.get("findings_count", 0)

        # 如果已經發現高風險漏洞，降低其他測試的優先級
        if findings_count > 3:
            for task in plan.get("tasks", []):
                if task.get("priority", 5) < 8:  # 不是高優先級任務
                    task["priority"] = max(task.get("priority", 5) - 1, 1)

        return plan
