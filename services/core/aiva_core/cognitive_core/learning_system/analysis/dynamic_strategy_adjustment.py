"""動態策略調整器 (RL 整合版)

基於回饋結果、指紋識別、WAF檢測等資訊進行策略調整，
實現自適應測試策略優化。

整合 RL 學習機制：
- 根據模組執行結果計算獎勵（透過 MQ 回傳的業務結果）
- 建議最佳 CLI 參數組合（在發送指令前）
- 支援 13 步驟流程整合

參考：LEARNING_INTEGRATION_WITH_13STEPS.md
"""

from dataclasses import dataclass
from typing import Any

from aiva_common.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RewardConfig:
    """執行結果的獎勵配置
    
    根據 LEARNING_INTEGRATION_WITH_13STEPS.md Step 8a 定義
    獎勵計算基於 MQ 回傳的業務結果（vulnerability, waf_status 等）
    """
    # 正向獎勵
    VULN_CONFIRMED: float = 10.0       # 確認高危漏洞 (CONFIRMED confidence)
    VULN_HIGH: float = 8.0             # 高危漏洞 (HIGH severity)
    VULN_MEDIUM: float = 5.0           # 中危漏洞 (MEDIUM severity)
    VULN_LOW: float = 2.0              # 低危漏洞 (LOW/INFO severity)
    WAF_BYPASS: float = 15.0           # 成功繞過 WAF
    
    # 懲罰
    SOFT_BLOCK: float = -1.0           # 被 WAF 軟封鎖
    EXECUTION_FAILURE: float = -5.0    # 執行失敗
    NO_RESULT: float = -2.0            # 無結果


class StrategyAdjuster:
    """動態策略調整器

    基於回饋結果、指紋識別、WAF檢測等資訊進行策略調整，
    實現自適應測試策略優化。
    """

    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self._learning_data: dict[str, list[dict[str, Any]]] = {}
        self._waf_patterns: dict[str, list[str]] = {}
        self._success_patterns: dict[str, list[dict[str, Any]]] = {}
        
        # RL 整合 (LEARNING_INTEGRATION_WITH_13STEPS.md)
        self.reward_config = reward_config or RewardConfig()
        self._parameter_history: dict[str, list[dict[str, Any]]] = {}  # 參數歷史
        self._reward_history: list[dict[str, Any]] = []  # 獎勵歷史

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
        
        整合 RL 獎勵計算 (Step 8a)
        結果透過 MQ 回傳，包含業務欄位而非 CLI stdout

        Args:
            feedback_data: 回饋數據，應包含：
                - scan_id: 掃描 ID
                - module: 模組名稱
                - success: 是否成功 (bool)
                - vulnerability: 漏洞資訊 (dict, 可選)
                - waf_status: WAF 狀態 (str, 可選)
                - parameters_used: 使用的 CLI 參數 (dict, 可選)
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

        # ===== RL 獎勵計算 (Step 8a) =====
        # 從 MQ 回傳的業務結果計算獎勵
        reward = self.calculate_reward(feedback_data)
        parameters_used = feedback_data.get("parameters_used", {})
        
        # 記錄獎勵歷史（用於 RL 訓練）
        self._reward_history.append({
            "scan_id": scan_id,
            "module": module,
            "parameters": parameters_used,
            "reward": reward,
            "success": success,
        })
        
        # 記錄參數歷史（用於參數優化）
        if parameters_used:
            if module not in self._parameter_history:
                self._parameter_history[module] = []
            self._parameter_history[module].append({
                "parameters": parameters_used,
                "reward": reward,
            })
        
        logger.info(
            f"RL: {module} reward={reward:.2f}, success={success}"
        )

    def calculate_reward(self, feedback_data: dict[str, Any]) -> float:
        """根據模組執行結果計算獎勵
        
        實現 LEARNING_INTEGRATION_WITH_13STEPS.md Step 8a:
        從 MQ 回傳的業務結果計算獎勵
        
        Args:
            feedback_data: 包含以下欄位:
                - success: 是否成功 (bool)
                - vulnerability: 漏洞資訊 (dict)
                    - severity: 嚴重性 (str)
                    - confidence: 可信度 (str)
                - waf_status: WAF 狀態 (str, 可選)
            
        Returns:
            獎勵值
        """
        cfg = self.reward_config
        reward = 0.0
        
        # 1. 執行失敗懲罰
        if not feedback_data.get("success", False):
            return cfg.EXECUTION_FAILURE
        
        # 2. 漏洞獎勵
        vulnerability = feedback_data.get("vulnerability", {})
        if vulnerability:
            severity = str(vulnerability.get("severity", "")).upper()
            confidence = str(vulnerability.get("confidence", "")).upper()
            
            # 已確認的漏洞給予最高獎勵
            if confidence == "CONFIRMED":
                reward += cfg.VULN_CONFIRMED
            elif severity in ("HIGH", "CRITICAL"):
                reward += cfg.VULN_HIGH
            elif severity == "MEDIUM":
                reward += cfg.VULN_MEDIUM
            elif severity in ("LOW", "INFO"):
                reward += cfg.VULN_LOW
        else:
            # 沒有發現漏洞
            reward += cfg.NO_RESULT
        
        # 3. WAF 繞過獎勵
        waf_status = str(feedback_data.get("waf_status", "unknown")).lower()
        if waf_status == "bypassed":
            reward += cfg.WAF_BYPASS
        elif waf_status == "blocked":
            reward += cfg.SOFT_BLOCK
        
        return reward

    def suggest_parameters(
        self, module: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """根據歷史經驗建議最佳 CLI 參數
        
        實現 LEARNING_INTEGRATION_WITH_13STEPS.md Step 9:
        策略修正 - 參數變異，Bandit 根據歷史經驗選擇參數
        
        此方法應在發送 CLI 指令前呼叫，讓 Task Planning 獲取建議參數
        
        Args:
            module: 模組名稱 (e.g., "xss", "sqli", "scanner")
            context: 目標上下文（可選）
                - fingerprints: 技術棧資訊
                - waf_detected: 是否偵測到 WAF
            
        Returns:
            建議的 CLI 參數字典
        """
        # 獲取此模組的參數歷史
        history = self._parameter_history.get(module, [])
        
        if not history:
            # 沒有歷史數據，返回預設參數
            return self._get_default_parameters(module)
        
        # 找出獎勵最高的參數組合
        best_entry = max(history, key=lambda x: x.get("reward", 0))
        best_params = dict(best_entry.get("parameters", {}))  # 複製避免修改原資料
        
        # 根據 context 調整參數
        if context:
            # WAF 檢測到時，調整為 stealth 模式
            if context.get("waf_detected"):
                best_params["mode"] = "stealth"
                best_params["delay"] = max(best_params.get("delay", 0), 500)
            
            # 根據技術棧調整
            tech_stack = context.get("fingerprints", {})
            if "php" in str(tech_stack.get("language", {})).lower():
                best_params["payload_type"] = "php_specific"
        
        logger.info(f"Suggested params for {module}: {best_params}")
        return best_params

    def _get_default_parameters(self, module: str) -> dict[str, Any]:
        """獲取預設 CLI 參數"""
        defaults = {
            "scanner": {"mode": "quick", "concurrency": 5, "depth": 2},
            "xss": {"mode": "quick", "payload": "basic"},
            "sqli": {"mode": "quick", "payload": "basic", "technique": "BEUST"},
            "ssrf": {"mode": "quick", "timeout": 10},
        }
        return defaults.get(module, {"mode": "quick"})

    def get_reward_statistics(self) -> dict[str, Any]:
        """獲取 RL 獎勵統計
        
        Returns:
            統計資訊
        """
        if not self._reward_history:
            return {"total_episodes": 0, "avg_reward": 0.0}
        
        rewards = [entry["reward"] for entry in self._reward_history]
        return {
            "total_episodes": len(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "positive_rate": sum(1 for r in rewards if r > 0) / len(rewards),
        }

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

