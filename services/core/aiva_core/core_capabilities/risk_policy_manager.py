"""風險策略管理器

從 config/risk_policies.yaml 讀取並評估風險策略
遵循配置驅動的風險評估原則
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class RiskPolicyManager:
    """風險策略管理器
    
    負責從 YAML 配置載入風險評估規則並執行評估
    """
    
    def __init__(self, policy_file: Path | None = None):
        """初始化風險策略管理器
        
        Args:
            policy_file: 風險策略配置文件路徑，默認使用 config/risk_policies.yaml
        """
        if policy_file is None:
            # 默認路徑：services/core/aiva_core/config/risk_policies.yaml
            aiva_core_root = Path(__file__).parent.parent
            policy_file = aiva_core_root / "config" / "risk_policies.yaml"
        
        self.policy_file = policy_file
        self.policy_config: dict[str, Any] = {}
        self._load_policy()
    
    def _load_policy(self) -> None:
        """載入風險策略配置
        
        Raises:
            FileNotFoundError: 如果配置文件不存在
            yaml.YAMLError: 如果 YAML 格式錯誤
        """
        if not self.policy_file.exists():
            raise FileNotFoundError(
                f"❌ 風險策略配置文件不存在: {self.policy_file}\n"
                f"請確認 config/risk_policies.yaml 文件存在"
            )
        
        try:
            with open(self.policy_file, encoding="utf-8") as f:
                self.policy_config = yaml.safe_load(f)
            
            logger.info(f"✅ 載入風險策略: {self.policy_config.get('policy_name', 'unknown')}")
            logger.info(f"   版本: {self.policy_config.get('version', 'unknown')}")
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"❌ 風險策略配置格式錯誤: {e}") from e
    
    def assess_risk(
        self,
        context: dict[str, Any],
        task_type: str | None = None
    ) -> tuple[str, float, list[dict[str, Any]]]:
        """評估風險等級
        
        Args:
            context: 風險上下文
                - target_type: 目標類型 (production/staging/development)
                - time_limit: 時間限制
                - business_hours: 是否工作時間
                - authorized: 是否授權
                - scope_verified: 範圍已驗證
                - contains_pii: 包含個人信息
                - contains_payment_data: 包含支付數據
                - contains_sensitive_data: 包含敏感數據
                - system_criticality: 系統關鍵度 (critical/high/medium/low)
                - waf_detected: 檢測到 WAF
                - rate_limit_detected: 檢測到速率限制
            task_type: 任務類型（scan/attack/exploit），用於向後兼容
        
        Returns:
            (risk_level, total_score, applied_rules) 元組
            - risk_level: 風險等級 (critical/high/medium/low)
            - total_score: 總風險分數
            - applied_rules: 應用的規則列表
        
        Example:
            >>> manager = RiskPolicyManager()
            >>> context = {
            ...     "target_type": "production",
            ...     "authorized": True,
            ...     "contains_pii": True,
            ...     "system_criticality": "high"
            ... }
            >>> level, score, rules = manager.assess_risk(context)
            >>> print(f"Risk Level: {level}, Score: {score}")
            Risk Level: high, Score: 8
        """
        rules_config = self.policy_config.get("rules", {})
        risk_levels_config = self.policy_config.get("risk_levels", {})
        
        total_score = 0.0
        applied_rules = []
        
        # 遍歷所有規則類別
        for category, rules in rules_config.items():
            for rule in rules:
                condition = rule.get("condition", "")
                
                # 簡單條件評估（支持基本比較運算）
                if self._evaluate_condition(condition, context):
                    score = rule.get("risk_score", 0)
                    total_score += score
                    
                    applied_rules.append({
                        "category": category,
                        "condition": condition,
                        "score": score,
                        "description": rule.get("description", ""),
                        "mitigation": rule.get("mitigation", "")
                    })
                    
                    logger.debug(f"   ✓ 規則匹配: {category} - {rule.get('description')} (+{score})")
        
        # 向後兼容：如果沒有任何規則匹配且提供了 task_type，使用簡單邏輯
        if total_score == 0 and task_type:
            logger.warning(f"⚠️ 沒有規則匹配，使用向後兼容邏輯 (task_type={task_type})")
            if task_type == "scan":
                return "low", 1.0, [{"legacy": True, "task_type": task_type}]
            elif task_type == "attack":
                return "high", 6.0, [{"legacy": True, "task_type": task_type}]
            else:
                return "medium", 3.0, [{"legacy": True, "task_type": task_type}]
        
        # 根據總分確定風險等級
        risk_level = self._determine_risk_level(total_score, risk_levels_config)
        
        logger.info(f"🎯 風險評估完成: {risk_level} (總分: {total_score})")
        return risk_level, total_score, applied_rules
    
    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """評估條件表達式
        
        支持的運算符：==, !=, is, is not
        
        Args:
            condition: 條件字符串，如 "target_type == 'production'"
            context: 上下文字典
        
        Returns:
            條件是否滿足
        """
        try:
            # 簡單安全的條件評估（避免使用 eval）
            # 支持格式: "key == 'value'" 或 "key is not None"
            
            if " == " in condition:
                key, value = condition.split(" == ")
                key = key.strip()
                value = value.strip().strip("'\"")
                return str(context.get(key, "")) == value
            
            elif " != " in condition:
                key, value = condition.split(" != ")
                key = key.strip()
                value = value.strip().strip("'\"")
                return str(context.get(key, "")) != value
            
            elif " is not " in condition:
                key, value = condition.split(" is not ")
                key = key.strip()
                if value.strip() == "None":
                    return context.get(key) is not None
                return False
            
            elif " is " in condition:
                key, value = condition.split(" is ")
                key = key.strip()
                if value.strip() == "None":
                    return context.get(key) is None
                return False
            
            else:
                logger.warning(f"⚠️ 不支持的條件格式: {condition}")
                return False
        
        except Exception as e:
            logger.error(f"❌ 條件評估失敗: {condition} - {e}")
            return False
    
    def _determine_risk_level(
        self,
        total_score: float,
        risk_levels_config: dict[str, Any]
    ) -> str:
        """根據總分確定風險等級
        
        Args:
            total_score: 總風險分數
            risk_levels_config: 風險等級配置
        
        Returns:
            風險等級字符串 (critical/high/medium/low)
        """
        # 按分數從高到低排序
        levels = [
            ("critical", risk_levels_config.get("critical", {})),
            ("high", risk_levels_config.get("high", {})),
            ("medium", risk_levels_config.get("medium", {})),
            ("low", risk_levels_config.get("low", {})),
        ]
        
        for level_name, level_config in levels:
            min_score = level_config.get("min_score", 0)
            max_score = level_config.get("max_score", float("inf"))
            
            if min_score <= total_score <= max_score:
                return level_name
        
        # 默認低風險
        return "low"
    
    def get_mitigation_actions(
        self,
        risk_level: str,
        applied_rules: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """獲取緩解措施建議
        
        Args:
            risk_level: 風險等級
            applied_rules: 應用的規則列表
        
        Returns:
            緩解措施字典
        """
        risk_levels_config = self.policy_config.get("risk_levels", {})
        level_config = risk_levels_config.get(risk_level, {})
        
        mitigations = [rule.get("mitigation", "") for rule in applied_rules if rule.get("mitigation")]
        
        return {
            "risk_level": risk_level,
            "mitigation_required": level_config.get("mitigation_required", False),
            "recommended_action": level_config.get("recommended_action", ""),
            "specific_mitigations": mitigations,
            "description": level_config.get("description", "")
        }

