"""風險策略管理器

負責加載和應用風險評估策略配置，支援動態策略切換。

符合 aiva_common 規範：
- 配置文件放在 config/ 目錄
- 使用 YAML 格式存儲策略
- 支援多環境/多客戶策略
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RiskRule:
    """風險規則"""
    condition: str
    risk_score: int
    description: str
    mitigation: str


@dataclass
class RiskLevel:
    """風險等級定義"""
    name: str
    min_score: int
    max_score: int | None
    description: str
    mitigation_required: bool
    recommended_action: str


class PolicyManager:
    """風險策略管理器
    
    功能：
    1. 加載 YAML 風險策略配置
    2. 評估情況並計算風險分數
    3. 提供動態策略切換（不同客戶/項目）
    4. 支援策略熱更新（生產環境）
    """

    def __init__(self, policy_path: str | None = None):
        """初始化策略管理器
        
        Args:
            policy_path: 策略文件路徑，默認使用 config/risk_policies.yaml
        """
        if policy_path is None:
            # 默認路徑：aiva_core/config/risk_policies.yaml
            config_dir = Path(__file__).parent.parent / "config"
            policy_path_obj = config_dir / "risk_policies.yaml"
        else:
            policy_path_obj = Path(policy_path)
        
        self.policy_path: Path = policy_path_obj
        self.policy: dict[str, Any] = {}
        self.rules: dict[str, list[RiskRule]] = {}
        self.risk_levels: dict[str, RiskLevel] = {}
        
        self._load_policy()

    def _load_policy(self):
        """加載風險策略配置"""
        try:
            if not self.policy_path.exists():
                raise FileNotFoundError(
                    f"❌ Policy file not found: {self.policy_path}\n"
                    f"   請創建策略配置文件"
                )
                self._use_fallback_policy()
                return
            
            with open(self.policy_path, encoding='utf-8') as f:
                self.policy = yaml.safe_load(f)
            
            # 解析規則
            rules_config = self.policy.get('rules', {})
            for category, category_rules in rules_config.items():
                self.rules[category] = [
                    RiskRule(
                        condition=rule['condition'],
                        risk_score=rule['risk_score'],
                        description=rule['description'],
                        mitigation=rule.get('mitigation', 'No mitigation specified')
                    )
                    for rule in category_rules
                ]
            
            # 解析風險等級
            levels_config = self.policy.get('risk_levels', {})
            for level_name, level_data in levels_config.items():
                self.risk_levels[level_name] = RiskLevel(
                    name=level_name,
                    min_score=level_data['min_score'],
                    max_score=level_data.get('max_score'),
                    description=level_data['description'],
                    mitigation_required=level_data['mitigation_required'],
                    recommended_action=level_data['recommended_action']
                )
            
            logger.info(
                f"✅ Loaded risk policy: {self.policy.get('policy_name')} "
                f"({len(self.rules)} categories, {sum(len(r) for r in self.rules.values())} rules)"
            )
            
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load policy from {self.policy_path}: {e}\n"
                f"   策略文件必須存在且格式正確"
            ) from e

    def assess_risk(self, situation: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
        """評估風險因素
        
        Args:
            situation: 當前情況（target_type, system_criticality 等）
            constraints: 約束條件（authorized, time_limit 等）
            
        Returns:
            風險評估結果：
            {
                'overall_risk': 'critical|high|medium|low',
                'risk_score': 5,
                'factors': ['description1', 'description2'],
                'mitigations': ['mitigation1', 'mitigation2'],
                'mitigation_required': True/False,
                'recommended_action': 'action description'
            }
        """
        factors = []
        mitigations = []
        risk_score = 0
        
        # 合併 situation 和 constraints 以便評估
        context = {**situation, **constraints}
        
        # 遍歷所有規則類別
        for category, category_rules in self.rules.items():
            for rule in category_rules:
                # 安全地評估條件
                try:
                    # 創建評估上下文
                    eval_context = {k: v for k, v in context.items()}
                    
                    # 評估條件（使用安全的 eval）
                    if self._evaluate_condition(rule.condition, eval_context):
                        risk_score += rule.risk_score
                        factors.append(rule.description)
                        mitigations.append(rule.mitigation)
                        
                        logger.debug(
                            f"Risk rule triggered: {rule.description} "
                            f"(+{rule.risk_score} points)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to evaluate rule '{rule.condition}': {e}")
        
        # 確定風險等級
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            'overall_risk': risk_level.name,
            'risk_score': risk_score,
            'factors': factors,
            'mitigations': mitigations,
            'mitigation_required': risk_level.mitigation_required,
            'recommended_action': risk_level.recommended_action,
            'policy_version': self.policy.get('version', 'unknown')
        }

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """安全地評估條件表達式
        
        Args:
            condition: 條件字符串（如 "target_type == 'production'"）
            context: 評估上下文
            
        Returns:
            條件是否滿足
        """
        try:
            # 簡單的條件解析（避免使用 eval）
            # 支援格式: "key == 'value'" 或 "key == True"
            
            if '==' in condition:
                left, right = condition.split('==')
                left = left.strip()
                right = right.strip().strip('"\'')
                
                # 獲取左側值
                left_value = context.get(left)
                
                # 比較
                if right.lower() == 'true':
                    return left_value is True
                elif right.lower() == 'false':
                    return left_value is False
                elif right.lower() == 'none':
                    return left_value is not None  # 注意："is not None" 表示存在
                else:
                    return str(left_value) == right
            
            elif ' is not None' in condition:
                key = condition.replace(' is not None', '').strip()
                return context.get(key) is not None
            
            else:
                # 直接布爾值檢查
                return bool(context.get(condition.strip(), False))
                
        except Exception as e:
            logger.warning(f"Condition evaluation failed for '{condition}': {e}")
            return False

    def _determine_risk_level(self, risk_score: int) -> RiskLevel:
        """根據風險分數確定風險等級"""
        # 從高到低檢查
        for level_name in ['critical', 'high', 'medium', 'low']:
            level = self.risk_levels.get(level_name)
            if level:
                if level.max_score is None:
                    # 無上限（如 critical >= 7）
                    if risk_score >= level.min_score:
                        return level
                else:
                    # 有上下限
                    if level.min_score <= risk_score <= level.max_score:
                        return level
        
        # 降級：如果沒有匹配，返回 low
        return self.risk_levels.get('low', RiskLevel(
            'low', 0, None, 'Unknown risk level', False, 'Review manually'
        ))

    def reload_policy(self, policy_path: str | None = None):
        """重新加載策略（熱更新支援）
        
        Args:
            policy_path: 新的策略文件路徑，None 則使用當前路徑
        """
        if policy_path:
            self.policy_path = Path(policy_path)
        
        logger.info(f"Reloading risk policy from {self.policy_path}")
        self._load_policy()

    def get_policy_info(self) -> dict[str, Any]:
        """獲取當前策略信息"""
        return {
            'policy_name': self.policy.get('policy_name', 'unknown'),
            'version': self.policy.get('version', 'unknown'),
            'description': self.policy.get('description', ''),
            'rule_categories': list(self.rules.keys()),
            'total_rules': sum(len(rules) for rules in self.rules.values()),
            'risk_levels': list(self.risk_levels.keys())
        }
