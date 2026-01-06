"""策略決策引擎

負責策略決策、風險評估和信心度計算
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class StrategyEngine:
    """策略決策引擎
    
    使用 5M 神經網路進行策略決策
    """

    def __init__(
        self,
        decision_engine: Any,
        experience_manager: Any,
    ):
        """初始化策略引擎
        
        Args:
            decision_engine: 5M 決策引擎
            experience_manager: 經驗管理器
        """
        self.decision_engine = decision_engine
        self.experience_manager = experience_manager

    async def make_strategy_decision(self, context: dict[str, Any]) -> dict[str, Any]:
        """策略決策

        Args:
            context: 決策上下文

        Returns:
            決策結果
        """
        logger.info("🤔 Making strategic decision with enhanced risk assessment...")

        try:
            situation = context.get("situation", {})
            options = context.get("options", [])
            constraints = context.get("constraints", {})

            # 1. 從經驗庫獲取相似情況的歷史決策
            historical_decisions = await self._get_similar_decisions(situation)

            # 2. 風險預評估
            risk_factors = self.assess_risk_factors(situation, constraints)

            # 3. 使用 5M Decision Engine 進行策略決策
            situation_features = self._encode_situation_for_neural(
                situation, options, risk_factors
            )
            
            # 4. 調用 5M 決策引擎
            situation_str = str(situation_features)
            context_str = str({
                "options": options,
                "constraints": constraints,
                "historical_count": len(historical_decisions),
            })
            neural_decision = self.decision_engine.generate_decision(
                task_description=situation_str,
                context=context_str,
            )
            
            # 從神經網路輸出構建決策響應
            decision_response = self._build_strategy_from_neural_decision(
                neural_decision=neural_decision,
                situation=situation,
                options=options,
                historical_decisions=historical_decisions,
                risk_factors=risk_factors,
            )

            # 5. 多維度信心度計算
            ai_confidence = decision_response.get("confidence", neural_decision.get("confidence", 0.5))
            historical_confidence = self._calculate_historical_confidence(historical_decisions)
            risk_adjusted_confidence = self._adjust_confidence_by_risk(
                base_confidence=(ai_confidence * 0.6) + (historical_confidence * 0.4),
                risk_factors=risk_factors,
            )

            # 6. 構建完整決策結果
            result = {
                "success": True,
                "decision": decision_response.get("decision", "proceed_with_caution"),
                "confidence": risk_adjusted_confidence,
                "reasoning": decision_response.get("reasoning", "Based on AI analysis"),
                "alternative_options": decision_response.get("alternative_options", []),
                "risks": decision_response.get("risks", []),
                "success_indicators": decision_response.get("success_indicators", []),
                "fallback_plan": decision_response.get("fallback_plan", "Abort and reassess"),
                "risk_assessment": {
                    "overall_risk": risk_factors.get("overall_risk", "medium"),
                    "key_factors": risk_factors.get("factors", []),
                    "mitigation_required": risk_factors.get("mitigation_required", False),
                },
                "historical_reference_count": len(historical_decisions),
                "decision_metadata": {
                    "ai_confidence": ai_confidence,
                    "historical_confidence": historical_confidence,
                    "risk_adjustment": risk_adjusted_confidence - ((ai_confidence * 0.6) + (historical_confidence * 0.4)),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.info(
                f"✅ Decision made: {result['decision']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"risk: {risk_factors.get('overall_risk', 'unknown')})"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Decision making failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "decision": "abort",
                "confidence": 0.0,
                "reasoning": "Decision process encountered an error. Aborting for safety.",
                "fallback_plan": "Manual review required",
            }

    def assess_risk_factors(
        self, situation: dict[str, Any], constraints: dict[str, Any]
    ) -> dict[str, Any]:
        """評估風險因素

        Returns:
            風險評估結果
        """
        factors = []
        risk_score = 0

        # 1. 目標環境風險
        if situation.get("target_type") == "production":
            factors.append("Production environment - High impact potential")
            risk_score += 3
        elif situation.get("target_type") == "staging":
            factors.append("Staging environment - Medium impact")
            risk_score += 1

        # 2. 時間約束風險
        if constraints.get("time_limit"):
            factors.append("Time-constrained operation - Reduced testing window")
            risk_score += 2

        # 3. 授權範圍風險
        if not constraints.get("authorized"):
            factors.append("⚠️ CRITICAL: Unauthorized testing - Legal risk")
            risk_score += 5

        # 4. 資料敏感度風險
        if situation.get("contains_sensitive_data"):
            factors.append("Sensitive data present - Privacy concerns")
            risk_score += 2

        # 5. 系統關鍵度風險
        if situation.get("system_criticality") == "high":
            factors.append("Critical system - Service disruption risk")
            risk_score += 3

        # 計算總體風險等級
        if risk_score >= 7:
            overall_risk = "critical"
            mitigation_required = True
        elif risk_score >= 4:
            overall_risk = "high"
            mitigation_required = True
        elif risk_score >= 2:
            overall_risk = "medium"
            mitigation_required = False
        else:
            overall_risk = "low"
            mitigation_required = False

        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "factors": factors,
            "mitigation_required": mitigation_required,
        }

    def _encode_situation_for_neural(
        self, situation: dict, options: list, risk_factors: dict
    ) -> dict[str, Any]:
        """將情況編碼為神經網路決策輸入"""
        return {
            "option_count": len(options) / 10.0,
            "risk_level": risk_factors.get("overall_risk", 0.5),
            "urgency": situation.get("urgency", 0.5),
            "complexity": situation.get("complexity", 0.5),
            "resource_available": situation.get("resources", 1.0),
        }

    def _build_strategy_from_neural_decision(
        self,
        neural_decision: dict,
        situation: dict,
        options: list,
        historical_decisions: list[dict],
        risk_factors: dict,
    ) -> dict[str, Any]:
        """從神經網路輸出構建策略決策響應"""
        attack_vector = neural_decision.get("attack_vector", "conservative")
        confidence = neural_decision.get("confidence", 0.5)
        recommended_tools = neural_decision.get("recommended_tools", [])
        
        # 根據神經網路輸出選擇最佳選項
        if options:
            selected_idx = int(confidence * len(options)) % len(options)
            selected_option = options[selected_idx]
        else:
            selected_option = attack_vector
        
        # 預定義的推理模板
        reasoning_templates = {
            "high_confidence": f"基於 5M 決策引擎分析，{attack_vector} 策略有 {confidence:.1%} 信心度。歷史數據支持此決策。",
            "medium_confidence": f"5M 引擎建議 {attack_vector} 策略，信心度 {confidence:.1%}。建議謹慎執行並監控結果。",
            "low_confidence": f"當前情況複雜，5M 引擎信心度為 {confidence:.1%}。建議採取保守的 {attack_vector} 策略。"
        }
        
        if confidence > 0.7:
            reasoning = reasoning_templates["high_confidence"]
        elif confidence > 0.4:
            reasoning = reasoning_templates["medium_confidence"]
        else:
            reasoning = reasoning_templates["low_confidence"]
        
        # 構建風險列表
        overall_risk = risk_factors.get("overall_risk", 0.5)
        risks = []
        if overall_risk > 0.7:
            risks.append({
                "description": "High-risk operation detected",
                "severity": "high",
                "mitigation": "Implement staged rollout with manual checkpoints"
            })
        if confidence < 0.5:
            risks.append({
                "description": "Low confidence decision",
                "severity": "medium", 
                "mitigation": "Prepare fallback strategies before execution"
            })
        
        return {
            "decision": selected_option,
            "reasoning": reasoning,
            "confidence": confidence,
            "alternative_options": [opt for opt in options if opt != selected_option][:3],
            "risks": risks,
            "success_indicators": [
                f"成功執行 {attack_vector} 策略",
                "達成預期目標",
                "無意外錯誤或警報"
            ],
            "fallback_plan": f"如果 {selected_option} 失敗，考慮 {options[0] if options else 'conservative approach'}",
            "recommended_tools": recommended_tools,
        }

    def _adjust_confidence_by_risk(
        self, base_confidence: float, risk_factors: dict[str, Any]
    ) -> float:
        """根據風險因素調整信心度"""
        overall_risk = risk_factors.get("overall_risk", "medium")

        if overall_risk == "critical":
            adjustment = -0.2
        elif overall_risk == "high":
            adjustment = -0.1
        elif overall_risk == "medium":
            adjustment = -0.05
        else:
            adjustment = 0.0

        adjusted = base_confidence + adjustment
        return max(0.1, min(adjusted, 0.95))

    async def _get_similar_decisions(self, situation: dict[str, Any]) -> list[dict]:
        """獲取相似情況的歷史決策"""
        try:
            all_experiences = self.experience_manager.get_high_quality_samples(
                min_quality=0.5,
                limit=100
            )
            similar_decisions = []
            for exp in all_experiences:
                exp_dict = exp.to_dict()
                if exp_dict.get("metadata", {}).get("context", {}).get("type") == situation.get("type"):
                    similar_decisions.append(exp_dict)
            return similar_decisions[:10]
        except Exception as e:
            logger.error(f"Failed to retrieve similar decisions: {e}")
            return []

    def _calculate_historical_confidence(self, historical_decisions: list[dict]) -> float:
        """根據歷史決策計算信心度"""
        if not historical_decisions:
            return 0.5

        success_count = len([d for d in historical_decisions if d.get("score", 0) > 0.7])
        return success_count / len(historical_decisions) if historical_decisions else 0.5

    def build_strategy_decision_prompt(
        self,
        situation: dict[str, Any],
        options: list[str],
        constraints: dict[str, Any],
        historical_decisions: list[dict],
        risk_factors: dict[str, Any],
    ) -> str:
        """構建策略決策提示詞"""
        prompt = f"""Analyze the following situation and make a strategic decision:

📋 **Situation Analysis**:
{situation}

⚖️ **Available Options**:
"""
        for idx, option in enumerate(options, 1):
            prompt += f"{idx}. {option}\n"

        if constraints:
            prompt += "\n🚧 **Constraints**:\n"
            for key, value in constraints.items():
                prompt += f"   - {key}: {value}\n"

        prompt += "\n⚠️ **Risk Assessment**:\n"
        prompt += f"   - Overall Risk Level: {risk_factors.get('overall_risk', 'unknown').upper()}\n"
        prompt += f"   - Risk Score: {risk_factors.get('risk_score', 0)}/10\n"
        if risk_factors.get("factors"):
            prompt += "   - Key Risk Factors:\n"
            for factor in risk_factors["factors"]:
                prompt += f"     • {factor}\n"

        if historical_decisions:
            success_rate = (
                len([d for d in historical_decisions if d.get("score", 0) > 0.7])
                / len(historical_decisions)
                * 100
            )
            prompt += "\n📊 **Historical Decisions** (similar situations):\n"
            prompt += f"   - Total References: {len(historical_decisions)}\n"
            prompt += f"   - Success Rate: {success_rate:.1f}%\n"
            prompt += "   - Top Cases:\n"
            for hist in historical_decisions[:2]:
                prompt += f"     • Decision: {hist.get('action', {}).get('decision', 'N/A')}\n"
                prompt += f"       Outcome: {'✅ Success' if hist.get('score', 0) > 0.7 else '⚠️ Partial'}\n"

        prompt += """
🎯 **Required Output**:
Please provide a comprehensive decision包含:
1. **Primary Decision**: Clear, actionable choice
2. **Reasoning**: Detailed explanation of decision logic
3. **Confidence Level**: 0.0-1.0 based on available information
4. **Alternative Options**: Backup choices if primary fails
5. **Risk Analysis**: Specific risks with severity (Low/Medium/High/Critical) and mitigation strategies
6. **Success Indicators**: Measurable criteria to validate decision effectiveness
7. **Fallback Plan**: What to do if decision leads to negative outcomes

⚖️ **Decision Criteria**:
- Prioritize safety and authorization compliance
- Balance effectiveness with risk level
- Consider time and resource constraints
- Learn from historical outcomes
"""
        return prompt
