"""攻擊計劃建構器

負責生成攻擊計畫和提示詞建構
"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class PlanBuilder:
    """攻擊計劃建構器
    
    負責使用 RAG 和 5M 引擎生成攻擊計畫
    """

    def __init__(
        self,
        rag_engine: Any,
        decision_engine: Any,
        experience_manager: Any,
    ):
        """初始化計劃建構器
        
        Args:
            rag_engine: RAG 引擎
            decision_engine: 5M 決策引擎
            experience_manager: 經驗管理器
        """
        self.rag_engine = rag_engine
        self.decision_engine = decision_engine
        self.experience_manager = experience_manager

    async def plan_attack(self, context: dict[str, Any]) -> dict[str, Any]:
        """生成攻擊計畫（RAG 增強）

        Args:
            context: 包含 target, objective 等

        Returns:
            攻擊計畫結果
        """
        logger.info("📋 Generating attack plan with RAG enhancement...")

        target = context.get("target")
        objective = context.get("objective", "Comprehensive security assessment")
        constraints = context.get("constraints", {})

        if not target:
            return {"success": False, "error": "Target not specified"}

        try:
            # 1. 使用 RAG 檢索相關知識
            from services.aiva_common.schemas import AttackTarget
            attack_target = AttackTarget(
                target_id=f"target_{uuid4().hex[:12]}",
                target_url=target,
                target_type="web",
                description=objective
            )
            rag_context = await self.rag_engine.enhance_attack_plan(
                target=attack_target,
                objective=objective,
            )

            # 2. 從經驗庫獲取歷史成功案例
            historical_experiences = self.experience_manager.get_high_quality_samples(
                min_quality=0.6,
                limit=50
            )
            # 轉換為 dict 格式
            historical_experiences = [exp.to_dict() for exp in historical_experiences]

            # 3. 使用 5M Decision Engine 生成計畫
            target_features = self._encode_target_for_neural(target, rag_context)
            
            task_desc_str = str(target_features)
            context_str = str({
                "objective": objective,
                "historical_count": len(historical_experiences),
                "constraints": constraints,
            })
            neural_decision = self.decision_engine.generate_decision(
                task_description=task_desc_str,
                context=context_str,
            )

            # 使用神經網路輸出構建計畫結構
            plan_response = self._build_plan_from_neural_decision(
                neural_decision=neural_decision,
                target=target,
                objective=objective,
                rag_context=rag_context,
                historical_experiences=historical_experiences,
            )

            # 4. 構建完整的攻擊計畫
            plan_id = f"plan_{uuid4().hex[:12]}"

            attack_plan = {
                "plan_id": plan_id,
                "target": target,
                "objective": objective,
                "phases": plan_response.get("phases", []),
                "risk_assessment": plan_response.get("risk_assessment", ""),
                "success_criteria": plan_response.get("success_criteria", []),
                "rag_context": {
                    "similar_techniques": rag_context.get("similar_techniques", []),
                    "successful_experiences_count": len(historical_experiences),
                },
                "confidence": self._calculate_plan_confidence(
                    rag_context, historical_experiences
                ),
                "created_at": datetime.now().isoformat(),
            }

            logger.info(
                f"✅ Generated plan {plan_id} with {len(attack_plan['phases'])} phases, "
                f"confidence: {attack_plan['confidence']:.2f}"
            )

            return {
                "success": True,
                "plan": attack_plan,
                "confidence": attack_plan["confidence"],
            }

        except Exception as e:
            logger.error(f"Failed to generate attack plan: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "fallback_message": "Plan generation failed, using basic strategy",
            }

    def _encode_target_for_neural(
        self, target: str, rag_context: dict[str, Any]
    ) -> dict[str, Any]:
        """將目標信息編碼為神經網路輸入特徵"""
        import hashlib
        
        target_hash = int(hashlib.md5(target.encode()).hexdigest()[:8], 16) / (16**8)
        
        similar_techs = rag_context.get("similar_techniques", [])
        tech_scores = [t.get("score", 0.5) for t in similar_techs[:5]]
        avg_similarity = sum(tech_scores) / len(tech_scores) if tech_scores else 0.5
        
        return {
            "target_hash": target_hash,
            "target_length": len(target) / 100.0,
            "has_ip": 1.0 if any(c.isdigit() for c in target.split('.')) else 0.0,
            "has_domain": 1.0 if '.' in target else 0.0,
            "rag_similarity": avg_similarity,
            "rag_tech_count": len(similar_techs) / 10.0,
        }

    def _build_plan_from_neural_decision(
        self,
        neural_decision: dict,
        target: str,
        objective: str,
        rag_context: dict[str, Any],
        historical_experiences: list[dict],
    ) -> dict[str, Any]:
        """從神經網路輸出構建完整的攻擊計畫"""
        attack_vector = neural_decision.get("attack_vector", "reconnaissance")
        confidence = neural_decision.get("confidence", 0.5)
        recommended_tools = neural_decision.get("recommended_tools", [])
        
        # 預定義的階段模板
        phase_templates = {
            "reconnaissance": {
                "name": "Reconnaissance",
                "description": "Information gathering and target enumeration",
                "steps": ["Domain/IP enumeration", "Port scanning", "Service detection", "Technology fingerprinting"],
                "expected_duration": "1-2 hours"
            },
            "vulnerability_scan": {
                "name": "Vulnerability Analysis",
                "description": "Identify potential vulnerabilities",
                "steps": ["Automated vulnerability scan", "Manual verification", "CVE mapping", "Risk assessment"],
                "expected_duration": "2-4 hours"
            },
            "exploitation": {
                "name": "Exploitation Planning",
                "description": "Plan and validate attack vectors",
                "steps": ["Exploit selection", "Payload preparation", "Environment setup", "Attack simulation"],
                "expected_duration": "3-6 hours"
            },
            "reporting": {
                "name": "Validation & Reporting",
                "description": "Document findings and recommendations",
                "steps": ["Finding documentation", "Evidence collection", "Risk scoring", "Remediation advice"],
                "expected_duration": "2-3 hours"
            }
        }
        
        # 根據攻擊向量選擇階段組合
        if attack_vector in ["recon", "reconnaissance", "information_gathering"]:
            selected_phases = ["reconnaissance", "vulnerability_scan"]
        elif attack_vector in ["exploit", "attack", "penetration"]:
            selected_phases = ["reconnaissance", "vulnerability_scan", "exploitation", "reporting"]
        else:
            selected_phases = ["reconnaissance", "vulnerability_scan", "reporting"]
        
        phases = [phase_templates[p] for p in selected_phases if p in phase_templates]
        
        # 根據推薦工具調整步驟
        if recommended_tools:
            for phase in phases:
                if "scan" in phase["name"].lower() and recommended_tools:
                    phase["steps"].insert(0, f"Use tools: {', '.join(recommended_tools[:3])}")
        
        # 風險評估
        if confidence > 0.8:
            risk_assessment = "Low - High confidence in approach based on similar historical cases"
        elif confidence > 0.6:
            risk_assessment = "Medium - Moderate confidence, recommend careful monitoring"
        else:
            risk_assessment = "High - Lower confidence, consider alternative approaches"
        
        return {
            "plan_id": str(uuid4()),
            "target": target,
            "objective": objective,
            "phases": phases,
            "risk_assessment": risk_assessment,
            "success_criteria": [
                "Complete information gathering for target",
                "Identify at least 3 potential vulnerabilities",
                "Validate findings with evidence",
                "Generate comprehensive report"
            ],
            "neural_confidence": confidence,
            "attack_vector": attack_vector,
        }

    def _calculate_plan_confidence(
        self, rag_context: dict[str, Any], historical_experiences: list[dict]
    ) -> float:
        """計算計畫信心度"""
        confidence = 0.3

        # 1. RAG 相似技術加成
        similar_techs = rag_context.get("similar_techniques", [])
        if similar_techs:
            tech_count_bonus = min(len(similar_techs) * 0.03, 0.15)
            avg_score = (
                sum(t.get("score", 0) for t in similar_techs) / len(similar_techs)
                if similar_techs else 0
            )
            score_bonus = avg_score * 0.1
            confidence += tech_count_bonus + score_bonus

        # 2. 歷史經驗加成
        if historical_experiences:
            exp_count = len(historical_experiences)
            count_factor = min(exp_count / 10, 1.0)
            success_exps = [e for e in historical_experiences if e.get("score", 0) > 0.7]
            success_rate = len(success_exps) / exp_count if exp_count > 0 else 0
            
            from datetime import timedelta
            recent_threshold = (datetime.now() - timedelta(days=7)).isoformat()
            recent_count = len([
                e for e in historical_experiences
                if e.get("timestamp", "") > recent_threshold
            ])
            recent_bonus = min(recent_count / exp_count * 0.05, 0.05) if exp_count > 0 else 0
            
            historical_bonus = (success_rate * count_factor * 0.3) + recent_bonus
            confidence += historical_bonus

        # 3. 組合效應加成
        if len(similar_techs) >= 3 and len(historical_experiences) >= 5:
            success_rate = len([
                e for e in historical_experiences if e.get("score", 0) > 0.7
            ]) / len(historical_experiences)
            if success_rate > 0.7:
                confidence += 0.05

        confidence = max(0.3, min(confidence, 0.95))

        logger.debug(
            f"Plan confidence calculated: {confidence:.3f} "
            f"(techs={len(similar_techs)}, exps={len(historical_experiences)})"
        )

        return confidence

    # ===== 提示詞建構方法 =====
    
    def build_plan_generation_prompt(
        self,
        target: str,
        objective: str,
        rag_context: dict[str, Any],
        historical_experiences: list[dict],
        constraints: dict[str, Any],
    ) -> str:
        """構建計畫生成提示詞"""
        prompt = self._build_base_prompt(target, objective)
        prompt += self._build_rag_section(rag_context)
        prompt += self._build_historical_section(historical_experiences)
        prompt += self._build_constraints_section(constraints)
        prompt += self._build_output_structure()
        return prompt

    def _build_base_prompt(self, target: str, objective: str) -> str:
        """構建基本提示詞部分"""
        return f"""Generate a comprehensive security testing attack plan for:

🎯 Target: {target}
📋 Objective: {objective}

"""

    def _build_rag_section(self, rag_context: dict[str, Any]) -> str:
        """構建 RAG 知識庫部分"""
        similar_techs = rag_context.get("similar_techniques", [])
        if not similar_techs:
            return ""
            
        section = "🔍 Similar Techniques from Knowledge Base:\n"
        for idx, tech in enumerate(similar_techs[:5], 1):
            section += f"{idx}. {tech.get('name', 'N/A')}\n"
            section += f"   - Description: {tech.get('description', 'N/A')}\n"
            section += f"   - Relevance Score: {tech.get('score', 0):.2f}\n"
            if tech.get("tags"):
                section += f"   - Tags: {', '.join(tech.get('tags', []))}\n"
        return section + "\n"

    def _build_historical_section(self, historical_experiences: list[dict]) -> str:
        """構建歷史經驗部分"""
        if not historical_experiences:
            return ""
            
        categories = self._categorize_experiences(historical_experiences)
        section = self._build_experience_stats(historical_experiences, categories)
        section += self._build_success_cases(categories['success'])
        section += self._build_failure_lessons(categories['failed'])
        return section + "\n"

    def _categorize_experiences(self, experiences: list[dict]) -> dict[str, list[dict]]:
        """將經驗按效果分類"""
        return {
            'success': [e for e in experiences if e.get("score", 0) > 0.7],
            'medium': [e for e in experiences if 0.4 <= e.get("score", 0) <= 0.7],
            'failed': [e for e in experiences if e.get("score", 0) < 0.4]
        }

    def _build_experience_stats(
        self, all_exps: list[dict], categories: dict[str, list[dict]]
    ) -> str:
        """構建經驗統計信息"""
        total = len(all_exps)
        success_rate = len(categories['success']) / total * 100
        medium_rate = len(categories['medium']) / total * 100
        failure_rate = len(categories['failed']) / total * 100
        
        return f"""📊 Historical Performance Analysis:
   - Total Experiences: {total}
   - ✅ Success Rate: {success_rate:.1f}%
   - ⚠️ Partial Success: {medium_rate:.1f}%
   - ❌ Failure Rate: {failure_rate:.1f}%
"""

    def _build_success_cases(self, success_exps: list[dict]) -> str:
        """構建成功案例部分"""
        if not success_exps:
            return ""
            
        section = "\n🌟 Top Successful Cases:\n"
        for exp in success_exps[:2]:
            context = exp.get("context", {})
            action = exp.get("action", {})
            section += f"   - Strategy: {action.get('decision', 'N/A')}\n"
            section += f"     Score: {exp.get('score', 0):.2f}, Type: {context.get('objective', 'N/A')}\n"
        return section

    def _build_failure_lessons(self, failed_exps: list[dict]) -> str:
        """構建失敗教訓部分"""
        if not failed_exps:
            return ""
            
        section = "\n⚠️ Lessons from Failed Attempts:\n"
        for exp in failed_exps[:2]:
            result = exp.get("result", {})
            section += f"   - Avoid: {result.get('error', 'Unknown error')}\n"
        return section

    def _build_constraints_section(self, constraints: dict[str, Any]) -> str:
        """構建約束條件部分"""
        if not constraints:
            return ""
            
        section = "🚧 Constraints:\n"
        for key, value in constraints.items():
            section += f"   - {key}: {value}\n"
        return section + "\n"

    def _build_output_structure(self) -> str:
        """構建輸出結構要求部分"""
        return """🎯 Required Output Structure:
1. **Multi-Phase Plan**:
   - Phase 1: Reconnaissance (information gathering)
   - Phase 2: Vulnerability Analysis (identify weaknesses)
   - Phase 3: Exploitation Planning (prepare attack vectors)
   - Phase 4: Validation & Reporting (verify findings)

2. **Risk Assessment**:
   - Identify potential risks for each phase
   - Categorize as Low/Medium/High/Critical
   - Suggest mitigation strategies

3. **Success Criteria**:
   - Measurable objectives for each phase
   - Clear indicators of completion
   - Fallback plans if primary approach fails

4. **Dynamic Adaptation**:
   - Conditional steps based on intermediate results
   - Alternative paths if obstacles encountered
   - Real-time adjustment triggers

⚖️ Focus on: Practical, safe, authorized security testing approaches.
🔒 Ensure: Compliance with ethical hacking standards and legal boundaries.
"""
