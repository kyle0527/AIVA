"""攻擊計劃建構器 - 負責生成攻擊計畫和提示詞建構"""

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class PlanBuilder:
    """攻擊計劃建構器
    
    負責使用 RAG 和 5M 引擎生成攻擊計畫
    使用 CLI 執行架構（subprocess），不直接依賴其他模組
    """

    def __init__(
        self,
        data_directory: Any = None,
        rag_engine: Any = None,
        decision_engine: Any = None,
        experience_manager: Any = None,
    ):
        """初始化計劃建構器
        
        Args:
            data_directory: 數據目錄路徑（用於存儲計劃）
            rag_engine: RAG 引擎實例
            decision_engine: 決策引擎實例
            experience_manager: 經驗管理器實例
        """
        self.data_directory = data_directory
        self.rag_engine = rag_engine
        self.decision_engine = decision_engine
        self.experience_manager = experience_manager
        # CLI 架構 - 透過 subprocess 調用 RAG/5M 服務
        self.feedback_history = []
        self.strategy_performance = {}

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
            from aiva_common.schemas import AttackTarget
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

            # 2.5. 分析反饋數據，獲取策略建議
            feedback_insights = self._analyze_feedback_for_planning(
                target=target,
                objective=objective
            )
            logger.info(f"📊 Feedback insights: {feedback_insights.get('summary', 'No data')}")

            # 3. 使用 5M Decision Engine 生成計畫（包含反饋洞察）
            target_features = self._encode_target_for_neural(target, rag_context, feedback_insights)
            
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
        self, target: str, rag_context: dict[str, Any], feedback_insights: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """將目標信息編碼為神經網路輸入特徵（包含反饋數據）"""
        import hashlib
        
        target_hash = int(hashlib.md5(target.encode()).hexdigest()[:8], 16) / (16**8)
        
        similar_techs = rag_context.get("similar_techniques", [])
        tech_scores = [t.get("score", 0.5) for t in similar_techs[:5]]
        avg_similarity = sum(tech_scores) / len(tech_scores) if tech_scores else 0.5
        
        # 整合反饋數據
        feedback_insights = feedback_insights or {}
        
        return {
            "target_hash": target_hash,
            "target_length": len(target) / 100.0,
            "has_ip": 1.0 if any(c.isdigit() for c in target.split('.')) else 0.0,
            "has_domain": 1.0 if '.' in target else 0.0,
            "rag_similarity": avg_similarity,
            "rag_tech_count": len(similar_techs) / 10.0,
            # 反饋驅動特徵
            "avg_success_rate": feedback_insights.get("avg_success_rate", 0.5),
            "waf_risk_score": feedback_insights.get("waf_risk_score", 0.5),
            "error_probability": feedback_insights.get("error_probability", 0.3),
            "recommended_strategy_score": feedback_insights.get("best_strategy_score", 0.5),
        }

    def _build_plan_from_neural_decision(
        self,
        neural_decision: dict,
        target: str,
        objective: str,
    ) -> dict[str, Any]:
        """從神經網路輸出構建完整的攻擊計畫
        
        Args:
            neural_decision: 神經網路決策結果
            target: 目標地址
            objective: 攻擊目標
            
        Returns:
            攻擊計畫字典
        """
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

    # ===== 反饋分析方法（關鍵決策支援）=====
    
    def _analyze_feedback_for_planning(
        self,
        target: str,
        objective: str
    ) -> dict[str, Any]:
        """分析歷史反饋數據，為規劃提供決策支援
        
        這是決策系統的核心方法，回答以下問題：
        1. 對於類似目標，什麼策略最成功？
        2. 常見的失敗模式是什麼？
        3. 預期的成功率和執行時間是多少？
        4. 需要避免什麼陷阱（WAF觸發、錯誤等）？
        
        Args:
            target: 目標 URL/IP
            objective: 攻擊目標描述
            
        Returns:
            反饋洞察字典，包含決策所需的關鍵信息
        """
        if not self.feedback_history:
            return self._get_default_feedback_insights()
        
        # 1. 找出類似目標的歷史反饋
        similar_feedbacks = self._find_similar_target_feedbacks(target, objective)
        
        # 2. 計算策略成功率
        strategy_stats = self._calculate_strategy_success_rates()
        
        # 3. 識別成功模式和失敗模式
        success_patterns = self._identify_success_patterns(similar_feedbacks)
        failure_patterns = self._identify_failure_patterns(similar_feedbacks)
        
        # 4. 計算風險指標
        waf_risk = self._calculate_waf_risk(similar_feedbacks)
        error_probability = self._calculate_error_probability(similar_feedbacks)
        
        # 5. 推薦最佳策略
        best_strategy = self._recommend_best_strategy(strategy_stats)
        
        insights = {
            "summary": f"基於 {len(similar_feedbacks)} 個類似目標的歷史數據",
            "similar_target_count": len(similar_feedbacks),
            "avg_success_rate": self._calculate_avg_success_rate(similar_feedbacks),
            "avg_execution_time": self._calculate_avg_execution_time(similar_feedbacks),
            "waf_risk_score": waf_risk,
            "error_probability": error_probability,
            "best_strategy": best_strategy,
            "best_strategy_score": strategy_stats.get(best_strategy, {}).get("avg_score", 0.5),
            "success_patterns": success_patterns,
            "failure_patterns": failure_patterns,
            "recommended_adjustments": self._generate_planning_adjustments(
                waf_risk, error_probability
            ),
            # 定義成功標準（關鍵！告訴系統什麼是成功）
            "success_criteria": self._define_success_criteria(objective, similar_feedbacks),
        }
        
        logger.info(
            f"📊 Feedback analysis: {insights['similar_target_count']} similar targets, "
            f"success_rate={insights['avg_success_rate']:.2%}, "
            f"best_strategy={best_strategy}"
        )
        
        return insights
    
    def _get_default_feedback_insights(self) -> dict[str, Any]:
        """當沒有歷史反饋時的默認洞察"""
        return {
            "summary": "無歷史數據，使用默認策略",
            "similar_target_count": 0,
            "avg_success_rate": 0.5,  # 中性默認值
            "avg_execution_time": 60.0,  # 假設 60 秒
            "waf_risk_score": 0.5,  # 未知風險
            "error_probability": 0.3,  # 保守估計
            "best_strategy": "reconnaissance",  # 默認從偵察開始
            "best_strategy_score": 0.5,
            "success_patterns": [],
            "failure_patterns": [],
            "recommended_adjustments": [],
            "success_criteria": self._get_default_success_criteria(),
        }
    
    def _find_similar_target_feedbacks(self, target: str, objective: str) -> list:
        """找出類似目標的歷史反饋"""
        similar = []
        target_domain = self._extract_domain(target)
        objective_keywords = set(objective.lower().split())
        
        for feedback in self.feedback_history:
            fb_target = feedback.metadata.get("target", "")
            fb_objective = feedback.metadata.get("objective", "")
            
            # 相似度計算
            score = 0
            
            # 同域名加分
            if target_domain and target_domain in fb_target:
                score += 0.5
            
            # 目標關鍵詞匹配
            fb_keywords = set(fb_objective.lower().split())
            keyword_overlap = len(objective_keywords & fb_keywords)
            if keyword_overlap > 0:
                score += min(keyword_overlap * 0.2, 0.5)
            
            # 相似度 > 0.3 則納入
            if score > 0.3:
                similar.append(feedback)
        
        return similar
    
    def _extract_domain(self, target: str) -> str:
        """從目標提取域名"""
        import re
        # 移除協議
        target = re.sub(r'^https?://', '', target)
        # 提取域名部分
        match = re.match(r'^([^/:]+)', target)
        return match.group(1) if match else ""
    
    def _calculate_strategy_success_rates(self) -> dict[str, dict]:
        """計算各策略的成功率統計"""
        stats = {}
        
        for strategy, scores in self.strategy_performance.items():
            if scores:
                stats[strategy] = {
                    "avg_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores),
                    "total_uses": len(scores),
                    "success_rate": len([s for s in scores if s > 0.6]) / len(scores),
                }
        
        return stats
    
    def _identify_success_patterns(self, feedbacks: list) -> list[dict]:
        """識別成功模式"""
        patterns = []
        success_feedbacks = [f for f in feedbacks if f.success_rate > 0.7]
        
        # 統計成功案例的共同特徵
        strategy_counts = {}
        for fb in success_feedbacks:
            for strategy in fb.strategies_used:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 找出高頻成功策略
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1])[:3]:
            if count >= 2:  # 至少出現 2 次
                patterns.append({
                    "type": "high_success_strategy",
                    "strategy": strategy,
                    "occurrence": count,
                    "recommendation": f"優先使用 {strategy} 策略",
                })
        
        return patterns
    
    def _identify_failure_patterns(self, feedbacks: list) -> list[dict]:
        """識別失敗模式"""
        patterns = []
        failed_feedbacks = [f for f in feedbacks if f.success_rate < 0.3]
        
        # 統計失敗原因
        waf_triggered_count = sum(1 for f in failed_feedbacks if f.waf_triggered)
        high_error_count = sum(1 for f in failed_feedbacks if f.error_rate > 0.5)
        
        if waf_triggered_count > len(failed_feedbacks) * 0.5:
            patterns.append({
                "type": "waf_block",
                "description": "WAF 阻擋是主要失敗原因",
                "mitigation": "使用混淆技術和延遲請求",
            })
        
        if high_error_count > len(failed_feedbacks) * 0.5:
            patterns.append({
                "type": "high_error_rate",
                "description": "執行錯誤率過高",
                "mitigation": "增加錯誤處理和重試邏輯",
            })
        
        return patterns
    
    def _calculate_waf_risk(self, feedbacks: list) -> float:
        """計算 WAF 風險分數"""
        if not feedbacks:
            return 0.5
        waf_count = sum(1 for f in feedbacks if f.waf_triggered)
        return waf_count / len(feedbacks)
    
    def _calculate_error_probability(self, feedbacks: list) -> float:
        """計算錯誤概率"""
        if not feedbacks:
            return 0.3
        error_rates = [f.error_rate for f in feedbacks]
        return sum(error_rates) / len(error_rates)
    
    def _calculate_avg_success_rate(self, feedbacks: list) -> float:
        """計算平均成功率"""
        if not feedbacks:
            return 0.5
        return sum(f.success_rate for f in feedbacks) / len(feedbacks)
    
    def _calculate_avg_execution_time(self, feedbacks: list) -> float:
        """計算平均執行時間"""
        if not feedbacks:
            return 60.0
        return sum(f.execution_time for f in feedbacks) / len(feedbacks)
    
    def _recommend_best_strategy(self, strategy_stats: dict) -> str:
        """推薦最佳策略
        
        Args:
            strategy_stats: 策略統計數據
            
        Returns:
            推薦的最佳策略名稱
        """
        if not strategy_stats:
            return "reconnaissance"
        
        # 按成功率排序
        sorted_strategies = sorted(
            strategy_stats.items(),
            key=lambda x: x[1].get("success_rate", 0) * x[1].get("avg_score", 0),
            reverse=True
        )
        
        if sorted_strategies:
            return sorted_strategies[0][0]
        return "reconnaissance"
    
    def _generate_planning_adjustments(
        self,
        waf_risk: float,
        error_probability: float,
    ) -> list[dict]:
        """生成規劃調整建議
        
        Args:
            waf_risk: WAF 風險分數
            error_probability: 錯誤概率
            
        Returns:
            調整建議列表
        """
        adjustments = []
        
        if waf_risk > 0.6:
            adjustments.append({
                "type": "waf_mitigation",
                "priority": "high",
                "action": "啟用 WAF 繞過模式",
                "details": {
                    "use_obfuscation": True,
                    "add_delay": True,
                    "rotate_user_agent": True,
                }
            })
        
        if error_probability > 0.4:
            adjustments.append({
                "type": "error_handling",
                "priority": "high",
                "action": "增強錯誤處理",
                "details": {
                    "add_retry": True,
                    "reduce_concurrency": True,
                    "validate_inputs": True,
                }
            })
        
        return adjustments
    
    def _define_success_criteria(self, objective: str, feedbacks: list) -> dict:
        """定義成功標準（關鍵方法：告訴系統什麼是成功）
        
        根據目標和歷史數據，定義明確的成功/失敗標準
        """
        # 基於目標類型定義基本標準
        objective_lower = objective.lower()
        
        criteria = {
            "min_vulnerabilities": 1,  # 至少發現 1 個漏洞才算成功
            "max_execution_time": 300,  # 最長執行時間 5 分鐘
            "max_error_rate": 0.3,  # 最大錯誤率 30%
            "waf_bypass_required": False,
        }
        
        # 根據目標調整
        if "xss" in objective_lower:
            criteria["vulnerability_types"] = ["xss", "cross-site scripting"]
            criteria["min_vulnerabilities"] = 1
        elif "sql" in objective_lower:
            criteria["vulnerability_types"] = ["sql injection", "sqli"]
            criteria["min_vulnerabilities"] = 1
        elif "comprehensive" in objective_lower or "全面" in objective_lower:
            criteria["min_vulnerabilities"] = 3
            criteria["max_execution_time"] = 600
        
        # 基於歷史數據調整預期
        if feedbacks:
            avg_vulns = sum(f.vulnerabilities_found for f in feedbacks) / len(feedbacks)
            if avg_vulns > 2:
                criteria["expected_vulnerabilities"] = int(avg_vulns)
        
        return criteria
    
    def _get_default_success_criteria(self) -> dict:
        """默認成功標準"""
        return {
            "min_vulnerabilities": 1,
            "max_execution_time": 300,
            "max_error_rate": 0.3,
            "waf_bypass_required": False,
        }


