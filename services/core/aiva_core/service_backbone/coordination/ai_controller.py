"""
AIVA AI 子系統控制器 - BioNeuronMasterController 的專門模組
負責特定 AI 功能的協調，避免與主控制器衝突
支援插件化的智能分析系統
"""

import asyncio
from datetime import datetime
import json
import logging
from typing import Any

try:
    from .plugins.ai_summary_plugin import AISummaryPlugin

    SUMMARY_PLUGIN_AVAILABLE = True
except ImportError:
    SUMMARY_PLUGIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class AISubsystemController:
    """AIVA AI 子系統控制器 - 避免與主控制器衝突
    
    重要：此類不再實例化 BioNeuronRAGAgent，而是使用主控制器共享的實例
    """

    def __init__(self, master_controller=None):
        """初始化 AI 子系統控制器
        
        Args:
            master_controller: 主控制器實例，用於共享 AI 資源
        """
        logger.info("🔧 初始化 AIVA AI 子系統控制器...")

        # 重要：不再創建獨立的 AI 實例，避免資源浪費
        self.master_controller = master_controller
        self._master_ai = None  # 延遲獲取主控 AI

        # 分散 AI 組件註冊
        self.ai_components = {
            "code_fixer": None,  # 延遲初始化 CodeFixer
            "smart_detectors": {},
            "detection_engines": {},
        }

        # AI 決策歷史 (與主控制器共享)
        self.decision_history = []
        
        # 🔌 插件系統 - 摘要功能
        self.summary_plugin: AISummaryPlugin | None = None
        if SUMMARY_PLUGIN_AVAILABLE:
            try:
                self.summary_plugin = AISummaryPlugin(enabled=True)
                logger.info("🔌 摘要插件已載入")
            except Exception as e:
                logger.warning(f"⚠️ 摘要插件載入失敗: {e}")
                self.summary_plugin = None
        else:
            logger.info("ℹ️ 摘要插件不可用")

        logger.info("✅ AI 子系統控制器初始化完成")
    
    @property
    def master_ai(self):
        """獲取主控 AI（從主控制器共享）"""
        if self.master_controller and hasattr(self.master_controller, 'bio_neuron_agent'):
            return self.master_controller.bio_neuron_agent
        return None

    async def process_specialized_request(
        self, user_input: str, **context
    ) -> dict[str, Any]:
        """處理專門的 AI 請求 - 透過主控制器協調"""
        logger.info(f"🔧 子系統 AI 處理: {user_input}")

        if not self.master_ai:
            return {
                "status": "error",
                "error": "No master AI controller available",
                "message": "子系統需要主控制器支援"
            }

        # 1. 分析任務複雜度
        task_analysis = self._analyze_task_complexity(user_input, context)

        # 2. 透過主控 AI 處理（避免重複實例化）
        try:
            if task_analysis["can_handle_directly"]:
                result = self._direct_processing(user_input, context)
            elif task_analysis["needs_code_fixing"]:
                result = self._coordinated_code_fixing(user_input, context)
            elif task_analysis["needs_specialized_detection"]:
                result = self._coordinated_detection(user_input, context)
            else:
                result = self._multi_ai_coordination(user_input, context)

            # 3. 記錄決策（與主控制器共享）
            self._record_specialized_decision(user_input, task_analysis, result)

        # 4. 🔌 插件化摘要生成
        if self.summary_plugin and self.summary_plugin.is_enabled():
            try:
                summary = await self.summary_plugin.generate_summary(
                    user_input, task_analysis, result, self.master_ai
                )
                if summary:
                    result["ai_summary"] = summary
            except Exception as e:
                logger.error(f"❌ 摘要插件執行失敗: {e}")

        return result

    def _analyze_task_complexity(
        self, user_input: str, context: dict
    ) -> dict[str, Any]:
        """分析任務複雜度 - 決定處理策略"""
        input_lower = user_input.lower()

        analysis = {
            "can_handle_directly": False,
            "needs_code_fixing": False,
            "needs_specialized_detection": False,
            "complexity_score": 0.0,
            "confidence": 0.0,
        }

        # 簡單任務判斷
        simple_patterns = ["讀取", "查看", "顯示", "列出", "狀態"]
        if any(pattern in input_lower for pattern in simple_patterns):
            analysis["can_handle_directly"] = True
            analysis["complexity_score"] = 0.2

        # 程式碼修復判斷
        fix_patterns = ["修復", "修正", "錯誤", "漏洞修復", "fix"]
        if any(pattern in input_lower for pattern in fix_patterns):
            analysis["needs_code_fixing"] = True
            analysis["complexity_score"] = 0.7

        # 專門檢測判斷
        detection_patterns = ["掃描", "檢測", "漏洞", "安全檢查"]
        if any(pattern in input_lower for pattern in detection_patterns):
            analysis["needs_specialized_detection"] = True
            analysis["complexity_score"] = 0.6

        analysis["confidence"] = min(analysis["complexity_score"] + 0.3, 1.0)
        return analysis

    async def _direct_processing(
        self, user_input: str, context: dict
    ) -> dict[str, Any]:
        """主控 AI 直接處理"""
        logger.info("📋 主控 AI 直接處理任務")

        result = self.master_ai.invoke(user_input, **context)

        return {
            "status": "success",
            "processing_method": "direct_master_ai",
            "result": result,
            "ai_conflicts": 0,
            "unified_control": True,
        }

    async def _coordinated_code_fixing(
        self, user_input: str, context: dict
    ) -> dict[str, Any]:
        """協調程式碼修復 - 主控 AI 監督下的修復"""
        logger.info("🔧 協調程式碼修復 (主控 AI 監督)")

        # 主控 AI 預處理
        preprocessed = self.master_ai.invoke(f"分析修復需求: {user_input}", **context)

        # 模擬程式碼修復 (實際會調用 CodeFixer，但保持主控監督)
        fix_result = {
            "fixed_code": "# 修復後的程式碼 (由主控 AI 協調)",
            "explanation": f'基於主控 AI 分析: {preprocessed.get("tool_result", {}).get("analysis", "未知")}',
            "confidence": 0.85,
        }

        # 主控 AI 驗證結果
        validation = self.master_ai.invoke(f"驗證修復結果: {fix_result}", **context)

        return {
            "status": "success",
            "processing_method": "coordinated_code_fixing",
            "original_analysis": preprocessed,
            "fix_result": fix_result,
            "validation": validation,
            "ai_conflicts": 0,
            "unified_control": True,
        }

    async def _coordinated_detection(
        self, user_input: str, context: dict
    ) -> dict[str, Any]:
        """協調漏洞檢測 - 統一調度多檢測引擎"""
        logger.info("🔍 協調漏洞檢測 (統一調度)")

        # 主控 AI 分析檢測需求
        detection_plan = self.master_ai.invoke(f"規劃檢測策略: {user_input}", **context)

        # 模擬多引擎檢測結果
        detection_results = {
            "sqli_results": {"vulnerabilities_found": 0, "confidence": 0.9},
            "xss_results": {"vulnerabilities_found": 1, "confidence": 0.8},
            "ssrf_results": {"vulnerabilities_found": 0, "confidence": 0.95},
        }

        # 主控 AI 整合結果
        integration = self.master_ai.invoke(
            f"整合檢測結果: {detection_results}", **context
        )

        return {
            "status": "success",
            "processing_method": "coordinated_detection",
            "detection_plan": detection_plan,
            "detection_results": detection_results,
            "integration": integration,
            "ai_conflicts": 0,
            "unified_control": True,
        }

    async def _multi_ai_coordination(
        self, user_input: str, context: dict
    ) -> dict[str, Any]:
        """多 AI 協同 - 主控 AI 統籌"""
        logger.info("🤝 多 AI 協同處理 (主控統籌)")

        # 主控 AI 制定協同計畫
        coordination_plan = self.master_ai.invoke(
            f"制定協同計畫: {user_input}", **context
        )

        # 模擬多 AI 協同執行
        coordination_results = {
            "master_ai_role": "總體規劃與最終決策",
            "code_fixer_role": "程式碼問題修復",
            "detectors_role": "安全漏洞檢測",
            "coordination_efficiency": 0.92,
        }

        # 主控 AI 最終整合
        final_result = self.master_ai.invoke(
            f"整合協同結果: {coordination_results}", **context
        )

        return {
            "status": "success",
            "processing_method": "multi_ai_coordination",
            "coordination_plan": coordination_plan,
            "coordination_results": coordination_results,
            "final_result": final_result,
            "ai_conflicts": 0,
            "unified_control": True,
        }

    # 🔌 插件管理方法
    def get_summary_plugin_status(self) -> dict[str, Any]:
        """獲取摘要插件狀態"""
        if self.summary_plugin:
            return self.summary_plugin.get_status()
        return {
            "plugin_name": "AI Summary Plugin",
            "enabled": False,
            "available": SUMMARY_PLUGIN_AVAILABLE,
            "message": "插件不可用" if not SUMMARY_PLUGIN_AVAILABLE else "插件未載入",
        }

    def enable_summary_plugin(self) -> dict[str, Any]:
        """啟用摘要插件"""
        if not SUMMARY_PLUGIN_AVAILABLE:
            return {"error": "摘要插件不可用"}

        if not self.summary_plugin:
            try:
                self.summary_plugin = AISummaryPlugin(enabled=True)
                return {"status": "success", "message": "摘要插件已啟用"}
            except Exception as e:
                return {"error": f"摘要插件啟用失敗: {e}"}
        else:
            self.summary_plugin.enable()
            return {"status": "success", "message": "摘要插件已啟用"}

    def disable_summary_plugin(self) -> dict[str, Any]:
        """禁用摘要插件"""
        if self.summary_plugin:
            self.summary_plugin.disable()
            return {"status": "success", "message": "摘要插件已禁用"}
        return {"message": "摘要插件未載入"}

    def configure_summary_plugin(self, **settings) -> dict[str, Any]:
        """配置摘要插件"""
        if not self.summary_plugin or not self.summary_plugin.is_enabled():
            return {"error": "摘要插件不可用或未啟用"}
        return self.summary_plugin.configure(**settings)

    def get_summary_statistics(self) -> dict[str, Any]:
        """獲取摘要統計 - 通過插件"""
        if self.summary_plugin:
            return self.summary_plugin.get_statistics()
        return {"error": "摘要插件不可用"}

    def reset_summary_plugin(self) -> dict[str, Any]:
        """重置摘要插件數據"""
        if self.summary_plugin and self.summary_plugin.is_enabled():
            self.summary_plugin.reset()
            return {"status": "success", "message": "摘要插件數據已重置"}
        return {"error": "摘要插件不可用或未啟用"}

    def unload_summary_plugin(self) -> dict[str, Any]:
        """卸載摘要插件"""
        if self.summary_plugin:
            self.summary_plugin.unload()
            self.summary_plugin = None
            return {"status": "success", "message": "摘要插件已卸載"}
        return {"message": "摘要插件未載入"}

    def _record_unified_decision(self, user_input: str, analysis: dict, result: dict):
        """記錄統一決策歷史"""
        decision_record = {
            "timestamp": asyncio.get_event_loop().time(),
            "user_input": user_input,
            "task_analysis": analysis,
            "processing_method": result.get("processing_method"),
            "ai_conflicts_avoided": result.get("ai_conflicts", 0) == 0,
            "unified_control_maintained": result.get("unified_control", False),
        }

        self.decision_history.append(decision_record)

        if len(self.decision_history) > 100:  # 保持歷史記錄在合理範圍
            self.decision_history.pop(0)

    def get_control_statistics(self) -> dict[str, Any]:
        """獲取統一控制統計"""
        if not self.decision_history:
            return {"no_decisions": True}

        total_decisions = len(self.decision_history)
        unified_decisions = sum(
            1 for d in self.decision_history if d["unified_control_maintained"]
        )
        conflict_free_decisions = sum(
            1 for d in self.decision_history if d["ai_conflicts_avoided"]
        )

        return {
            "total_decisions": total_decisions,
            "unified_control_rate": unified_decisions / total_decisions,
            "conflict_free_rate": conflict_free_decisions / total_decisions,
            "processing_methods": {
                method: sum(
                    1 for d in self.decision_history if d["processing_method"] == method
                )
                for method in {d["processing_method"] for d in self.decision_history}
            },
            "recommendation": (
                "統一控制效果良好"
                if unified_decisions / total_decisions > 0.9
                else "需要優化統一控制"
            ),
        }

    def _classify_request_type(self, user_input: str) -> str:
        """分類請求類型"""
        input_lower = user_input.lower()

        if any(word in input_lower for word in ["讀取", "查看", "顯示", "列出"]):
            return "資訊查詢"
        elif any(word in input_lower for word in ["修復", "修正", "錯誤"]):
            return "程式碼修復"
        elif any(word in input_lower for word in ["掃描", "檢測", "漏洞"]):
            return "安全檢測"
        elif any(word in input_lower for word in ["協調", "整合", "統一"]):
            return "系統協調"
        elif any(word in input_lower for word in ["分析", "優化", "改善"]):
            return "系統優化"
        else:
            return "綜合處理"

    def _get_complexity_level(self, score: float) -> str:
        """獲取複雜度等級"""
        if score < 0.3:
            return "簡單"
        elif score < 0.6:
            return "中等"
        elif score < 0.8:
            return "複雜"
        else:
            return "高度複雜"

    def _calculate_efficiency_score(self, task_analysis: dict, result: dict) -> float:
        """計算處理效率分數"""
        base_score = 0.7

        # 根據統一控制加分
        if result.get("unified_control", False):
            base_score += 0.15

        # 根據無衝突加分
        if result.get("ai_conflicts", 0) == 0:
            base_score += 0.1

        # 根據成功狀態加分
        if result.get("status") == "success":
            base_score += 0.05

        return min(base_score, 1.0)

    def _extract_recommendations(self, ai_analysis: dict) -> list[str]:
        """從 AI 分析中提取建議"""
        try:
            analysis_text = ai_analysis.get("tool_result", {}).get("analysis", "")

            # 簡單的建議提取邏輯
            recommendations = []
            if "建議" in analysis_text:
                recommendations.append("參考 AI 分析建議進行優化")
            if "改善" in analysis_text:
                recommendations.append("考慮實施改善措施")
            if "優化" in analysis_text:
                recommendations.append("探索進一步優化方案")

            return recommendations if recommendations else ["繼續保持當前處理方式"]

        except Exception:
            return ["請人工檢查處理結果"]

    def _identify_learning_points(
        self, user_input: str, task_analysis: dict, result: dict
    ) -> list[str]:
        """識別學習要點"""
        learning_points = []

        # 根據處理方式識別學習點
        method = result.get("processing_method", "")
        if "direct" in method:
            learning_points.append("主控 AI 能夠獨立處理此類簡單任務")
        elif "coordinated" in method:
            learning_points.append("協調處理模式在復雜任務中效果良好")
        elif "multi_ai" in method:
            learning_points.append("多 AI 協同能處理高複雜度任務")

        # 根據複雜度識別學習點
        complexity = task_analysis.get("complexity_score", 0)
        if complexity > 0.7:
            learning_points.append("高複雜度任務需要更精細的分析")
        elif complexity < 0.3:
            learning_points.append("簡單任務可以進一步自動化")

        return learning_points

    def _create_brief_summary(self, summary: dict) -> dict:
        """創建簡要摘要"""
        return {
            "type": "簡要摘要",
            "status": summary["basic_info"].get("success_rate", 0) > 0.5,
            "method": summary["processing_summary"]["method_used"],
            "efficiency": summary["processing_summary"]["efficiency_score"],
        }

    def _enhance_detailed_summary(self, summary: dict, result: dict) -> dict:
        """增強詳細摘要"""
        summary["detailed_analysis"] = {
            "processing_steps": self._extract_processing_steps(result),
            "resource_usage": self._estimate_resource_usage(result),
            "improvement_potential": self._assess_improvement_potential(summary),
            "technical_details": {
                "ai_components_used": list(self.ai_components.keys()),
                "coordination_method": result.get("processing_method", "unknown"),
                "decision_confidence": summary["ai_insights"]["confidence"],
            },
        }
        return summary

    def _extract_processing_steps(self, result: dict) -> list[str]:
        """提取處理步驟"""
        method = result.get("processing_method", "")

        if method == "direct_master_ai":
            return ["接收請求", "主控 AI 直接分析", "生成結果", "返回答案"]
        elif method == "coordinated_code_fixing":
            return [
                "接收請求",
                "主控 AI 預處理",
                "協調修復組件",
                "驗證結果",
                "返回修復方案",
            ]
        elif method == "coordinated_detection":
            return [
                "接收請求",
                "規劃檢測策略",
                "執行多引擎檢測",
                "整合結果",
                "返回檢測報告",
            ]
        elif method == "multi_ai_coordination":
            return [
                "接收請求",
                "制定協同計畫",
                "多 AI 協同執行",
                "最終整合",
                "返回綜合結果",
            ]
        else:
            return ["接收請求", "分析處理", "生成結果"]

    def _estimate_resource_usage(self, result: dict) -> dict:
        """估算資源使用情況"""
        method = result.get("processing_method", "")

        usage_map = {
            "direct_master_ai": {"cpu": "low", "memory": "low", "ai_calls": 1},
            "coordinated_code_fixing": {
                "cpu": "medium",
                "memory": "medium",
                "ai_calls": 3,
            },
            "coordinated_detection": {"cpu": "high", "memory": "medium", "ai_calls": 4},
            "multi_ai_coordination": {"cpu": "high", "memory": "high", "ai_calls": 5},
        }

        return usage_map.get(
            method, {"cpu": "unknown", "memory": "unknown", "ai_calls": 1}
        )

    def _assess_improvement_potential(self, summary: dict) -> str:
        """評估改善潛力"""
        efficiency = summary["processing_summary"]["efficiency_score"]

        if efficiency >= 0.9:
            return "優秀，微調即可"
        elif efficiency >= 0.7:
            return "良好，有小幅改善空間"
        elif efficiency >= 0.5:
            return "可接受，需要優化"
        else:
            return "需要重大改善"

    def _record_summary_history(self, summary: dict):
        """記錄摘要歷史"""
        self.summary_history.append(summary)

        # 保持歷史記錄在合理範圍
        if len(self.summary_history) > 50:
            self.summary_history.pop(0)

    def _record_unified_decision(self, user_input: str, analysis: dict, result: dict):
        """記錄統一決策歷史"""
        decision_record = {
            "timestamp": asyncio.get_event_loop().time(),
            "user_input": user_input,
            "task_analysis": analysis,
            "processing_method": result.get("processing_method"),
            "ai_conflicts_avoided": result.get("ai_conflicts", 0) == 0,
            "unified_control_maintained": result.get("unified_control", False),
        }

        self.decision_history.append(decision_record)

        if len(self.decision_history) > 100:  # 保持歷史記錄在合理範圍
            self.decision_history.pop(0)

    def get_ai_summary_statistics(self) -> dict[str, Any]:
        """獲取 AI 摘要統計"""
        if not self.summary_history:
            return {"no_summaries": True}

        total_summaries = len(self.summary_history)
        success_summaries = sum(
            1
            for s in self.summary_history
            if s.get("basic_info", {}).get("success_rate", 0) > 0.5
        )

        # 統計請求類型分布
        request_types = {}
        efficiency_scores = []

        for summary in self.summary_history:
            req_type = summary.get("basic_info", {}).get("request_type", "unknown")
            request_types[req_type] = request_types.get(req_type, 0) + 1

            efficiency = summary.get("processing_summary", {}).get(
                "efficiency_score", 0
            )
            if efficiency > 0:
                efficiency_scores.append(efficiency)

        avg_efficiency = (
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        )

        return {
            "summary_statistics": {
                "total_summaries": total_summaries,
                "success_rate": (
                    success_summaries / total_summaries if total_summaries > 0 else 0
                ),
                "average_efficiency": round(avg_efficiency, 3),
            },
            "request_type_distribution": request_types,
            "efficiency_analysis": {
                "min_efficiency": min(efficiency_scores) if efficiency_scores else 0,
                "max_efficiency": max(efficiency_scores) if efficiency_scores else 0,
                "avg_efficiency": avg_efficiency,
            },
            "recommendations": self._generate_summary_recommendations(
                avg_efficiency, request_types
            ),
        }

    def _generate_summary_recommendations(
        self, avg_efficiency: float, request_types: dict
    ) -> list[str]:
        """生成摘要分析建議"""
        recommendations = []

        if avg_efficiency < 0.6:
            recommendations.append("建議優化處理效率，目標提升至 70% 以上")
        elif avg_efficiency < 0.8:
            recommendations.append("處理效率良好，可進一步微調至 85% 以上")
        else:
            recommendations.append("處理效率優秀，保持當前水準")

        # 分析最常見的請求類型
        if request_types:
            most_common = max(request_types, key=request_types.get)
            recommendations.append(f"最常處理「{most_common}」類型請求，可針對性優化")

        return recommendations

    def configure_summary_settings(self, **settings) -> dict[str, Any]:
        """配置摘要生成設定"""
        old_config = self.summary_config.copy()

        # 更新配置
        for key, value in settings.items():
            if key in self.summary_config:
                self.summary_config[key] = value

        logger.info(f"📋 摘要配置已更新: {settings}")

        return {
            "status": "success",
            "old_config": old_config,
            "new_config": self.summary_config,
            "changes_applied": list(settings.keys()),
        }

    def get_latest_summaries(self, count: int = 5) -> list[dict]:
        """獲取最近的摘要記錄"""
        return self.summary_history[-count:] if self.summary_history else []

    def export_summary_report(self, format_type: str = "json") -> dict[str, Any]:
        """匯出摘要報告"""
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_summaries": len(self.summary_history),
                "summary_period": "全部歷史記錄",
                "format": format_type,
            },
            "summary_statistics": self.get_ai_summary_statistics(),
            "control_statistics": self.get_control_statistics(),
            "recent_summaries": self.get_latest_summaries(10),
            "configuration": {
                "summary_config": self.summary_config,
                "ai_components": list(self.ai_components.keys()),
            },
        }

        logger.info(f"📊 摘要報告已生成 ({format_type} 格式)")

        return {
            "status": "success",
            "report_data": report_data,
            "export_format": format_type,
            "file_suggestion": f"aiva_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}",
        }

    async def generate_comprehensive_summary(
        self, time_period: str = "recent"
    ) -> dict[str, Any]:
        """生成綜合摘要分析"""
        logger.info(f"📈 生成綜合摘要分析 (期間: {time_period})")

        # 選擇分析時間範圍
        if time_period == "recent":
            data_to_analyze = (
                self.summary_history[-10:]
                if len(self.summary_history) >= 10
                else self.summary_history
            )
        elif time_period == "all":
            data_to_analyze = self.summary_history
        else:
            data_to_analyze = self.summary_history[-5:]  # 默認最近5個

        if not data_to_analyze:
            return {"error": "沒有可分析的摘要數據"}

        # 使用主控 AI 進行綜合分析
        analysis_prompt = f"""
        請對以下 AIVA AI 系統的處理摘要進行綜合分析:
        
        分析數據量: {len(data_to_analyze)} 條記錄
        時間期間: {time_period}
        
        摘要數據: {json.dumps(data_to_analyze, ensure_ascii=False, indent=2)}
        
        請提供:
        1. 系統性能趨勢分析
        2. 處理效率評估
        3. 常見問題模式
        4. 改善建議
        5. 未來發展方向
        """

        try:
            ai_comprehensive_analysis = self.master_ai.invoke(analysis_prompt)

            comprehensive_summary = {
                "analysis_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "data_points": len(data_to_analyze),
                    "time_period": time_period,
                    "analysis_depth": "comprehensive",
                },
                "ai_insights": ai_comprehensive_analysis,
                "quantitative_analysis": self._perform_quantitative_analysis(
                    data_to_analyze
                ),
                "trend_analysis": self._analyze_trends(data_to_analyze),
                "recommendations": self._generate_comprehensive_recommendations(
                    data_to_analyze
                ),
            }

            # 記錄綜合分析
            self.summary_history.append(
                {
                    "type": "comprehensive_analysis",
                    "timestamp": datetime.now().isoformat(),
                    "analysis_result": comprehensive_summary,
                }
            )

            logger.info("✅ 綜合摘要分析完成")
            return comprehensive_summary

        except Exception as e:
            logger.error(f"❌ 綜合摘要分析失敗: {e}")
            return {"error": f"綜合分析失敗: {str(e)}"}

    def _perform_quantitative_analysis(self, summaries: list[dict]) -> dict:
        """執行定量分析"""
        if not summaries:
            return {}

        efficiency_scores = []
        success_rates = []
        complexity_levels = []

        for summary in summaries:
            if "processing_summary" in summary:
                eff = summary["processing_summary"].get("efficiency_score", 0)
                if eff > 0:
                    efficiency_scores.append(eff)

            if "basic_info" in summary:
                sr = summary["basic_info"].get("success_rate", 0)
                success_rates.append(sr)

                cl = summary["basic_info"].get("complexity_level", "")
                if cl:
                    complexity_levels.append(cl)

        return {
            "efficiency": {
                "average": (
                    sum(efficiency_scores) / len(efficiency_scores)
                    if efficiency_scores
                    else 0
                ),
                "min": min(efficiency_scores) if efficiency_scores else 0,
                "max": max(efficiency_scores) if efficiency_scores else 0,
                "trend": (
                    "improving"
                    if len(efficiency_scores) > 2
                    and efficiency_scores[-1] > efficiency_scores[0]
                    else "stable"
                ),
            },
            "success_rate": {
                "average": (
                    sum(success_rates) / len(success_rates) if success_rates else 0
                ),
                "total_attempts": len(success_rates),
            },
            "complexity_distribution": {
                level: complexity_levels.count(level)
                for level in set(complexity_levels)
            },
        }

    def _analyze_trends(self, summaries: list[dict]) -> dict:
        """分析趨勢"""
        if len(summaries) < 3:
            return {"insufficient_data": True}

        recent = summaries[-3:]
        older = summaries[:-3] if len(summaries) > 3 else []

        # 比較最近和較早期的表現
        recent_avg_eff = sum(
            s.get("processing_summary", {}).get("efficiency_score", 0) for s in recent
        ) / len(recent)
        older_avg_eff = (
            sum(
                s.get("processing_summary", {}).get("efficiency_score", 0)
                for s in older
            )
            / len(older)
            if older
            else recent_avg_eff
        )

        trend_direction = (
            "improving"
            if recent_avg_eff > older_avg_eff
            else "declining" if recent_avg_eff < older_avg_eff else "stable"
        )

        return {
            "performance_trend": trend_direction,
            "recent_efficiency": recent_avg_eff,
            "historical_efficiency": older_avg_eff,
            "improvement_rate": abs(recent_avg_eff - older_avg_eff),
        }

    def _generate_comprehensive_recommendations(
        self, summaries: list[dict]
    ) -> list[str]:
        """生成綜合建議"""
        recommendations = []

        # 分析效率分布
        efficiency_scores = [
            s.get("processing_summary", {}).get("efficiency_score", 0)
            for s in summaries
        ]
        avg_efficiency = (
            sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        )

        if avg_efficiency < 0.6:
            recommendations.append("🔧 系統效率偏低，建議檢查和優化 AI 組件協調機制")
        elif avg_efficiency > 0.85:
            recommendations.append("✨ 系統效率優秀，可考慮處理更複雜的任務")

        # 分析請求類型多樣性
        request_types = [
            s.get("basic_info", {}).get("request_type", "") for s in summaries
        ]
        unique_types = len(set(request_types))

        if unique_types < 3:
            recommendations.append("📈 建議擴展處理更多類型的任務以提升系統適應性")
        elif unique_types > 5:
            recommendations.append("🎯 系統處理多樣化任務能力強，可專注於深度優化")

        return recommendations

    def get_control_statistics(self) -> dict[str, Any]:
        """獲取統一控制統計"""
        if not self.decision_history:
            return {"no_decisions": True}

        total_decisions = len(self.decision_history)
        unified_decisions = sum(
            1 for d in self.decision_history if d["unified_control_maintained"]
        )
        conflict_free_decisions = sum(
            1 for d in self.decision_history if d["ai_conflicts_avoided"]
        )

        return {
            "total_decisions": total_decisions,
            "unified_control_rate": unified_decisions / total_decisions,
            "conflict_free_rate": conflict_free_decisions / total_decisions,
            "processing_methods": {
                method: sum(
                    1 for d in self.decision_history if d["processing_method"] == method
                )
                for method in {d["processing_method"] for d in self.decision_history}
            },
            "recommendation": (
                "統一控制效果良好"
                if unified_decisions / total_decisions > 0.9
                else "需要優化統一控制"
            ),
        }


# 使用示例
async def demonstrate_unified_control():
    """展示統一 AI 控制的效果"""
    print("🎯 AIVA 統一 AI 控制展示")
    print("=" * 40)

    controller = UnifiedAIController()

    test_requests = [
        "讀取 app.py 檔案",
        "修復 SQL 注入漏洞",
        "執行全面安全掃描",
        "協調 Go 和 Rust 模組",
        "分析並優化系統架構",
    ]

    for request in test_requests:
        print(f"\n👤 用戶請求: {request}")
        result = await controller.process_unified_request(request)
        print(f"🤖 處理方式: {result['processing_method']}")
        print(f"✅ 統一控制: {result['unified_control']}")
        print(f"🔄 AI 衝突: {result['ai_conflicts']}")

        # 顯示 AI 摘要
        if "ai_summary" in result:
            summary = result["ai_summary"]
            print(
                f"📋 AI 摘要: {summary.get('basic_info', {}).get('request_type', 'N/A')}"
            )
            print(
                f"⚡ 效率分數: {summary.get('processing_summary', {}).get('efficiency_score', 0):.2f}"
            )

    print("\n📊 統一控制統計:")
    stats = controller.get_control_statistics()
    print(f"統一控制率: {stats['unified_control_rate']:.1%}")
    print(f"無衝突率: {stats['conflict_free_rate']:.1%}")
    print(f"建議: {stats['recommendation']}")

    print("\n📈 摘要統計:")
    summary_stats = controller.get_ai_summary_statistics()
    if "no_summaries" not in summary_stats:
        print(f"摘要總數: {summary_stats['summary_statistics']['total_summaries']}")
        print(
            f"平均效率: {summary_stats['summary_statistics']['average_efficiency']:.2f}"
        )
        print(f"成功率: {summary_stats['summary_statistics']['success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(demonstrate_unified_control())
