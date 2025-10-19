"""
AI 摘要插件 - 可插拔的智能分析模組
獨立的摘要生成和分析系統，可隨時啟用或禁用
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AISummaryPlugin:
    """AI 摘要插件 - 獨立的摘要生成系統"""
    
    def __init__(self, enabled: bool = True):
        """初始化摘要插件"""
        self.enabled = enabled
        self.plugin_name = "AI Summary Plugin"
        self.version = "1.0.0"
        
        # 摘要配置
        self.config = {
            'auto_generate': True,
            'include_metrics': True,
            'include_recommendations': True,
            'summary_depth': 'detailed'  # 'brief', 'standard', 'detailed'
        }
        
        # 摘要歷史
        self.summary_history = []
        
        logger.info(f"🔌 {self.plugin_name} v{self.version} {'已啟用' if enabled else '已禁用'}")
    
    def is_enabled(self) -> bool:
        """檢查插件是否啟用"""
        return self.enabled
    
    def enable(self):
        """啟用插件"""
        self.enabled = True
        logger.info(f"✅ {self.plugin_name} 已啟用")
    
    def disable(self):
        """禁用插件"""
        self.enabled = False
        logger.info(f"❌ {self.plugin_name} 已禁用")
    
    def get_status(self) -> dict[str, Any]:
        """獲取插件狀態"""
        return {
            'plugin_name': self.plugin_name,
            'version': self.version,
            'enabled': self.enabled,
            'summary_count': len(self.summary_history),
            'config': self.config
        }

    async def generate_summary(self, user_input: str, task_analysis: dict, result: dict, master_ai) -> Optional[dict[str, Any]]:
        """生成 AI 處理摘要"""
        if not self.enabled:
            return None
            
        logger.info("📝 [Plugin] 生成 AI 處理摘要...")
        
        try:
            # 準備摘要數據
            summary_data = {
                'timestamp': datetime.now().isoformat(),
                'request': user_input,
                'complexity': task_analysis.get('complexity_score', 0),
                'method': result.get('processing_method'),
                'success': result.get('status') == 'success'
            }
            
            # 使用主控 AI 生成智能摘要
            summary_prompt = self._build_summary_prompt(summary_data, task_analysis, result)
            ai_analysis = master_ai.invoke(summary_prompt)
            
            # 構建完整摘要
            summary = {
                'plugin_info': {
                    'generated_by': self.plugin_name,
                    'version': self.version,
                    'timestamp': datetime.now().isoformat()
                },
                'basic_info': {
                    'request_type': self._classify_request_type(user_input),
                    'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'complexity_level': self._get_complexity_level(task_analysis.get('complexity_score', 0)),
                    'success_rate': 1.0 if result.get('status') == 'success' else 0.0
                },
                'processing_summary': {
                    'method_used': result.get('processing_method', 'unknown'),
                    'ai_coordination': result.get('unified_control', False),
                    'conflicts_avoided': result.get('ai_conflicts', 0) == 0,
                    'efficiency_score': self._calculate_efficiency_score(task_analysis, result)
                },
                'ai_insights': {
                    'analysis': ai_analysis.get('tool_result', {}).get('analysis', '智能分析不可用'),
                    'recommendations': self._extract_recommendations(ai_analysis),
                    'confidence': task_analysis.get('confidence', 0.0),
                    'learning_points': self._identify_learning_points(user_input, task_analysis, result)
                }
            }
            
            # 根據配置調整摘要深度
            if self.config['summary_depth'] == 'brief':
                summary = self._create_brief_summary(summary)
            elif self.config['summary_depth'] == 'detailed':
                summary = self._enhance_detailed_summary(summary, result)
            
            # 記錄摘要歷史
            self._record_summary_history(summary)
            
            logger.info("✅ [Plugin] AI 摘要生成完成")
            return summary
            
        except Exception as e:
            logger.error(f"❌ [Plugin] 摘要生成失敗: {e}")
            return {
                'plugin_info': {
                    'generated_by': self.plugin_name,
                    'error': f'摘要生成失敗: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                },
                'basic_info': {
                    'request_type': 'error_summary',
                    'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

    def _build_summary_prompt(self, summary_data: dict, task_analysis: dict, result: dict) -> str:
        """構建摘要生成提示"""
        return f"""
        請分析以下 AI 處理結果並生成智能摘要：
        
        用戶請求: {summary_data['request']}
        處理方式: {summary_data['method']}
        複雜度分數: {summary_data['complexity']}
        成功狀態: {summary_data['success']}
        
        任務分析: {json.dumps(task_analysis, ensure_ascii=False, indent=2)}
        處理結果: {json.dumps(result, ensure_ascii=False, indent=2)}
        
        請提供:
        1. 處理效果分析
        2. 改善建議
        3. 學習要點
        4. 未來優化方向
        """

    def _classify_request_type(self, user_input: str) -> str:
        """分類請求類型"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ['讀取', '查看', '顯示', '列出']):
            return '資訊查詢'
        elif any(word in input_lower for word in ['修復', '修正', '錯誤']):
            return '程式碼修復'
        elif any(word in input_lower for word in ['掃描', '檢測', '漏洞']):
            return '安全檢測'
        elif any(word in input_lower for word in ['協調', '整合', '統一']):
            return '系統協調'
        elif any(word in input_lower for word in ['分析', '優化', '改善']):
            return '系統優化'
        else:
            return '綜合處理'

    def _get_complexity_level(self, score: float) -> str:
        """獲取複雜度等級"""
        if score < 0.3:
            return '簡單'
        elif score < 0.6:
            return '中等'
        elif score < 0.8:
            return '複雜'
        else:
            return '高度複雜'

    def _calculate_efficiency_score(self, task_analysis: dict, result: dict) -> float:
        """計算處理效率分數"""
        base_score = 0.7
        
        # 根據統一控制加分
        if result.get('unified_control', False):
            base_score += 0.15
            
        # 根據無衝突加分
        if result.get('ai_conflicts', 0) == 0:
            base_score += 0.1
            
        # 根據成功狀態加分
        if result.get('status') == 'success':
            base_score += 0.05
            
        return min(base_score, 1.0)

    def _extract_recommendations(self, ai_analysis: dict) -> list[str]:
        """從 AI 分析中提取建議"""
        try:
            analysis_text = ai_analysis.get('tool_result', {}).get('analysis', '')
            
            recommendations = []
            if '建議' in analysis_text:
                recommendations.append('參考 AI 分析建議進行優化')
            if '改善' in analysis_text:
                recommendations.append('考慮實施改善措施')
            if '優化' in analysis_text:
                recommendations.append('探索進一步優化方案')
                
            return recommendations if recommendations else ['繼續保持當前處理方式']
            
        except Exception:
            return ['請人工檢查處理結果']

    def _identify_learning_points(self, user_input: str, task_analysis: dict, result: dict) -> list[str]:
        """識別學習要點"""
        learning_points = []
        
        # 根據處理方式識別學習點
        method = result.get('processing_method', '')
        if 'direct' in method:
            learning_points.append('主控 AI 能夠獨立處理此類簡單任務')
        elif 'coordinated' in method:
            learning_points.append('協調處理模式在復雜任務中效果良好')
        elif 'multi_ai' in method:
            learning_points.append('多 AI 協同能處理高複雜度任務')
            
        # 根據複雜度識別學習點
        complexity = task_analysis.get('complexity_score', 0)
        if complexity > 0.7:
            learning_points.append('高複雜度任務需要更精細的分析')
        elif complexity < 0.3:
            learning_points.append('簡單任務可以進一步自動化')
            
        return learning_points

    def _create_brief_summary(self, summary: dict) -> dict:
        """創建簡要摘要"""
        return {
            'plugin_info': summary['plugin_info'],
            'type': '簡要摘要',
            'status': summary['basic_info'].get('success_rate', 0) > 0.5,
            'method': summary['processing_summary']['method_used'],
            'efficiency': summary['processing_summary']['efficiency_score']
        }

    def _enhance_detailed_summary(self, summary: dict, result: dict) -> dict:
        """增強詳細摘要"""
        summary['detailed_analysis'] = {
            'processing_steps': self._extract_processing_steps(result),
            'resource_usage': self._estimate_resource_usage(result),
            'improvement_potential': self._assess_improvement_potential(summary)
        }
        return summary

    def _extract_processing_steps(self, result: dict) -> list[str]:
        """提取處理步驟"""
        method = result.get('processing_method', '')
        
        if method == 'direct_master_ai':
            return ['接收請求', '主控 AI 直接分析', '生成結果', '返回答案']
        elif method == 'coordinated_code_fixing':
            return ['接收請求', '主控 AI 預處理', '協調修復組件', '驗證結果', '返回修復方案']
        elif method == 'coordinated_detection':
            return ['接收請求', '規劃檢測策略', '執行多引擎檢測', '整合結果', '返回檢測報告']
        elif method == 'multi_ai_coordination':
            return ['接收請求', '制定協同計畫', '多 AI 協同執行', '最終整合', '返回綜合結果']
        else:
            return ['接收請求', '分析處理', '生成結果']

    def _estimate_resource_usage(self, result: dict) -> dict:
        """估算資源使用情況"""
        method = result.get('processing_method', '')
        
        usage_map = {
            'direct_master_ai': {'cpu': 'low', 'memory': 'low', 'ai_calls': 1},
            'coordinated_code_fixing': {'cpu': 'medium', 'memory': 'medium', 'ai_calls': 3},
            'coordinated_detection': {'cpu': 'high', 'memory': 'medium', 'ai_calls': 4},
            'multi_ai_coordination': {'cpu': 'high', 'memory': 'high', 'ai_calls': 5}
        }
        
        return usage_map.get(method, {'cpu': 'unknown', 'memory': 'unknown', 'ai_calls': 1})

    def _assess_improvement_potential(self, summary: dict) -> str:
        """評估改善潛力"""
        efficiency = summary['processing_summary']['efficiency_score']
        
        if efficiency >= 0.9:
            return '優秀，微調即可'
        elif efficiency >= 0.7:
            return '良好，有小幅改善空間'
        elif efficiency >= 0.5:
            return '可接受，需要優化'
        else:
            return '需要重大改善'

    def _record_summary_history(self, summary: dict):
        """記錄摘要歷史"""
        if not self.enabled:
            return
            
        self.summary_history.append(summary)
        
        # 保持歷史記錄在合理範圍
        if len(self.summary_history) > 50:
            self.summary_history.pop(0)

    def get_statistics(self) -> dict[str, Any]:
        """獲取摘要統計"""
        if not self.enabled or not self.summary_history:
            return {'plugin_enabled': self.enabled, 'no_summaries': True}
            
        total_summaries = len(self.summary_history)
        success_summaries = sum(1 for s in self.summary_history 
                               if s.get('basic_info', {}).get('success_rate', 0) > 0.5)
        
        return {
            'plugin_enabled': self.enabled,
            'plugin_name': self.plugin_name,
            'version': self.version,
            'total_summaries': total_summaries,
            'success_rate': success_summaries / total_summaries if total_summaries > 0 else 0,
            'config': self.config
        }

    def configure(self, **settings) -> dict[str, Any]:
        """配置插件設定"""
        if not self.enabled:
            return {'error': '插件未啟用'}
            
        old_config = self.config.copy()
        
        # 更新配置
        for key, value in settings.items():
            if key in self.config:
                self.config[key] = value
                
        logger.info(f"🔌 [Plugin] 配置已更新: {settings}")
        
        return {
            'status': 'success',
            'old_config': old_config,
            'new_config': self.config,
            'changes_applied': list(settings.keys())
        }

    def reset(self):
        """重置插件數據"""
        if not self.enabled:
            return
            
        self.summary_history.clear()
        logger.info(f"🔌 [Plugin] {self.plugin_name} 數據已重置")

    def unload(self):
        """卸載插件"""
        self.enabled = False
        self.summary_history.clear()
        logger.info(f"🔌 [Plugin] {self.plugin_name} 已卸載")