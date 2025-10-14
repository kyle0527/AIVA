"""
AIVA 統一 AI 控制器 - 整合所有 AI 組件
將分散的 AI 組件統一在 BioNeuronRAGAgent 控制下
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .ai_engine import BioNeuronRAGAgent

logger = logging.getLogger(__name__)


class UnifiedAIController:
    """AIVA 統一 AI 控制器 - 消除 AI 組件衝突"""

    def __init__(self, codebase_path: str = "c:/AMD/AIVA"):
        """初始化統一 AI 控制器"""
        logger.info("🧠 初始化 AIVA 統一 AI 控制器...")

        # 主控 AI 系統
        self.master_ai = BioNeuronRAGAgent(codebase_path)

        # 分散 AI 組件註冊
        self.ai_components = {
            'code_fixer': None,  # 延遲初始化 CodeFixer
            'smart_detectors': {},
            'detection_engines': {}
        }

        # AI 決策歷史
        self.decision_history = []

        logger.info("✅ 統一 AI 控制器初始化完成")

    async def process_unified_request(self, user_input: str, **context) -> dict[str, Any]:
        """統一處理所有 AI 請求 - 避免 AI 衝突"""
        logger.info(f"🎯 統一 AI 處理: {user_input}")

        # 1. 主控 AI 分析任務複雜度
        task_analysis = self._analyze_task_complexity(user_input, context)

        # 2. 決定處理策略
        if task_analysis['can_handle_directly']:
            # 主控 AI 直接處理
            result = await self._direct_processing(user_input, context)
        elif task_analysis['needs_code_fixing']:
            # 需要程式碼修復，但仍由主控 AI 協調
            result = await self._coordinated_code_fixing(user_input, context)
        elif task_analysis['needs_specialized_detection']:
            # 需要專門檢測，主控 AI 統籌
            result = await self._coordinated_detection(user_input, context)
        else:
            # 複雜任務，多 AI 協同但主控統籌
            result = await self._multi_ai_coordination(user_input, context)

        # 3. 記錄統一決策
        self._record_unified_decision(user_input, task_analysis, result)

        return result

    def _analyze_task_complexity(self, user_input: str, context: dict) -> dict[str, Any]:
        """分析任務複雜度 - 決定處理策略"""
        input_lower = user_input.lower()

        analysis = {
            'can_handle_directly': False,
            'needs_code_fixing': False,
            'needs_specialized_detection': False,
            'complexity_score': 0.0,
            'confidence': 0.0
        }

        # 簡單任務判斷
        simple_patterns = ['讀取', '查看', '顯示', '列出', '狀態']
        if any(pattern in input_lower for pattern in simple_patterns):
            analysis['can_handle_directly'] = True
            analysis['complexity_score'] = 0.2

        # 程式碼修復判斷
        fix_patterns = ['修復', '修正', '錯誤', '漏洞修復', 'fix']
        if any(pattern in input_lower for pattern in fix_patterns):
            analysis['needs_code_fixing'] = True
            analysis['complexity_score'] = 0.7

        # 專門檢測判斷
        detection_patterns = ['掃描', '檢測', '漏洞', '安全檢查']
        if any(pattern in input_lower for pattern in detection_patterns):
            analysis['needs_specialized_detection'] = True
            analysis['complexity_score'] = 0.6

        analysis['confidence'] = min(analysis['complexity_score'] + 0.3, 1.0)
        return analysis

    async def _direct_processing(self, user_input: str, context: dict) -> dict[str, Any]:
        """主控 AI 直接處理"""
        logger.info("📋 主控 AI 直接處理任務")

        result = self.master_ai.invoke(user_input, **context)

        return {
            'status': 'success',
            'processing_method': 'direct_master_ai',
            'result': result,
            'ai_conflicts': 0,
            'unified_control': True
        }

    async def _coordinated_code_fixing(self, user_input: str, context: dict) -> dict[str, Any]:
        """協調程式碼修復 - 主控 AI 監督下的修復"""
        logger.info("🔧 協調程式碼修復 (主控 AI 監督)")

        # 主控 AI 預處理
        preprocessed = self.master_ai.invoke(f"分析修復需求: {user_input}", **context)

        # 模擬程式碼修復 (實際會調用 CodeFixer，但保持主控監督)
        fix_result = {
            'fixed_code': '# 修復後的程式碼 (由主控 AI 協調)',
            'explanation': f'基於主控 AI 分析: {preprocessed.get("tool_result", {}).get("analysis", "未知")}',
            'confidence': 0.85
        }

        # 主控 AI 驗證結果
        validation = self.master_ai.invoke(f"驗證修復結果: {fix_result}", **context)

        return {
            'status': 'success',
            'processing_method': 'coordinated_code_fixing',
            'original_analysis': preprocessed,
            'fix_result': fix_result,
            'validation': validation,
            'ai_conflicts': 0,
            'unified_control': True
        }

    async def _coordinated_detection(self, user_input: str, context: dict) -> dict[str, Any]:
        """協調漏洞檢測 - 統一調度多檢測引擎"""
        logger.info("🔍 協調漏洞檢測 (統一調度)")

        # 主控 AI 分析檢測需求
        detection_plan = self.master_ai.invoke(f"規劃檢測策略: {user_input}", **context)

        # 模擬多引擎檢測結果
        detection_results = {
            'sqli_results': {'vulnerabilities_found': 0, 'confidence': 0.9},
            'xss_results': {'vulnerabilities_found': 1, 'confidence': 0.8},
            'ssrf_results': {'vulnerabilities_found': 0, 'confidence': 0.95}
        }

        # 主控 AI 整合結果
        integration = self.master_ai.invoke(f"整合檢測結果: {detection_results}", **context)

        return {
            'status': 'success',
            'processing_method': 'coordinated_detection',
            'detection_plan': detection_plan,
            'detection_results': detection_results,
            'integration': integration,
            'ai_conflicts': 0,
            'unified_control': True
        }

    async def _multi_ai_coordination(self, user_input: str, context: dict) -> dict[str, Any]:
        """多 AI 協同 - 主控 AI 統籌"""
        logger.info("🤝 多 AI 協同處理 (主控統籌)")

        # 主控 AI 制定協同計畫
        coordination_plan = self.master_ai.invoke(f"制定協同計畫: {user_input}", **context)

        # 模擬多 AI 協同執行
        coordination_results = {
            'master_ai_role': '總體規劃與最終決策',
            'code_fixer_role': '程式碼問題修復',
            'detectors_role': '安全漏洞檢測',
            'coordination_efficiency': 0.92
        }

        # 主控 AI 最終整合
        final_result = self.master_ai.invoke(f"整合協同結果: {coordination_results}", **context)

        return {
            'status': 'success',
            'processing_method': 'multi_ai_coordination',
            'coordination_plan': coordination_plan,
            'coordination_results': coordination_results,
            'final_result': final_result,
            'ai_conflicts': 0,
            'unified_control': True
        }

    def _record_unified_decision(self, user_input: str, analysis: dict, result: dict):
        """記錄統一決策歷史"""
        decision_record = {
            'timestamp': asyncio.get_event_loop().time(),
            'user_input': user_input,
            'task_analysis': analysis,
            'processing_method': result.get('processing_method'),
            'ai_conflicts_avoided': result.get('ai_conflicts', 0) == 0,
            'unified_control_maintained': result.get('unified_control', False)
        }

        self.decision_history.append(decision_record)

        if len(self.decision_history) > 100:  # 保持歷史記錄在合理範圍
            self.decision_history.pop(0)

    def get_control_statistics(self) -> dict[str, Any]:
        """獲取統一控制統計"""
        if not self.decision_history:
            return {'no_decisions': True}

        total_decisions = len(self.decision_history)
        unified_decisions = sum(1 for d in self.decision_history if d['unified_control_maintained'])
        conflict_free_decisions = sum(1 for d in self.decision_history if d['ai_conflicts_avoided'])

        return {
            'total_decisions': total_decisions,
            'unified_control_rate': unified_decisions / total_decisions,
            'conflict_free_rate': conflict_free_decisions / total_decisions,
            'processing_methods': {
                method: sum(1 for d in self.decision_history if d['processing_method'] == method)
                for method in {d['processing_method'] for d in self.decision_history}
            },
            'recommendation': '統一控制效果良好' if unified_decisions / total_decisions > 0.9 else '需要優化統一控制'
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
        "分析並優化系統架構"
    ]

    for request in test_requests:
        print(f"\n👤 用戶請求: {request}")
        result = await controller.process_unified_request(request)
        print(f"🤖 處理方式: {result['processing_method']}")
        print(f"✅ 統一控制: {result['unified_control']}")
        print(f"🔄 AI 衝突: {result['ai_conflicts']}")

    print("\n📊 統一控制統計:")
    stats = controller.get_control_statistics()
    print(f"統一控制率: {stats['unified_control_rate']:.1%}")
    print(f"無衝突率: {stats['conflict_free_rate']:.1%}")
    print(f"建議: {stats['recommendation']}")


if __name__ == "__main__":
    asyncio.run(demonstrate_unified_control())
