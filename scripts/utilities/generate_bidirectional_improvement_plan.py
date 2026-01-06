#!/usr/bin/env python3
"""
模組雙向連結改進方案分析
基於現有架構提出最小侵入性的改進建議
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    base_path = Path(r"C:\D\fold7\AIVA-git")
    classification_file = base_path / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
    
    print("\n" + "="*70)
    print("🔧 模組雙向連結改進方案")
    print("="*70)
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flows = data.get('flows', [])
    
    # 分析現有模組能力
    def get_module(path):
        if 'services/core/aiva_core/' in path or 'services\\core\\aiva_core\\' in path:
            parts = path.replace('\\', '/').split('services/core/aiva_core/')[1].split('/')
            return parts[0] if parts else 'unknown'
        return 'unknown'
    
    # 收集每個模組的能力（終點）
    module_capabilities = defaultdict(set)
    
    for flow in flows:
        full_path = flow.get('full_path', [])
        if len(full_path) < 2:
            continue
        
        end_path = full_path[-1]
        end_module = get_module(end_path)
        
        if end_module != 'unknown':
            module_capabilities[end_module].add(end_path)
    
    print("\n📦 各模組現有能力終點數量:\n")
    for module in ['cognitive_core', 'internal_exploration', 'task_planning', 
                   'external_learning', 'core_capabilities', 'service_backbone']:
        count = len(module_capabilities[module])
        print(f"   {module:<25} {count:3} 個能力終點")
    
    print("\n\n" + "="*70)
    print("💡 改進方案：維持架構下的最小侵入性改進")
    print("="*70)
    
    print("\n## 方案概述\n")
    print("基於現有 840 flows 架構，通過「反向通道」設計實現雙向連結，")
    print("無需大幅修改現有代碼，只需添加少量反向調用接口。\n")
    
    print("\n" + "="*70)
    print("🎯 問題 1: 三個單向模組無法主動調用其他模組")
    print("="*70)
    
    print("\n### 受影響模組:")
    print("   • cognitive_core (124 入站, 0 出站)")
    print("   • internal_exploration (201 入站, 0 出站)")
    print("   • task_planning (48 入站, 0 出站)")
    
    print("\n### 解決方案: 添加「反向通道接口」\n")
    
    # 方案 1: cognitive_core 反向通道
    print("─" * 70)
    print("1️⃣  cognitive_core 反向通道設計")
    print("─" * 70)
    
    print("\n   目的: 讓認知核心能夠:")
    print("   • 主動調用 task_planning 規劃任務")
    print("   • 主動調用 core_capabilities 執行決策")
    print("   • 主動調用 external_learning 請求學習")
    
    print("\n   實施步驟:")
    print("   a) 在 cognitive_core 新增反向調用器:")
    print("      位置: services/core/aiva_core/cognitive_core/outbound/")
    print("      文件:")
    print("        • task_planner_client.py      → 調用 task_planning")
    print("        • capability_executor_client.py → 調用 core_capabilities")
    print("        • learning_requester_client.py → 調用 external_learning")
    
    print("\n   b) 每個反向調用器實現:")
    print("      ```python")
    print("      class TaskPlannerClient:")
    print("          '''cognitive_core → task_planning 的反向通道'''")
    print("          def request_plan(self, goal, context):")
    print("              # 調用 task_planning.planner.plan_generator")
    print("              return self._call_module('task_planning', 'plan_generator', ...)")
    print("      ```")
    
    print("\n   c) 優勢:")
    print("      ✓ 不修改現有 flows，只添加新的調用路徑")
    print("      ✓ 通過統一接口管理跨模組調用")
    print("      ✓ 保持模組邊界清晰")
    
    # 方案 2: internal_exploration 反向通道
    print("\n" + "─" * 70)
    print("2️⃣  internal_exploration 反向通道設計")
    print("─" * 70)
    
    print("\n   目的: 讓內部探索能夠:")
    print("   • 主動請求 external_learning 進行模型訓練")
    print("   • 主動調用 cognitive_core 進行決策驗證")
    print("   • 主動請求 service_backbone 存儲分析結果")
    
    print("\n   實施步驟:")
    print("   a) 在 internal_exploration 新增反向調用器:")
    print("      位置: services/core/aiva_core/internal_exploration/outbound/")
    print("      文件:")
    print("        • learning_trigger_client.py   → 調用 external_learning")
    print("        • decision_validator_client.py → 調用 cognitive_core")
    print("        • storage_client.py            → 調用 service_backbone")
    
    print("\n   b) 實現自主分析循環:")
    print("      ```python")
    print("      class LearningTriggerClient:")
    print("          '''internal_exploration → external_learning 的反向通道'''")
    print("          def trigger_training(self, analysis_data):")
    print("              # 分析完成後主動請求訓練")
    print("              return self._call_module('external_learning', 'model_trainer', ...)")
    print("      ```")
    
    print("\n   c) 優勢:")
    print("      ✓ 形成「分析 → 學習」閉環")
    print("      ✓ 內部探索可以自主驅動系統改進")
    print("      ✓ 減少對外部模組的依賴")
    
    # 方案 3: task_planning 反向通道
    print("\n" + "─" * 70)
    print("3️⃣  task_planning 反向通道設計")
    print("─" * 70)
    
    print("\n   目的: 讓任務規劃能夠:")
    print("   • 主動調用 core_capabilities 驗證能力可用性")
    print("   • 主動請求 cognitive_core 進行決策確認")
    print("   • 主動調用 service_backbone 查詢資源狀態")
    
    print("\n   實施步驟:")
    print("   a) 在 task_planning 新增反向調用器:")
    print("      位置: services/core/aiva_core/task_planning/outbound/")
    print("      文件:")
    print("        • capability_checker_client.py → 調用 core_capabilities")
    print("        • decision_confirmer_client.py → 調用 cognitive_core")
    print("        • resource_query_client.py     → 調用 service_backbone")
    
    print("\n   b) 實現規劃驗證循環:")
    print("      ```python")
    print("      class CapabilityCheckerClient:")
    print("          '''task_planning → core_capabilities 的反向通道'''")
    print("          def check_capability_available(self, capability_name):")
    print("              # 規劃前驗證能力是否可用")
    print("              return self._call_module('core_capabilities', 'capability_registry', ...)")
    print("      ```")
    
    print("\n   c) 優勢:")
    print("      ✓ 規劃器可以主動驗證計劃可行性")
    print("      ✓ 減少無效計劃的產生")
    print("      ✓ 提高系統執行效率")
    
    print("\n\n" + "="*70)
    print("🎯 問題 2: 核心模組間缺少直接連接")
    print("="*70)
    
    print("\n### 缺少的連接對:")
    print("   • cognitive_core ↔ task_planning")
    print("   • cognitive_core ↔ core_capabilities")
    print("   • task_planning ↔ core_capabilities")
    
    print("\n### 解決方案: 建立「直通通道」\n")
    
    print("─" * 70)
    print("直通通道 1: cognitive_core → task_planning")
    print("─" * 70)
    
    print("\n   場景: 決策層直接下達規劃指令")
    print("\n   實現:")
    print("   • 在 cognitive_core 的反向調用器中添加直通方法")
    print("   • 使用現有的 task_planner_client.py")
    print("   • 調用路徑: cognitive_core.decision → task_planning.planner")
    
    print("\n   代碼示例:")
    print("   ```python")
    print("   # cognitive_core/outbound/task_planner_client.py")
    print("   def direct_plan_request(self, objective):")
    print("       '''繞過中間層，直接請求規劃'''")
    print("       from ...task_planning.planner import plan_generator")
    print("       return plan_generator.generate_plan(objective)")
    print("   ```")
    
    print("\n" + "─" * 70)
    print("直通通道 2: cognitive_core → core_capabilities")
    print("─" * 70)
    
    print("\n   場景: 決策層直接執行關鍵能力")
    print("\n   實現:")
    print("   • 在 cognitive_core 的反向調用器中添加能力執行方法")
    print("   • 使用現有的 capability_executor_client.py")
    print("   • 調用路徑: cognitive_core.decision → core_capabilities.capability_registry")
    
    print("\n" + "─" * 70)
    print("直通通道 3: task_planning → core_capabilities")
    print("─" * 70)
    
    print("\n   場景: 規劃層直接調度能力執行")
    print("\n   實現:")
    print("   • 在 task_planning 的反向調用器中添加執行方法")
    print("   • 使用現有的 capability_checker_client.py")
    print("   • 調用路徑: task_planning.executor → core_capabilities.attack")
    
    print("\n\n" + "="*70)
    print("📋 實施計劃總覽")
    print("="*70)
    
    print("\n### 階段 1: 添加反向調用器基礎設施 (1-2 天)")
    print("\n   步驟:")
    print("   1. 創建統一的反向調用基類")
    print("      位置: services/core/aiva_core/common/outbound_client_base.py")
    print("      功能: 提供跨模組調用的統一接口和錯誤處理")
    
    print("\n   2. 為三個單向模組創建 outbound/ 目錄")
    print("      • cognitive_core/outbound/")
    print("      • internal_exploration/outbound/")
    print("      • task_planning/outbound/")
    
    print("\n### 階段 2: 實現核心反向調用器 (2-3 天)")
    print("\n   優先級:")
    print("   🔴 高優先級:")
    print("      • cognitive_core → task_planning (決策到規劃)")
    print("      • task_planning → core_capabilities (規劃到執行)")
    print("      • internal_exploration → external_learning (分析到學習)")
    
    print("\n   🟡 中優先級:")
    print("      • cognitive_core → core_capabilities (決策到執行)")
    print("      • cognitive_core → external_learning (決策到學習)")
    print("      • internal_exploration → cognitive_core (分析到決策)")
    
    print("\n   🟢 低優先級:")
    print("      • task_planning → cognitive_core (規劃到決策)")
    print("      • internal_exploration → service_backbone (分析到存儲)")
    
    print("\n### 階段 3: 更新 flow 配置 (1 天)")
    print("\n   步驟:")
    print("   1. 為每個新的反向調用添加 flow 定義")
    print("   2. 更新 latest_classification.json")
    print("   3. 驗證新 flows 的正確性")
    
    print("\n### 階段 4: 測試和驗證 (2-3 天)")
    print("\n   測試項目:")
    print("   ✓ 單元測試: 每個反向調用器的獨立測試")
    print("   ✓ 集成測試: 跨模組調用的端到端測試")
    print("   ✓ 性能測試: 確保反向調用不影響系統性能")
    print("   ✓ 回歸測試: 確保不破壞現有功能")
    
    print("\n\n" + "="*70)
    print("📊 改進前後對比")
    print("="*70)
    
    print("\n| 指標 | 改進前 | 改進後 | 變化 |")
    print("|------|--------|--------|------|")
    print("| 單向模組數 | 3 | 0 | ✅ -3 |")
    print("| 核心模組連接 | 3/6 對 | 6/6 對 | ✅ +3 |")
    print("| 連接密度 | 30.0% | ~45-50% | ✅ +15-20% |")
    print("| 孤立模組 | 0 | 0 | ✅ 維持 |")
    print("| 預計新增 flows | - | ~50-80 | 📈 增加 |")
    
    print("\n\n" + "="*70)
    print("💡 關鍵優勢")
    print("="*70)
    
    print("\n1. 🔧 最小侵入性")
    print("   • 不修改現有 840 flows")
    print("   • 只添加新的反向調用接口")
    print("   • 現有功能完全不受影響")
    
    print("\n2. 🏗️ 架構保持")
    print("   • 模組邊界清晰")
    print("   • 依賴關係明確")
    print("   • 易於維護和擴展")
    
    print("\n3. 🔄 漸進式實施")
    print("   • 可以分階段實施")
    print("   • 每個階段獨立驗證")
    print("   • 降低風險")
    
    print("\n4. 🎯 靈活性提升")
    print("   • 模組間可以雙向通信")
    print("   • 支持更複雜的工作流")
    print("   • 提高系統自主性")
    
    print("\n\n" + "="*70)
    print("⚠️ 注意事項")
    print("="*70)
    
    print("\n1. 避免循環依賴")
    print("   • 使用依賴注入模式")
    print("   • 明確調用方向和時序")
    print("   • 添加循環檢測機制")
    
    print("\n2. 性能考慮")
    print("   • 反向調用可能增加延遲")
    print("   • 添加調用監控和日誌")
    print("   • 設置超時和重試機制")
    
    print("\n3. 錯誤處理")
    print("   • 統一的錯誤處理策略")
    print("   • 優雅降級機制")
    print("   • 詳細的錯誤日誌")
    
    print("\n4. 文檔更新")
    print("   • 更新架構文檔")
    print("   • 添加反向調用使用示例")
    print("   • 更新 flow 圖表")
    
    # 保存報告
    output_file = base_path / "docs" / "BIDIRECTIONAL_CONNECTIVITY_IMPROVEMENT_PLAN.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 模組雙向連結改進方案\n\n")
        f.write("**方案日期**: 2026-01-01\n\n")
        
        f.write("## 目標\n\n")
        f.write("在維持現有 840 flows 架構的前提下，通過添加「反向通道」實現模組間的完整雙向連結。\n\n")
        
        f.write("## 現狀問題\n\n")
        f.write("### 1. 三個單向模組\n\n")
        f.write("- **cognitive_core**: 124 入站, 0 出站 - 只能被調用\n")
        f.write("- **internal_exploration**: 201 入站, 0 出站 - 只能被調用\n")
        f.write("- **task_planning**: 48 入站, 0 出站 - 只能被調用\n\n")
        
        f.write("### 2. 核心模組間缺少連接\n\n")
        f.write("- cognitive_core ↔ task_planning\n")
        f.write("- cognitive_core ↔ core_capabilities\n")
        f.write("- task_planning ↔ core_capabilities\n\n")
        
        f.write("## 解決方案\n\n")
        f.write("### 方案概述\n\n")
        f.write("通過「反向通道接口」設計實現雙向連結，核心思想：\n\n")
        f.write("1. **不修改現有 flows** - 保持向後兼容\n")
        f.write("2. **添加反向調用器** - 為單向模組添加主動調用能力\n")
        f.write("3. **建立直通通道** - 為核心模組建立直接連接\n")
        f.write("4. **統一接口管理** - 使用統一的跨模組調用基類\n\n")
        
        f.write("### 技術實現\n\n")
        f.write("#### 1. 反向調用基礎設施\n\n")
        f.write("創建統一的反向調用基類：\n\n")
        f.write("```python\n")
        f.write("# services/core/aiva_core/common/outbound_client_base.py\n")
        f.write("class OutboundClientBase:\n")
        f.write("    '''跨模組調用的統一基類'''\n")
        f.write("    \n")
        f.write("    def _call_module(self, target_module, target_capability, **kwargs):\n")
        f.write("        '''統一的跨模組調用接口'''\n")
        f.write("        # 1. 驗證目標模組和能力\n")
        f.write("        # 2. 構建調用上下文\n")
        f.write("        # 3. 執行調用\n")
        f.write("        # 4. 處理錯誤和超時\n")
        f.write("        # 5. 記錄調用日誌\n")
        f.write("        pass\n")
        f.write("```\n\n")
        
        f.write("#### 2. cognitive_core 反向調用器\n\n")
        f.write("為認知核心添加主動調用能力：\n\n")
        f.write("**文件結構**:\n")
        f.write("```\n")
        f.write("cognitive_core/\n")
        f.write("├── outbound/\n")
        f.write("│   ├── __init__.py\n")
        f.write("│   ├── task_planner_client.py      # → task_planning\n")
        f.write("│   ├── capability_executor_client.py # → core_capabilities\n")
        f.write("│   └── learning_requester_client.py  # → external_learning\n")
        f.write("```\n\n")
        
        f.write("**實現示例**:\n")
        f.write("```python\n")
        f.write("# cognitive_core/outbound/task_planner_client.py\n")
        f.write("from ...common.outbound_client_base import OutboundClientBase\n\n")
        f.write("class TaskPlannerClient(OutboundClientBase):\n")
        f.write("    '''cognitive_core → task_planning 的反向通道'''\n")
        f.write("    \n")
        f.write("    def request_plan(self, objective, context):\n")
        f.write("        '''請求任務規劃'''\n")
        f.write("        return self._call_module(\n")
        f.write("            target_module='task_planning',\n")
        f.write("            target_capability='plan_generator',\n")
        f.write("            objective=objective,\n")
        f.write("            context=context\n")
        f.write("        )\n")
        f.write("    \n")
        f.write("    def verify_plan(self, plan):\n")
        f.write("        '''驗證計劃可行性'''\n")
        f.write("        return self._call_module(\n")
        f.write("            target_module='task_planning',\n")
        f.write("            target_capability='plan_validator',\n")
        f.write("            plan=plan\n")
        f.write("        )\n")
        f.write("```\n\n")
        
        f.write("#### 3. internal_exploration 反向調用器\n\n")
        f.write("為內部探索添加主動學習能力：\n\n")
        f.write("**文件結構**:\n")
        f.write("```\n")
        f.write("internal_exploration/\n")
        f.write("├── outbound/\n")
        f.write("│   ├── __init__.py\n")
        f.write("│   ├── learning_trigger_client.py    # → external_learning\n")
        f.write("│   ├── decision_validator_client.py  # → cognitive_core\n")
        f.write("│   └── storage_client.py             # → service_backbone\n")
        f.write("```\n\n")
        
        f.write("#### 4. task_planning 反向調用器\n\n")
        f.write("為任務規劃添加能力驗證：\n\n")
        f.write("**文件結構**:\n")
        f.write("```\n")
        f.write("task_planning/\n")
        f.write("├── outbound/\n")
        f.write("│   ├── __init__.py\n")
        f.write("│   ├── capability_checker_client.py  # → core_capabilities\n")
        f.write("│   ├── decision_confirmer_client.py  # → cognitive_core\n")
        f.write("│   └── resource_query_client.py      # → service_backbone\n")
        f.write("```\n\n")
        
        f.write("## 實施計劃\n\n")
        f.write("### 階段 1: 基礎設施 (1-2 天)\n\n")
        f.write("- [ ] 創建 `OutboundClientBase` 基類\n")
        f.write("- [ ] 為三個單向模組創建 `outbound/` 目錄\n")
        f.write("- [ ] 實現統一的調用接口和錯誤處理\n\n")
        
        f.write("### 階段 2: 核心反向調用器 (2-3 天)\n\n")
        f.write("**高優先級**:\n")
        f.write("- [ ] cognitive_core → task_planning\n")
        f.write("- [ ] task_planning → core_capabilities\n")
        f.write("- [ ] internal_exploration → external_learning\n\n")
        
        f.write("**中優先級**:\n")
        f.write("- [ ] cognitive_core → core_capabilities\n")
        f.write("- [ ] cognitive_core → external_learning\n")
        f.write("- [ ] internal_exploration → cognitive_core\n\n")
        
        f.write("**低優先級**:\n")
        f.write("- [ ] task_planning → cognitive_core\n")
        f.write("- [ ] internal_exploration → service_backbone\n\n")
        
        f.write("### 階段 3: Flow 配置更新 (1 天)\n\n")
        f.write("- [ ] 為每個新反向調用添加 flow 定義\n")
        f.write("- [ ] 更新 `latest_classification.json`\n")
        f.write("- [ ] 驗證新 flows 的正確性\n\n")
        
        f.write("### 階段 4: 測試驗證 (2-3 天)\n\n")
        f.write("- [ ] 單元測試: 每個反向調用器\n")
        f.write("- [ ] 集成測試: 跨模組調用\n")
        f.write("- [ ] 性能測試: 延遲和吞吐量\n")
        f.write("- [ ] 回歸測試: 確保不破壞現有功能\n\n")
        
        f.write("## 預期效果\n\n")
        f.write("| 指標 | 改進前 | 改進後 | 變化 |\n")
        f.write("|------|--------|--------|------|\n")
        f.write("| 單向模組數 | 3 | 0 | ✅ -3 |\n")
        f.write("| 核心模組連接 | 3/6 對 | 6/6 對 | ✅ +3 |\n")
        f.write("| 連接密度 | 30.0% | ~45-50% | ✅ +15-20% |\n")
        f.write("| 孤立模組 | 0 | 0 | ✅ 維持 |\n")
        f.write("| 預計新增 flows | 0 | 50-80 | 📈 增加 |\n\n")
        
        f.write("## 關鍵優勢\n\n")
        f.write("1. **最小侵入性**: 不修改現有 840 flows，完全向後兼容\n")
        f.write("2. **架構保持**: 模組邊界清晰，依賴關係明確\n")
        f.write("3. **漸進式實施**: 可分階段實施，每階段獨立驗證\n")
        f.write("4. **靈活性提升**: 支持更複雜的工作流和自主決策\n\n")
        
        f.write("## 注意事項\n\n")
        f.write("### 1. 避免循環依賴\n")
        f.write("- 使用依賴注入模式\n")
        f.write("- 明確調用方向和時序\n")
        f.write("- 添加循環檢測機制\n\n")
        
        f.write("### 2. 性能考慮\n")
        f.write("- 反向調用可能增加延遲\n")
        f.write("- 添加調用監控和日誌\n")
        f.write("- 設置超時和重試機制\n\n")
        
        f.write("### 3. 錯誤處理\n")
        f.write("- 統一的錯誤處理策略\n")
        f.write("- 優雅降級機制\n")
        f.write("- 詳細的錯誤日誌\n\n")
        
        f.write("### 4. 文檔更新\n")
        f.write("- 更新架構文檔\n")
        f.write("- 添加反向調用使用示例\n")
        f.write("- 更新 flow 圖表\n\n")
        
        f.write("## 下一步行動\n\n")
        f.write("1. **評審此方案**: 與團隊討論可行性\n")
        f.write("2. **選擇試點**: 先實施一個高優先級的反向調用器\n")
        f.write("3. **驗證效果**: 評估試點的效果和影響\n")
        f.write("4. **全面推廣**: 根據試點結果調整並全面實施\n")
    
    print(f"\n✅ 詳細改進方案已保存至: {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
