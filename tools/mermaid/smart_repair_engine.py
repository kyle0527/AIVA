#!/usr/bin/env python3
"""
AIVA Mermaid 智能修復引擎 | AIVA Mermaid Smart Repair Engine
=============================================================

基於官方 Mermaid.js v11.12.0 標準的完整解決方案
Complete solution based on official Mermaid.js v11.12.0 standards

核心功能 Core Features:
1. 🔍 官方標準驗證：使用真正的 Mermaid.js 引擎
2. 🛠️ 智能自動修復：基於經驗的錯誤修復規則
3. 📊 錯誤模式學習：從驗證結果持續學習
4. 🎯 預防性檢查：在生成階段防止錯誤
5. 📈 持續性改進：基於使用反饋不斷優化

使用說明 Usage:
1. 替換原有的 validate_syntax 函數
2. 自動檢測並修復常見問題  
3. 保存修復經驗供未來使用
4. 生成詳細的診斷報告
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到 Python 路徑
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from tools.mermaid.mermaid_diagnostic_system import MermaidDiagnosticSystem, RepairRule
except ImportError as e:
    print(f"警告：無法導入診斷系統: {e}")
    print("將使用基礎驗證模式")


class MermaidSmartValidator:
    """Mermaid 智能驗證器 - 集成診斷系統"""
    
    def __init__(self):
        """初始化智能驗證器"""
        try:
            self.diagnostic_system = MermaidDiagnosticSystem()
            self.has_advanced_features = True
            print("✅ 智能診斷系統已啟用")
        except Exception as e:
            print(f"⚠️ 智能診斷系統啟用失敗: {e}")
            print("使用基礎驗證模式")
            self.has_advanced_features = False
    
    def validate_and_repair(self, mermaid_code: str, context: str = "unknown") -> tuple[bool, str, str]:
        """
        驗證並修復 Mermaid 代碼
        
        Args:
            mermaid_code: 要驗證的代碼
            context: 上下文信息 (文件路徑等)
            
        Returns:
            tuple[bool, str, str]: (是否成功, 狀態消息, 修復後代碼)
        """
        if self.has_advanced_features:
            return self._smart_validate_and_repair(mermaid_code, context)
        else:
            return self._basic_validate(mermaid_code)
    
    def _smart_validate_and_repair(self, mermaid_code: str, context: str) -> tuple[bool, str, str]:
        """使用智能診斷系統進行驗證和修復"""
        try:
            # 檢測圖表類型
            diagram_type = self._detect_diagram_type(mermaid_code)
            
            # 進行診斷和修復
            result = self.diagnostic_system.diagnose_and_repair(
                context, 
                mermaid_code, 
                diagram_type
            )
            
            if result.success:
                status_msg = f"✅ 驗證通過 (應用了 {len(result.applied_rules)} 個修復規則)" if result.applied_rules else "✅ 驗證通過"
                return True, status_msg, result.after_content
            else:
                error_summary = f"❌ 驗證失敗: {result.final_status}"
                if result.applied_rules:
                    error_summary += f"\n已嘗試應用修復規則: {', '.join(result.applied_rules)}"
                return False, error_summary, result.after_content
                
        except Exception as e:
            return False, f"❌ 智能驗證過程出錯: {e}", mermaid_code
    
    def _basic_validate(self, mermaid_code: str) -> tuple[bool, str, str]:
        """基礎驗證 (降級方案)"""
        lines = [line.strip() for line in mermaid_code.split('\n') if line.strip()]
        
        if not lines:
            return False, "❌ 空的 Mermaid 代碼", mermaid_code
        
        # 檢查圖表類型
        valid_types = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
                      'stateDiagram', 'gantt', 'pie', 'gitgraph', 'erDiagram']
        
        first_line = lines[0].lower()
        has_valid_type = any(first_line.startswith(dt) for dt in valid_types)
        
        if not has_valid_type:
            return False, f"❌ 無效的圖表類型: {lines[0]}", mermaid_code
        
        # 基礎括號檢查
        open_count = mermaid_code.count('[') + mermaid_code.count('(') + mermaid_code.count('{')
        close_count = mermaid_code.count(']') + mermaid_code.count(')') + mermaid_code.count('}')
        
        if open_count != close_count:
            return False, "❌ 括號不匹配", mermaid_code
        
        # 基礎修復嘗試
        fixed_code = self._apply_basic_fixes(mermaid_code)
        
        return True, "✅ 基礎驗證通過", fixed_code
    
    def _detect_diagram_type(self, content: str) -> str:
        """檢測圖表類型"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return "unknown"
        
        first_line = lines[0].lower()
        
        diagram_types = {
            'graph': 'graph',
            'flowchart': 'flowchart', 
            'sequencediagram': 'sequenceDiagram',
            'classdiagram': 'classDiagram',
            'statediagram': 'stateDiagram',
            'gantt': 'gantt',
            'pie': 'pie'
        }
        
        for key, value in diagram_types.items():
            if first_line.startswith(key):
                return value
                
        return "unknown"
    
    def _apply_basic_fixes(self, mermaid_code: str) -> str:
        """
        應用基礎修復 (基於實際經驗改進)
        優先修復最常見和最嚴重的問題
        """
        import re
        
        # 1. 修復嵌套 mermaid 代碼塊 (最嚴重的問題)
        # 移除內層的 ```mermaid 標記
        fixed_code = re.sub(
            r'```mermaid\n((?:(?!```mermaid)(?!```).*\n)*?)```mermaid\n((?:(?!```).*\n)*?)```',
            r'```mermaid\n\1\2```',
            mermaid_code,
            flags=re.MULTILINE
        )
        
        # 2. 修復行尾多餘空格 (只處理真正有問題的)
        fixed_code = re.sub(r'class\s+([\w,]+)\s+(\w+)\s{2,}$', r'class \1 \2', fixed_code, flags=re.MULTILINE)
        
        # 3. 標準化 direction 指令
        fixed_code = re.sub(r'direction\s+(LR|RL|TB|BT)\s+', r'direction \1\n', fixed_code)
        
        # 4. 移除多餘的空行 (但保留必要的結構)
        fixed_code = re.sub(r'\n\s*\n\s*\n', '\n\n', fixed_code)
        
        # 5. 修復未關閉的代碼塊
        if fixed_code.count('```mermaid') > fixed_code.count('```mermaid') - fixed_code.count('```'):
            if not fixed_code.strip().endswith('```'):
                fixed_code = fixed_code.rstrip() + '\n```'
        
        return fixed_code
    
    def get_repair_statistics(self) -> dict:
        """獲取修復統計信息"""
        if self.has_advanced_features:
            return self.diagnostic_system.get_rule_statistics()
        else:
            return {"message": "智能統計功能需要完整的診斷系統"}
    
    def add_custom_repair_rule(self, rule: RepairRule):
        """添加自定義修復規則"""
        if self.has_advanced_features:
            self.diagnostic_system.add_custom_rule(rule)
        else:
            print("⚠️ 自定義規則功能需要完整的診斷系統")


def integrate_with_mermaid_optimizer():
    """
    集成到現有的 Mermaid 優化器
    
    這個函數會替換原有的 validate_syntax 方法，
    提供智能驗證和修復功能
    """
    smart_validator = MermaidSmartValidator()
    
    def enhanced_validate_syntax(self, mermaid_code: str) -> tuple[bool, str]:
        """增強的語法驗證方法 - 替換原有方法"""
        success, message, fixed_code = smart_validator.validate_and_repair(mermaid_code, "mermaid_optimizer")
        
        # 如果代碼被修復了，更新原始代碼
        if fixed_code != mermaid_code and hasattr(self, '_last_generated_code'):
            self._last_generated_code = fixed_code
        
        return success, message
    
    return enhanced_validate_syntax, smart_validator


def patch_mermaid_optimizer():
    """
    修補現有的 MermaidOptimizer 類
    為其添加智能驗證功能
    """
    try:
        # 嘗試導入現有的優化器
        from tools.features.mermaid_optimizer import MermaidOptimizer
        
        # 創建智能驗證器
        smart_validator = MermaidSmartValidator()
        
        # 保存原始方法
        original_validate = MermaidOptimizer.validate_syntax
        
        def enhanced_validate(self, mermaid_code: str) -> tuple[bool, str]:
            """增強版驗證方法"""
            # 首先嘗試智能驗證和修復
            success, message, fixed_code = smart_validator.validate_and_repair(
                mermaid_code, 
                getattr(self, '_current_context', 'mermaid_optimizer')
            )
            
            # 如果修復成功，保存修復後的代碼
            if fixed_code != mermaid_code:
                self._last_fixed_code = fixed_code
                if hasattr(self, 'last_generated'):
                    self.last_generated = fixed_code
            
            return success, message
        
        # 添加新方法
        def get_last_fixed_code(self):
            """獲取最後修復的代碼"""
            return getattr(self, '_last_fixed_code', None)
        
        def get_repair_stats(self):
            """獲取修復統計"""
            return smart_validator.get_repair_statistics()
        
        # 修補類
        MermaidOptimizer.validate_syntax = enhanced_validate
        MermaidOptimizer.get_last_fixed_code = get_last_fixed_code
        MermaidOptimizer.get_repair_stats = get_repair_stats
        
        print("✅ Mermaid Optimizer 已成功集成智能驗證功能")
        return True
        
    except ImportError as e:
        print(f"⚠️ 無法找到 MermaidOptimizer: {e}")
        return False
    except Exception as e:
        print(f"❌ 修補過程出錯: {e}")
        return False


def test_smart_repair_engine():
    """測試智能修復引擎"""
    print("🧪 測試 AIVA Mermaid 智能修復引擎")
    print("=" * 50)
    
    # 創建驗證器
    validator = MermaidSmartValidator()
    
    # 測試案例 1：class 空格問題
    test_case_1 = """
graph TB
    A[開始] --> B[處理]
    
    classDef highlight fill:#ff0000,stroke:#000000,stroke-width:2px
    class A,B highlight  
"""
    
    print("測試案例 1: class 應用空格問題")
    success, message, fixed = validator.validate_and_repair(test_case_1, "test_case_1")
    print(f"結果: {message}")
    if fixed != test_case_1:
        print("✨ 代碼已修復!")
        print(f"修復前: {repr(test_case_1.split('class A,B highlight')[1])}")
        print(f"修復後: {repr(fixed.split('class A,B highlight')[1])}")
    print()
    
    # 測試案例 2：嵌套 mermaid 塊
    test_case_2 = """
```mermaid
graph LR
    A --> B
```mermaid
    B --> C
```
```
"""
    
    print("測試案例 2: 嵌套 mermaid 代碼塊")
    success, message, fixed = validator.validate_and_repair(test_case_2, "test_case_2")
    print(f"結果: {message}")
    print()
    
    # 顯示統計信息
    stats = validator.get_repair_statistics()
    print("📊 修復統計信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # 測試智能修復引擎
    test_smart_repair_engine()
    
    # 嘗試修補現有優化器
    print("\n🔧 嘗試集成到現有優化器...")
    success = patch_mermaid_optimizer()
    
    if success:
        print("✅ 集成成功！現在 MermaidOptimizer 具有智能驗證能力")
    else:
        print("⚠️ 集成失敗，請手動集成或檢查路徑")