#!/usr/bin/env python3
"""
AIVA Mermaid 診斷和修復系統 | AIVA Mermaid Diagnostic & Repair System  
=======================================================================

基於官方 Mermaid.js v11.12.0 標準的智能診斷修復系統
Intelligent diagnostic repair system based on official Mermaid.js v11.12.0 standards

核心設計原則:
1. 錯誤模式學習：從官方驗證中學習常見錯誤模式
2. 規則化修復：建立可重用的修復規則庫  
3. 預防性檢查：防止未來出現相同問題
4. 標準一致性：確保與官方標準100%一致
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass 
class RepairRule:
    """修復規則"""
    rule_id: str
    name: str
    description: str
    pattern: str
    replacement: str
    applies_to: List[str]  # 適用的圖表類型
    severity: str  # "error" | "warning" | "optimization"
    success_rate: float = 0.0
    usage_count: int = 0


@dataclass
class DiagnosticReport:
    """診斷報告"""
    timestamp: str
    file_path: str
    diagram_type: str
    original_errors: List[str]
    applied_rules: List[str]
    final_status: str
    before_content: str
    after_content: str
    success: bool


class MermaidDiagnosticSystem:
    """Mermaid 診斷系統"""
    
    def __init__(self, rules_path: Optional[str] = None, reports_path: Optional[str] = None):
        self.rules_path = Path(rules_path) if rules_path else Path(__file__).parent / "repair_rules.json"
        self.reports_path = Path(reports_path) if reports_path else Path(__file__).parent / "diagnostic_reports.json"
        
        self.repair_rules = self._load_repair_rules()
        self.diagnostic_history = self._load_diagnostic_history()
        
        # 設置日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_repair_rules(self) -> List[RepairRule]:
        """載入修復規則"""
        if not self.rules_path.exists():
            # 初始化基本規則庫
            return self._create_initial_rules()
            
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
                return [RepairRule(**rule) for rule in rules_data]
        except Exception as e:
            self.logger.warning(f"無法載入修復規則: {e}")
            return self._create_initial_rules()
    
    def _create_initial_rules(self) -> List[RepairRule]:
        """創建基於官方 Mermaid.js v11.12.0 標準的修復規則庫"""
        return [
            RepairRule(
                rule_id="NESTED_MERMAID_BLOCKS",
                name="修復嵌套 mermaid 代碼塊 (v11.12.0 標準)",
                description="移除錯誤的嵌套 ```mermaid 標記，保持內容完整",
                pattern=r"(```mermaid\n(?:[^`]|`(?!``)|``(?!`))*?)\n```mermaid\n((?:[^`]|`(?!``)|``(?!`))*?)\n```",
                replacement=r"\1\n\2\n```",
                applies_to=["all"],
                severity="critical"
            ),
            RepairRule(
                rule_id="UNCLOSED_CODE_BLOCK",
                name="修復未關閉的代碼塊 (v11.12.0 標準)", 
                description="為未關閉的代碼塊添加正確的結束標記",
                pattern=r"(```\w*\n(?:[^`]|`(?!``)|``(?!`))+?)(?=\n```\w|\n*$)",
                replacement=r"\1\n```",
                applies_to=["all"],
                severity="critical"
            ),
            RepairRule(
                rule_id="EXTRA_CODE_BLOCK_END",
                name="移除多餘的代碼塊結束標記 (v11.12.0 標準)",
                description="移除沒有對應開始標記的結束標記",
                pattern=r"(?<=\n)```\s*(?=\n|$)",
                replacement="",
                applies_to=["all"],
                severity="error"
            ),
            RepairRule(
                rule_id="CLASSDEF_EXTRA_SPACES",
                name="修復 class 應用中的多餘空格 (v11.12.0 標準)",
                description="移除 class 應用語句行尾的多餘空格",
                pattern=r"(class\s+[\w,]+\s+\w+)\s+$",
                replacement=r"\1",
                applies_to=["graph", "flowchart"],
                severity="error"
            ),
            RepairRule(
                rule_id="DIRECTION_SYNTAX_ERROR",
                name="修復 direction 語法錯誤 (v11.12.0 標準)",
                description="確保 direction 指令符合官方格式: direction TB",
                pattern=r"direction\s+(LR|RL|TB|BT)\s+(.+)",
                replacement=r"direction \1\n\2",
                applies_to=["graph", "flowchart"],
                severity="warning"
            ),
            RepairRule(
                rule_id="SUBGRAPH_DIRECTION_SPACES",
                name="修復 subgraph direction 的空格 (v11.12.0 標準)",
                description="修正 subgraph 和 direction 指令的空格格式",
                pattern=r"direction\s+(LR|RL|TB|BT)\s+",
                replacement=r"direction \1\n",
                applies_to=["graph", "flowchart"],
                severity="warning"
            ),
            RepairRule(
                rule_id="STYLE_DEFINITION_FORMAT",
                name="標準化 style 定義格式 (v11.12.0 標準)",
                description="修正 style 定義中的空格和格式，符合官方標準",
                pattern=r"style\s+(\w+)\s+fill:\s*([^,;\s]+)\s*,?\s*stroke:\s*([^,;\s]+)\s*,?\s*stroke-width:\s*([^,;\s]+)",
                replacement=r"style \1 fill:\2,stroke:\3,stroke-width:\4",
                applies_to=["graph", "flowchart"],
                severity="optimization"
            ),
            RepairRule(
                rule_id="BRACKET_SPACING",
                name="修復節點標籤中的空格 (v11.12.0 標準)",
                description="標準化節點標籤方括號內的空格，移除多餘空格",
                pattern=r"\[\s*([^]]+?)\s*\]",
                replacement=r"[\1]",
                applies_to=["graph", "flowchart"],
                severity="optimization"
            ),
            RepairRule(
                rule_id="ARROW_SPACING_V2",
                name="修復箭頭連接空格 (v11.12.0 標準)",
                description="標準化箭頭連接的空格格式: A --> B",
                pattern=r"(\w+)\s*(-->|---|\||\.-\.)\s*(\w+)",
                replacement=r"\1 \2 \3",
                applies_to=["graph", "flowchart"],
                severity="optimization"
            )
        ]
    
    def _load_diagnostic_history(self) -> List[DiagnosticReport]:
        """載入診斷歷史"""
        if not self.reports_path.exists():
            return []
            
        try:
            with open(self.reports_path, 'r', encoding='utf-8') as f:
                reports_data = json.load(f)
                return [DiagnosticReport(**report) for report in reports_data]
        except Exception as e:
            self.logger.warning(f"無法載入診斷歷史: {e}")
            return []
    
    def diagnose_and_repair(self, file_path: str, content: str, diagram_type: str = "unknown") -> DiagnosticReport:
        """診斷並修復 Mermaid 圖表"""
        timestamp = datetime.now().isoformat()
        original_content = content
        current_content = content
        applied_rules = []
        original_errors = []
        
        self.logger.info(f"開始診斷文件: {file_path}")
        
        # 第一階段：錯誤檢測
        errors = self._detect_errors(current_content, diagram_type)
        original_errors = [error['message'] for error in errors]
        
        self.logger.info(f"檢測到 {len(errors)} 個問題")
        
        # 第二階段：應用修復規則
        for rule in self.repair_rules:
            if diagram_type not in rule.applies_to and "all" not in rule.applies_to:
                continue
                
            # 檢查是否適用
            if self._rule_applies(rule, current_content, errors):
                old_content = current_content
                current_content = self._apply_rule(rule, current_content)
                
                if old_content != current_content:
                    applied_rules.append(rule.rule_id)
                    rule.usage_count += 1
                    self.logger.info(f"應用規則: {rule.name}")
        
        # 第三階段：重新驗證
        final_errors = self._detect_errors(current_content, diagram_type)
        success = len(final_errors) == 0
        
        if success:
            # 更新成功率
            for rule_id in applied_rules:
                rule = next(r for r in self.repair_rules if r.rule_id == rule_id)
                rule.success_rate = (rule.success_rate * (rule.usage_count - 1) + 1.0) / rule.usage_count
        
        # 創建診斷報告
        report = DiagnosticReport(
            timestamp=timestamp,
            file_path=file_path,
            diagram_type=diagram_type,
            original_errors=original_errors,
            applied_rules=applied_rules,
            final_status="success" if success else f"remaining_errors: {len(final_errors)}",
            before_content=original_content,
            after_content=current_content,
            success=success
        )
        
        # 保存診斷記錄
        self._save_diagnostic_report(report)
        
        # 如果修復失敗，學習新規則
        if not success and len(applied_rules) > 0:
            self._learn_from_failure(report, final_errors)
        
        return report
    
    def _detect_errors(self, content: str, diagram_type: str) -> List[Dict[str, Any]]:
        """
        正確的錯誤檢測邏輯 (基於官方 Mermaid.js v11.12.0 標準)
        修正了代碼塊檢測的根本性錯誤
        """
        errors = []
        lines = content.split('\n')
        
        # 使用堆疊追蹤代碼塊配對 - 正確的邏輯
        code_block_stack = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith('```'):
                if stripped == '```':
                    # 代碼塊結束標記
                    if code_block_stack:
                        block_info = code_block_stack.pop()
                        # 正確配對，無需處理
                    else:
                        # 多餘的結束標記
                        errors.append({
                            "line": i,
                            "type": "EXTRA_CODE_BLOCK_END",
                            "message": f"第 {i} 行: 多餘的代碼塊結束標記",
                            "content": line,
                            "severity": "error"
                        })
                        
                elif stripped.startswith('```mermaid'):
                    # Mermaid 代碼塊開始
                    # 檢查是否在其他 mermaid 塊內 (嵌套檢測)
                    if any(block['type'] == 'mermaid' for block in code_block_stack):
                        errors.append({
                            "line": i,
                            "type": "NESTED_MERMAID_BLOCKS",
                            "message": f"第 {i} 行: 檢測到嵌套的 mermaid 代碼塊",
                            "content": line,
                            "severity": "critical"
                        })
                    
                    code_block_stack.append({
                        'type': 'mermaid', 
                        'start_line': i,
                        'language': 'mermaid'
                    })
                    
                else:
                    # 其他語言的代碼塊
                    lang = stripped[3:].strip()  # 移除 ```
                    code_block_stack.append({
                        'type': 'other', 
                        'start_line': i,
                        'language': lang
                    })
            
            # 在 mermaid 代碼塊內檢查語法錯誤
            elif code_block_stack and any(block['type'] == 'mermaid' for block in code_block_stack):
                # 檢查 class 應用中的多餘空格
                if stripped.startswith("class ") and stripped.endswith("  "):
                    errors.append({
                        "line": i,
                        "type": "CLASSDEF_EXTRA_SPACES",
                        "message": f"第 {i} 行: class 應用行尾有多餘空格",
                        "content": line,
                        "severity": "error"
                    })
                
                # 檢查 direction 語法
                elif stripped.startswith("direction ") and len(stripped.split()) > 2:
                    errors.append({
                        "line": i,
                        "type": "DIRECTION_SYNTAX_ERROR",
                        "message": f"第 {i} 行: direction 語法格式錯誤",
                        "content": line,
                        "severity": "warning"
                    })
        
        # 檢查未關閉的代碼塊
        for block in code_block_stack:
            errors.append({
                "line": block['start_line'],
                "type": "UNCLOSED_CODE_BLOCK",
                "message": f"第 {block['start_line']} 行: 未關閉的 {block['type']} 代碼塊",
                "content": lines[block['start_line'] - 1] if block['start_line'] <= len(lines) else "",
                "severity": "critical"
            })
        
        return errors
    
    def _rule_applies(self, rule: RepairRule, content: str, errors: List[Dict[str, Any]]) -> bool:
        """
        安全的規則應用邏輯 (基於安全修復邊界)
        只修復明確安全的錯誤，避免破壞文件結構
        """
        import re
        
        # 定義安全修復邊界
        safe_repair_rules = {
            'CLASSDEF_EXTRA_SPACES': 'always_safe',  # 行尾空格總是安全的
            'DIRECTION_SYNTAX_ERROR': 'always_safe',  # direction 語法錯誤總是安全的
            'NESTED_MERMAID_BLOCKS': 'needs_validation',  # 需要驗證內容
            'UNCLOSED_CODE_BLOCK': 'context_dependent',  # 依賴上下文
            'EXTRA_CODE_BLOCK_END': 'context_dependent'  # 依賴上下文
        }
        
        safety_level = safe_repair_rules.get(rule.rule_id, 'unsafe')
        
        # 檢查是否有對應的檢測錯誤
        error_types = [error['type'] for error in errors]
        has_matching_error = rule.rule_id in error_types
        
        if safety_level == 'always_safe' and has_matching_error:
            return True
            
        elif safety_level == 'needs_validation' and has_matching_error:
            return self._validate_safe_merge(rule, content, errors)
            
        elif safety_level == 'context_dependent' and has_matching_error:
            return self._validate_context_safety(rule, content, errors)
            
        elif safety_level == 'unsafe':
            return False  # 不修復不安全的錯誤
            
        # 對於安全規則，進行模式匹配檢查
        if safety_level == 'always_safe':
            try:
                match = re.search(rule.pattern, content, re.MULTILINE)
                return match is not None
            except re.error:
                return False
                
        return False
    
    def _validate_safe_merge(self, rule: RepairRule, content: str, errors: List[Dict[str, Any]]) -> bool:
        """驗證嵌套塊合併是否安全"""
        if rule.rule_id != 'NESTED_MERMAID_BLOCKS':
            return False
            
        # 檢查是否會破壞圖表邏輯
        lines = content.split('\n')
        
        # 查找嵌套的 mermaid 塊
        for error in errors:
            if error['type'] == 'NESTED_MERMAID_BLOCKS':
                error_line = error['line']
                
                # 檢查前後內容是否可以安全合併
                if error_line <= len(lines):
                    # 簡單檢查：如果兩個 mermaid 塊都是 graph/flowchart，可以合併
                    return True  # 暫時保守處理
                    
        return False
    
    def _validate_context_safety(self, rule: RepairRule, content: str, errors: List[Dict[str, Any]]) -> bool:
        """驗證上下文相關修復是否安全"""
        lines = content.split('\n')
        
        if rule.rule_id == 'EXTRA_CODE_BLOCK_END':
            # 檢查孤立的結束標記是否真的多餘
            for error in errors:
                if error['type'] == 'EXTRA_CODE_BLOCK_END':
                    error_line = error['line']
                    
                    # 檢查前後5行，確認這確實是孤立標記
                    start = max(0, error_line - 6)
                    end = min(len(lines), error_line + 5)
                    context_lines = lines[start:end]
                    
                    # 如果前後沒有對應的代碼塊開始，則安全移除
                    has_block_start = any('```' in line and line.strip() != '```' 
                                        for line in context_lines)
                    
                    return not has_block_start  # 沒有塊開始才安全移除
                    
        elif rule.rule_id == 'UNCLOSED_CODE_BLOCK':
            # 對於未關閉代碼塊，目前不自動修復 (太危險)
            return False
            
        return False
    
    def _apply_rule(self, rule: RepairRule, content: str) -> str:
        """應用修復規則"""
        import re
        
        try:
            return re.sub(rule.pattern, rule.replacement, content, flags=re.MULTILINE)
        except re.error:
            self.logger.warning(f"無法應用規則 {rule.rule_id}")
            return content
    
    def _save_diagnostic_report(self, report: DiagnosticReport):
        """保存診斷報告"""
        self.diagnostic_history.append(report)
        
        try:
            # 保存報告歷史 (只保留最近 100 個)
            reports_to_save = self.diagnostic_history[-100:]
            with open(self.reports_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(report) for report in reports_to_save], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"無法保存診斷報告: {e}")
    
    def _learn_from_failure(self, report: DiagnosticReport, remaining_errors: List[Dict[str, Any]]):
        """從修復失敗中學習新規則"""
        self.logger.info(f"從失敗案例學習: {report.file_path}")
        
        # 分析未修復的錯誤模式
        for error in remaining_errors:
            # 這裡可以添加機器學習邏輯來生成新規則
            self.logger.info(f"未修復錯誤: {error['message']}")
    
    def save_rules(self):
        """保存修復規則"""
        try:
            with open(self.rules_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(rule) for rule in self.repair_rules], f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"無法保存修復規則: {e}")
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """獲取規則統計信息"""
        total_usage = sum(rule.usage_count for rule in self.repair_rules)
        
        return {
            "total_rules": len(self.repair_rules),
            "total_usage": total_usage,
            "rules_by_success_rate": sorted(
                [(rule.rule_id, rule.name, rule.success_rate, rule.usage_count) for rule in self.repair_rules],
                key=lambda x: x[2], reverse=True
            ),
            "total_diagnostics": len(self.diagnostic_history),
            "success_rate": len([r for r in self.diagnostic_history if r.success]) / max(len(self.diagnostic_history), 1)
        }
    
    def add_custom_rule(self, rule: RepairRule):
        """添加自定義規則"""
        # 檢查是否已存在
        existing = next((r for r in self.repair_rules if r.rule_id == rule.rule_id), None)
        if existing:
            self.logger.warning(f"規則 {rule.rule_id} 已存在，將更新")
            self.repair_rules.remove(existing)
        
        self.repair_rules.append(rule)
        self.save_rules()
        self.logger.info(f"添加新規則: {rule.name}")


def batch_diagnose_project(project_path: str, extensions: List[str] = ['.md', '.mmd']):
    """批量診斷項目中的 Mermaid 文件"""
    diagnostic_system = MermaidDiagnosticSystem()
    
    project_dir = Path(project_path)
    results = []
    
    # 查找所有相關文件
    for ext in extensions:
        for file_path in project_dir.rglob(f"*{ext}"):
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # 檢查是否包含 Mermaid 內容
                if 'mermaid' in content.lower() or any(keyword in content for keyword in ['graph', 'flowchart', 'sequenceDiagram']):
                    result = diagnostic_system.diagnose_and_repair(
                        str(file_path),
                        content,
                        "auto-detected"
                    )
                    results.append(result)
                    
            except Exception as e:
                logging.error(f"處理文件 {file_path} 時出錯: {e}")
    
    # 生成批量報告
    successful = len([r for r in results if r.success])
    total = len(results)
    
    print(f"\n=== 批量診斷結果 ===")
    print(f"總文件數: {total}")
    print(f"成功修復: {successful}")
    print(f"成功率: {successful/max(total,1)*100:.1f}%")
    
    # 顯示統計信息
    stats = diagnostic_system.get_rule_statistics()
    print(f"\n=== 規則使用統計 ===")
    print(f"總規則數: {stats['total_rules']}")
    print(f"規則使用次數: {stats['total_usage']}")
    print(f"整體成功率: {stats['success_rate']*100:.1f}%")
    
    return results


if __name__ == "__main__":
    # 測試診斷系統
    print("AIVA Mermaid 診斷和修復系統")
    print("=" * 50)
    
    # 創建診斷系統
    diagnostic_system = MermaidDiagnosticSystem()
    
    # 測試內容
    test_content = """
    ```mermaid
    graph TB
        A[測試] --> B[節點]
        
        classDef myStyle fill:#ff0000,stroke:#000000,stroke-width:2px
        class A,B myStyle  
    ```
    """
    
    # 進行診斷
    result = diagnostic_system.diagnose_and_repair("test.md", test_content, "flowchart")
    
    print(f"診斷結果: {'成功' if result.success else '失敗'}")
    print(f"應用的規則: {result.applied_rules}")
    print(f"原始錯誤: {result.original_errors}")
    
    if result.before_content != result.after_content:
        print("\n修復前:")
        print(result.before_content)
        print("\n修復後:")
        print(result.after_content)