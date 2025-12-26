#!/usr/bin/env python3
"""
連接建議分析器 - 分析哪些函數應該連接以及影響評估

功能：
1. 分析函數的語義相關性
2. 根據函數名稱、參數類型、返回值推斷連接關係
3. 評估缺失連接對系統運作的影響等級
4. 生成具體的修復建議
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class FunctionInfo:
    """函數信息"""
    name: str
    file: str
    params: List[Tuple[str, Optional[str]]]  # (param_name, type_hint)
    return_type: Optional[str]
    calls: List[str]
    called_by: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    line_number: int = 0


@dataclass
class ConnectionRecommendation:
    """連接建議"""
    caller: str
    caller_file: str
    callee: str
    callee_file: str
    confidence: str  # CRITICAL, HIGH, MEDIUM, LOW
    reason: List[str]
    impact: str  # BLOCKING, HIGH, MEDIUM, LOW, NONE
    impact_description: str
    suggested_location: str
    code_example: Optional[str] = None


class ConnectionRecommendationAnalyzer:
    """連接建議分析器"""
    
    def __init__(self, source_root: str):
        self.source_root = Path(source_root)
        self.functions: Dict[str, FunctionInfo] = {}
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)
        
        # 關鍵詞模式匹配
        self.critical_patterns = {
            'security': ['xss', 'sql_injection', 'idor', 'csrf', 'auth', 'validate', 'sanitize'],
            'core': ['init', 'start', 'stop', 'execute', 'run', 'main', 'process'],
            'data': ['save', 'load', 'persist', 'store', 'retrieve', 'fetch', 'query'],
            'state': ['record', 'track', 'monitor', 'log', 'update', 'sync'],
            'ml': ['train', 'predict', 'infer', 'fit', 'evaluate', 'forward', 'backward'],
        }
        
    def extract_all_functions(self):
        """提取所有函數信息"""
        print("🔍 提取函數信息...")
        
        for py_file in self.source_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=str(py_file))
                    self._extract_from_ast(tree, py_file)
            except Exception as e:
                print(f"  ⚠️  無法解析 {py_file.name}: {e}")
                
        print(f"  ✅ 提取了 {len(self.functions)} 個函數")
        
    def _should_skip_file(self, file_path: Path) -> bool:
        """判斷是否跳過文件"""
        skip_patterns = ['__pycache__', 'test_', '_test.py', '.venv', 'venv', 'node_modules']
        return any(pattern in str(file_path) for pattern in skip_patterns)
        
    def _extract_from_ast(self, tree: ast.AST, file_path: Path):
        """從AST提取函數信息"""
        relative_path = file_path.relative_to(self.source_root)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = self._parse_function(node, str(relative_path))
                full_name = f"{relative_path}::{func_info.name}"
                self.functions[full_name] = func_info
                
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                self._extract_imports(node, str(relative_path))
                
    def _parse_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str) -> FunctionInfo:
        """解析函數定義"""
        params = []
        for arg in node.args.args:
            param_name = arg.arg
            type_hint = ast.unparse(arg.annotation) if arg.annotation else None
            params.append((param_name, type_hint))
            
        return_type = ast.unparse(node.returns) if node.returns else None
        
        # 提取函數調用
        calls = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    calls.append(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    calls.append(n.func.attr)
                    
        docstring = ast.get_docstring(node)
        decorators = [ast.unparse(d) for d in node.decorator_list]
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        return FunctionInfo(
            name=node.name,
            file=file_path,
            params=params,
            return_type=return_type,
            calls=calls,
            docstring=docstring,
            decorators=decorators,
            is_async=is_async,
            line_number=node.lineno
        )
        
    def _extract_imports(self, node, file_path: str):
        """提取導入信息"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.file_imports[file_path].add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            self.file_imports[file_path].add(node.module)
                
    def analyze_missing_connections(self) -> List[ConnectionRecommendation]:
        """分析缺失的連接"""
        print("\n🔗 分析缺失連接...")
        recommendations = []
        
        # 找出孤立函數（有接口但無調用）
        orphaned_functions = self._find_orphaned_functions()
        print(f"  發現 {len(orphaned_functions)} 個孤立函數")
        
        # 找出潛在的調用者
        for orphan_key, orphan_func in orphaned_functions.items():
            potential_callers = self._find_potential_callers(orphan_func)
            
            for caller_key, confidence, reasons in potential_callers:
                caller_func = self.functions[caller_key]
                impact, impact_desc = self._evaluate_impact(orphan_func)
                
                recommendation = ConnectionRecommendation(
                    caller=caller_func.name,
                    caller_file=caller_func.file,
                    callee=orphan_func.name,
                    callee_file=orphan_func.file,
                    confidence=confidence,
                    reason=reasons,
                    impact=impact,
                    impact_description=impact_desc,
                    suggested_location=self._suggest_location(caller_func),
                    code_example=self._generate_code_example(orphan_func)
                )
                
                recommendations.append(recommendation)
                
        # 按影響等級和置信度排序
        recommendations.sort(key=lambda x: (
            self._impact_score(x.impact),
            self._confidence_score(x.confidence)
        ), reverse=True)
        
        print(f"  ✅ 生成了 {len(recommendations)} 個連接建議")
        return recommendations
        
    def _find_orphaned_functions(self) -> Dict[str, FunctionInfo]:
        """找出孤立的函數（有參數或返回值但未被調用）"""
        orphaned = {}
        
        # 構建調用圖
        all_called = set()
        for func_key, func_info in self.functions.items():
            for call in func_info.calls:
                all_called.add(call)
                
        # 找出未被調用的函數
        for func_key, func_info in self.functions.items():
            # 跳過特殊函數
            if func_info.name.startswith('_') and func_info.name != '__init__':
                continue
            if func_info.name in ['main', '__main__']:
                continue
                
            # 檢查是否被調用
            is_called = func_info.name in all_called
            
            # 有接口但未被調用
            has_interface = len(func_info.params) > 0 or func_info.return_type is not None
            
            if not is_called and has_interface:
                orphaned[func_key] = func_info
                
        return orphaned
        
    def _find_potential_callers(self, orphan_func: FunctionInfo) -> List[Tuple[str, str, List[str]]]:
        """找出潛在的調用者"""
        potential = []
        
        for func_key, func_info in self.functions.items():
            if func_info.file == orphan_func.file and func_info.name == orphan_func.name:
                continue
                
            confidence, reasons = self._calculate_connection_confidence(func_info, orphan_func)
            
            if confidence != "NONE":
                potential.append((func_key, confidence, reasons))
                
        return potential
        
    def _calculate_connection_confidence(
        self, 
        potential_caller: FunctionInfo, 
        callee: FunctionInfo
    ) -> Tuple[str, List[str]]:
        """計算連接的置信度"""
        reasons = []
        score = 0
        
        # 1. 文件相關性（同一文件或相關文件）
        if potential_caller.file == callee.file:
            score += 20
            reasons.append("同一文件中定義")
        elif self._are_files_related(potential_caller.file, callee.file):
            score += 10
            reasons.append("相關文件")
            
        # 2. 命名相關性
        name_similarity = self._calculate_name_similarity(potential_caller.name, callee.name)
        if name_similarity > 0.6:
            score += 15
            reasons.append(f"函數名稱相似度: {name_similarity:.0%}")
            
        # 3. 語義相關性
        semantic_match = self._check_semantic_match(potential_caller, callee)
        if semantic_match:
            score += 25
            reasons.append(f"語義相關: {semantic_match}")
            
        # 4. 參數類型匹配
        if self._check_parameter_compatibility(potential_caller, callee):
            score += 20
            reasons.append("參數類型兼容")
            
        # 5. 調用鏈分析
        if self._check_call_chain_logic(potential_caller, callee):
            score += 15
            reasons.append("符合調用鏈邏輯")
            
        # 6. Docstring 提示
        if self._check_docstring_hints(potential_caller, callee):
            score += 10
            reasons.append("文檔字符串提示相關")
            
        # 根據分數確定置信度
        if score >= 60:
            return "CRITICAL", reasons
        elif score >= 40:
            return "HIGH", reasons
        elif score >= 20:
            return "MEDIUM", reasons
        elif score > 0:
            return "LOW", reasons
        else:
            return "NONE", []
            
    def _are_files_related(self, file1: str, file2: str) -> bool:
        """判斷兩個文件是否相關"""
        # 同一目錄
        if Path(file1).parent == Path(file2).parent:
            return True
            
        # 相似的文件名前綴
        name1 = Path(file1).stem
        name2 = Path(file2).stem
        
        common_prefixes = ['aiva_', 'ai_', 'ml_', 'rl_', 'neural_']
        for prefix in common_prefixes:
            if name1.startswith(prefix) and name2.startswith(prefix):
                return True
                
        return False
        
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """計算函數名稱相似度"""
        # 移除常見前綴/後綴
        name1 = re.sub(r'^(get_|set_|call_|handle_|process_)', '', name1)
        name2 = re.sub(r'^(get_|set_|call_|handle_|process_)', '', name2)
        
        # 分詞
        words1 = set(re.findall(r'[a-z]+', name1.lower()))
        words2 = set(re.findall(r'[a-z]+', name2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
        
    def _check_semantic_match(self, caller: FunctionInfo, callee: FunctionInfo) -> Optional[str]:
        """檢查語義匹配"""
        callee_lower = callee.name.lower()
        caller_lower = caller.name.lower()
        
        for category, keywords in self.critical_patterns.items():
            callee_matches = [kw for kw in keywords if kw in callee_lower]
            caller_matches = [kw for kw in keywords if kw in caller_lower]
            
            if callee_matches and caller_matches:
                return f"{category} 相關 ({', '.join(callee_matches)})"
                
        return None
        
    def _check_parameter_compatibility(self, caller: FunctionInfo, callee: FunctionInfo) -> bool:
        """檢查參數兼容性"""
        if not callee.params:
            return True
            
        # 檢查調用者是否有可能提供這些參數
        callee_param_types = {ptype for _, ptype in callee.params if ptype}
        
        if not callee_param_types:
            return True
            
        # 簡單的類型匹配檢查
        for _, param_type in caller.params:
            if param_type and param_type in callee_param_types:
                return True
                
        return False
        
    def _check_call_chain_logic(self, caller: FunctionInfo, callee: FunctionInfo) -> bool:
        """檢查調用鏈邏輯"""
        # 檢查常見的調用模式
        patterns = [
            (r'init', r'(setup|start|register|load)'),
            (r'process', r'(validate|transform|save)'),
            (r'execute', r'(prepare|run|cleanup)'),
            (r'train', r'(forward|backward|update)'),
            (r'analyze', r'(collect|process|report)'),
        ]
        
        caller_lower = caller.name.lower()
        callee_lower = callee.name.lower()
        
        for caller_pattern, callee_pattern in patterns:
            if re.search(caller_pattern, caller_lower) and re.search(callee_pattern, callee_lower):
                return True
                
        return False
        
    def _check_docstring_hints(self, caller: FunctionInfo, callee: FunctionInfo) -> bool:
        """檢查文檔字符串提示"""
        if not caller.docstring or not callee.name:
            return False
            
        # 在調用者的文檔中提到被調用者
        return callee.name.lower() in caller.docstring.lower()
        
    def _evaluate_impact(self, orphan_func: FunctionInfo) -> Tuple[str, str]:
        """評估缺失連接的影響"""
        func_lower = orphan_func.name.lower()
        
        # BLOCKING: 關鍵安全功能
        security_keywords = ['xss', 'sql', 'injection', 'csrf', 'auth', 'validate', 'sanitize', 'idor']
        if any(kw in func_lower for kw in security_keywords):
            return "BLOCKING", "❌ 安全功能未啟用，系統存在嚴重安全風險"
            
        # HIGH: 核心功能
        core_keywords = ['init', 'start', 'main', 'execute', 'process', 'train', 'predict']
        if any(kw in func_lower for kw in core_keywords):
            return "HIGH", "⚠️  核心功能未連接，可能導致功能失效"
            
        # MEDIUM: 數據處理
        data_keywords = ['save', 'load', 'store', 'persist', 'record', 'log']
        if any(kw in func_lower for kw in data_keywords):
            return "MEDIUM", "⚡ 數據處理功能未使用，可能丟失重要信息"
            
        # LOW: 輔助功能
        return "LOW", "ℹ️  輔助功能未使用，對核心功能影響較小"
        
    def _suggest_location(self, caller: FunctionInfo) -> str:
        """建議插入位置"""
        return f"在 {caller.file} 的 {caller.name} 函數中，第 {caller.line_number} 行附近"
        
    def _generate_code_example(self, callee: FunctionInfo) -> Optional[str]:
        """生成代碼示例"""
        # 生成參數
        args = []
        for param_name, param_type in callee.params:
            if param_name == 'self':
                continue
            args.append(param_name)
            
        args_str = ', '.join(args)
        
        # 生成調用代碼
        if callee.return_type:
            return f"result = self.{callee.name}({args_str})"
        else:
            return f"self.{callee.name}({args_str})"
            
    def _impact_score(self, impact: str) -> int:
        """影響等級分數"""
        scores = {"BLOCKING": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
        return scores.get(impact, 0)
        
    def _confidence_score(self, confidence: str) -> int:
        """置信度分數"""
        scores = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
        return scores.get(confidence, 0)
        
    def generate_report(self, recommendations: List[ConnectionRecommendation], output_file: str):
        """生成報告"""
        print(f"\n📝 生成報告: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 🔗 連接建議與影響評估報告\n\n")
            f.write(f"生成時間: {self._get_timestamp()}\n\n")
            
            # 統計信息
            f.write("## 📊 統計摘要\n\n")
            f.write(f"- **總建議數:** {len(recommendations)}\n")
            
            impact_stats: dict[str, int] = defaultdict(int)
            confidence_stats: dict[str, int] = defaultdict(int)
            for rec in recommendations:
                impact_stats[rec.impact] += 1
                confidence_stats[rec.confidence] += 1
                
            f.write("\n### 按影響等級\n\n")
            for level in ["BLOCKING", "HIGH", "MEDIUM", "LOW"]:
                count = impact_stats[level]
                if count > 0:
                    f.write(f"- **{level}:** {count} 個\n")
                    
            f.write("\n### 按置信度\n\n")
            for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                count = confidence_stats[level]
                if count > 0:
                    f.write(f"- **{level}:** {count} 個\n")
                    
            # 詳細建議
            f.write("\n---\n\n## 🎯 詳細建議\n\n")
            
            for i, rec in enumerate(recommendations, 1):
                self._write_recommendation(f, i, rec)
                
            # 總結
            f.write("\n---\n\n## 📋 修復總結\n\n")
            f.write("### 🔴 必須立即修復（BLOCKING）\n\n")
            blocking = [r for r in recommendations if r.impact == "BLOCKING"]
            if blocking:
                for rec in blocking[:10]:
                    f.write(f"- `{rec.callee}` 應該被 `{rec.caller}` 調用\n")
                if len(blocking) > 10:
                    f.write(f"\n*... 還有 {len(blocking) - 10} 個 BLOCKING 問題*\n")
            else:
                f.write("✅ 無 BLOCKING 問題\n")
                
            f.write("\n### ⚠️  建議優先修復（HIGH）\n\n")
            high = [r for r in recommendations if r.impact == "HIGH"]
            if high:
                for rec in high[:10]:
                    f.write(f"- `{rec.callee}` 應該被 `{rec.caller}` 調用\n")
                if len(high) > 10:
                    f.write(f"\n*... 還有 {len(high) - 10} 個 HIGH 問題*\n")
            else:
                f.write("✅ 無 HIGH 問題\n")
                
        print("  ✅ 報告已保存")
        
    def _write_recommendation(self, f, index: int, rec: ConnectionRecommendation):
        """寫入單個建議"""
        # 標題
        impact_emoji = {
            "BLOCKING": "🔴",
            "HIGH": "⚠️ ",
            "MEDIUM": "⚡",
            "LOW": "ℹ️ "
        }
        
        confidence_emoji = {
            "CRITICAL": "🎯",
            "HIGH": "✅",
            "MEDIUM": "🟡",
            "LOW": "❓"
        }
        
        f.write(f"### {index}. {impact_emoji.get(rec.impact, '')} {confidence_emoji.get(rec.confidence, '')} "
                f"`{rec.caller}` → `{rec.callee}`\n\n")
        
        # 基本信息
        f.write(f"**影響等級:** {rec.impact}\n\n")
        f.write(f"**置信度:** {rec.confidence}\n\n")
        f.write(f"{rec.impact_description}\n\n")
        
        # 文件位置
        f.write(f"**調用者文件:** `{rec.caller_file}`\n\n")
        f.write(f"**被調用者文件:** `{rec.callee_file}`\n\n")
        
        # 原因
        f.write("**建議原因:**\n\n")
        for reason in rec.reason:
            f.write(f"- {reason}\n")
        f.write("\n")
        
        # 建議位置
        f.write(f"**建議修改位置:**\n\n{rec.suggested_location}\n\n")
        
        # 代碼示例
        if rec.code_example:
            f.write(f"**代碼示例:**\n\n```python\n{rec.code_example}\n```\n\n")
            
        f.write("---\n\n")
        
    def _get_timestamp(self) -> str:
        """獲取時間戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="連接建議分析器")
    parser.add_argument("--source-root", required=True, help="源代碼根目錄")
    parser.add_argument("--output", default="connection_recommendations.md", help="輸出文件")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔗 連接建議與影響評估分析器")
    print("=" * 70)
    
    analyzer = ConnectionRecommendationAnalyzer(args.source_root)
    
    # 提取函數
    analyzer.extract_all_functions()
    
    # 分析連接
    recommendations = analyzer.analyze_missing_connections()
    
    # 生成報告
    analyzer.generate_report(recommendations, args.output)
    
    print("\n" + "=" * 70)
    print("✅ 分析完成！")
    print("=" * 70)
    
    # 打印摘要
    blocking = len([r for r in recommendations if r.impact == "BLOCKING"])
    high = len([r for r in recommendations if r.impact == "HIGH"])
    
    if blocking > 0:
        print(f"\n❌ 發現 {blocking} 個 BLOCKING 問題 - 必須立即修復！")
    if high > 0:
        print(f"⚠️  發現 {high} 個 HIGH 問題 - 建議優先修復")
        
    print(f"\n📄 完整報告: {args.output}")


if __name__ == "__main__":
    main()
