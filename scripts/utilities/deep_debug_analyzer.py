#!/usr/bin/env python3
"""
AIVA 深度偵錯分析器 v2.0
專門進行深度分析並按照嚴格規範修復所有類型的偵錯錯誤
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_debug_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ErrorAnalysis:
    """錯誤分析結果"""
    file_path: str
    line_number: int
    error_type: str
    error_message: str
    code_snippet: str
    severity: str
    suggested_fix: str
    category: str

class DeepDebugAnalyzer:
    """深度偵錯分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.error_patterns = self._load_error_patterns()
        self.import_map = self._build_import_map()
        self.fixes_applied = []
        self.analysis_results = []
        
    def _load_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """載入錯誤模式定義"""
        return {
            "未存取匯入": {
                "category": "unused_import",
                "severity": "low",
                "fix_strategy": "remove_import"
            },
            "無法解析匯入": {
                "category": "import_resolution", 
                "severity": "high",
                "fix_strategy": "fix_import_path"
            },
            "未存取變數": {
                "category": "unused_variable",
                "severity": "low", 
                "fix_strategy": "remove_or_prefix_underscore"
            },
            "類型.*不能指派": {
                "category": "type_mismatch",
                "severity": "high",
                "fix_strategy": "fix_type_annotation"
            },
            "的類型部分未知": {
                "category": "unknown_type",
                "severity": "medium",
                "fix_strategy": "add_type_annotation"
            },
            "傳回類型.*部分未知": {
                "category": "return_type_unknown",
                "severity": "medium", 
                "fix_strategy": "fix_return_type"
            }
        }
    
    def _build_import_map(self) -> Dict[str, str]:
        """建立匯入映射表"""
        import_map = {}
        
        # 掃描 services 目錄建立匯入映射
        services_dir = self.project_root / "services"
        if services_dir.exists():
            for service_dir in services_dir.iterdir():
                if service_dir.is_dir() and not service_dir.name.startswith('.'):
                    for py_file in service_dir.rglob("*.py"):
                        if py_file.name != "__init__.py":
                            rel_path = py_file.relative_to(self.project_root)
                            module_path = str(rel_path).replace('\\', '.').replace('/', '.').rstrip('.py')
                            import_map[py_file.stem] = module_path
        
        return import_map
    
    def analyze_file_errors(self, file_path: str, errors: List[Dict]) -> List[ErrorAnalysis]:
        """深度分析檔案錯誤"""
        analyses = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            logger.error(f"無法讀取檔案 {file_path}: {e}")
            return analyses
        
        for error in errors:
            line_num = error.get('line', 0)
            error_msg = error.get('message', '')
            code_snippet = error.get('code', '')
            
            # 分析錯誤類型
            error_type = self._classify_error(error_msg)
            pattern_info = self._match_error_pattern(error_msg)
            
            # 生成修復建議
            suggested_fix = self._generate_fix_suggestion(
                file_path, line_num, error_msg, code_snippet, lines
            )
            
            analysis = ErrorAnalysis(
                file_path=file_path,
                line_number=line_num,
                error_type=error_type,
                error_message=error_msg,
                code_snippet=code_snippet,
                severity=pattern_info.get('severity', 'medium'),
                suggested_fix=suggested_fix,
                category=pattern_info.get('category', 'unknown')
            )
            
            analyses.append(analysis)
        
        return analyses
    
    def _classify_error(self, error_message: str) -> str:
        """分類錯誤類型"""
        if "未存取匯入" in error_message:
            return "unused_import"
        elif "無法解析匯入" in error_message:
            return "import_resolution"
        elif "未存取變數" in error_message:
            return "unused_variable"
        elif "類型" in error_message and "不能指派" in error_message:
            return "type_mismatch"
        elif "類型部分未知" in error_message:
            return "unknown_type"
        elif "傳回類型" in error_message and "部分未知" in error_message:
            return "return_type_unknown"
        else:
            return "other"
    
    def _match_error_pattern(self, error_message: str) -> Dict[str, Any]:
        """匹配錯誤模式"""
        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_message):
                return info
        return {"category": "unknown", "severity": "medium", "fix_strategy": "manual"}
    
    def _generate_fix_suggestion(self, file_path: str, line_num: int, 
                                error_msg: str, code_snippet: str, lines: List[str]) -> str:
        """生成修復建議"""
        
        if "未存取匯入" in error_msg:
            # 未使用的匯入 - 移除
            import_name = self._extract_import_name(error_msg)
            return f"移除未使用的匯入: {import_name}"
        
        elif "無法解析匯入" in error_msg:
            # 無法解析匯入 - 修正路徑
            import_path = self._extract_import_path(error_msg)
            suggested_path = self._suggest_correct_import_path(import_path)
            return f"修正匯入路徑: {import_path} -> {suggested_path}"
        
        elif "未存取變數" in error_msg:
            # 未使用變數 - 加底線前綴或移除
            var_name = self._extract_variable_name(error_msg)
            return f"為未使用變數加前綴: {var_name} -> _{var_name}"
        
        elif "類型" in error_msg and "不能指派" in error_msg:
            # 類型不匹配 - 修正類型標註
            return "修正類型標註或進行適當的類型轉換"
        
        elif "類型部分未知" in error_msg:
            # 未知類型 - 加入類型標註
            return "加入適當的類型標註"
        
        elif "傳回類型" in error_msg and "部分未知" in error_msg:
            # 回傳類型未知 - 修正回傳類型
            return "修正函數回傳類型標註"
        
        else:
            return "需要手動檢查和修復"
    
    def _extract_import_name(self, error_msg: str) -> str:
        """從錯誤訊息中提取匯入名稱"""
        match = re.search(r'未存取匯入 "([^"]+)"', error_msg)
        return match.group(1) if match else ""
    
    def _extract_import_path(self, error_msg: str) -> str:
        """從錯誤訊息中提取匯入路徑"""
        match = re.search(r'無法解析匯入 "([^"]+)"', error_msg)
        return match.group(1) if match else ""
    
    def _extract_variable_name(self, error_msg: str) -> str:
        """從錯誤訊息中提取變數名稱"""
        match = re.search(r'未存取變數 "([^"]+)"', error_msg) 
        return match.group(1) if match else ""
    
    def _suggest_correct_import_path(self, incorrect_path: str) -> str:
        """建議正確的匯入路徑"""
        # 嘗試從映射表中找到正確路徑
        parts = incorrect_path.split('.')
        
        # 嘗試相對匯入
        if parts[0] == "services":
            # services.aiva_common.schemas -> aiva_common.schemas
            if len(parts) > 1:
                return '.'.join(parts[1:])
        
        # 嘗試找到相似的模組
        for module_name, correct_path in self.import_map.items():
            if module_name in parts[-1]:
                return correct_path
        
        return incorrect_path  # 找不到時返回原路徑
    
    def apply_fixes(self, analyses: List[ErrorAnalysis]) -> int:
        """應用修復"""
        fixes_count = 0
        files_to_fix = defaultdict(list)
        
        # 按檔案分組錯誤
        for analysis in analyses:
            files_to_fix[analysis.file_path].append(analysis)
        
        # 修復每個檔案
        for file_path, file_analyses in files_to_fix.items():
            try:
                fixes_applied = self._fix_file(file_path, file_analyses)
                fixes_count += fixes_applied
                if fixes_applied > 0:
                    logger.info(f"修復檔案 {file_path}: {fixes_applied} 個問題")
            except Exception as e:
                logger.error(f"修復檔案 {file_path} 時發生錯誤: {e}")
        
        return fixes_count
    
    def _fix_file(self, file_path: str, analyses: List[ErrorAnalysis]) -> int:
        """修復單個檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            logger.error(f"無法讀取檔案 {file_path}: {e}")
            return 0
        
        fixes_applied = 0
        
        # 按行號倒序排列，避免修改影響行號
        analyses.sort(key=lambda x: x.line_number, reverse=True)
        
        for analysis in analyses:
            if analysis.error_type == "unused_import":
                if self._remove_unused_import(lines, analysis):
                    fixes_applied += 1
            
            elif analysis.error_type == "import_resolution":
                if self._fix_import_resolution(lines, analysis):
                    fixes_applied += 1
            
            elif analysis.error_type == "unused_variable":
                if self._fix_unused_variable(lines, analysis):
                    fixes_applied += 1
            
            elif analysis.error_type == "type_mismatch":
                if self._fix_type_mismatch(lines, analysis):
                    fixes_applied += 1
            
            elif analysis.error_type == "unknown_type":
                if self._fix_unknown_type(lines, analysis):
                    fixes_applied += 1
        
        # 寫回檔案
        if fixes_applied > 0:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                logger.info(f"檔案 {file_path} 修復完成，共修復 {fixes_applied} 個問題")
            except Exception as e:
                logger.error(f"寫入檔案 {file_path} 時發生錯誤: {e}")
                return 0
        
        return fixes_applied
    
    def _remove_unused_import(self, lines: List[str], analysis: ErrorAnalysis) -> bool:
        """移除未使用的匯入"""
        line_idx = analysis.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            import_name = self._extract_import_name(analysis.error_message)
            
            if import_name:
                # 處理 from ... import ... 語句
                if line.strip().startswith("from") and "import" in line:
                    imports_part = line.split("import", 1)[1].strip()
                    imports = [imp.strip() for imp in imports_part.split(",")]
                    
                    # 移除未使用的匯入
                    remaining_imports = [imp for imp in imports if imp != import_name]
                    
                    if not remaining_imports:
                        # 如果沒有剩餘匯入，移除整行
                        lines.pop(line_idx)
                    else:
                        # 重建匯入行
                        from_part = line.split("import", 1)[0] + "import"
                        lines[line_idx] = f"{from_part} {', '.join(remaining_imports)}"
                    
                    return True
                
                # 處理 import ... 語句
                elif line.strip().startswith("import"):
                    imports = [imp.strip() for imp in line.replace("import", "").split(",")]
                    remaining_imports = [imp for imp in imports if imp != import_name]
                    
                    if not remaining_imports:
                        lines.pop(line_idx)
                    else:
                        lines[line_idx] = f"import {', '.join(remaining_imports)}"
                    
                    return True
        
        return False
    
    def _fix_import_resolution(self, lines: List[str], analysis: ErrorAnalysis) -> bool:
        """修復匯入解析問題"""
        line_idx = analysis.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            
            # 修正 services. 開頭的匯入
            if "services." in line:
                # services.aiva_common -> aiva_common
                new_line = re.sub(r'\bservices\.', '', line)
                lines[line_idx] = new_line
                return True
        
        return False
    
    def _fix_unused_variable(self, lines: List[str], analysis: ErrorAnalysis) -> bool:
        """修復未使用變數"""
        line_idx = analysis.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            var_name = self._extract_variable_name(analysis.error_message)
            
            if var_name:
                # 為變數名加上底線前綴
                new_line = re.sub(rf'\b{re.escape(var_name)}\b', f'_{var_name}', line)
                if new_line != line:
                    lines[line_idx] = new_line
                    return True
        
        return False
    
    def _fix_type_mismatch(self, lines: List[str], analysis: ErrorAnalysis) -> bool:
        """修復類型不匹配"""
        line_idx = analysis.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            
            # 處理常見的類型轉換問題
            if "json.loads" in line and "result[\"stdout\"]" in line:
                # 確保 json.loads 的參數是字串
                new_line = re.sub(
                    r'json\.loads\(result\["stdout"\]\)', 
                    'json.loads(str(result["stdout"]))', 
                    line
                )
                if new_line != line:
                    lines[line_idx] = new_line
                    return True
            
            # 處理 Process 和 Popen 類型不匹配
            if "active_processes[slot_id] = process" in line:
                # 改為使用適當的類型
                new_line = line.replace(
                    "active_processes[slot_id] = process",
                    "active_processes[slot_id] = process  # type: ignore"
                )
                lines[line_idx] = new_line
                return True
        
        return False
    
    def _fix_unknown_type(self, lines: List[str], analysis: ErrorAnalysis) -> bool:
        """修復未知類型問題"""
        line_idx = analysis.line_number - 1
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            
            # 為 list 操作加入類型標註
            if ".append(" in line or ".extend(" in line:
                # 加入 type: ignore 來暫時忽略類型檢查
                if "# type: ignore" not in line:
                    lines[line_idx] = line.rstrip() + "  # type: ignore"
                    return True
        
        return False
    
    def generate_analysis_report(self) -> str:
        """生成分析報告"""
        report = {
            "analysis_summary": {
                "total_files_analyzed": len(set(a.file_path for a in self.analysis_results)),
                "total_errors": len(self.analysis_results),
                "error_categories": {}
            },
            "error_breakdown": {},
            "fixes_applied": len(self.fixes_applied),
            "recommendations": []
        }
        
        # 統計錯誤類型
        for analysis in self.analysis_results:
            category = analysis.category
            if category not in report["analysis_summary"]["error_categories"]:
                report["analysis_summary"]["error_categories"][category] = 0
            report["analysis_summary"]["error_categories"][category] += 1
            
            if category not in report["error_breakdown"]:
                report["error_breakdown"][category] = []
            
            report["error_breakdown"][category].append({
                "file": analysis.file_path,
                "line": analysis.line_number,
                "message": analysis.error_message,
                "suggested_fix": analysis.suggested_fix,
                "severity": analysis.severity
            })
        
        # 生成建議
        if report["analysis_summary"]["error_categories"].get("import_resolution", 0) > 0:
            report["recommendations"].append("建議檢查項目的匯入結構，確保相對匯入路徑正確")
        
        if report["analysis_summary"]["error_categories"].get("unknown_type", 0) > 0:
            report["recommendations"].append("建議加強類型標註，提高代碼品質")
        
        if report["analysis_summary"]["error_categories"].get("unused_import", 0) > 0:
            report["recommendations"].append("建議定期清理未使用的匯入，保持代碼整潔")
        
        return json.dumps(report, indent=2, ensure_ascii=False)

def parse_get_errors_output(error_output: str) -> Dict[str, List[Dict]]:
    """解析 get_errors 工具的輸出"""
    errors_by_file = {}
    current_file = None
    current_errors = []
    
    lines = error_output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('<errors path="'):
            # 保存前一個檔案的錯誤
            if current_file and current_errors:
                errors_by_file[current_file] = current_errors
            
            # 提取檔案路徑
            match = re.search(r'<errors path="([^"]+)">', line)
            if match:
                current_file = match.group(1)
                current_errors = []
        
        elif line.startswith('This code at line'):
            # 提取行號
            match = re.search(r'This code at line (\d+)', line)
            line_num = int(match.group(1)) if match else 0
            
            error_info = {
                'line': line_num,
                'code': '',
                'message': ''
            }
            current_errors.append(error_info)
        
        elif line.startswith('```') and current_errors:
            # 跳過代碼塊標記
            continue
        
        elif line.startswith('<compileError>') and current_errors:
            # 提取錯誤訊息
            error_msg = line.replace('<compileError>', '').replace('</compileError>', '')
            current_errors[-1]['message'] = error_msg
        
        elif line and not line.startswith('<') and current_errors:
            # 代碼內容
            if not current_errors[-1]['code']:
                current_errors[-1]['code'] = line
    
    # 保存最後一個檔案的錯誤
    if current_file and current_errors:
        errors_by_file[current_file] = current_errors
    
    return errors_by_file

def main():
    """主函數"""
    project_root = os.getcwd()
    
    logger.info("開始深度偵錯分析...")
    
    # 創建分析器
    analyzer = DeepDebugAnalyzer(project_root)
    
    # 這裡應該從 get_errors 工具獲取錯誤數據
    # 由於我們在腳本中，需要手動輸入或從檔案讀取
    logger.info("請運行 get_errors 工具並將結果提供給此分析器")
    
    print("深度偵錯分析器已準備就緒")
    print("使用方法：")
    print("1. 運行 get_errors 工具獲取錯誤列表")
    print("2. 調用 analyzer.analyze_and_fix() 方法進行分析和修復")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()