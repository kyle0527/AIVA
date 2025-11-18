#!/usr/bin/env python3
"""
AIVA 通用修復指南實施工具
基於修復指南進行安全的批量修復操作
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime

class SafeBatchProcessor:
    """安全的批量修復處理器 - 基於通用指南"""
    
    def __init__(self):
        self.fixes_applied = {
            'empty_f_strings': 0,
            'unused_variables': 0,
            'import_improvements': 0
        }
    
    def fix_empty_f_strings(self, content: str) -> Tuple[str, int]:
        """批量修復空F-string - 低風險操作"""
        fixes = 0
        
        # 安全的F-string修復模式
        patterns = [
            (r'print\(f"([^{]*?)"\)', r'print("\1")'),  # print(f"text") -> print("text")
            (r'print\(f\'([^{]*?)\'\)', r"print('\1')"), # print(f'text') -> print('text')
            (r'logger\.info\(f"([^{]*?)"\)', r'logger.info("\1")'),  # logger.info(f"text")
            (r'logger\.error\(f"([^{]*?)"\)', r'logger.error("\1")'), # logger.error(f"text")
            (r'logger\.warning\(f"([^{]*?)"\)', r'logger.warning("\1")'), # logger.warning
        ]
        
        for pattern, replacement in patterns:
            matches = list(re.finditer(pattern, content))
            for match in matches:
                # 檢查是否真的沒有變量插值
                if '{' not in match.group(1):
                    content = content.replace(match.group(0), re.sub(pattern, replacement, match.group(0)))
                    fixes += 1
        
        return content, fixes
    
    def fix_unused_variables(self, content: str) -> Tuple[str, int]:
        """批量修復未使用變數 - 低風險操作"""
        fixes = 0
        
        # 安全地移除明顯未使用的變數
        patterns = [
            # 目標參數未使用的情況
            (r'(\s+)(target_code|target_url|target_host|target_info) = kwargs\.get\([^)]+\)\s*\n(?!\s*[^#\n]*\2)', 
             r'\1_ = kwargs.get("target", "")  # 參數未使用，標記為忽略\n'),
             
            # 明顯未使用的topology變數
            (r'(\s+)topology = network_data\.get\([^)]+\)\s*\n(?!\s*[^#\n]*topology)', 
             r'\1# topology = network_data.get("topology", {})  # 暫時移除未使用變數\n'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                fixes += 1
        
        return content, fixes
    
    def validate_changes(self, original: str, modified: str) -> bool:
        """驗證修改的安全性"""
        try:
            # 簡單的語法檢查
            compile(modified, '<string>', 'exec')
            
            # 檢查關鍵結構是否保持不變
            original_lines = len(original.split('\n'))
            modified_lines = len(modified.split('\n'))
            
            # 行數變化不應該太大 (允許5%的變化)
            if abs(original_lines - modified_lines) / original_lines > 0.05:
                return False
                
            return True
        except SyntaxError:
            return False
    
    def apply_safe_batch_fixes(self, file_path: str) -> Dict[str, int]:
        """應用安全的批量修復"""
        path = Path(file_path)
        
        if not path.exists():
            return {'error': 'File not found'}
        
        # 讀取原始內容
        with open(path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 創建備份
        backup_path = path.with_suffix('.py.batch_backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # 應用修復
        content = original_content
        
        # F-string修復
        content, f_fixes = self.fix_empty_f_strings(content)
        self.fixes_applied['empty_f_strings'] += f_fixes
        
        # 未使用變數修復
        content, var_fixes = self.fix_unused_variables(content)
        self.fixes_applied['unused_variables'] += var_fixes
        
        # 驗證修改
        if self.validate_changes(original_content, content):
            # 寫入修復後的內容
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'f_string_fixes': f_fixes,
                'unused_var_fixes': var_fixes,
                'total_fixes': f_fixes + var_fixes,
                'status': 'success'
            }
        else:
            # 恢復原始內容
            with open(path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            return {
                'error': 'Validation failed, changes reverted',
                'status': 'failed'
            }

def main():
    """主函數：按照通用指南進行批量修復"""
    print("🔄 階段三：批量處理標準化問題")
    print("=" * 50)
    
    processor = SafeBatchProcessor()
    
    # 目標文件
    target_files = [
        'c:/D/fold7/AIVA-git/aiva_capability_orchestrator.py',
        'c:/D/fold7/AIVA-git/services/core/aiva_core/ai_engine/real_neural_core.py'
    ]
    
    total_results = []
    
    for file_path in target_files:
        print(f"\n🔧 批量修復檔案: {Path(file_path).name}")
        result = processor.apply_safe_batch_fixes(file_path)
        
        if 'error' in result:
            print(f"   ❌ 修復失敗: {result['error']}")
        else:
            print(f"   ✅ 修復完成:")
            print(f"      - F-string修復: {result['f_string_fixes']}")
            print(f"      - 未使用變數修復: {result['unused_var_fixes']}")
            print(f"      - 總修復數: {result['total_fixes']}")
        
        total_results.append({
            'file': file_path,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    # 生成批量修復報告
    report = {
        'batch_repair_session': {
            'timestamp': datetime.now().isoformat(),
            'total_files_processed': len(target_files),
            'successful_repairs': len([r for r in total_results if r['result'].get('status') == 'success']),
            'total_fixes_applied': processor.fixes_applied,
            'file_results': total_results
        }
    }
    
    with open('c:/D/fold7/AIVA-git/batch_repair_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 批量修復完成!")
    print(f"   - 處理檔案: {len(target_files)}")
    print(f"   - 成功修復: {len([r for r in total_results if r['result'].get('status') == 'success'])}")
    print(f"   - F-string修復: {processor.fixes_applied['empty_f_strings']}")
    print(f"   - 變數修復: {processor.fixes_applied['unused_variables']}")
    print(f"   - 報告已生成: batch_repair_report.json")

if __name__ == "__main__":
    main()