#!/usr/bin/env python3
"""
AIVA 檔案整理工具
整理散落在根目錄的腳本和報告到相應的分類目錄
並為所有報告添加建立/修正時間戳
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import re

class AIVAFileOrganizer:
    """AIVA 檔案整理器"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 定義檔案分類規則
        self.script_patterns = {
            'ai_analysis': [
                'ai_*.py', 'analyze_ai_*.py', '*ai_manager*.py', 
                'intelligent_*.py', 'ai_component_*.py', 'ai_system_*.py'
            ],
            'testing': [
                'test_*.py', '*_test.py', 'comprehensive_*test*.py',
                'validation*.py', 'autonomous_test*.py', 'pentest*.py'
            ],
            'analysis': [
                'analyze_*.py', 'scanner_*.py', 'check_*.py',
                'verify_*.py', 'validate_*.py'
            ],
            'utilities': [
                'aiva_launcher.py', 'aiva_package_validator.py',
                'health_check.py', 'launch_*.py', 'fix_*.py',
                'migrate_*.py', 'apply_*.py'
            ],
            'cross_language': [
                '*cross_language*.py', 'schema_codegen*.py',
                'comprehensive_schema_test.py'
            ]
        }
        
        self.report_patterns = {
            'architecture': [
                'ARCHITECTURE_*.md', '*ARCHITECTURE*.md', 'SYSTEM_*.md'
            ],
            'ai_analysis': [
                'AI_*.md', '*AI_*.md', 'AIVA_AI_*.md'
            ],
            'schema': [
                'SCHEMA_*.md', '*SCHEMA*.md', 'CROSS_LANGUAGE_*.md'
            ],
            'testing': [
                'TEST_*.md', '*TEST*.md', 'TESTING_*.md', 'COMPLETE_*.md'
            ],
            'documentation': [
                'DOCUMENTATION_*.md', 'DOC_*.md', '*GUIDE*.md'
            ],
            'project_status': [
                'PROJECT_*.md', 'TODO*.md', 'DEPLOYMENT*.md',
                'PROGRESS_*.md', 'STATUS_*.md'
            ]
        }
        
    def get_root_files(self) -> List[Path]:
        """獲取根目錄中的所有檔案"""
        files = []
        for item in self.root_path.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                files.append(item)
        return files
    
    def classify_file(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """分類檔案"""
        filename = file_path.name
        
        # 分類腳本檔案
        if filename.endswith('.py'):
            for category, patterns in self.script_patterns.items():
                for pattern in patterns:
                    if self._match_pattern(filename, pattern):
                        return 'scripts', category
            return 'scripts', 'misc'
        
        # 分類報告檔案
        elif filename.endswith('.md'):
            for category, patterns in self.report_patterns.items():
                for pattern in patterns:
                    if self._match_pattern(filename, pattern):
                        return 'reports', category
            return 'reports', 'misc'
        
        # 分類 JSON 報告檔案
        elif filename.endswith('.json') and ('report' in filename.lower() or 'analysis' in filename.lower()):
            return 'reports', 'data'
        
        # 分類日誌檔案
        elif filename.endswith('.log'):
            return 'logs', 'misc'
        
        return None, None
    
    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """檢查檔案名是否符合模式"""
        # 簡單的模式匹配，支援 * 通配符
        import fnmatch
        return fnmatch.fnmatch(filename.lower(), pattern.lower())
    
    def add_timestamp_to_report(self, file_path: Path) -> bool:
        """為報告檔案添加時間戳"""
        try:
            if not file_path.suffix.lower() in ['.md', '.json']:
                return False
                
            # 讀取檔案內容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 檢查是否已經有時間戳
            if 'Created:' in content or 'Last Modified:' in content:
                # 更新 Last Modified
                if 'Last Modified:' in content:
                    content = re.sub(
                        r'Last Modified: .*',
                        f'Last Modified: {self.current_date}',
                        content
                    )
                else:
                    # 在 Created 後添加 Last Modified
                    content = re.sub(
                        r'(Created: .*\n)',
                        f'\\1Last Modified: {self.current_date}\n',
                        content
                    )
            else:
                # 添加時間戳標頭
                timestamp_header = f"""---
Created: {self.current_date}
Last Modified: {self.current_date}
Document Type: {"Report" if file_path.suffix == ".md" else "Data"}
---

"""
                content = timestamp_header + content
            
            # 寫回檔案
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return True
            
        except Exception as e:
            print(f"❌ 無法為 {file_path.name} 添加時間戳: {e}")
            return False
    
    def create_target_directory(self, base_dir: str, category: str) -> Path:
        """創建目標目錄"""
        target_dir = self.root_path / base_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir
    
    def move_file(self, source: Path, target_dir: Path) -> bool:
        """移動檔案"""
        try:
            target_path = target_dir / source.name
            
            # 如果目標檔案已存在，添加時間戳後綴
            if target_path.exists():
                stem = target_path.stem
                suffix = target_path.suffix
                target_path = target_dir / f"{stem}_{self.timestamp}{suffix}"
            
            shutil.move(str(source), str(target_path))
            print(f"✅ 移動: {source.name} → {target_dir.relative_to(self.root_path)}/")
            return True
            
        except Exception as e:
            print(f"❌ 移動失敗 {source.name}: {e}")
            return False
    
    def organize_files(self) -> Dict[str, Any]:
        """整理檔案"""
        print(f"🚀 開始整理 AIVA 檔案 ({self.current_date})")
        print("=" * 60)
        
        files = self.get_root_files()
        stats = {
            'total_files': len(files),
            'processed': 0,
            'moved': 0,
            'timestamped': 0,
            'skipped': 0,
            'by_category': {}
        }
        
        for file_path in files:
            stats['processed'] += 1
            
            # 跳過特殊檔案
            if file_path.name in ['README.md', 'pyproject.toml', 'requirements.txt', 
                                 'docker-compose.yml', '.env.example']:
                stats['skipped'] += 1
                continue
            
            # 分類檔案
            base_dir, category = self.classify_file(file_path)
            
            if base_dir is None or category is None:
                stats['skipped'] += 1
                continue
            
            # 為報告添加時間戳
            if base_dir == 'reports':
                if self.add_timestamp_to_report(file_path):
                    stats['timestamped'] += 1
            
            # 創建目標目錄並移動檔案
            target_dir = self.create_target_directory(base_dir, category)
            if self.move_file(file_path, target_dir):
                stats['moved'] += 1
                
                # 統計分類
                key = f"{base_dir}/{category}"
                if key not in stats['by_category']:
                    stats['by_category'][key] = 0
                stats['by_category'][key] += 1
        
        return stats
    
    def generate_organization_report(self, stats: Dict) -> str:
        """生成整理報告"""
        report = f"""# AIVA 檔案整理報告

---
Created: {self.current_date}
Last Modified: {self.current_date}
Document Type: Report
---

## 整理摘要

- **整理日期**: {self.current_date}
- **處理檔案**: {stats['processed']} 個
- **成功移動**: {stats['moved']} 個
- **添加時間戳**: {stats['timestamped']} 個
- **跳過檔案**: {stats['skipped']} 個

## 檔案分類統計

"""
        
        for category, count in sorted(stats['by_category'].items()):
            report += f"- **{category}**: {count} 個檔案\n"
        
        report += f"""

## 目錄結構

整理後的目錄結構：

```
AIVA-git/
├── scripts/
│   ├── ai_analysis/        # AI 分析相關腳本
│   ├── testing/            # 測試相關腳本
│   ├── analysis/           # 分析工具腳本
│   ├── utilities/          # 工具腳本
│   ├── cross_language/     # 跨語言相關腳本
│   └── misc/               # 其他腳本
├── reports/
│   ├── architecture/       # 架構相關報告
│   ├── ai_analysis/        # AI 分析報告
│   ├── schema/             # Schema 相關報告
│   ├── testing/            # 測試報告
│   ├── documentation/      # 文檔相關報告
│   ├── project_status/     # 專案狀態報告
│   ├── data/               # JSON 數據報告
│   └── misc/               # 其他報告
└── logs/                   # 日誌檔案
```

## 時間戳標準

所有報告檔案現在都包含以下時間戳資訊：

```markdown
---
Created: YYYY-MM-DD
Last Modified: YYYY-MM-DD
Document Type: Report/Data
---
```

## 維護建議

1. **新檔案命名**: 建議按照分類命名規則命名新檔案
2. **定期整理**: 建議每月運行一次整理工具
3. **時間戳維護**: 修改報告時請更新 Last Modified 時間
4. **分類維護**: 如有新的檔案類型，請更新分類規則

---

**整理工具**: `organize_aiva_files.py`  
**下次建議整理時間**: {datetime.now().replace(month=datetime.now().month+1 if datetime.now().month < 12 else 1).strftime('%Y-%m-%d')}
"""
        
        return report
    
    def save_organization_report(self, stats: Dict) -> bool:
        """保存整理報告"""
        try:
            report_content = self.generate_organization_report(stats)
            report_path = self.root_path / 'reports' / 'project_status' / f'FILE_ORGANIZATION_REPORT_{self.timestamp}.md'
            
            # 確保目錄存在
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n📄 整理報告已保存: {report_path.relative_to(self.root_path)}")
            return True
            
        except Exception as e:
            print(f"❌ 保存整理報告失敗: {e}")
            return False

def main():
    """主函數"""
    organizer = AIVAFileOrganizer()
    
    print("🔍 正在分析根目錄檔案...")
    stats = organizer.organize_files()
    
    print("\n" + "=" * 60)
    print("📊 整理完成統計")
    print("=" * 60)
    print(f"處理檔案: {stats['processed']} 個")
    print(f"成功移動: {stats['moved']} 個")
    print(f"添加時間戳: {stats['timestamped']} 個")
    print(f"跳過檔案: {stats['skipped']} 個")
    
    # 保存整理報告
    organizer.save_organization_report(stats)
    
    print(f"\n✨ AIVA 檔案整理完成！")
    print(f"📁 所有檔案已按分類整理到相應目錄")
    print(f"🕒 所有報告已添加時間戳資訊")

if __name__ == "__main__":
    main()