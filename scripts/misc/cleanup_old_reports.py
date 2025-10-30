#!/usr/bin/env python3
"""
AIVA 報告清理工具
分析並清理超過一週的老舊且已完成的報告
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class AIVAReportCleaner:
    """AIVA 報告清理器"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.reports_dir = self.root_path / "reports"
        self.current_date = datetime.now()
        self.one_week_ago = self.current_date - timedelta(days=7)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 定義已完成的報告關鍵字
        self.completion_keywords = [
            'COMPLETE', 'COMPLETION', 'FINISHED', 'FINAL', 'DONE',
            '完成', '結束', '完結', 'SUCCESS', 'RESOLVED'
        ]
        
        # 定義重要報告關鍵字（不應該刪除）
        self.important_keywords = [
            'GUIDE', 'DOCUMENTATION', 'README', 'MANUAL', 'HANDBOOK',
            'REFERENCE', 'API', 'SCHEMA', 'ARCHITECTURE', 'ANALYSIS'
        ]
        
        # 保留的報告類型
        self.keep_categories = [
            'documentation',  # 文檔類報告
            'architecture',   # 架構報告
            'schema'         # Schema 報告
        ]
        
    def extract_date_from_content(self, file_path: Path) -> Optional[datetime]:
        """從檔案內容中提取日期資訊"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(500)  # 只讀前500字符
            
            # 尋找 Created: 或 Last Modified: 日期
            date_pattern = r'(?:Created|Last Modified):\s*(\d{4}-\d{2}-\d{2})'
            matches = re.findall(date_pattern, content)
            
            if matches:
                # 使用最新的日期
                latest_date_str = max(matches)
                return datetime.strptime(latest_date_str, '%Y-%m-%d')
            
            # 如果沒有找到標準格式，嘗試其他格式
            date_patterns = [
                r'(\d{4}年\d{1,2}月\d{1,2}日)',
                r'(\d{4}/\d{1,2}/\d{1,2})',
                r'(\d{4}-\d{1,2}-\d{1,2})',
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    date_str = matches[0]
                    # 嘗試解析不同格式
                    for fmt in ['%Y年%m月%d日', '%Y/%m/%d', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
            
        except Exception as e:
            print(f"⚠️  無法讀取 {file_path.name}: {e}")
        
        return None
    
    def extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """從檔案名中提取日期"""
        # 尋找檔案名中的時間戳格式
        patterns = [
            r'(\d{8})_\d{6}',  # YYYYMMDD_HHMMSS
            r'(\d{8})',        # YYYYMMDD
            r'(\d{4}\d{2}\d{2})',  # YYYYMMDD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                try:
                    return datetime.strptime(date_str, '%Y%m%d')
                except ValueError:
                    continue
        
        return None
    
    def get_file_modification_date(self, file_path: Path) -> datetime:
        """獲取檔案系統修改時間"""
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def is_completed_report(self, file_path: Path) -> bool:
        """判斷是否為已完成的報告"""
        filename = file_path.name.upper()
        
        # 檢查檔案名是否包含完成關鍵字
        for keyword in self.completion_keywords:
            if keyword in filename:
                return True
        
        # 檢查檔案內容（前200字符）
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(200).upper()
            
            for keyword in self.completion_keywords:
                if keyword in content:
                    return True
        except:
            pass
        
        return False
    
    def is_important_report(self, file_path: Path) -> bool:
        """判斷是否為重要報告（不應刪除）"""
        filename = file_path.name.upper()
        
        # 檢查檔案名是否包含重要關鍵字
        for keyword in self.important_keywords:
            if keyword in filename:
                return True
        
        # 檢查是否在保留的分類中
        parent_dir = file_path.parent.name
        if parent_dir in self.keep_categories:
            return True
        
        return False
    
    def analyze_reports(self) -> Dict[str, List[Dict]]:
        """分析所有報告檔案"""
        analysis = {
            'old_completed': [],     # 超過一週的已完成報告
            'old_important': [],     # 超過一週的重要報告
            'recent': [],            # 最近的報告
            'unknown_date': [],      # 無法確定日期的報告
            'cleanup_candidates': [] # 清理候選
        }
        
        print("🔍 分析報告檔案...")
        
        for report_file in self.reports_dir.rglob('*.md'):
            if report_file.is_file():
                # 獲取日期資訊
                content_date = self.extract_date_from_content(report_file)
                filename_date = self.extract_date_from_filename(report_file.name)
                file_mod_date = self.get_file_modification_date(report_file)
                
                # 選擇最可靠的日期
                report_date = content_date or filename_date or file_mod_date
                
                file_info = {
                    'path': report_file,
                    'relative_path': report_file.relative_to(self.root_path),
                    'name': report_file.name,
                    'size': report_file.stat().st_size,
                    'date': report_date,
                    'date_source': 'content' if content_date else ('filename' if filename_date else 'filesystem'),
                    'is_completed': self.is_completed_report(report_file),
                    'is_important': self.is_important_report(report_file),
                    'category': report_file.parent.name
                }
                
                # 分類
                if report_date < self.one_week_ago:
                    if file_info['is_important']:
                        analysis['old_important'].append(file_info)
                    elif file_info['is_completed']:
                        analysis['old_completed'].append(file_info)
                        analysis['cleanup_candidates'].append(file_info)
                    else:
                        analysis['old_completed'].append(file_info)
                elif report_date is None:
                    analysis['unknown_date'].append(file_info)
                else:
                    analysis['recent'].append(file_info)
        
        return analysis
    
    def generate_cleanup_plan(self, analysis: Dict) -> str:
        """生成清理計劃"""
        plan = f"""# AIVA 報告清理計劃

---
Created: {self.current_date.strftime('%Y-%m-%d')}
Last Modified: {self.current_date.strftime('%Y-%m-%d')}
Document Type: Plan
---

## 清理摘要

- **分析日期**: {self.current_date.strftime('%Y-%m-%d')}
- **清理標準**: 超過 7 天且已完成的報告
- **總報告數**: {sum(len(v) for v in analysis.values())}

## 分類統計

### 📊 報告分類
- **最近報告** (< 7天): {len(analysis['recent'])} 個
- **舊重要報告** (> 7天, 重要): {len(analysis['old_important'])} 個
- **舊已完成報告** (> 7天, 已完成): {len(analysis['old_completed'])} 個
- **日期不明報告**: {len(analysis['unknown_date'])} 個
- **清理候選**: {len(analysis['cleanup_candidates'])} 個

### 🗑️ 建議清理的報告

"""
        
        if analysis['cleanup_candidates']:
            total_size = 0
            for report in sorted(analysis['cleanup_candidates'], key=lambda x: x['date']):
                size_kb = report['size'] / 1024
                total_size += size_kb
                plan += f"- **{report['name']}**\n"
                plan += f"  - 路徑: `{report['relative_path']}`\n"
                plan += f"  - 日期: {report['date'].strftime('%Y-%m-%d')} ({report['date_source']})\n"
                plan += f"  - 大小: {size_kb:.1f} KB\n"
                plan += f"  - 分類: {report['category']}\n"
                plan += "\n"
            
            plan += f"**預計釋放空間**: {total_size:.1f} KB\n\n"
        else:
            plan += "✅ 沒有發現需要清理的報告\n\n"
        
        plan += """### 🔒 保留的重要報告

"""
        
        if analysis['old_important']:
            for report in sorted(analysis['old_important'], key=lambda x: x['date']):
                plan += f"- **{report['name']}** (保留原因: 重要文檔)\n"
                plan += f"  - 路徑: `{report['relative_path']}`\n"
                plan += f"  - 日期: {report['date'].strftime('%Y-%m-%d')}\n\n"
        
        plan += """### ⚠️ 日期不明的報告

"""
        
        if analysis['unknown_date']:
            for report in analysis['unknown_date']:
                plan += f"- **{report['name']}**\n"
                plan += f"  - 路徑: `{report['relative_path']}`\n"
                plan += f"  - 建議: 手動檢查並添加時間戳\n\n"
        
        plan += """## 清理操作

### 自動清理
```bash
# 執行清理計劃
python scripts/misc/cleanup_old_reports.py --execute

# 僅預覽清理
python scripts/misc/cleanup_old_reports.py --preview
```

### 手動清理
建議的清理步驟：
1. 備份重要報告
2. 移動到 `_archive` 目錄而非直接刪除
3. 30天後再永久刪除

## 安全措施

- ✅ 自動備份到 `_archive/cleanup_{timestamp}/`
- ✅ 保留所有重要文檔和指南
- ✅ 不刪除最近 7 天內的報告
- ✅ 提供復原機制

---

**清理工具**: `cleanup_old_reports.py`  
**備份位置**: `_archive/cleanup_{timestamp}/`
"""
        
        return plan
    
    def execute_cleanup(self, analysis: Dict, dry_run: bool = True) -> Dict:
        """執行清理操作"""
        cleanup_stats = {
            'planned': len(analysis['cleanup_candidates']),
            'moved': 0,
            'failed': 0,
            'total_size': 0
        }
        
        if not analysis['cleanup_candidates']:
            print("✅ 沒有需要清理的報告")
            return cleanup_stats
        
        # 創建備份目錄
        backup_dir = self.root_path / "_archive" / f"cleanup_{self.timestamp}"
        
        if not dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 創建備份目錄: {backup_dir.relative_to(self.root_path)}")
        
        print(f"\n{'🔍 預覽' if dry_run else '🗑️ 執行'} 清理操作:")
        print("-" * 50)
        
        for report in analysis['cleanup_candidates']:
            try:
                cleanup_stats['total_size'] += report['size']
                
                if dry_run:
                    print(f"📄 將移動: {report['relative_path']}")
                else:
                    # 創建備份目錄結構
                    backup_file_dir = backup_dir / report['path'].parent.relative_to(self.reports_dir)
                    backup_file_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 移動檔案到備份目錄
                    backup_file_path = backup_file_dir / report['path'].name
                    report['path'].rename(backup_file_path)
                    
                    print(f"✅ 已移動: {report['relative_path']} → _archive/")
                    cleanup_stats['moved'] += 1
                    
            except Exception as e:
                print(f"❌ 處理失敗 {report['name']}: {e}")
                cleanup_stats['failed'] += 1
        
        return cleanup_stats
    
    def save_cleanup_plan(self, plan: str) -> bool:
        """保存清理計劃"""
        try:
            plan_file = self.reports_dir / "project_status" / f"REPORT_CLEANUP_PLAN_{self.timestamp}.md"
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(plan_file, 'w', encoding='utf-8') as f:
                f.write(plan)
            
            print(f"📋 清理計劃已保存: {plan_file.relative_to(self.root_path)}")
            return True
            
        except Exception as e:
            print(f"❌ 保存清理計劃失敗: {e}")
            return False

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIVA 報告清理工具')
    parser.add_argument('--execute', action='store_true', help='執行清理操作')
    parser.add_argument('--preview', action='store_true', help='僅預覽清理計劃')
    args = parser.parse_args()
    
    cleaner = AIVAReportCleaner()
    
    print("🚀 開始分析 AIVA 報告檔案")
    print("=" * 50)
    
    # 分析報告
    analysis = cleaner.analyze_reports()
    
    # 生成清理計劃
    plan = cleaner.generate_cleanup_plan(analysis)
    cleaner.save_cleanup_plan(plan)
    
    # 顯示統計
    print(f"\n📊 分析完成統計:")
    print(f"   最近報告: {len(analysis['recent'])} 個")
    print(f"   舊重要報告: {len(analysis['old_important'])} 個")
    print(f"   舊已完成報告: {len(analysis['old_completed'])} 個")
    print(f"   清理候選: {len(analysis['cleanup_candidates'])} 個")
    
    # 執行或預覽清理
    if args.execute:
        print(f"\n🗑️ 執行清理操作...")
        stats = cleaner.execute_cleanup(analysis, dry_run=False)
        print(f"\n✨ 清理完成:")
        print(f"   計劃清理: {stats['planned']} 個")
        print(f"   成功移動: {stats['moved']} 個")
        print(f"   失敗: {stats['failed']} 個")
        print(f"   釋放空間: {stats['total_size']/1024:.1f} KB")
    elif args.preview or len(analysis['cleanup_candidates']) > 0:
        print(f"\n🔍 預覽清理操作...")
        cleaner.execute_cleanup(analysis, dry_run=True)
        print(f"\n💡 執行清理: python scripts/misc/cleanup_old_reports.py --execute")

if __name__ == "__main__":
    main()