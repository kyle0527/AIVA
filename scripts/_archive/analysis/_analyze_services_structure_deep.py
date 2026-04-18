#!/usr/bin/env python3


"""深度分析 services 目錄結構與實際內容的一致性"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set
import re
project_root = Path(__file__).resolve().parent.parent.parent.parent

class ServicesAnalyzer:
    def __init__(self, services_dir: str):
        self.services_dir = Path(services_dir)
        self.results = {
            'modules': {},
            'empty_dirs': [],
            'readme_issues': [],
            'file_classification_issues': [],
            'link_hierarchy': {}
        }
        
    def analyze_module(self, module_path: Path) -> Dict:
        """分析單個模組"""
        module_name = module_path.name
        
        # 統計文件類型
        py_files = list(module_path.rglob("*.py"))
        py_files = [f for f in py_files if '__pycache__' not in str(f) and 'node_modules' not in str(f)]
        
        go_files = list(module_path.rglob("*.go"))
        rs_files = list(module_path.rglob("*.rs"))
        ts_files = list(module_path.rglob("*.ts"))
        ts_files = [f for f in ts_files if 'node_modules' not in str(f)]
        
        md_files = list(module_path.rglob("*.md"))
        md_files = [f for f in md_files if 'node_modules' not in str(f)]
        
        readme_path = module_path / "README.md"
        has_readme = readme_path.exists()
        
        # 分析 README 內容
        readme_analysis = {}
        if has_readme:
            readme_analysis = self.analyze_readme(readme_path, module_path)
        
        return {
            'name': module_name,
            'path': str(module_path.relative_to(self.services_dir)),
            'code_files': {
                'python': len(py_files),
                'go': len(go_files),
                'rust': len(rs_files),
                'typescript': len(ts_files)
            },
            'doc_files': len(md_files),
            'has_readme': has_readme,
            'readme_analysis': readme_analysis,
            'subdirs': [d.name for d in module_path.iterdir() if d.is_dir() and d.name not in ['__pycache__', 'node_modules', '.pytest_cache']]
        }
        
    def analyze_readme(self, readme_path: Path, module_path: Path) -> Dict:
        """分析 README 內容"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return {'error': 'Cannot read file'}
        
        # 統計基本信息
        lines = content.count('\n')
        size = len(content)
        
        # 提取標題
        h1_matches = re.findall(r'^#\s+(.+)$', content, re.MULTILINE)
        h2_matches = re.findall(r'^##\s+(.+)$', content, re.MULTILINE)
        
        # 檢查是否有目錄
        has_toc = '## 📑 目錄' in content or '## 目錄' in content
        
        # 提取所有連結
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        # 檢查連結的有效性
        broken_links = []
        for link_text, link_url in links:
            if link_url.startswith('http'):
                continue
            if link_url.startswith('#'):
                continue
            
            # 相對路徑連結
            link_path = (module_path / link_url).resolve()
            if not link_path.exists():
                broken_links.append((link_text, link_url))
        
        # 檢查是否包含大量技術細節（應該獨立成文件）
        has_code_blocks = content.count('```') > 10
        has_long_sections = any(len(section) > 5000 for section in content.split('##'))
        
        # 檢查是否有子模組連結
        subdir_links = []
        for link_text, link_url in links:
            if '/README.md' in link_url or link_url.endswith('/'):
                subdir_links.append((link_text, link_url))
        
        return {
            'size': size,
            'lines': lines,
            'h1_count': len(h1_matches),
            'h2_count': len(h2_matches),
            'has_toc': has_toc,
            'total_links': len(links),
            'broken_links': broken_links,
            'subdir_links': subdir_links,
            'needs_split': has_code_blocks or has_long_sections,
            'title': h1_matches[0] if h1_matches else 'No title'
        }
    
    def find_empty_dirs(self) -> List[str]:
        """找出空資料夾"""
        empty_dirs = []
        for root, dirs, files in os.walk(self.services_dir):
            # 排除特殊目錄
            dirs[:] = [d for d in dirs if d not in ['__pycache__', 'node_modules', '.pytest_cache', '.git']]
            
            if not dirs and not files:
                empty_dirs.append(root)
            elif not files:
                # 檢查子目錄是否都是空的
                has_content = False
                for subdir in dirs:
                    subdir_path = Path(root) / subdir
                    if any(subdir_path.rglob('*')):
                        has_content = True
                        break
                if not has_content:
                    empty_dirs.append(root)
        
        return empty_dirs
    
    def analyze_all(self):
        """執行完整分析"""
        print("="*80)
        print("🔍 Services 目錄深度分析")
        print("="*80)
        
        # 分析主要模組
        main_modules = ['aiva_common', 'core', 'features', 'integration', 'scan']
        
        for module_name in main_modules:
            module_path = self.services_dir / module_name
            if module_path.exists():
                print(f"\n📦 分析模組: {module_name}")
                analysis = self.analyze_module(module_path)
                self.results['modules'][module_name] = analysis
                
                # 顯示摘要
                print(f"  代碼文件: Python={analysis['code_files']['python']}, " +
                      f"Go={analysis['code_files']['go']}, " +
                      f"Rust={analysis['code_files']['rust']}, " +
                      f"TypeScript={analysis['code_files']['typescript']}")
                print(f"  文檔文件: {analysis['doc_files']} 個")
                print(f"  README: {'✅' if analysis['has_readme'] else '❌'}")
                
                if analysis['has_readme']:
                    readme = analysis['readme_analysis']
                    print(f"    - 大小: {readme.get('size', 0):,} bytes")
                    print(f"    - 連結: {readme.get('total_links', 0)} 個")
                    if readme.get('broken_links'):
                        print(f"    - ⚠️ 損壞連結: {len(readme['broken_links'])} 個")
                    if readme.get('needs_split'):
                        print(f"    - ⚠️ 建議拆分（內容過多）")
        
        # 查找空資料夾
        print(f"\n📂 檢查空資料夾...")
        empty_dirs = self.find_empty_dirs()
        self.results['empty_dirs'] = [str(Path(d).relative_to(self.services_dir)) for d in empty_dirs]
        
        if empty_dirs:
            print(f"  找到 {len(empty_dirs)} 個空資料夾:")
            for d in empty_dirs[:10]:
                rel_path = Path(d).relative_to(self.services_dir)
                print(f"    - {rel_path}")
            if len(empty_dirs) > 10:
                print(f"    ... 還有 {len(empty_dirs) - 10} 個")
        else:
            print("  ✅ 沒有空資料夾")
        
        return self.results
    
    def generate_report(self) -> str:
        """生成分析報告"""
        report = []
        report.append("# 🔍 Services 目錄深度分析報告\n\n")
        report.append("---\n")
        report.append("**分析時間**: 2025年11月27日\n\n")
        
        report.append("## 📑 目錄\n\n")
        report.append("- [模組分析](#模組分析)\n")
        report.append("- [空資料夾](#空資料夾)\n")
        report.append("- [README 問題](#readme-問題)\n")
        report.append("- [建議行動](#建議行動)\n\n")
        
        report.append("---\n\n")
        report.append("## 📦 模組分析\n\n")
        
        for module_name, analysis in self.results['modules'].items():
            report.append(f"### {module_name}\n\n")
            report.append(f"**路徑**: `services/{analysis['path']}`\n\n")
            
            # 代碼統計
            code = analysis['code_files']
            total_code = sum(code.values())
            if total_code > 0:
                report.append("**代碼文件**:\n")
                if code['python'] > 0:
                    report.append(f"- Python: {code['python']} 個\n")
                if code['go'] > 0:
                    report.append(f"- Go: {code['go']} 個\n")
                if code['rust'] > 0:
                    report.append(f"- Rust: {code['rust']} 個\n")
                if code['typescript'] > 0:
                    report.append(f"- TypeScript: {code['typescript']} 個\n")
                report.append("\n")
            else:
                report.append("**代碼文件**: ❌ 無代碼文件\n\n")
            
            # README 分析
            if analysis['has_readme']:
                readme = analysis['readme_analysis']
                report.append("**README 分析**:\n")
                report.append(f"- 標題: *{readme.get('title', 'No title')}*\n")
                report.append(f"- 大小: {readme.get('size', 0):,} bytes ({readme.get('lines', 0)} 行)\n")
                report.append(f"- 目錄: {'✅' if readme.get('has_toc') else '❌'}\n")
                report.append(f"- 連結: {readme.get('total_links', 0)} 個\n")
                
                if readme.get('broken_links'):
                    report.append(f"\n⚠️ **損壞的連結** ({len(readme['broken_links'])} 個):\n")
                    for link_text, link_url in readme['broken_links'][:5]:
                        report.append(f"  - `[{link_text}]({link_url})`\n")
                    if len(readme['broken_links']) > 5:
                        report.append(f"  - ... 還有 {len(readme['broken_links']) - 5} 個\n")
                
                if readme.get('needs_split'):
                    report.append(f"\n⚠️ **建議拆分**: README 內容過多，建議提取為獨立文件\n")
                
                report.append("\n")
            else:
                report.append("**README**: ❌ 缺少 README.md\n\n")
            
            # 子目錄
            if analysis['subdirs']:
                report.append(f"**子目錄** ({len(analysis['subdirs'])} 個):\n")
                for subdir in analysis['subdirs'][:10]:
                    report.append(f"- `{subdir}/`\n")
                if len(analysis['subdirs']) > 10:
                    report.append(f"- ... 還有 {len(analysis['subdirs']) - 10} 個\n")
                report.append("\n")
        
        report.append("## 📂 空資料夾\n\n")
        if self.results['empty_dirs']:
            report.append(f"找到 {len(self.results['empty_dirs'])} 個空資料夾:\n\n")
            for d in self.results['empty_dirs']:
                report.append(f"- `{d}`\n")
            report.append("\n")
        else:
            report.append("✅ 沒有空資料夾\n\n")
        
        report.append("## 📋 建議行動\n\n")
        report.append("### 1. 刪除空資料夾\n\n")
        if self.results['empty_dirs']:
            report.append("```powershell\n")
            for d in self.results['empty_dirs']:
                report.append(f'Remove-Item -Path "{project_root}/services\\{d}" -Force -Recurse\n')
            report.append("```\n\n")
        else:
            report.append("無需操作\n\n")
        
        report.append("### 2. 修正損壞的連結\n\n")
        has_broken_links = False
        for module_name, analysis in self.results['modules'].items():
            if analysis['has_readme'] and analysis['readme_analysis'].get('broken_links'):
                has_broken_links = True
                report.append(f"**{module_name}**: {len(analysis['readme_analysis']['broken_links'])} 個損壞連結\n")
        
        if not has_broken_links:
            report.append("✅ 沒有損壞的連結\n")
        report.append("\n")
        
        report.append("### 3. 拆分過大的 README\n\n")
        needs_split = []
        for module_name, analysis in self.results['modules'].items():
            if analysis['has_readme'] and analysis['readme_analysis'].get('needs_split'):
                needs_split.append(module_name)
        
        if needs_split:
            report.append("以下模組的 README 建議拆分:\n\n")
            for module in needs_split:
                report.append(f"- `{module}/README.md`\n")
            report.append("\n")
        else:
            report.append("✅ 所有 README 大小適中\n\n")
        
        report.append("---\n\n")
        report.append("*報告生成時間: 2025年11月27日*\n")
        
        return ''.join(report)

def main():
    services_dir = str(project_root / "services")
    
    analyzer = ServicesAnalyzer(services_dir)
    results = analyzer.analyze_all()
    
    # 生成報告
    report = analyzer.generate_report()
    report_file = (project_root / "SERVICES_STRUCTURE_ANALYSIS_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 報告已保存: {report_file}")
    
    # 保存 JSON
    json_file = (project_root / "_services_structure_analysis.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 數據已保存: {json_file}")
    print("\n" + "="*80)
    print("✨ 分析完成！")
    print("="*80)

if __name__ == "__main__":
    main()
