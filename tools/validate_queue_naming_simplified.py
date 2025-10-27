#!/usr/bin/env python3
"""
AIVA 隊列命名驗證工具 - 簡化版
專注檢查實際的 AIVA workers，排除第三方庫
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# 標準隊列命名規範
STANDARD_RESULT_QUEUE = "findings.new"
STANDARD_TASK_QUEUE_PATTERNS = [
    "tasks.function.*",  # 功能檢測任務
    "tasks.scan.*",      # 掃描任務  
    "tasks.analysis.*"   # 分析任務
]

# 要檢查的目錄和文件模式
WORKER_DIRECTORIES = [
    "services/features/*/cmd/worker/",  # Go workers
    "services/features/*/src/",         # Rust workers  
    "services/scan/*/src/",             # 掃描服務
    "services/scan/*/*.py",             # Python 掃描服務
    "services/features/*/*.py"          # Python workers
]

def find_queue_definitions(root_path: Path) -> List[Tuple[str, str, str, str]]:
    """
    查找隊列定義
    返回: [(文件路徑, 隊列名, 語言, 上下文)]
    """
    findings = []
    
    # 要檢查的文件模式
    patterns = {
        "go": [
            r'resultQueue\s*:?=\s*["\']([^"\']+)["\']',
            r'taskQueue\s*:?=\s*["\']([^"\']+)["\']',
            r'RESULT_QUEUE\s*:?=\s*["\']([^"\']+)["\']',
            r'TASK_QUEUE\s*:?=\s*["\']([^"\']+)["\']',
        ],
        "rust": [
            r'const\s+RESULT_QUEUE:\s*&str\s*=\s*["\']([^"\']+)["\']',
            r'const\s+TASK_QUEUE:\s*&str\s*=\s*["\']([^"\']+)["\']',
            r'const\s+FINDING_QUEUE:\s*&str\s*=\s*["\']([^"\']+)["\']',
        ],
        "typescript": [
            r'const\s+RESULT_QUEUE\s*=\s*[^\'\"]*["\']([^"\']+)["\'];?',
            r'const\s+TASK_QUEUE\s*=\s*[^\'\"]*["\']([^"\']+)["\'];?',
        ],
        "python": [
            r'result_queue\s*=\s*["\']([^"\']+)["\']',
            r'task_queue\s*=\s*["\']([^"\']+)["\']',
            r'RESULT_QUEUE\s*=\s*["\']([^"\']+)["\']',
            r'TASK_QUEUE\s*=\s*["\']([^"\']+)["\']',
        ]
    }
    
    # 掃描指定的工作目錄
    for worker_dir in WORKER_DIRECTORIES:
        for file_path in root_path.glob(worker_dir):
            if file_path.is_file():
                _scan_file(file_path, patterns, findings)
            elif file_path.is_dir():
                # 掃描目錄中的所有相關文件
                for ext in ["*.go", "*.rs", "*.py", "*.ts", "*.js"]:
                    for file in file_path.rglob(ext):
                        # 排除第三方庫
                        if "node_modules" in str(file) or "__pycache__" in str(file):
                            continue
                        _scan_file(file, patterns, findings)
    
    return findings

def _scan_file(file_path: Path, patterns: Dict[str, List[str]], findings: List):
    """掃描單個文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        # 確定文件語言
        lang = _detect_language(file_path)
        if lang not in patterns:
            return
            
        # 搜索隊列定義
        for pattern in patterns[lang]:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                queue_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                
                # 獲取上下文
                context_line = lines[line_num - 1] if line_num <= len(lines) else ""
                
                findings.append((
                    str(file_path.relative_to(file_path.parents[5])),  # 相對路徑
                    queue_name,
                    lang,
                    context_line.strip()
                ))
                
    except Exception as e:
        print(f"警告: 無法讀取文件 {file_path}: {e}")

def _detect_language(file_path: Path) -> str:
    """根據文件擴展名檢測語言"""
    ext = file_path.suffix.lower()
    if ext == '.go':
        return 'go'
    elif ext == '.rs':
        return 'rust'
    elif ext in ['.ts', '.js']:
        return 'typescript'
    elif ext == '.py':
        return 'python'
    return 'unknown'

def validate_queues(findings: List[Tuple[str, str, str, str]]) -> Dict:
    """驗證隊列名稱是否符合標準"""
    report = {
        'total': len(findings),
        'compliant': [],
        'non_compliant': [],
        'statistics': {}
    }
    
    for file_path, queue_name, lang, context in findings:
        if _is_result_queue(context):
            # 檢查結果隊列
            if queue_name == STANDARD_RESULT_QUEUE:
                report['compliant'].append((file_path, queue_name, lang, 'result', context))
            else:
                report['non_compliant'].append((file_path, queue_name, lang, 'result', context, 
                                              f"應為 '{STANDARD_RESULT_QUEUE}'"))
        elif _is_task_queue(context):
            # 檢查任務隊列 (目前任務隊列命名較靈活，只記錄)
            report['compliant'].append((file_path, queue_name, lang, 'task', context))
    
    # 統計
    report['statistics'] = {
        'total_queues': len(findings),
        'compliant_count': len(report['compliant']),
        'non_compliant_count': len(report['non_compliant']),
        'compliance_rate': len(report['compliant']) / len(findings) * 100 if findings else 100
    }
    
    return report

def _is_result_queue(context: str) -> bool:
    """判斷是否為結果隊列定義"""
    result_indicators = ['result', 'finding', 'output']
    return any(indicator in context.lower() for indicator in result_indicators)

def _is_task_queue(context: str) -> bool:
    """判斷是否為任務隊列定義"""
    task_indicators = ['task', 'input', 'job']
    return any(indicator in context.lower() for indicator in task_indicators)

def generate_report(report: Dict) -> str:
    """生成可讀的報告"""
    lines = [
        "# AIVA 隊列命名一致性檢查報告",
        "=" * 50,
        "",
        f"📊 **統計信息**",
        f"- 總隊列定義數: {report['statistics']['total_queues']}",
        f"- 符合標準: {report['statistics']['compliant_count']}",
        f"- 需要修復: {report['statistics']['non_compliant_count']}",
        f"- 合規率: {report['statistics']['compliance_rate']:.1f}%",
        ""
    ]
    
    if report['non_compliant']:
        lines.extend([
            "❌ **需要修復的隊列**",
            ""
        ])
        for file_path, queue_name, lang, queue_type, context, suggestion in report['non_compliant']:
            lines.extend([
                f"- 📁 `{file_path}`",
                f"  - 當前隊列名: `{queue_name}` ❌",
                f"  - 語言: {lang}",
                f"  - 類型: {queue_type}",
                f"  - 建議: {suggestion}",
                f"  - 上下文: `{context}`",
                ""
            ])
    
    if report['compliant']:
        lines.extend([
            "✅ **符合標準的隊列**",
            ""
        ])
        for file_path, queue_name, lang, queue_type, context in report['compliant']:
            lines.extend([
                f"- 📁 `{file_path}`",
                f"  - 隊列名: `{queue_name}` ✅",
                f"  - 語言: {lang}",
                f"  - 類型: {queue_type}",
                ""
            ])
    
    return "\n".join(lines)

def main():
    """主函數"""
    root_path = Path(__file__).parent.parent
    print(f"🔍 掃描 AIVA workers 中的隊列定義...")
    print(f"📂 根目錄: {root_path}")
    
    # 查找隊列定義
    findings = find_queue_definitions(root_path)
    print(f"📋 找到 {len(findings)} 個隊列定義")
    
    # 驗證
    report = validate_queues(findings)
    
    # 生成報告
    report_content = generate_report(report)
    
    # 保存報告
    report_path = root_path / "reports" / "queue_naming_simplified.md"
    os.makedirs(report_path.parent, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 輸出結果
    print("\n" + "=" * 50)
    print(f"📄 報告已保存至: {report_path}")
    print(f"📊 合規率: {report['statistics']['compliance_rate']:.1f}%")
    
    if report['non_compliant']:
        print(f"❌ 發現 {len(report['non_compliant'])} 個需要修復的問題")
        return 1
    else:
        print("✅ 所有隊列命名都符合標準！")
        return 0

if __name__ == "__main__":
    exit(main())