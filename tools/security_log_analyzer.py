"""
🔍 安全日誌分析器 - Security Log Analyzer
基於 OWASP Juice Shop 真實攻擊數據的智能分析工具

功能:
1. 自動識別攻擊類型 (SQL Injection, XSS, Auth Bypass, etc.)
2. 統計攻擊成功率和頻率
3. 生成安全報告和建議
4. 提取攻擊特徵用於 AI 訓練
"""

import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class SecurityLogAnalyzer:
    """安全日誌分析器"""
    
    # 攻擊模式定義
    ATTACK_PATTERNS = {
        'SQL Injection': [
            r'SQLITE_ERROR',
            r'UNION',
            r'incomplete input',
            r'syntax error.*(?:UNION|SELECT|WHERE)',
            r'WHERE parameter.*has invalid.*value'
        ],
        'XSS Attack': [
            r'near "XSS"',
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick='
        ],
        'Authentication Bypass': [
            r'UnauthorizedError',
            r'No Authorization header',
            r'Invalid token',
            r'no header in signature'
        ],
        'Path Traversal': [
            r'Unexpected path',
            r'\.\./\.\.',
            r'%2e%2e',
            r'/etc/passwd',
            r'/api/.*\.php'
        ],
        'File Upload Attack': [
            r'uploadTypeChallenge',
            r'upload.*\.exe',
            r'upload.*\.php',
            r'Content-Type.*image'
        ],
        'Error-Based Attack': [
            r'errorHandlingChallenge',
            r'Error:.*at\s+/',
            r'Stack trace'
        ],
        'Parameter Pollution': [
            r'invalid "undefined" value',
            r'captchaId.*undefined',
            r'parameter.*null'
        ],
        'Blocked Activity': [
            r'Blocked illegal activity',
            r'Forbidden',
            r'Access denied'
        ]
    }
    
    # 成功攻擊標記
    SUCCESS_INDICATORS = [
        r'Solved \d+-star',
        r'Challenge solved',
        r'Cheat score',
        r'info: Solved'
    ]
    
    def __init__(self, log_file: str):
        """初始化分析器"""
        self.log_file = Path(log_file)
        self.log_lines = []
        self.attack_stats = defaultdict(lambda: {'count': 0, 'timestamps': [], 'samples': []})
        self.success_attacks = []
        self.timeline = []
        
    def load_log(self) -> bool:
        """載入日誌文件"""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                self.log_lines = f.readlines()
            logger.info(f"✓ 載入日誌: {len(self.log_lines)} 行")
            return True
        except Exception as e:
            logger.error(f"✗ 載入失敗: {e}")
            return False
    
    def parse_timestamp(self, line: str) -> str:
        """提取時間戳"""
        match = re.match(r'^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', line)
        return match.group(1) if match else ""
    
    def analyze_attacks(self) -> Dict:
        """分析所有攻擊類型"""
        logger.info("🔍 開始分析攻擊模式...")
        
        for line_num, line in enumerate(self.log_lines, 1):
            timestamp = self.parse_timestamp(line)
            
            # 檢查每種攻擊類型
            for attack_type, patterns in self.ATTACK_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        self.attack_stats[attack_type]['count'] += 1
                        if timestamp:
                            self.attack_stats[attack_type]['timestamps'].append(timestamp)
                        if len(self.attack_stats[attack_type]['samples']) < 3:
                            # 保存前 3 個樣本
                            self.attack_stats[attack_type]['samples'].append(line.strip())
                        break
            
            # 檢查成功攻擊
            for pattern in self.SUCCESS_INDICATORS:
                if re.search(pattern, line, re.IGNORECASE):
                    self.success_attacks.append({
                        'timestamp': timestamp,
                        'line_num': line_num,
                        'content': line.strip()
                    })
        
        logger.info(f"✓ 發現 {len(self.attack_stats)} 種攻擊類型")
        logger.info(f"✓ 檢測到 {len(self.success_attacks)} 次成功攻擊")
        
        return dict(self.attack_stats)
    
    def generate_report(self, output_file: str = None) -> str:
        """生成安全分析報告"""
        if not output_file:
            output_file = Path("_out") / "security_analysis_report.md"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 統計總數
        total_attacks = sum(stats['count'] for stats in self.attack_stats.values())
        
        report_lines = [
            f"# 🔐 安全日誌分析報告",
            f"",
            f"**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**日誌文件**: `{self.log_file.name}`  ",
            f"**總行數**: {len(self.log_lines):,}  ",
            f"**檢測攻擊總數**: {total_attacks:,}  ",
            f"**成功攻擊次數**: {len(self.success_attacks)}  ",
            f"",
            f"---",
            f"",
            f"## 📊 攻擊類型統計",
            f""
        ]
        
        # 按攻擊次數排序
        sorted_attacks = sorted(self.attack_stats.items(), 
                               key=lambda x: x[1]['count'], 
                               reverse=True)
        
        for attack_type, stats in sorted_attacks:
            count = stats['count']
            percentage = (count / total_attacks * 100) if total_attacks > 0 else 0
            
            report_lines.extend([
                f"### {attack_type}",
                f"",
                f"- **次數**: {count:,} ({percentage:.1f}%)",
                f"- **首次出現**: {stats['timestamps'][0] if stats['timestamps'] else 'N/A'}",
                f"- **最後出現**: {stats['timestamps'][-1] if stats['timestamps'] else 'N/A'}",
                f"",
                f"**攻擊樣本**:",
                f"```"
            ])
            
            for i, sample in enumerate(stats['samples'][:3], 1):
                report_lines.append(f"{i}. {sample[:120]}...")
            
            report_lines.extend([
                f"```",
                f""
            ])
        
        # 成功攻擊詳情
        report_lines.extend([
            f"---",
            f"",
            f"## ✅ 成功攻擊記錄",
            f""
        ])
        
        for i, success in enumerate(self.success_attacks, 1):
            report_lines.extend([
                f"### 成功 #{i}",
                f"",
                f"- **時間**: {success['timestamp']}",
                f"- **行號**: {success['line_num']}",
                f"- **內容**: `{success['content'][:100]}...`",
                f""
            ])
        
        # 安全建議
        report_lines.extend([
            f"---",
            f"",
            f"## 💡 安全建議",
            f"",
            f"### 高優先級",
            f""
        ])
        
        if 'SQL Injection' in self.attack_stats:
            report_lines.append(f"1. **SQL Injection 防護**: 檢測到 {self.attack_stats['SQL Injection']['count']} 次 SQL 注入嘗試")
            report_lines.append(f"   - 使用參數化查詢")
            report_lines.append(f"   - 實施輸入驗證和清理")
            report_lines.append(f"   - 啟用 WAF 規則")
            report_lines.append(f"")
        
        if 'XSS Attack' in self.attack_stats:
            report_lines.append(f"2. **XSS 防護**: 檢測到 {self.attack_stats['XSS Attack']['count']} 次 XSS 攻擊")
            report_lines.append(f"   - 輸出編碼所有用戶數據")
            report_lines.append(f"   - 實施 CSP (Content Security Policy)")
            report_lines.append(f"   - 使用 HTTPOnly cookies")
            report_lines.append(f"")
        
        if 'Authentication Bypass' in self.attack_stats:
            report_lines.append(f"3. **身份驗證加強**: {self.attack_stats['Authentication Bypass']['count']} 次繞過嘗試")
            report_lines.append(f"   - 強制所有 API 端點驗證")
            report_lines.append(f"   - 實施速率限制")
            report_lines.append(f"   - 使用多因素驗證 (MFA)")
            report_lines.append(f"")
        
        # AI 訓練建議
        report_lines.extend([
            f"### 🤖 AI 訓練優化建議",
            f"",
            f"1. **攻擊模式識別訓練**: 基於 {total_attacks} 個真實攻擊樣本",
            f"2. **異常檢測模型**: 訓練識別 {len(self.attack_stats)} 種攻擊類型",
            f"3. **成功率預測**: 使用 {len(self.success_attacks)} 個成功案例優化",
            f"4. **時序分析**: 利用時間戳數據進行攻擊鏈重建",
            f""
        ])
        
        # 寫入報告
        report_content = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"✓ 報告已生成: {output_file}")
        return str(output_file)
    
    def export_training_data(self, output_file: str = None) -> str:
        """匯出 AI 訓練數據"""
        if not output_file:
            output_file = Path("_out") / "attack_training_data.json"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_data = {
            'metadata': {
                'source': str(self.log_file),
                'generated_at': datetime.now().isoformat(),
                'total_attacks': sum(s['count'] for s in self.attack_stats.values()),
                'attack_types': len(self.attack_stats),
                'success_rate': len(self.success_attacks) / len(self.log_lines) * 100
            },
            'attack_patterns': {},
            'success_cases': [],
            'feature_vectors': []
        }
        
        # 整理攻擊模式數據
        for attack_type, stats in self.attack_stats.items():
            training_data['attack_patterns'][attack_type] = {
                'count': stats['count'],
                'samples': stats['samples'],
                'frequency': stats['count'] / len(self.log_lines)
            }
        
        # 成功案例
        training_data['success_cases'] = [
            {
                'timestamp': s['timestamp'],
                'content': s['content']
            }
            for s in self.success_attacks
        ]
        
        # 特徵向量 (用於機器學習)
        for attack_type in self.attack_stats.keys():
            training_data['feature_vectors'].append({
                'label': attack_type,
                'count': self.attack_stats[attack_type]['count'],
                'normalized_frequency': self.attack_stats[attack_type]['count'] / len(self.log_lines)
            })
        
        # 寫入 JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 訓練數據已匯出: {output_file}")
        return str(output_file)


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🔍 安全日誌分析器')
    parser.add_argument('log_file', nargs='?', 
                       default='AI_OPTIMIZATION_REQUIREMENTS.txt',
                       help='日誌文件路徑')
    parser.add_argument('--report', '-r', 
                       help='報告輸出路徑 (預設: _out/security_analysis_report.md)')
    parser.add_argument('--export', '-e', 
                       help='訓練數據輸出路徑 (預設: _out/attack_training_data.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='顯示詳細信息')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("🔍 安全日誌分析器 - Security Log Analyzer")
    print("=" * 60)
    print()
    
    # 初始化分析器
    analyzer = SecurityLogAnalyzer(args.log_file)
    
    # 載入日誌
    if not analyzer.load_log():
        print("❌ 無法載入日誌文件")
        return
    
    # 分析攻擊
    print("\n📊 分析中...\n")
    analyzer.analyze_attacks()
    
    # 生成報告
    print("\n📝 生成報告...\n")
    report_file = analyzer.generate_report(args.report)
    print(f"✓ 報告: {report_file}")
    
    # 匯出訓練數據
    print("\n💾 匯出訓練數據...\n")
    training_file = analyzer.export_training_data(args.export)
    print(f"✓ 訓練數據: {training_file}")
    
    # 顯示摘要
    print("\n" + "=" * 60)
    print("📊 分析摘要")
    print("=" * 60)
    
    total = sum(s['count'] for s in analyzer.attack_stats.values())
    print(f"總攻擊次數: {total:,}")
    print(f"攻擊類型數: {len(analyzer.attack_stats)}")
    print(f"成功攻擊: {len(analyzer.success_attacks)}")
    print()
    
    print("前 5 種最常見攻擊:")
    sorted_attacks = sorted(analyzer.attack_stats.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)[:5]
    
    for i, (attack_type, stats) in enumerate(sorted_attacks, 1):
        percentage = (stats['count'] / total * 100) if total > 0 else 0
        print(f"  {i}. {attack_type}: {stats['count']:,} ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ 分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
