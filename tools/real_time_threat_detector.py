"""
⚡ 實時威脅檢測器 - Real-time Threat Detector
即時監控和檢測安全威脅

功能:
1. 實時日誌監控
2. 即時攻擊檢測和警報
3. 自動防禦響應建議
4. 威脅儀表板
"""

import re
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class RealTimeThreatDetector:
    """實時威脅檢測器"""
    
    # 攻擊特徵庫 (基於 OWASP Juice Shop 真實數據)
    THREAT_SIGNATURES = {
        'Critical': {
            'SQL Injection': [
                r"UNION\s+SELECT",
                r"'\s+OR\s+'1'\s*=\s*'1",
                r"';?\s*DROP\s+TABLE",
                r"SQLITE_ERROR.*UNION"
            ],
            'Remote Code Execution': [
                r"eval\s*\(",
                r"exec\s*\(",
                r"system\s*\(",
                r"shell_exec"
            ]
        },
        'High': {
            'XSS Attack': [
                r"<script[^>]*>",
                r"javascript:",
                r"onerror\s*=",
                r"near \"XSS\""
            ],
            'Path Traversal': [
                r"\.\./\.\./",
                r"%2e%2e%2f",
                r"/etc/passwd",
                r"Unexpected path.*\.php"
            ],
            'Authentication Bypass': [
                r"No Authorization header",
                r"Invalid token",
                r"UnauthorizedError"
            ]
        },
        'Medium': {
            'File Upload Vulnerability': [
                r"upload.*\.php",
                r"upload.*\.exe",
                r"uploadTypeChallenge"
            ],
            'Parameter Pollution': [
                r"undefined.*value",
                r"null.*parameter"
            ]
        },
        'Low': {
            'Information Disclosure': [
                r"Error:.*at\s+/",
                r"Stack trace",
                r"errorHandlingChallenge"
            ]
        }
    }
    
    def __init__(self, alert_threshold: int = 5):
        """初始化檢測器"""
        self.alert_threshold = alert_threshold  # 觸發警報的最小次數
        self.threat_counter = {severity: {} for severity in self.THREAT_SIGNATURES.keys()}
        self.recent_threats = deque(maxlen=100)  # 最近 100 個威脅
        self.alerts = []
        self.start_time = datetime.now()
        
    def analyze_log_line(self, line: str) -> Optional[Dict]:
        """分析單行日誌"""
        timestamp = datetime.now().isoformat()
        
        for severity, threat_types in self.THREAT_SIGNATURES.items():
            for threat_type, patterns in threat_types.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # 檢測到威脅
                        threat = {
                            'timestamp': timestamp,
                            'severity': severity,
                            'type': threat_type,
                            'pattern': pattern,
                            'content': line.strip()[:200]
                        }
                        
                        # 更新計數器
                        if threat_type not in self.threat_counter[severity]:
                            self.threat_counter[severity][threat_type] = 0
                        self.threat_counter[severity][threat_type] += 1
                        
                        # 添加到最近威脅
                        self.recent_threats.append(threat)
                        
                        # 檢查是否需要警報
                        if self.threat_counter[severity][threat_type] >= self.alert_threshold:
                            self._trigger_alert(threat)
                        
                        return threat
        
        return None
    
    def _trigger_alert(self, threat: Dict):
        """觸發安全警報"""
        alert = {
            'alert_id': len(self.alerts) + 1,
            'timestamp': datetime.now().isoformat(),
            'severity': threat['severity'],
            'threat_type': threat['type'],
            'count': self.threat_counter[threat['severity']][threat['type']],
            'recommendation': self._get_recommendation(threat['type'])
        }
        
        self.alerts.append(alert)
        
        # 記錄警報
        logger.warning(f"🚨 ALERT #{alert['alert_id']} | {alert['severity']} | {alert['threat_type']} (x{alert['count']})")
    
    def _get_recommendation(self, threat_type: str) -> str:
        """獲取防禦建議"""
        recommendations = {
            'SQL Injection': "立即啟用 WAF SQL 注入規則，審查所有數據庫查詢",
            'XSS Attack': "實施 CSP 策略，檢查所有輸出編碼",
            'Authentication Bypass': "強制重新驗證所有會話，啟用 MFA",
            'Path Traversal': "限制文件訪問路徑，啟用白名單",
            'Remote Code Execution': "隔離受影響系統，立即修補漏洞",
            'File Upload Vulnerability': "暫停文件上傳功能，掃描已上傳文件",
            'Parameter Pollution': "驗證所有輸入參數，實施嚴格類型檢查",
            'Information Disclosure': "禁用詳細錯誤消息，審查日誌配置"
        }
        
        return recommendations.get(threat_type, "請審查安全配置並更新防護規則")
    
    def monitor_file(self, log_file: str, interval: float = 1.0, duration: int = 60):
        """監控日誌文件"""
        log_path = Path(log_file)
        
        if not log_path.exists():
            logger.error(f"✗ 日誌文件不存在: {log_file}")
            return
        
        print("=" * 70)
        print("⚡ 實時威脅檢測器啟動")
        print("=" * 70)
        print(f"監控文件: {log_file}")
        print(f"警報閾值: {self.alert_threshold}")
        print(f"監控時長: {duration} 秒")
        print(f"掃描間隔: {interval} 秒")
        print("=" * 70)
        print()
        
        start = time.time()
        lines_processed = 0
        
        # 讀取現有內容
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 分析所有行
        for line in lines:
            threat = self.analyze_log_line(line)
            if threat:
                print(f"🔍 [{threat['severity']}] {threat['type']}")
            
            lines_processed += 1
            
            # 模擬實時處理
            if lines_processed % 100 == 0:
                time.sleep(interval / 10)
                self._print_status(lines_processed)
        
        # 最終報告
        print("\n" + "=" * 70)
        print("📊 監控報告")
        print("=" * 70)
        self._print_summary()
    
    def _print_status(self, lines: int):
        """打印狀態"""
        total_threats = sum(sum(counts.values()) for counts in self.threat_counter.values())
        print(f"  處理: {lines:,} 行 | 威脅: {total_threats} | 警報: {len(self.alerts)}")
    
    def _print_summary(self):
        """打印摘要"""
        print(f"\n處理時間: {(datetime.now() - self.start_time).total_seconds():.2f} 秒")
        print(f"總威脅數: {sum(sum(counts.values()) for counts in self.threat_counter.values())}")
        print(f"總警報數: {len(self.alerts)}")
        
        print("\n威脅統計:")
        for severity in ['Critical', 'High', 'Medium', 'Low']:
            if self.threat_counter[severity]:
                print(f"\n  【{severity}】")
                for threat_type, count in self.threat_counter[severity].items():
                    print(f"    • {threat_type}: {count}")
        
        if self.alerts:
            print(f"\n🚨 警報詳情:")
            for alert in self.alerts[-5:]:  # 顯示最後 5 個
                print(f"\n  Alert #{alert['alert_id']} - {alert['severity']}")
                print(f"    類型: {alert['threat_type']}")
                print(f"    次數: {alert['count']}")
                print(f"    建議: {alert['recommendation']}")
    
    def export_report(self, output_file: str = "_out/threat_detection_report.json"):
        """匯出檢測報告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'start_time': self.start_time.isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'statistics': {
                'total_threats': sum(sum(counts.values()) for counts in self.threat_counter.values()),
                'total_alerts': len(self.alerts),
                'threats_by_severity': {
                    severity: sum(counts.values())
                    for severity, counts in self.threat_counter.items()
                }
            },
            'threat_breakdown': self.threat_counter,
            'alerts': self.alerts,
            'recent_threats': list(self.recent_threats)[-20:]  # 最近 20 個
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ 報告已匯出: {output_file}")
        return str(output_path)


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description='⚡ 實時威脅檢測器')
    parser.add_argument('log_file', nargs='?',
                       default='AI_OPTIMIZATION_REQUIREMENTS.txt',
                       help='監控的日誌文件')
    parser.add_argument('--threshold', '-t', type=int, default=5,
                       help='警報閾值 (預設: 5)')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                       help='掃描間隔秒數 (預設: 1.0)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                       help='監控時長秒數 (預設: 60)')
    parser.add_argument('--output', '-o',
                       default='_out/threat_detection_report.json',
                       help='報告輸出文件')
    
    args = parser.parse_args()
    
    # 初始化檢測器
    detector = RealTimeThreatDetector(alert_threshold=args.threshold)
    
    # 監控文件
    detector.monitor_file(
        args.log_file,
        interval=args.interval,
        duration=args.duration
    )
    
    # 匯出報告
    print(f"\n💾 匯出報告...\n")
    report_file = detector.export_report(args.output)
    print(f"✓ 報告: {report_file}")
    
    print("\n" + "=" * 70)
    print("✓ 監控完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
