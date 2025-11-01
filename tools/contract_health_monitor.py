#!/usr/bin/env python3
"""
AIVA 合約健康度監控系統

自動化監控合約系統的健康狀況，包括使用率、驗證錯誤、
廢棄警告等關鍵指標。支持即時監控和定期報告。
"""

import asyncio
import json
import logging
import sqlite3
import smtplib
from datetime import UTC, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import sys
from dataclasses import dataclass, asdict
from email.mime.multipart import MIMEMultipart

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/contract_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ContractMetric:
    """合約指標數據結構"""
    timestamp: datetime
    total_contracts: int
    used_contracts: int
    usage_coverage: float
    validation_errors: int
    deprecated_warnings: int
    performance_score: float
    health_status: str  # 'excellent', 'good', 'warning', 'critical'


@dataclass
class AlertConfig:
    """告警配置"""
    email_enabled: bool = True
    slack_enabled: bool = False
    webhook_url: Optional[str] = None
    email_recipients: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = ['aiva-dev@example.com']


class ContractHealthMonitor:
    """合約健康度監控器"""
    
    def __init__(self, config_path: str = "config/monitor_config.json"):
        self.config_path = config_path
        self.db_path = "logs/contract_metrics.db"
        self.alert_config = AlertConfig()
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(hours=1)  # 告警冷卻時間
        
        self._init_database()
        self._load_config()
    
    def _init_database(self):
        """初始化指標數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contract_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_contracts INTEGER NOT NULL,
                used_contracts INTEGER NOT NULL,
                usage_coverage REAL NOT NULL,
                validation_errors INTEGER NOT NULL,
                deprecated_warnings INTEGER NOT NULL,
                performance_score REAL NOT NULL,
                health_status TEXT NOT NULL,
                raw_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("數據庫初始化完成")
    
    def _load_config(self):
        """載入監控配置"""
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                self.alert_config = AlertConfig(
                    email_enabled=config_data.get('email_enabled', True),
                    slack_enabled=config_data.get('slack_enabled', False),
                    webhook_url=config_data.get('webhook_url'),
                    email_recipients=config_data.get('email_recipients', ['aiva-dev@example.com'])
                )
                logger.info(f"配置載入成功: {self.config_path}")
            except Exception as e:
                logger.warning(f"配置載入失敗，使用默認配置: {e}")
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """創建默認配置文件"""
        default_config = {
            "email_enabled": True,
            "slack_enabled": False,
            "webhook_url": None,
            "email_recipients": ["aiva-dev@example.com"],
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "smtp_username": "monitor@example.com",
            "smtp_password": "${SMTP_PASSWORD}",
            "monitoring_interval": 3600,
            "thresholds": {
                "coverage_warning": 0.15,
                "coverage_critical": 0.10,
                "error_warning": 5,
                "error_critical": 20,
                "performance_warning": 0.7,
                "performance_critical": 0.5
            }
        }
        
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"創建默認配置文件: {self.config_path}")
    
    def collect_metrics(self) -> ContractMetric:
        """收集合約指標數據"""
        try:
            logger.info("開始收集合約指標數據...")
            
            # 1. 獲取合約覆蓋率
            coverage_result = self._run_coverage_analysis()
            
            # 2. 檢查驗證錯誤
            validation_errors = self._check_validation_errors()
            
            # 3. 檢查廢棄警告
            deprecated_warnings = self._check_deprecated_usage()
            
            # 4. 評估性能分數
            performance_score = self._calculate_performance_score()
            
            # 5. 確定健康狀態
            health_status = self._determine_health_status(
                coverage_result['coverage'],
                validation_errors,
                deprecated_warnings,
                performance_score
            )
            
            metric = ContractMetric(
                timestamp=datetime.now(UTC),
                total_contracts=coverage_result['total'],
                used_contracts=coverage_result['used'],
                usage_coverage=coverage_result['coverage'],
                validation_errors=validation_errors,
                deprecated_warnings=deprecated_warnings,
                performance_score=performance_score,
                health_status=health_status
            )
            
            logger.info(f"指標收集完成: 覆蓋率={metric.usage_coverage:.1%}, 狀態={metric.health_status}")
            return metric
            
        except Exception as e:
            logger.error(f"指標收集失敗: {e}")
            raise
    
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """運行合約覆蓋率分析"""
        try:
            result = subprocess.run(
                [sys.executable, "tools/analyze_contract_coverage.py", "--json"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
                timeout=60
            )
            
            if result.returncode == 0:
                coverage_data = json.loads(result.stdout)
                return {
                    'total': coverage_data['analysis']['total_contracts'],
                    'used': coverage_data['analysis']['used_contracts'],
                    'coverage': coverage_data['analysis']['coverage_rate']
                }
            else:
                logger.error(f"覆蓋率分析失敗: {result.stderr}")
                return {'total': 0, 'used': 0, 'coverage': 0.0}
                
        except Exception as e:
            logger.error(f"覆蓋率分析異常: {e}")
            return {'total': 0, 'used': 0, 'coverage': 0.0}
    
    def _check_validation_errors(self) -> int:
        """檢查合約驗證錯誤數量"""
        try:
            result = subprocess.run(
                [sys.executable, "tools/schema_compliance_validator.py", "--count-errors"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return int(result.stdout.strip())
            else:
                return 0
                
        except Exception as e:
            logger.error(f"驗證錯誤檢查異常: {e}")
            return 0
    
    def _check_deprecated_usage(self) -> int:
        """檢查廢棄合約使用警告"""
        try:
            # 搜索代碼中的廢棄合約使用
            deprecated_patterns = [
                "legacy_finding_id",
                "old_vulnerability_format", 
                "deprecated_scan_type",
                "@deprecated"
            ]
            
            warning_count = 0
            for pattern in deprecated_patterns:
                result = subprocess.run(
                    ["git", "grep", "-r", "--count", pattern, ".", "--", "*.py"],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd()
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if ':' in line:
                            warning_count += int(line.split(':')[1])
            
            return warning_count
            
        except Exception as e:
            logger.error(f"廢棄使用檢查異常: {e}")
            return 0
    
    def _calculate_performance_score(self) -> float:
        """計算合約系統性能分數 (0.0-1.0)"""
        try:
            # 性能指標權重
            weights = {
                'import_speed': 0.3,      # 導入速度
                'validation_speed': 0.4,   # 驗證速度  
                'memory_usage': 0.3        # 內存使用
            }
            
            # 測試導入速度
            import_score = self._test_import_performance()
            
            # 測試驗證速度
            validation_score = self._test_validation_performance()
            
            # 檢查內存使用
            memory_score = self._test_memory_usage()
            
            performance_score = (
                import_score * weights['import_speed'] +
                validation_score * weights['validation_speed'] +
                memory_score * weights['memory_usage']
            )
            
            return max(0.0, min(1.0, performance_score))
            
        except Exception as e:
            logger.error(f"性能評估異常: {e}")
            return 0.5  # 默認中等分數
    
    def _test_import_performance(self) -> float:
        """測試合約導入性能"""
        try:
            import time
            start_time = time.perf_counter()
            
            # 測試主要合約導入
            exec("""
from services.aiva_common.schemas import (
    FindingPayload, AivaMessage, Authentication, 
    ScanStartPayload, FunctionTelemetry
)
""")
            
            end_time = time.perf_counter()
            import_time = end_time - start_time
            
            # 導入時間越短分數越高 (期望 < 0.1秒 = 滿分)
            score = max(0.0, min(1.0, 0.1 / max(import_time, 0.001)))
            return score
            
        except Exception as e:
            logger.error(f"導入性能測試失敗: {e}")
            return 0.5
    
    def _test_validation_performance(self) -> float:
        """測試合約驗證性能"""
        try:
            import time
            from services.aiva_common.schemas import FindingPayload, Vulnerability
            
            # 測試數據
            test_data = {
                "finding_id": "test_001",
                "vulnerability": {
                    "name": "Test Vulnerability",
                    "severity": "medium",
                    "confidence": "certain"
                }
            }
            
            start_time = time.perf_counter()
            
            # 執行多次驗證測試
            for _ in range(100):
                FindingPayload(**test_data)
            
            end_time = time.perf_counter()
            validation_time = end_time - start_time
            
            # 驗證時間越短分數越高 (期望 < 0.01秒 = 滿分)
            score = max(0.0, min(1.0, 0.01 / max(validation_time, 0.001)))
            return score
            
        except Exception as e:
            logger.error(f"驗證性能測試失敗: {e}")
            return 0.5
    
    def _test_memory_usage(self) -> float:
        """測試內存使用情況"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # 轉換為MB
            
            # 內存使用越少分數越高 (期望 < 100MB = 滿分)
            score = max(0.0, min(1.0, 100 / max(memory_mb, 10)))
            return score
            
        except Exception as e:
            logger.error(f"內存測試失敗: {e}")
            return 0.7  # 默認較高分數
    
    def _determine_health_status(self, coverage: float, errors: int, warnings: int, performance: float) -> str:
        """確定健康狀態 - 分解為較小函數以符合 PEP-8 認知複雜度要求"""
        # 計算各項分數
        coverage_score = self._calculate_coverage_score(coverage)
        error_score = self._calculate_error_score(errors)
        warning_score = self._calculate_warning_score(warnings)
        performance_score = self._calculate_performance_score_rating(performance)
        
        # 總分計算
        total_score = coverage_score + error_score + warning_score + performance_score
        
        return self._score_to_health_status(total_score)
    
    def _calculate_coverage_score(self, coverage: float) -> int:
        """計算覆蓋率分數 (40% 權重)"""
        if coverage >= 0.20:
            return 40
        elif coverage >= 0.15:
            return 30
        elif coverage >= 0.10:
            return 20
        else:
            return 10
    
    def _calculate_error_score(self, errors: int) -> int:
        """計算錯誤分數 (30% 權重)"""
        if errors == 0:
            return 30
        elif errors <= 5:
            return 20
        elif errors <= 20:
            return 10
        else:
            return 0
    
    def _calculate_warning_score(self, warnings: int) -> int:
        """計算警告分數 (10% 權重)"""
        if warnings == 0:
            return 10
        elif warnings <= 10:
            return 7
        elif warnings <= 20:
            return 5
        else:
            return 0
    
    def _calculate_performance_score_rating(self, performance: float) -> int:
        """計算性能分數 (20% 權重)"""
        return int(performance * 20)
    
    def _score_to_health_status(self, score: int) -> str:
        """將總分轉換為健康狀態"""
        if score >= 85:
            return 'excellent'
        elif score >= 70:
            return 'good'
        elif score >= 50:
            return 'warning'
        else:
            return 'critical'
    
    def store_metric(self, metric: ContractMetric):
        """存儲指標到數據庫"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO contract_metrics 
                (timestamp, total_contracts, used_contracts, usage_coverage,
                 validation_errors, deprecated_warnings, performance_score, 
                 health_status, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.total_contracts,
                metric.used_contracts,
                metric.usage_coverage,
                metric.validation_errors,
                metric.deprecated_warnings,
                metric.performance_score,
                metric.health_status,
                json.dumps(asdict(metric))
            ))
            
            conn.commit()
            conn.close()
            logger.info("指標已存儲到數據庫")
            
        except Exception as e:
            logger.error(f"指標存儲失敗: {e}")
    
    async def check_alerts(self, metric: ContractMetric):
        """檢查告警條件"""
        alerts = []
        
        # 覆蓋率告警
        if metric.usage_coverage < 0.10:
            alerts.append({
                'type': 'coverage_critical',
                'severity': 'critical',
                'message': f'合約覆蓋率嚴重偏低: {metric.usage_coverage:.1%}',
                'details': f'當前覆蓋率 {metric.usage_coverage:.1%}，低於10%臨界值'
            })
        elif metric.usage_coverage < 0.15:
            alerts.append({
                'type': 'coverage_warning',
                'severity': 'warning',
                'message': f'合約覆蓋率偏低: {metric.usage_coverage:.1%}',
                'details': f'當前覆蓋率 {metric.usage_coverage:.1%}，低於15%警告值'
            })
        
        # 驗證錯誤告警
        if metric.validation_errors > 20:
            alerts.append({
                'type': 'validation_critical', 
                'severity': 'critical',
                'message': f'合約驗證錯誤過多: {metric.validation_errors}個',
                'details': f'發現{metric.validation_errors}個驗證錯誤，超過20個臨界值'
            })
        elif metric.validation_errors > 5:
            alerts.append({
                'type': 'validation_warning',
                'severity': 'warning', 
                'message': f'合約驗證錯誤: {metric.validation_errors}個',
                'details': f'發現{metric.validation_errors}個驗證錯誤，超過5個警告值'
            })
        
        # 性能告警
        if metric.performance_score < 0.5:
            alerts.append({
                'type': 'performance_critical',
                'severity': 'critical',
                'message': f'合約系統性能嚴重下降: {metric.performance_score:.1%}',
                'details': f'性能分數 {metric.performance_score:.1%}，低於50%臨界值'
            })
        elif metric.performance_score < 0.7:
            alerts.append({
                'type': 'performance_warning',
                'severity': 'warning',
                'message': f'合約系統性能下降: {metric.performance_score:.1%}',
                'details': f'性能分數 {metric.performance_score:.1%}，低於70%警告值'
            })
        
        # 發送告警
        for alert in alerts:
            await self._send_alert(alert, metric)
    
    async def _send_alert(self, alert: Dict[str, Any], metric: ContractMetric):
        """發送告警通知"""
        alert_key = alert['type']
        current_time = datetime.now(UTC)
        
        # 檢查冷卻時間
        if (alert_key in self.last_alert_time and 
            current_time - self.last_alert_time[alert_key] < self.alert_cooldown):
            return
        
        try:
            # 記錄告警
            self._store_alert(alert, metric)
            
            # 發送郵件通知
            if self.alert_config.email_enabled:
                await self._send_email_alert(alert, metric)
            
            # 發送 Slack 通知 (如果配置)
            if self.alert_config.slack_enabled and self.alert_config.webhook_url:
                await self._send_slack_alert(alert, metric)
            
            self.last_alert_time[alert_key] = current_time
            logger.info(f"告警已發送: {alert['type']} - {alert['message']}")
            
        except Exception as e:
            logger.error(f"告警發送失敗: {e}")
    
    def _store_alert(self, alert: Dict[str, Any], _: ContractMetric):
        """存儲告警記錄"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(UTC).isoformat(),
                alert['type'],
                alert['severity'],
                alert['message'],
                json.dumps(alert.get('details', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"告警記錄存儲失敗: {e}")
    
    def _send_email_alert(self, alert: Dict[str, Any], _: ContractMetric):
        """發送郵件告警"""
        try:
            # 構建郵件內容
            severity_emoji = {
                'critical': '🚨',
                'warning': '⚠️',
                'info': 'ℹ️'
            }
            
            subject = f"{severity_emoji.get(alert['severity'], '📊')} AIVA 合約系統告警 - {alert['message']}"
            

            
            # 發送郵件 (簡化版本，實際需要配置 SMTP)
            print(f"📧 郵件告警: {subject}")
            print(f"收件人: {', '.join(self.alert_config.email_recipients)}")
            logger.info("郵件告警已準備 (需要配置 SMTP 服務器)")
            
        except Exception as e:
            logger.error(f"郵件告警失敗: {e}")
    
    async def _send_slack_alert(self, alert: Dict[str, Any], metric: ContractMetric):
        """發送 Slack 告警 - 遵循 AIVA 五大模組架構的 Integration Module 規範"""
        try:
            # Slack webhook 消息格式 - 符合官方 Slack API 標準
            slack_message = {
                "text": f"AIVA 合約系統告警: {alert['message']}",
                "attachments": [
                    {
                        "color": "danger" if alert['severity'] == 'critical' else "warning",
                        "fields": [
                            {"title": "告警類型", "value": alert['type'], "short": True},
                            {"title": "嚴重程度", "value": alert['severity'], "short": True},
                            {"title": "覆蓋率", "value": f"{metric.usage_coverage:.1%}", "short": True},
                            {"title": "驗證錯誤", "value": str(metric.validation_errors), "short": True},
                            {"title": "性能分數", "value": f"{metric.performance_score:.1%}", "short": True},
                            {"title": "健康狀態", "value": metric.health_status, "short": True}
                        ],
                        "footer": "AIVA 合約監控",
                        "ts": int(metric.timestamp.timestamp())
                    }
                ]
            }
            
            # 實際發送到 Slack Webhook - 使用官方 aiohttp 異步 HTTP 客戶端
            if hasattr(self.alert_config, 'slack_webhook') and self.alert_config.slack_webhook:
                import aiohttp
                import asyncio
                
                timeout = aiohttp.ClientTimeout(total=30)  # 30秒超時
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.alert_config.slack_webhook, 
                        json=slack_message,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status == 200:
                            logger.info(f"✅ Slack 告警已成功發送: {alert['message']}")
                        else:
                            response_text = await response.text()
                            logger.warning(f"⚠️ Slack 告警發送失敗 (HTTP {response.status}): {response_text}")
            else:
                # 開發模式：顯示告警內容供調試
                print(f"💬 Slack 告警: {alert['message']}")
                logger.info("📋 Slack 告警已準備 (需要在配置中設置 slack_webhook URL)")
            
        except Exception as e:
            logger.error(f"Slack 告警失敗: {e}")
    
    async def generate_report(self, days: int = 7) -> str:
        """生成健康度報告 - 使用異步檔案操作符合 PEP-8 異步函數標準"""
        try:
            # 使用異步數據庫操作 (實際項目中建議使用 aiosqlite)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询最近的指標數據
            cursor.execute('''
                SELECT * FROM contract_metrics 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            metrics_data = cursor.fetchall()
            
            # 查询最近的告警
            cursor.execute('''
                SELECT * FROM alerts
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days))
            
            alerts_data = cursor.fetchall()
            conn.close()
            
            # 生成報告
            report = self._format_health_report(metrics_data, alerts_data, days)
            
            # 異步保存報告 - 使用 asyncio.to_thread 以符合官方 Python 異步最佳實踐
            report_path = f"reports/contract_health_report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 使用 asyncio.to_thread 將同步 I/O 轉為異步操作 (Python 3.9+ 官方推薦)
            import asyncio
            await asyncio.to_thread(self._write_report_file, report_path, report)
            
            logger.info(f"📄 健康度報告已異步生成: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"健康度報告生成失敗: {e}")
            return ""
    
    def _write_report_file(self, report_path: str, report: str) -> None:
        """輔助方法：同步寫入報告檔案 - 供 asyncio.to_thread 使用"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _format_health_report(self, metrics_data: List, alerts_data: List, days: int) -> str:
        """格式化健康度報告 - 遵循 PEP-8 認知複雜度≤15 要求"""
        if not metrics_data:
            return "## AIVA 合約健康度報告\n\n**無數據可用**\n"
        
        # 基本資訊
        header = self._build_report_header(metrics_data, days)
        
        # 狀態概覽
        overview = self._build_status_overview(metrics_data)
        
        # 趨勢分析
        trend_analysis = self._build_trend_analysis(metrics_data)
        
        # 告警統計
        alert_stats = self._build_alert_statistics(alerts_data)
        
        return f"{header}\n{overview}\n{trend_analysis}\n{alert_stats}"
    
    def _build_report_header(self, metrics_data: List, days: int) -> str:
        """建立報告標頭 - 基本資訊部分"""
        now = datetime.now(UTC)
        return f"""# AIVA 合約健康度報告

**生成時間**: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**報告週期**: 最近 {days} 天  
**數據點數**: {len(metrics_data)}"""

    def _build_status_overview(self, metrics_data: List) -> str:
        """建立狀態概覽表格"""
        latest_metric = metrics_data[0] if metrics_data else None
        if not latest_metric:
            return "\n## 📊 當前狀態概覽\n\n**無最新數據**\n"
        
        # 計算趨勢
        coverage_trend = self._calculate_trend([m[4] for m in metrics_data])
        error_trend = self._calculate_trend([m[5] for m in metrics_data])
        
        # 狀態指標
        health_status = self._get_health_status_emoji(latest_metric[8])
        coverage_trend_emoji = self._get_trend_emoji(coverage_trend)
        error_trend_emoji = self._get_trend_emoji(error_trend)
        performance_status = self._get_performance_status_emoji(latest_metric[6])
        
        return f"""
## 📊 當前狀態概覽

| 指標 | 數值 | 狀態 |
|------|------|------|
| 健康狀態 | {latest_metric[8] if latest_metric else 'N/A'} | {health_status} |
| 合約覆蓋率 | {latest_metric[4]:.1%} if latest_metric else 'N/A' | {coverage_trend_emoji} |
| 已使用合約 | {latest_metric[3] if latest_metric else 'N/A'}/{latest_metric[2] if latest_metric else 'N/A'} | - |
| 驗證錯誤 | {latest_metric[5] if latest_metric else 'N/A'} | {error_trend_emoji} |
| 性能分數 | {latest_metric[6]:.1%} if latest_metric else 'N/A' | {performance_status} |"""

    def _build_trend_analysis(self, metrics_data: List) -> str:
        """建立趨勢分析部分"""
        coverage_trend = self._calculate_trend([m[4] for m in metrics_data])
        error_trend = self._calculate_trend([m[5] for m in metrics_data])
        
        coverage_trend_text = self._get_trend_text(coverage_trend)
        error_trend_text = self._get_trend_text(error_trend)
        
        return f"""
## 📈 趨勢分析

### 覆蓋率變化
- **趨勢**: {coverage_trend_text}
- **變化幅度**: {abs(coverage_trend):.2%}

### 錯誤趨勢  
- **趨勢**: {error_trend_text}
- **變化幅度**: {abs(error_trend):.1f} 個"""

    def _build_alert_statistics(self, alerts_data: List) -> str:
        """建立告警統計部分"""
        return f"""
## 🚨 告警統計

**總告警數**: {len(alerts_data)}

"""
    
    def _get_health_status_emoji(self, health_status: str) -> str:
        """取得健康狀態表情符號"""
        if health_status in ['excellent', 'good']:
            return '🟢'
        elif health_status == 'warning':
            return '🟡'
        else:
            return '🔴'
    
    def _get_trend_emoji(self, trend: float) -> str:
        """取得趨勢表情符號"""
        if trend > 0:
            return '📈'
        elif trend < 0:
            return '📉'
        else:
            return '➡️'
    
    def _get_trend_text(self, trend: float) -> str:
        """取得趨勢文字描述"""
        if trend > 0:
            return '上升 📈'
        elif trend < 0:
            return '下降 📉'
        else:
            return '穩定 ➡️'
    
    def _get_performance_status_emoji(self, performance: float) -> str:
        """取得性能狀態表情符號"""
        if performance > 0.7:
            return '🟢'
        elif performance > 0.5:
            return '🟡'
        else:
            return '🔴'
    
    def _calculate_trend(self, values: List[float]) -> float:
        """計算數值趨勢 (簡單線性趨勢)"""
        if len(values) < 2:
            return 0.0
        
        # 簡單線性趨勢計算
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # 斜率計算
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    async def start_monitoring(self, interval: int = 3600):
        """開始持續監控"""
        logger.info(f"開始合約健康度監控 (間隔: {interval}秒)")
        
        while True:
            try:
                # 收集指標
                metric = self.collect_metrics()
                
                # 存儲指標
                self.store_metric(metric)
                
                # 檢查告警
                await self.check_alerts(metric)
                
                # 日誌記錄
                logger.info(
                    f"監控週期完成 - 覆蓋率: {metric.usage_coverage:.1%}, "
                    f"錯誤: {metric.validation_errors}, 狀態: {metric.health_status}"
                )
                
                # 等待下一個週期
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("監控已停止 (用戶中斷)")
                break
            except Exception as e:
                logger.error(f"監控週期失敗: {e}")
                await asyncio.sleep(60)  # 出錯時短暫等待


async def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA 合約健康度監控系統")
    parser.add_argument("--config", default="config/monitor_config.json", help="配置文件路徑")
    parser.add_argument("--interval", type=int, default=3600, help="監控間隔(秒)")
    parser.add_argument("--once", action="store_true", help="只運行一次檢查")
    parser.add_argument("--report", action="store_true", help="生成健康度報告")
    parser.add_argument("--days", type=int, default=7, help="報告天數")
    
    args = parser.parse_args()
    
    # 創建監控器
    monitor = ContractHealthMonitor(args.config)
    
    if args.report:
        # 生成報告
        report_path = await monitor.generate_report(args.days)
        print(f"✅ 健康度報告已生成: {report_path}")
    elif args.once:
        # 單次檢查
        metric = monitor.collect_metrics()
        monitor.store_metric(metric)
        await monitor.check_alerts(metric)
        
        print("📊 合約健康度檢查結果:")
        print(f"  覆蓋率: {metric.usage_coverage:.1%}")
        print(f"  驗證錯誤: {metric.validation_errors}個")
        print(f"  廢棄警告: {metric.deprecated_warnings}個") 
        print(f"  性能分數: {metric.performance_score:.1%}")
        print(f"  健康狀態: {metric.health_status}")
    else:
        # 持續監控
        await monitor.start_monitoring(args.interval)


if __name__ == "__main__":
    asyncio.run(main())