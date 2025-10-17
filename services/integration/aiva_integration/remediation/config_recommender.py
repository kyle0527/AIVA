"""
Config Recommender - 配置建議器

根據掃描結果提供安全配置建議
使用 ruamel.yaml 處理配置文件
"""

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
import structlog

logger = structlog.get_logger(__name__)


class ConfigRecommender:
    """
    安全配置建議器

    分析當前配置並提供安全改進建議
    """

    def __init__(self, preserve_comments: bool = True):
        """
        初始化配置建議器

        Args:
            preserve_comments: 是否保留 YAML 註釋
        """
        self.preserve_comments = preserve_comments
        self.yaml = YAML()
        if preserve_comments:
            self.yaml.preserve_quotes = True
            self.yaml.default_flow_style = False
        self.recommendations: list[dict[str, Any]] = []

        logger.info("config_recommender_initialized")

    def analyze_security_config(
        self,
        config: dict[str, Any],
        config_type: str = "general",
    ) -> dict[str, Any]:
        """
        分析安全配置

        Args:
            config: 配置字典
            config_type: 配置類型 (web_server, database, application, etc.)

        Returns:
            分析結果
        """
        logger.info("analyzing_config", config_type=config_type)

        analysis = {
            "analysis_id": hashlib.sha256(
                f"{config_type}{datetime.now()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "config_type": config_type,
            "issues": [],
            "recommendations": [],
            "risk_level": "LOW",
        }

        # 根據配置類型進行分析
        if config_type == "web_server":
            analysis = self._analyze_web_server_config(config, analysis)
        elif config_type == "database":
            analysis = self._analyze_database_config(config, analysis)
        elif config_type == "application":
            analysis = self._analyze_application_config(config, analysis)
        else:
            analysis = self._analyze_general_config(config, analysis)

        # 評估整體風險
        analysis["risk_level"] = self._calculate_risk_level(analysis["issues"])

        self.recommendations.append(analysis)
        logger.info(
            "config_analyzed",
            analysis_id=analysis["analysis_id"],
            issues=len(analysis["issues"]),
            risk=analysis["risk_level"],
        )
        return analysis

    def _analyze_web_server_config(
        self,
        config: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """分析 Web 服務器配置"""
        # 檢查 HTTPS
        if not config.get("https", {}).get("enabled", False):
            analysis["issues"].append({
                "severity": "HIGH",
                "issue": "HTTPS not enabled",
                "current_value": False,
                "recommended_value": True,
            })
            analysis["recommendations"].append({
                "key": "https.enabled",
                "current": False,
                "recommended": True,
                "reason": "HTTPS encrypts traffic and protects against eavesdropping",
            })

        # 檢查 HSTS
        if not config.get("security_headers", {}).get("hsts", False):
            analysis["issues"].append({
                "severity": "MEDIUM",
                "issue": "HSTS header not enabled",
            })
            analysis["recommendations"].append({
                "key": "security_headers.hsts",
                "recommended": True,
                "reason": "HSTS prevents protocol downgrade attacks",
            })

        # 檢查 CSP
        if not config.get("security_headers", {}).get("csp"):
            analysis["issues"].append({
                "severity": "MEDIUM",
                "issue": "Content Security Policy not configured",
            })
            analysis["recommendations"].append({
                "key": "security_headers.csp",
                "recommended": "default-src 'self'; script-src 'self'",
                "reason": "CSP helps prevent XSS attacks",
            })

        return analysis

    def _analyze_database_config(
        self,
        config: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """分析數據庫配置"""
        # 檢查 SSL
        if not config.get("ssl", {}).get("enabled", False):
            analysis["issues"].append({
                "severity": "HIGH",
                "issue": "Database SSL not enabled",
            })
            analysis["recommendations"].append({
                "key": "ssl.enabled",
                "recommended": True,
                "reason": "SSL encrypts database connections",
            })

        # 檢查默認端口
        default_ports = {
            "mysql": 3306,
            "postgresql": 5432,
            "mongodb": 27017,
        }
        db_type = config.get("type", "").lower()
        current_port = config.get("port")

        if db_type in default_ports and current_port == default_ports[db_type]:
            analysis["issues"].append({
                "severity": "LOW",
                "issue": f"Using default port for {db_type}",
                "current_value": current_port,
            })
            analysis["recommendations"].append({
                "key": "port",
                "current": current_port,
                "recommended": f"Non-default port (not {current_port})",
                "reason": "Custom ports reduce automated attack surface",
            })

        # 檢查密碼強度要求
        if not config.get("password_policy", {}).get("enforce_strength", False):
            analysis["recommendations"].append({
                "key": "password_policy.enforce_strength",
                "recommended": True,
                "reason": "Strong password policies prevent brute force attacks",
            })

        return analysis

    def _analyze_application_config(
        self,
        config: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """分析應用程序配置"""
        # 檢查 debug 模式
        if config.get("debug", False):
            analysis["issues"].append({
                "severity": "HIGH",
                "issue": "Debug mode enabled in production",
                "current_value": True,
                "recommended_value": False,
            })
            analysis["recommendations"].append({
                "key": "debug",
                "current": True,
                "recommended": False,
                "reason": "Debug mode exposes sensitive information",
            })

        # 檢查 secret key
        secret_key = config.get("secret_key", "")
        if not secret_key or secret_key in ["changeme", "secret", "default"]:
            analysis["issues"].append({
                "severity": "CRITICAL",
                "issue": "Weak or default secret key",
            })
            analysis["recommendations"].append({
                "key": "secret_key",
                "recommended": "[generated random string]",
                "reason": "Strong secret keys protect session security",
            })

        # 檢查 CORS
        cors_origins = config.get("cors", {}).get("origins", [])
        if "*" in cors_origins:
            analysis["issues"].append({
                "severity": "MEDIUM",
                "issue": "CORS allows all origins",
            })
            analysis["recommendations"].append({
                "key": "cors.origins",
                "current": ["*"],
                "recommended": ["https://trusted-domain.com"],
                "reason": "Restrictive CORS prevents unauthorized access",
            })

        return analysis

    def _analyze_general_config(
        self,
        config: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """分析通用配置"""
        # 檢查日誌級別
        log_level = config.get("logging", {}).get("level", "").upper()
        if log_level in ["DEBUG", "TRACE"]:
            analysis["recommendations"].append({
                "key": "logging.level",
                "current": log_level,
                "recommended": "INFO",
                "reason": "Verbose logging can expose sensitive information",
            })

        return analysis

    def _calculate_risk_level(self, issues: list[dict[str, Any]]) -> str:
        """計算整體風險等級"""
        if not issues:
            return "LOW"

        severity_scores = {
            "CRITICAL": 10,
            "HIGH": 7,
            "MEDIUM": 4,
            "LOW": 1,
        }

        total_score = sum(
            severity_scores.get(issue.get("severity", "LOW"), 1)
            for issue in issues
        )

        if total_score >= 10:
            return "CRITICAL"
        if total_score >= 7:
            return "HIGH"
        if total_score >= 4:
            return "MEDIUM"
        return "LOW"

    def generate_secure_config(
        self,
        base_config: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        根據建議生成安全配置

        Args:
            base_config: 基礎配置
            analysis: 分析結果

        Returns:
            安全配置
        """
        logger.info("generating_secure_config")

        secure_config = base_config.copy()

        # 應用所有建議
        for rec in analysis["recommendations"]:
            keys = rec["key"].split(".")
            current = secure_config

            # 導航到嵌套鍵
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 設置推薦值
            current[keys[-1]] = rec["recommended"]

        logger.info("secure_config_generated")
        return secure_config

    def export_recommendations_to_yaml(
        self,
        output_path: str | Path,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        導出建議到 YAML 文件

        Args:
            output_path: 輸出路徑
            analysis: 分析結果

        Returns:
            導出結果
        """
        path = Path(output_path)

        try:
            with open(path, "w", encoding="utf-8") as f:
                self.yaml.dump(
                    {
                        "analysis_id": analysis["analysis_id"],
                        "timestamp": analysis["timestamp"],
                        "config_type": analysis["config_type"],
                        "risk_level": analysis["risk_level"],
                        "recommendations": analysis["recommendations"],
                    },
                    f,
                )

            logger.info("recommendations_exported", file=str(path))
            return {
                "success": True,
                "file": str(path),
            }

        except Exception as e:
            logger.error("export_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    def get_recommendations(self) -> list[dict[str, Any]]:
        """獲取所有建議"""
        return self.recommendations


def main():
    """測試範例"""
    print("[U+2699][U+FE0F]  Config Recommender Demo")
    print("=" * 60)

    recommender = ConfigRecommender()

    # 分析 Web 服務器配置
    web_config = {
        "https": {"enabled": False},
        "security_headers": {"hsts": False},
        "port": 80,
    }

    analysis = recommender.analyze_security_config(
        config=web_config,
        config_type="web_server",
    )

    print(f"\n[STATS] Analysis ID: {analysis['analysis_id']}")
    print(f"[TARGET] Config Type: {analysis['config_type']}")
    print(f"[WARN]  Risk Level: {analysis['risk_level']}")
    print(f"\n[SEARCH] Issues Found: {len(analysis['issues'])}")

    for issue in analysis["issues"]:
        print(f"   - [{issue['severity']}] {issue['issue']}")

    print(f"\n[TIP] Recommendations: {len(analysis['recommendations'])}")
    for rec in analysis["recommendations"][:3]:
        print(f"   - {rec['key']}: {rec.get('recommended')}")
        print(f"     Reason: {rec['reason']}")

    # 生成安全配置
    secure_config = recommender.generate_secure_config(web_config, analysis)
    print("\n[OK] Secure config generated")
    print(f"   HTTPS enabled: {secure_config['https']['enabled']}")

    print("\n[OK] Demo completed")


if __name__ == "__main__":
    main()
