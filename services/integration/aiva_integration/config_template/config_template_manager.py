from __future__ import annotations

from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ConfigTemplateManager:
    """配置模板管理器 - 管理掃描配置模板"""

    def __init__(self) -> None:
        self._templates: dict[str, dict[str, Any]] = {
            "web_app": {
                "name": "Web 應用掃描模板",
                "modules": ["xss", "sqli", "csrf"],
                "depth": 3,
                "timeout": 300,
            },
            "api": {
                "name": "API 掃描模板",
                "modules": ["sqli", "ssrf"],
                "depth": 2,
                "timeout": 180,
            },
            "comprehensive": {
                "name": "全面掃描模板",
                "modules": ["xss", "sqli", "ssrf", "csrf", "lfi"],
                "depth": 5,
                "timeout": 600,
            },
        }

    def get_template(self, template_name: str) -> dict[str, Any]:
        """獲取配置模板"""
        return self._templates.get(template_name, self._templates["web_app"])

    def list_templates(self) -> list[str]:
        """列出可用模板"""
        return list(self._templates.keys())

    def create_custom_template(self, name: str, config: dict[str, Any]) -> None:
        """創建自定義模板"""
        self._templates[name] = config
        logger.info(f"Created custom template: {name}")
