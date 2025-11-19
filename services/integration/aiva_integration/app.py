

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException

from services.aiva_common.enums import Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, FindingPayload
from services.aiva_common.utils import get_logger

from .analysis.compliance_policy_checker import CompliancePolicyChecker
from .analysis.risk_assessment_engine import RiskAssessmentEngine
from .analysis.vuln_correlation_analyzer import VulnerabilityCorrelationAnalyzer
from .config_template.config_template_manager import ConfigTemplateManager
from .perf_feedback.improvement_suggestion_generator import (
    ImprovementSuggestionGenerator,
)
from .perf_feedback.scan_metadata_analyzer import ScanMetadataAnalyzer
from .reception.data_reception_layer import DataReceptionLayer
from .reception.unified_storage_adapter import UnifiedStorageAdapter
from .reporting.formatter_exporter import FormatterExporter
from .reporting.report_content_generator import ReportContentGenerator
from .reporting.report_template_selector import ReportTemplateSelector

app = FastAPI(title="AIVA Integration Module")
logger = get_logger(__name__)

# 升級到統一存儲架構 (遵循 aiva_common 標準)
# 原：獨立的 SqlResultDatabase，數據分散存儲
# 新：統一的 StorageManager + PostgreSQL 後端，集中管理所有數據
import os

# 研發階段直接使用預設配置
from urllib.parse import urlparse
database_url = "postgresql://postgres:postgres@localhost:5432/aiva_db"
db_url = urlparse(database_url)

storage_adapter = UnifiedStorageAdapter(
    data_root="./data/integration",
    db_config={
        "host": db_url.hostname or "localhost",
        "port": db_url.port or 5432,
        "database": db_url.path.lstrip('/') or "aiva_db",
        "user": db_url.username or "postgres",
        "password": db_url.password or "postgres",
    }
)
recv = DataReceptionLayer(storage_adapter)
corr = VulnerabilityCorrelationAnalyzer()
risk = RiskAssessmentEngine()
comp = CompliancePolicyChecker()
rptgen = ReportContentGenerator()
fmt = FormatterExporter()
rptsel = ReportTemplateSelector()
smeta = ScanMetadataAnalyzer()
impr = ImprovementSuggestionGenerator()
ctm = ConfigTemplateManager()


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_consume_logs())


@app.get("/findings/{finding_id}")
async def get_finding(finding_id: str) -> dict[str, Any]:
    """
    獲取指定 ID 的漏洞發現

    Args:
        finding_id: 漏洞發現的唯一識別碼

    Returns:
        漏洞發現的詳細資料

    Raises:
        HTTPException: 當找不到對應的漏洞發現時,返回 404
    """
    result = await storage_adapter.get_finding(finding_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Finding with ID '{finding_id}' not found",
        )

    return result.model_dump()


async def _consume_logs() -> None:
    broker = await get_broker()
    subscriber = await broker.subscribe(Topic.LOG_RESULTS_ALL)
    async for mqmsg in subscriber:
        msg = AivaMessage.model_validate_json(mqmsg.body)
        finding = FindingPayload(**msg.payload)
        await recv.store_finding(finding)
