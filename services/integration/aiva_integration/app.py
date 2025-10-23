from __future__ import annotations

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
from .reception.sql_result_database import SqlResultDatabase
from .reporting.formatter_exporter import FormatterExporter
from .reporting.report_content_generator import ReportContentGenerator
from .reporting.report_template_selector import ReportTemplateSelector

app = FastAPI(title="AIVA Integration Module")
logger = get_logger(__name__)

db = SqlResultDatabase("sqlite:///aiva_integration.db")
recv = DataReceptionLayer(db)
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
    result = await db.get_finding(finding_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Finding with ID '{finding_id}' not found",
        )

    return result.model_dump()


async def _consume_logs() -> None:
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.LOG_RESULTS_ALL):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        finding = FindingPayload(**msg.payload)
        recv.store_finding(finding)
