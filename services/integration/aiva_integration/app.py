from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI

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
from .reception.test_result_database import TestResultDatabase
from .reporting.formatter_exporter import FormatterExporter
from .reporting.report_content_generator import ReportContentGenerator
from .reporting.report_template_selector import ReportTemplateSelector

app = FastAPI(title="AIVA Integration Module")
logger = get_logger(__name__)

db = TestResultDatabase()
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
    result = db.get_finding(finding_id)
    if isinstance(result, dict):
        return result
    # Convert Pydantic model to dict if needed
    try:
        return result.model_dump()
    except Exception:
        return {"error": "not_found", "finding_id": finding_id}


async def _consume_logs() -> None:
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.LOG_RESULTS_ALL):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        finding = FindingPayload(**msg.payload)
        recv.store_finding(finding)
