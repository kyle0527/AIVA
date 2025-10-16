"""
模組相關枚舉 - 模組名稱、主題等
"""

from __future__ import annotations

from enum import Enum


class ModuleName(str, Enum):
    API_GATEWAY = "ApiGateway"
    CORE = "CoreModule"
    SCAN = "ScanModule"
    INTEGRATION = "IntegrationModule"
    FUNCTION = "FunctionModule"
    FUNC_XSS = "FunctionXSS"
    FUNC_SQLI = "FunctionSQLI"
    FUNC_SSRF = "FunctionSSRF"
    FUNC_IDOR = "FunctionIDOR"
    OAST = "OASTService"
    THREAT_INTEL = "ThreatIntelModule"
    AUTHZ = "AuthZModule"
    POSTEX = "PostExModule"
    REMEDIATION = "RemediationModule"
    BIZLOGIC = "BizLogicModule"


class Topic(str, Enum):
    # Scan Topics
    TASK_SCAN_START = "tasks.scan.start"
    RESULTS_SCAN_COMPLETED = "results.scan.completed"

    # Function Topics
    TASK_FUNCTION_START = "tasks.function.start"
    TASK_FUNCTION_XSS = "tasks.function.xss"
    TASK_FUNCTION_SQLI = "tasks.function.sqli"
    TASK_FUNCTION_SSRF = "tasks.function.ssrf"
    FUNCTION_IDOR_TASK = "tasks.function.idor"
    RESULTS_FUNCTION_COMPLETED = "results.function.completed"

    # AI Training Topics
    TASK_AI_TRAINING_START = "tasks.ai.training.start"
    TASK_AI_TRAINING_EPISODE = "tasks.ai.training.episode"
    TASK_AI_TRAINING_STOP = "tasks.ai.training.stop"
    RESULTS_AI_TRAINING_PROGRESS = "results.ai.training.progress"
    RESULTS_AI_TRAINING_COMPLETED = "results.ai.training.completed"
    RESULTS_AI_TRAINING_FAILED = "results.ai.training.failed"

    # AI Experience & Learning Topics
    EVENT_AI_EXPERIENCE_CREATED = "events.ai.experience.created"
    EVENT_AI_TRACE_COMPLETED = "events.ai.trace.completed"
    EVENT_AI_MODEL_UPDATED = "events.ai.model.updated"
    COMMAND_AI_MODEL_DEPLOY = "commands.ai.model.deploy"

    # RAG Knowledge Topics
    TASK_RAG_KNOWLEDGE_UPDATE = "tasks.rag.knowledge.update"
    TASK_RAG_QUERY = "tasks.rag.query"
    RESULTS_RAG_RESPONSE = "results.rag.response"

    # General Topics
    FINDING_DETECTED = "findings.detected"
    LOG_RESULTS_ALL = "log.results.all"
    STATUS_TASK_UPDATE = "status.task.update"
    FEEDBACK_CORE_STRATEGY = "feedback.core.strategy"
    MODULE_HEARTBEAT = "module.heartbeat"
    COMMAND_TASK_CANCEL = "command.task.cancel"
    CONFIG_GLOBAL_UPDATE = "config.global.update"

    # ThreatIntel Topics
    TASK_THREAT_INTEL_LOOKUP = "tasks.threat_intel.lookup"
    TASK_IOC_ENRICHMENT = "tasks.threat_intel.ioc_enrichment"
    TASK_MITRE_MAPPING = "tasks.threat_intel.mitre_mapping"
    RESULTS_THREAT_INTEL = "results.threat_intel"

    # Scan Progress & Failure Topics (新增)
    RESULTS_SCAN_PROGRESS = "results.scan.progress"
    RESULTS_SCAN_FAILED = "results.scan.failed"
    EVENT_SCAN_ASSET_DISCOVERED = "events.scan.asset.discovered"

    # Function Progress & Failure Topics (新增)
    RESULTS_FUNCTION_PROGRESS = "results.function.progress"
    RESULTS_FUNCTION_FAILED = "results.function.failed"
    EVENT_FUNCTION_VULN_FOUND = "events.function.vuln.found"

    # Integration Analysis Topics (新增)
    TASK_INTEGRATION_ANALYSIS_START = "tasks.integration.analysis.start"
    RESULTS_INTEGRATION_ANALYSIS_PROGRESS = "results.integration.analysis.progress"
    RESULTS_INTEGRATION_ANALYSIS_COMPLETED = "results.integration.analysis.completed"
    COMMAND_INTEGRATION_REPORT_GENERATE = "commands.integration.report.generate"
    EVENT_INTEGRATION_REPORT_GENERATED = "events.integration.report.generated"

    # AI Scenario Topics (新增)
    EVENT_AI_SCENARIO_LOADED = "events.ai.scenario.loaded"

    # AuthZ Topics
    TASK_AUTHZ_CHECK = "tasks.authz.check"
    TASK_AUTHZ_ANALYZE = "tasks.authz.analyze"
    RESULTS_AUTHZ = "results.authz"

    # PostEx Topics (僅用於授權測試環境)
    TASK_POSTEX_TEST = "tasks.postex.test"
    TASK_POSTEX_PRIVILEGE_ESCALATION = "tasks.postex.privilege_escalation"
    TASK_POSTEX_LATERAL_MOVEMENT = "tasks.postex.lateral_movement"
    TASK_POSTEX_DATA_EXFILTRATION = "tasks.postex.data_exfiltration"
    TASK_POSTEX_PERSISTENCE = "tasks.postex.persistence"
    RESULTS_POSTEX = "results.postex"

    # Remediation Topics
    TASK_REMEDIATION_GENERATE = "tasks.remediation.generate"
    RESULTS_REMEDIATION = "results.remediation"

