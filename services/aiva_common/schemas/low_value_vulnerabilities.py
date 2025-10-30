"""
低價值高概率漏洞檢測 Schema 模型 - HackerOne 穩定收入策略

此模組專門針對容易發現、獎金穩定的漏洞類型，實現了詳細的檢測模式、
測試向量和自動化策略，用於實現 80% 資源投入的穩定收入目標。

重點漏洞類型：
- Information Disclosure ($50-$200) - 60% 成功率
- Reflected XSS ($100-$300) - 45% 成功率  
- CSRF ($100-$300) - 40% 成功率
- Simple IDOR ($200-$500) - 35% 成功率
- Open Redirect ($50-$150) - 55% 成功率
- Host Header Injection ($75-$250) - 50% 成功率
"""



from datetime import datetime, UTC
from typing import List, Dict, Optional, Any, Literal, Union
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, field_validator


from ..enums.security import (
    BugBountyCategory, BountyPriorityTier, VulnerabilityDifficulty,
    TestingApproach, ProgramType, ResponseTimeCategory
)


# ==================== 低價值高概率漏洞基礎模型 ====================


class LowValueVulnerabilityType(str, Enum):
    """低價值高概率漏洞類型枚舉"""
    
    # 信息洩露類 - 最穩定收入 ($50-$200, 60% 成功率)
    INFO_DISCLOSURE_ERROR_MESSAGES = "info_disclosure_error_messages"
    INFO_DISCLOSURE_COMMENTS = "info_disclosure_comments"  
    INFO_DISCLOSURE_HEADERS = "info_disclosure_headers"
    INFO_DISCLOSURE_PATHS = "info_disclosure_paths"
    INFO_DISCLOSURE_DEBUG_INFO = "info_disclosure_debug_info"
    
    # XSS 類 - 中等穩定收入 ($100-$300, 45% 成功率)
    REFLECTED_XSS_BASIC = "reflected_xss_basic"
    REFLECTED_XSS_URL_PARAMS = "reflected_xss_url_params"
    REFLECTED_XSS_FORM_INPUTS = "reflected_xss_form_inputs"
    DOM_XSS_SIMPLE = "dom_xss_simple"
    
    # CSRF 類 - 中等穩定收入 ($100-$300, 40% 成功率)
    CSRF_MISSING_TOKEN = "csrf_missing_token"
    CSRF_WEAK_TOKEN = "csrf_weak_token"
    CSRF_GET_REQUEST = "csrf_get_request"
    CSRF_JSON_BYPASS = "csrf_json_bypass"
    
    # IDOR 類 - 較高收入 ($200-$500, 35% 成功率)
    IDOR_SIMPLE_ID = "idor_simple_id"
    IDOR_UUID_EXPOSURE = "idor_uuid_exposure"
    IDOR_FILE_ACCESS = "idor_file_access"
    IDOR_USER_DATA = "idor_user_data"
    
    # 重定向類 - 低收入高成功率 ($50-$150, 55% 成功率)
    OPEN_REDIRECT_PARAM = "open_redirect_param"
    OPEN_REDIRECT_REFERER = "open_redirect_referer"
    OPEN_REDIRECT_HOST = "open_redirect_host"
    
    # Host Header 類 - 中等收入 ($75-$250, 50% 成功率)
    HOST_HEADER_INJECTION = "host_header_injection"
    HOST_HEADER_PASSWORD_RESET = "host_header_password_reset"
    HOST_HEADER_CACHE_POISONING = "host_header_cache_poisoning"
    
    # CORS 類 - 低收入 ($75-$200, 45% 成功率)
    CORS_WILDCARD_ORIGIN = "cors_wildcard_origin"
    CORS_NULL_ORIGIN = "cors_null_origin"
    CORS_SUBDOMAIN_BYPASS = "cors_subdomain_bypass"
    
    # Clickjacking 類 - 低收入 ($50-$150, 40% 成功率)
    CLICKJACKING_NO_FRAME_OPTIONS = "clickjacking_no_frame_options"
    CLICKJACKING_WEAK_CSP = "clickjacking_weak_csp"


class VulnerabilityPattern(BaseModel):
    """漏洞檢測模式基礎類"""
    
    pattern_id: str = Field(description="模式唯一標識符")
    name: str = Field(description="模式名稱")
    vulnerability_type: LowValueVulnerabilityType = Field(description="漏洞類型")
    description: str = Field(description="模式描述")
    
    # 獎金預估
    min_bounty_usd: int = Field(description="最低獎金（美元）")
    max_bounty_usd: int = Field(description="最高獎金（美元）")
    avg_bounty_usd: int = Field(description="平均獎金（美元）")
    success_rate: float = Field(ge=0.0, le=1.0, description="發現成功率")
    
    # 難度和時間估算
    difficulty: VulnerabilityDifficulty = Field(description="發現難度")
    avg_discovery_time_minutes: int = Field(description="平均發現時間（分鐘）")
    max_discovery_time_minutes: int = Field(description="最大發現時間（分鐘）")
    
    # 測試配置
    testing_approach: TestingApproach = Field(description="測試方法")
    automation_level: float = Field(ge=0.0, le=1.0, description="自動化程度")
    
    # 適用程式類型
    suitable_program_types: List[ProgramType] = Field(description="適合的程式類型")
    priority_tier: BountyPriorityTier = Field(description="優先級層級")
    
    # 檢測條件
    detection_patterns: List[str] = Field(description="檢測模式")
    test_vectors: List[str] = Field(description="測試向量")
    false_positive_indicators: List[str] = Field(default_factory=list, description="誤報指標")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ==================== 信息洩露檢測模式 ====================


class InfoDisclosurePattern(VulnerabilityPattern):
    """信息洩露檢測模式 - 最穩定收入來源"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.INFO_DISCLOSURE_ERROR_MESSAGES,
        LowValueVulnerabilityType.INFO_DISCLOSURE_COMMENTS,
        LowValueVulnerabilityType.INFO_DISCLOSURE_HEADERS,
        LowValueVulnerabilityType.INFO_DISCLOSURE_PATHS,
        LowValueVulnerabilityType.INFO_DISCLOSURE_DEBUG_INFO
    ]
    
    # 信息類型
    sensitive_info_types: List[str] = Field(description="敏感信息類型")
    disclosure_locations: List[str] = Field(description="洩露位置")
    
    # 檢測配置
    response_patterns: List[str] = Field(description="回應模式")
    header_patterns: List[str] = Field(description="標頭模式")
    content_patterns: List[str] = Field(description="內容模式")
    
    # 誤報過濾
    benign_patterns: List[str] = Field(description="良性模式")
    context_requirements: List[str] = Field(description="上下文需求")


class ErrorMessageDisclosure(InfoDisclosurePattern):
    """錯誤訊息洩露模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.INFO_DISCLOSURE_ERROR_MESSAGES] = LowValueVulnerabilityType.INFO_DISCLOSURE_ERROR_MESSAGES
    
    # 錯誤類型
    error_types: List[str] = Field(description="錯誤類型")
    stack_trace_indicators: List[str] = Field(description="堆疊追蹤指標")
    database_error_patterns: List[str] = Field(description="資料庫錯誤模式")
    framework_error_patterns: List[str] = Field(description="框架錯誤模式")
    
    # 觸發方法
    trigger_methods: List[str] = Field(description="觸發方法")
    invalid_input_vectors: List[str] = Field(description="無效輸入向量")


class DebugInfoDisclosure(InfoDisclosurePattern):
    """除錯信息洩露模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.INFO_DISCLOSURE_DEBUG_INFO] = LowValueVulnerabilityType.INFO_DISCLOSURE_DEBUG_INFO
    
    # 除錯信息類型
    debug_endpoints: List[str] = Field(description="除錯端點")
    debug_parameters: List[str] = Field(description="除錯參數")
    development_artifacts: List[str] = Field(description="開發文件")
    
    # 環境檢測
    development_indicators: List[str] = Field(description="開發環境指標")
    staging_indicators: List[str] = Field(description="測試環境指標")


# ==================== XSS 檢測模式 ====================


class XSSPattern(VulnerabilityPattern):
    """XSS 檢測模式基礎類"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.REFLECTED_XSS_BASIC,
        LowValueVulnerabilityType.REFLECTED_XSS_URL_PARAMS,
        LowValueVulnerabilityType.REFLECTED_XSS_FORM_INPUTS,
        LowValueVulnerabilityType.DOM_XSS_SIMPLE
    ]
    
    # XSS 負載
    basic_payloads: List[str] = Field(description="基礎 XSS 負載")
    encoded_payloads: List[str] = Field(description="編碼 XSS 負載")
    filter_bypass_payloads: List[str] = Field(description="過濾器繞過負載")
    
    # 反射點
    reflection_contexts: List[str] = Field(description="反射上下文")
    injection_points: List[str] = Field(description="注入點")
    
    # 檢測配置
    confirmation_patterns: List[str] = Field(description="確認模式")
    false_positive_patterns: List[str] = Field(description="誤報模式")


class ReflectedXSSBasic(XSSPattern):
    """基礎反射型 XSS 模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.REFLECTED_XSS_BASIC] = LowValueVulnerabilityType.REFLECTED_XSS_BASIC
    
    # 基礎測試
    simple_test_cases: List[str] = Field(description="簡單測試案例")
    parameter_pollution_tests: List[str] = Field(description="參數污染測試")
    
    # WAF 繞過
    waf_bypass_techniques: List[str] = Field(description="WAF 繞過技術")
    encoding_variations: List[str] = Field(description="編碼變體")


class DOMXSSSimple(XSSPattern):
    """簡單 DOM XSS 模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.DOM_XSS_SIMPLE] = LowValueVulnerabilityType.DOM_XSS_SIMPLE
    
    # DOM 操作
    dom_sources: List[str] = Field(description="DOM 來源")
    dom_sinks: List[str] = Field(description="DOM 匯聚點")
    javascript_patterns: List[str] = Field(description="JavaScript 模式")
    
    # 觸發方法
    hash_triggers: List[str] = Field(description="Hash 觸發器")
    postmessage_triggers: List[str] = Field(description="PostMessage 觸發器")


# ==================== CSRF 檢測模式 ====================


class CSRFPattern(VulnerabilityPattern):
    """CSRF 檢測模式基礎類"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.CSRF_MISSING_TOKEN,
        LowValueVulnerabilityType.CSRF_WEAK_TOKEN,
        LowValueVulnerabilityType.CSRF_GET_REQUEST,
        LowValueVulnerabilityType.CSRF_JSON_BYPASS
    ]
    
    # 測試目標
    sensitive_actions: List[str] = Field(description="敏感操作")
    state_changing_endpoints: List[str] = Field(description="狀態變更端點")
    
    # 檢測方法
    token_locations: List[str] = Field(description="令牌位置")
    bypass_techniques: List[str] = Field(description="繞過技術")
    
    # 驗證配置
    proof_of_concept_templates: List[str] = Field(description="概念驗證模板")


class CSRFMissingToken(CSRFPattern):
    """缺少 CSRF 令牌模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.CSRF_MISSING_TOKEN] = LowValueVulnerabilityType.CSRF_MISSING_TOKEN
    
    # 檢測步驟
    original_request_analysis: List[str] = Field(description="原始請求分析")
    token_removal_tests: List[str] = Field(description="令牌移除測試")
    referer_bypass_tests: List[str] = Field(description="Referer 繞過測試")


class CSRFJSONBypass(CSRFPattern):
    """JSON CSRF 繞過模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.CSRF_JSON_BYPASS] = LowValueVulnerabilityType.CSRF_JSON_BYPASS
    
    # JSON 特定測試
    content_type_variations: List[str] = Field(description="Content-Type 變體")
    json_payload_templates: List[str] = Field(description="JSON 負載模板")
    form_encoding_bypass: List[str] = Field(description="表單編碼繞過")


# ==================== IDOR 檢測模式 ====================


class IDORPattern(VulnerabilityPattern):
    """IDOR 檢測模式基礎類"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.IDOR_SIMPLE_ID,
        LowValueVulnerabilityType.IDOR_UUID_EXPOSURE,
        LowValueVulnerabilityType.IDOR_FILE_ACCESS,
        LowValueVulnerabilityType.IDOR_USER_DATA
    ]
    
    # 識別模式
    id_patterns: List[str] = Field(description="ID 模式")
    parameter_names: List[str] = Field(description="參數名稱")
    
    # 測試方法
    enumeration_techniques: List[str] = Field(description="枚舉技術")
    privilege_escalation_tests: List[str] = Field(description="權限提升測試")
    
    # 驗證配置
    access_control_bypass: List[str] = Field(description="存取控制繞過")


class IDORSimpleID(IDORPattern):
    """簡單 ID IDOR 模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.IDOR_SIMPLE_ID] = LowValueVulnerabilityType.IDOR_SIMPLE_ID
    
    # 簡單 ID 測試
    sequential_id_tests: List[str] = Field(description="順序 ID 測試")
    predictable_patterns: List[str] = Field(description="可預測模式")
    brute_force_ranges: List[str] = Field(description="暴力破解範圍")


class IDORUserData(IDORPattern):
    """使用者資料 IDOR 模式"""
    
    vulnerability_type: Literal[LowValueVulnerabilityType.IDOR_USER_DATA] = LowValueVulnerabilityType.IDOR_USER_DATA
    
    # 使用者資料類型
    sensitive_data_types: List[str] = Field(description="敏感資料類型")
    profile_endpoints: List[str] = Field(description="個人資料端點")
    data_export_endpoints: List[str] = Field(description="資料匯出端點")


# ==================== 其他低價值高概率漏洞 ====================


class OpenRedirectPattern(VulnerabilityPattern):
    """開放重定向檢測模式"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.OPEN_REDIRECT_PARAM,
        LowValueVulnerabilityType.OPEN_REDIRECT_REFERER,
        LowValueVulnerabilityType.OPEN_REDIRECT_HOST
    ]
    
    # 重定向參數
    redirect_parameters: List[str] = Field(description="重定向參數")
    redirect_patterns: List[str] = Field(description="重定向模式")
    
    # 測試負載
    malicious_urls: List[str] = Field(description="惡意 URL")
    bypass_techniques: List[str] = Field(description="繞過技術")


class HostHeaderInjectionPattern(VulnerabilityPattern):
    """Host Header 注入檢測模式"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.HOST_HEADER_INJECTION,
        LowValueVulnerabilityType.HOST_HEADER_PASSWORD_RESET,
        LowValueVulnerabilityType.HOST_HEADER_CACHE_POISONING
    ]
    
    # 注入測試
    malicious_hosts: List[str] = Field(description="惡意主機")
    port_variations: List[str] = Field(description="埠口變體")
    
    # 功能測試
    password_reset_tests: List[str] = Field(description="密碼重設測試")
    cache_poisoning_tests: List[str] = Field(description="快取中毒測試")


class CORSMisconfigurationPattern(VulnerabilityPattern):
    """CORS 錯誤配置檢測模式"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.CORS_WILDCARD_ORIGIN,
        LowValueVulnerabilityType.CORS_NULL_ORIGIN,
        LowValueVulnerabilityType.CORS_SUBDOMAIN_BYPASS
    ]
    
    # CORS 測試
    origin_variations: List[str] = Field(description="Origin 變體")
    credential_tests: List[str] = Field(description="憑證測試")
    preflight_bypass: List[str] = Field(description="預檢繞過")


class ClickjackingPattern(VulnerabilityPattern):
    """點擊劫持檢測模式"""
    
    vulnerability_type: Literal[
        LowValueVulnerabilityType.CLICKJACKING_NO_FRAME_OPTIONS,
        LowValueVulnerabilityType.CLICKJACKING_WEAK_CSP
    ]
    
    # 防護檢測
    frame_options_tests: List[str] = Field(description="Frame Options 測試")
    csp_header_tests: List[str] = Field(description="CSP 標頭測試")
    
    # PoC 生成
    iframe_templates: List[str] = Field(description="iframe 模板")
    ui_redressing_techniques: List[str] = Field(description="UI 重建技術")


# ==================== 測試執行和結果 ====================


class LowValueVulnerabilityTest(BaseModel):
    """低價值漏洞測試配置"""
    
    test_id: str = Field(description="測試唯一標識符")
    name: str = Field(description="測試名稱")
    target_url: HttpUrl = Field(description="測試目標 URL")
    
    # 測試配置
    patterns: List[VulnerabilityPattern] = Field(description="使用的檢測模式")
    max_test_time_minutes: int = Field(default=120, description="最大測試時間（分鐘）")
    parallel_tests: int = Field(default=3, description="並行測試數")
    
    # 篩選條件
    min_success_rate: float = Field(default=0.3, description="最低成功率")
    min_bounty_usd: int = Field(default=50, description="最低獎金（美元）")
    max_difficulty: VulnerabilityDifficulty = Field(default=VulnerabilityDifficulty.MEDIUM, description="最高難度")
    
    # 程式資訊
    program_type: ProgramType = Field(description="程式類型")
    expected_response_time: ResponseTimeCategory = Field(description="預期回應時間")
    
    # 優先級策略
    prioritize_by: Literal["success_rate", "bounty_amount", "discovery_time", "roi"] = Field(
        default="roi",
        description="優先級策略"
    )
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LowValueVulnerabilityResult(BaseModel):
    """低價值漏洞測試結果"""
    
    result_id: str = Field(description="結果唯一標識符")
    test_id: str = Field(description="關聯測試 ID")
    pattern_id: str = Field(description="使用的模式 ID")
    
    # 漏洞資訊
    vulnerability_found: bool = Field(description="是否發現漏洞")
    vulnerability_type: LowValueVulnerabilityType = Field(description="漏洞類型")
    confidence_score: int = Field(ge=0, le=100, description="置信分數")
    
    # 測試詳情
    endpoint: str = Field(description="測試端點")
    method: str = Field(description="HTTP 方法")
    payload: str = Field(description="使用的負載")
    response_snippet: str = Field(description="回應片段")
    
    # 時間統計
    discovery_time_minutes: int = Field(description="發現時間（分鐘）")
    total_requests: int = Field(description="總請求數")
    
    # 獎金預估
    estimated_bounty_usd: int = Field(description="預估獎金（美元）")
    bounty_confidence: float = Field(ge=0.0, le=1.0, description="獎金置信度")
    
    # ROI 計算
    time_investment_hours: float = Field(description="時間投入（小時）")
    expected_roi: float = Field(description="預期投資回報率")
    
    # 報告準備
    ready_for_submission: bool = Field(default=False, description="準備提交")
    report_draft: Optional[str] = Field(default=None, description="報告草稿")
    
    # 驗證狀態
    manually_verified: bool = Field(default=False, description="手動驗證")
    false_positive_risk: float = Field(ge=0.0, le=1.0, description="誤報風險")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BugBountyStrategy(BaseModel):
    """Bug Bounty 策略配置"""
    
    strategy_id: str = Field(description="策略唯一標識符")
    name: str = Field(description="策略名稱")
    description: str = Field(description="策略描述")
    
    # 資源分配
    low_value_allocation_percent: int = Field(default=80, ge=0, le=100, description="低價值漏洞資源分配百分比")
    high_value_allocation_percent: int = Field(default=20, ge=0, le=100, description="高價值漏洞資源分配百分比")
    
    # 目標設定
    daily_income_target_usd: int = Field(description="每日收入目標（美元）")
    weekly_income_target_usd: int = Field(description="每週收入目標（美元）")
    monthly_income_target_usd: int = Field(description="每月收入目標（美元）")
    
    # 測試配置
    max_programs_per_day: int = Field(default=5, description="每日最大測試程式數")
    max_hours_per_program: float = Field(default=2.0, description="每個程式最大測試時間（小時）")
    
    # 成功率要求
    min_overall_success_rate: float = Field(default=0.4, description="最低整體成功率")
    target_false_positive_rate: float = Field(default=0.1, description="目標誤報率")
    
    # 優先級設定
    preferred_vulnerability_types: List[LowValueVulnerabilityType] = Field(description="偏好漏洞類型")
    avoided_vulnerability_types: List[LowValueVulnerabilityType] = Field(default_factory=list, description="避免漏洞類型")
    
    # 程式篩選
    preferred_program_types: List[ProgramType] = Field(description="偏好程式類型")
    max_response_time: ResponseTimeCategory = Field(description="最大回應時間")
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("low_value_allocation_percent", "high_value_allocation_percent")
    @classmethod
    def validate_allocation_sum(cls, v, info):
        """驗證資源分配總和為 100%"""
        if hasattr(info, 'data') and info.data:
            low_val = info.data.get('low_value_allocation_percent', 0)
            high_val = info.data.get('high_value_allocation_percent', 0)
            if low_val + high_val != 100:
                raise ValueError("低價值和高價值資源分配總和必須為 100%")
        return v


# ==================== 獎金預測和分析 ====================


class BountyPrediction(BaseModel):
    """獎金預測模型"""
    
    prediction_id: str = Field(description="預測唯一標識符")
    vulnerability_type: LowValueVulnerabilityType = Field(description="漏洞類型")
    program_type: ProgramType = Field(description="程式類型")
    
    # 歷史數據
    historical_bounties: List[int] = Field(description="歷史獎金數據")
    success_count: int = Field(description="成功次數")
    total_attempts: int = Field(description="總嘗試次數")
    
    # 預測結果
    predicted_bounty_min: int = Field(description="預測最低獎金")
    predicted_bounty_max: int = Field(description="預測最高獎金")
    predicted_bounty_avg: int = Field(description="預測平均獎金")
    success_probability: float = Field(ge=0.0, le=1.0, description="成功概率")
    
    # 時間預測
    avg_discovery_time_hours: float = Field(description="平均發現時間（小時）")
    avg_response_time_days: int = Field(description="平均回應時間（天）")
    
    # 競爭分析
    competition_level: Literal["low", "medium", "high"] = Field(description="競爭程度")
    duplicate_risk: float = Field(ge=0.0, le=1.0, description="重複風險")
    
    # 信心區間
    confidence_interval_95: tuple[int, int] = Field(description="95% 信心區間")
    prediction_confidence: float = Field(ge=0.0, le=1.0, description="預測信心")
    
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ROIAnalysis(BaseModel):
    """投資回報率分析"""
    
    analysis_id: str = Field(description="分析唯一標識符")
    strategy_id: str = Field(description="關聯策略 ID")
    
    # 時間投入
    time_period_days: int = Field(description="分析期間（天）")
    total_hours_invested: float = Field(description="總投入時間（小時）")
    
    # 收入分析
    total_bounties_earned: int = Field(description="總獲得獎金（美元）")
    average_bounty_per_finding: int = Field(description="每個發現平均獎金")
    successful_submissions: int = Field(description="成功提交數")
    rejected_submissions: int = Field(description="被拒絕提交數")
    
    # ROI 計算
    hourly_rate_usd: float = Field(description="每小時收入率（美元）")
    success_rate: float = Field(ge=0.0, le=1.0, description="成功率")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="誤報率")
    
    # 效率指標
    avg_time_per_finding_hours: float = Field(description="每個發現平均時間（小時）")
    findings_per_day: float = Field(description="每日發現數")
    
    # 趨勢分析
    roi_trend: Literal["improving", "stable", "declining"] = Field(description="ROI 趨勢")
    recommended_adjustments: List[str] = Field(description="建議調整")
    
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))