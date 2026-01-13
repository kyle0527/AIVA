"""標準任務參數包 - 統一所有模組的通信接口

此模組定義所有 AIVA 組件之間傳遞的標準數據結構：
- TaskContext: 統一任務上下文
- ScanTaskContext: 掃描任務專用
- AttackTaskContext: 攻擊任務專用

設計理念：
> 統一參數包 = 統一語言，讓所有模組都能無縫通信

日期: 2025-12-17
版本: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone

# UTC 兼容性处理（Python 3.11+ 使用 UTC，较旧版本使用 timezone.utc）
try:
    from datetime import UTC  # type: ignore
except ImportError:
    UTC = timezone.utc  # type: ignore


class TaskIntent(str, Enum):
    """任務意圖"""
    SCAN = "scan"  # 掃描
    ATTACK = "attack"  # 攻擊
    ANALYZE = "analyze"  # 分析
    EXPLOIT = "exploit"  # 利用
    RECON = "reconnaissance"  # 偵察


class TaskPhase(str, Enum):
    """任務階段（基於 Cyber Kill Chain）"""
    RECON = "reconnaissance"  # 偵察
    WEAPONIZATION = "weaponization"  # 武器化
    DELIVERY = "delivery"  # 投遞
    EXPLOITATION = "exploitation"  # 利用
    INSTALLATION = "installation"  # 安裝
    COMMAND_AND_CONTROL = "command_and_control"  # C&C
    ACTIONS_ON_OBJECTIVES = "actions_on_objectives"  # 目標行動


class StealthLevel(str, Enum):
    """隱匿級別"""
    NONE = "none"  # 無隱匿
    LOW = "low"  # 低隱匿
    MEDIUM = "medium"  # 中等隱匿
    HIGH = "high"  # 高隱匿
    PARANOID = "paranoid"  # 極度隱匿


@dataclass
class TaskConstraints:
    """任務約束條件"""
    # 速率限制
    rate_limit: int = 1000  # packets/sec
    max_requests_per_minute: int = 60
    
    # 時間約束
    timeout: int = 30  # 秒
    max_duration: Optional[int] = None  # 最大持續時間（秒）
    
    # 隱匿設置
    stealth_level: StealthLevel = StealthLevel.MEDIUM
    use_proxy: bool = False
    rotate_user_agent: bool = True
    
    # 範圍限制
    max_depth: int = 3  # 最大爬取深度
    max_urls: int = 1000  # 最大 URL 數量
    allowed_domains: list[str] = field(default_factory=list)
    
    # 安全設置
    safe_mode: bool = True  # 安全模式（避免破壞性操作）
    require_confirmation: bool = False  # 是否需要確認


@dataclass
class AIDecisionParams:
    """AI 決策參數"""
    selected_tool: Optional[str] = None  # AI 決定的工具 (nmap/masscan/sqlmap)
    confidence_score: float = 0.0  # AI 對此決策的信心分數 (0-1)
    reasoning: str = ""  # 決策推理
    alternatives: list[str] = field(default_factory=list)  # 備選方案


@dataclass
class TaskContext:
    """標準任務上下文 - 基礎類
    
    所有模組之間只傳遞這個物件，不再傳遞散亂的變數。
    這是所有任務的基礎結構。
    """
    # === 核心信息 ===
    task_id: str  # 任務 ID
    intent: TaskIntent  # 任務意圖
    target: str  # 目標（IP/URL/Domain）
    
    # === 階段信息 ===
    phase: TaskPhase = TaskPhase.RECON
    
    # === AI 決策 ===
    ai_decision: AIDecisionParams = field(default_factory=AIDecisionParams)
    
    # === 約束條件 ===
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    
    # === 元數據 ===
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # === 用戶輸入原文 ===
    user_input: Optional[str] = None  # 保存原始用戶命令


@dataclass
class ScanTaskContext(TaskContext):
    """掃描任務上下文
    
    專門用於掃描相關任務，繼承 TaskContext 並添加掃描特定字段。
    """
    # === 掃描特定參數 ===
    ports: list[int] = field(default_factory=list)  # 目標端口列表
    scan_type: str = "syn"  # 掃描類型 (syn/tcp/udp/comprehensive)
    
    # === 引擎選擇 ===
    preferred_engine: Optional[str] = None  # 首選引擎 (nmap/masscan)
    fallback_engines: list[str] = field(default_factory=list)  # 備用引擎
    
    # === Phase0/Phase1 控制 ===
    enable_phase0: bool = True  # 是否啟用 Phase0 快速偵察
    enable_phase1: bool = True  # 是否啟用 Phase1 深度掃描
    phase0_timeout: int = 600  # Phase0 超時（秒）
    phase1_timeout: int = 1800  # Phase1 超時（秒）


@dataclass
class AttackTaskContext(TaskContext):
    """攻擊任務上下文
    
    專門用於攻擊/利用相關任務。
    """
    # === 攻擊目標 ===
    objective: str = "comprehensive_assessment"  # 攻擊目標
    vulnerability_type: Optional[str] = None  # 漏洞類型 (xss/sqli/ssrf等)
    
    # === 攻擊參數 ===
    payloads: list[str] = field(default_factory=list)  # 攻擊載荷
    attack_vectors: list[str] = field(default_factory=list)  # 攻擊向量
    
    # === RAG 增強 ===
    use_rag: bool = True  # 是否使用 RAG 增強
    rag_context: dict[str, Any] = field(default_factory=dict)  # RAG 檢索結果
    
    # === 學習設置 ===
    enable_learning: bool = True  # 是否啟用學習
    is_training_scenario: bool = False  # 是否為訓練場景


@dataclass
class AnalysisTaskContext(TaskContext):
    """分析任務上下文
    
    專門用於結果分析、報告生成等。
    """
    # === 分析數據 ===
    raw_results: dict[str, Any] = field(default_factory=dict)  # 原始結果
    
    # === 分析選項 ===
    analysis_depth: str = "standard"  # 分析深度 (quick/standard/deep)
    include_recommendations: bool = True  # 是否包含修復建議
    generate_report: bool = True  # 是否生成報告


# === 工廠函數 ===

def create_scan_context(
    target: str,
    user_input: str,
    **kwargs
) -> ScanTaskContext:
    """創建掃描任務上下文的便捷函數"""
    import uuid
    
    return ScanTaskContext(
        task_id=str(uuid.uuid4()),
        intent=TaskIntent.SCAN,
        target=target,
        user_input=user_input,
        **kwargs
    )


def create_attack_context(
    target: str,
    objective: str,
    user_input: str,
    **kwargs
) -> AttackTaskContext:
    """創建攻擊任務上下文的便捷函數"""
    import uuid
    
    return AttackTaskContext(
        task_id=str(uuid.uuid4()),
        intent=TaskIntent.ATTACK,
        target=target,
        objective=objective,
        user_input=user_input,
        **kwargs
    )


def parse_user_input_to_context(user_input: str) -> TaskContext:
    """解析用戶輸入為任務上下文
    
    這個函數負責「聽懂人類語言」，提取關鍵信息。
    
    示例：
        "快速掃描 192.168.1.1" -> ScanTaskContext(rate_limit=5000)
        "偷偷掃描 example.com" -> ScanTaskContext(stealth_level=HIGH)
        "檢查 http://target.com 的 XSS" -> AttackTaskContext(vulnerability_type="xss")
    """
    import re
    import uuid
    
    # 提取目標（IP/URL/Domain）
    target_match = re.search(r'(?:https?://)?(?:[\w\-]+\.)+[\w\-]+|\d+\.\d+\.\d+\.\d+', user_input)
    target = target_match.group(0) if target_match else "unknown"
    
    # 判斷意圖
    if any(word in user_input for word in ["掃描", "scan", "掃"]):
        intent = TaskIntent.SCAN
        context_type = ScanTaskContext
    elif any(word in user_input for word in ["攻擊", "attack", "exploit", "利用"]):
        intent = TaskIntent.ATTACK
        context_type = AttackTaskContext
    else:
        intent = TaskIntent.SCAN  # 默認為掃描
        context_type = ScanTaskContext
    
    # 創建基礎上下文
    if context_type == ScanTaskContext:
        context = ScanTaskContext(
            task_id=str(uuid.uuid4()),
            intent=intent,
            target=target,
            user_input=user_input
        )
    else:
        # 先創建 AttackTaskContext，然後設定 vulnerability_type
        # 提取目標（XSS/SQLi等）
        objective = "comprehensive_assessment"
        vulnerability_type = None
        if "xss" in user_input.lower():
            objective = "檢查 XSS 漏洞"
            vulnerability_type = "xss"
        elif "sql" in user_input.lower():
            objective = "檢查 SQL 注入"
            vulnerability_type = "sqli"
        
        context = AttackTaskContext(
            task_id=str(uuid.uuid4()),
            intent=intent,
            target=target,
            objective=objective,
            user_input=user_input
        )
        
        # 如果有 vulnerability_type，則設定它
        if vulnerability_type:
            context.vulnerability_type = vulnerability_type
    
    # 解析修飾詞
    constraints = TaskConstraints()
    
    # 速度相關
    if any(word in user_input for word in ["快", "fast", "快速", "急"]):
        constraints.rate_limit = 5000
        constraints.max_requests_per_minute = 300
    
    # 隱匿相關
    if any(word in user_input for word in ["偷偷", "stealth", "隱匿", "低調", "悄悄"]):
        constraints.stealth_level = StealthLevel.HIGH
        constraints.use_proxy = True
        constraints.rate_limit = 100  # 降低速率
    
    # 深度相關
    if any(word in user_input for word in ["深入", "deep", "詳細", "全面"]):
        constraints.max_depth = 5
        constraints.max_urls = 5000
    elif any(word in user_input for word in ["簡單", "quick", "快速"]):
        constraints.max_depth = 1
        constraints.max_urls = 100
    
    context.constraints = constraints
    
    return context
