"""Decision Schema - 決策數據合約

定義 cognitive_core 與 task_planning 之間的決策交接數據結構

Compliance Note:
- 創建日期: 2025-11-16
- 目的: 解決問題三「決策交接不明確」
- 符合架構原則: 使用 aiva_common 統一數據合約定義
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """高階意圖類型"""
    
    # 漏洞測試類
    TEST_VULNERABILITY = "test_vulnerability"
    EXPLOIT_TARGET = "exploit_target"
    
    # 偵察掃描類
    SCAN_SURFACE = "scan_surface"
    ENUMERATE_ASSETS = "enumerate_assets"
    
    # 分析類
    ANALYZE_RESULTS = "analyze_results"
    CORRELATE_FINDINGS = "correlate_findings"
    
    # 權限提升類
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"


class TargetInfo(BaseModel):
    """目標信息"""
    
    target_id: str = Field(description="目標唯一標識")
    target_type: str = Field(description="目標類型: url, ip, domain, etc.")
    target_value: str = Field(description="目標值")
    context: dict[str, Any] = Field(default_factory=dict, description="目標上下文信息")


class DecisionConstraints(BaseModel):
    """決策約束條件"""
    
    time_limit: int | None = Field(default=None, description="時間限制（秒）")
    risk_level: str = Field(default="medium", description="風險等級: low, medium, high")
    stealth_mode: bool = Field(default=False, description="隱蔽模式")
    resource_limits: dict[str, Any] = Field(default_factory=dict, description="資源限制")
    forbidden_actions: list[str] = Field(default_factory=list, description="禁止的行為")


class HighLevelIntent(BaseModel):
    """高階意圖 - cognitive_core 的決策輸出
    
    這是「大腦」輸出給「規劃器」的數據合約
    
    職責劃分：
    - cognitive_core (大腦): 決定「做什麼」(What) 和「為什麼」(Why)
    - task_planning (規劃器): 決定「怎麼做」(How) - 生成 AST
    """
    
    # 基本信息
    intent_id: str = Field(default_factory=lambda: str(uuid4()), description="意圖唯一標識")
    intent_type: IntentType = Field(description="意圖類型")
    
    # 目標信息
    target: TargetInfo = Field(description="測試目標")
    
    # 決策參數
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="意圖參數 (例如: vulnerability_type, test_depth, etc.)"
    )
    
    # 約束條件
    constraints: DecisionConstraints = Field(
        default_factory=DecisionConstraints,
        description="執行約束"
    )
    
    # 決策元信息
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="決策信心度 (0-1)"
    )
    reasoning: str = Field(default="", description="決策推理過程")
    alternatives: list[dict[str, Any]] = Field(
        default_factory=list,
        description="備選方案"
    )
    
    # 上下文信息
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="決策上下文 (例如: previous_results, discovered_vulns, etc.)"
    )
    
    # 元數據
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="enhanced_decision_agent", description="決策來源")
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent_id": "intent_123",
                "intent_type": "test_vulnerability",
                "target": {
                    "target_id": "target_001",
                    "target_type": "url",
                    "target_value": "https://example.com/login",
                    "context": {"form_fields": ["username", "password"]}
                },
                "parameters": {
                    "vulnerability_type": "sql_injection",
                    "test_depth": "comprehensive",
                    "payload_count": 50
                },
                "constraints": {
                    "time_limit": 300,
                    "risk_level": "medium",
                    "stealth_mode": False
                },
                "confidence": 0.85,
                "reasoning": "發現 login 表單缺乏輸入驗證，高機率存在 SQL 注入漏洞"
            }
        }


class DecisionToASTContract(BaseModel):
    """決策到 AST 的轉換合約
    
    定義 task_planning 如何將 HighLevelIntent 轉換為 AttackPlan (AST)
    """
    
    intent: HighLevelIntent = Field(description="輸入：高階意圖")
    
    # 規劃器生成的 AST 信息
    ast_plan_id: str | None = Field(default=None, description="生成的 AST 計劃 ID")
    ast_generated: bool = Field(default=False, description="AST 是否已生成")
    
    # 轉換元數據
    transformation_notes: str = Field(default="", description="轉換說明")
    planner_decisions: dict[str, Any] = Field(
        default_factory=dict,
        description="規劃器的具體決策 (例如: tool_selection, step_ordering, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent": {
                    "intent_id": "intent_123",
                    "intent_type": "test_vulnerability",
                    "target": {
                        "target_id": "target_001",
                        "target_type": "url",
                        "target_value": "https://example.com/login"
                    },
                    "parameters": {
                        "vulnerability_type": "sql_injection"
                    }
                },
                "ast_plan_id": "plan_456",
                "ast_generated": True,
                "transformation_notes": "生成了 5 步 SQL 注入測試計劃",
                "planner_decisions": {
                    "tool_selection": "sqlmap",
                    "step_count": 5,
                    "estimated_duration": 180
                }
            }
        }


class DecisionFeedback(BaseModel):
    """決策反饋 - 用於外部學習閉環
    
    執行完成後，將結果反饋給 cognitive_core 用於學習
    """
    
    intent_id: str = Field(description="原始意圖 ID")
    plan_id: str = Field(description="執行的計劃 ID")
    
    # 執行結果
    success: bool = Field(description="執行是否成功")
    findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="發現的漏洞或結果"
    )
    
    # 偏差分析
    deviations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="計劃 vs 實際執行的偏差"
    )
    
    # 學習信號
    should_adjust_confidence: bool = Field(
        default=False,
        description="是否需要調整決策信心度"
    )
    should_retrain: bool = Field(
        default=False,
        description="是否需要觸發模型訓練"
    )
    
    # 元數據
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time: float = Field(description="執行時間（秒）")
    
    class Config:
        json_schema_extra = {
            "example": {
                "intent_id": "intent_123",
                "plan_id": "plan_456",
                "success": True,
                "findings": [
                    {
                        "vulnerability_type": "sql_injection",
                        "severity": "high",
                        "location": "/login"
                    }
                ],
                "deviations": [
                    {
                        "expected": "5_steps",
                        "actual": "3_steps",
                        "reason": "某些步驟跳過"
                    }
                ],
                "should_adjust_confidence": True,
                "should_retrain": False,
                "execution_time": 145.3
            }
        }
