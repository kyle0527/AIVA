"""AI Commander 類型定義

包含 AI 任務類型和組件類型枚舉，以及 CLI 命令模型導入
"""

from enum import Enum

# 導入新架構的 CLI 命令模型（階段 1：CLI 參數包驅動架構）
from aiva_common.schemas.commands import CLICommand

# 重新導出 CLICommand 供其他模組使用
__all__ = ["AITaskType", "AIComponent", "CLICommand"]


class AITaskType(str, Enum):
    """AI 任務類型"""

    # 決策類
    ATTACK_PLANNING = "attack_planning"  # 攻擊計畫生成
    STRATEGY_DECISION = "strategy_decision"  # 策略決策
    RISK_ASSESSMENT = "risk_assessment"  # 風險評估

    # 執行類
    VULNERABILITY_DETECTION = "vulnerability_detection"  # 漏洞檢測
    EXPLOIT_EXECUTION = "exploit_execution"  # 漏洞利用
    CODE_ANALYSIS = "code_analysis"  # 代碼分析
    ATTACK_EXECUTION = "attack_execution"  # 攻擊執行
    TWO_PHASE_SCAN = "two_phase_scan"  # 兩階段掃描

    # 學習類
    EXPERIENCE_LEARNING = "experience_learning"  # 經驗學習
    MODEL_TRAINING = "model_training"  # 模型訓練
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知識檢索
    CAPABILITY_QUERY = "capability_query"  # 能力查詢 (v11.0)

    # 協調類
    MULTI_LANG_COORDINATION = "multi_lang_coordination"  # 多語言協調
    TASK_DELEGATION = "task_delegation"  # 任務委派


class AIComponent(str, Enum):
    """AI 組件類型"""

    DECISION_ENGINE_5M = "decision_engine_5m"  # 5M 參數決策引擎
    RAG_ENGINE = "rag_engine"  # RAG 引擎
    TRAINING_SYSTEM = "training_system"  # 訓練系統
    MULTILANG_COORDINATOR = "multilang_coordinator"  # 多語言協調器

    # 語言專屬 AI
    GO_AI_MODULE = "go_ai_module"  # Go AI 模組
    RUST_AI_MODULE = "rust_ai_module"  # Rust AI 模組
    TS_AI_MODULE = "ts_ai_module"  # TypeScript AI 模組
