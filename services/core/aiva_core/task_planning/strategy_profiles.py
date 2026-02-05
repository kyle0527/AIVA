"""攻擊策略配置檔案 - 統一策略架構 v2.0

設計原則：
- 環境無關設計：策略不區分 sandbox/production
- 經驗可遷移：學習結果直接適用於任何目標
- 基於目標屬性：根據目標敏感度調整，而非環境

架構理念（參考 Transfer Learning 最佳實踐）：
> "學了能否用" 的關鍵是避免 Domain Gap
> 策略參數應基於目標特性，而非執行環境

四個核心維度（任何環境通用）：
1. intensity: 攻擊強度
2. stealth: 隱蔽性  
3. coverage: 技術覆蓋
4. persistence: 執行韌性
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AttackStrategyProfile:
    """攻擊策略配置 - 環境無關設計
    
    所有屬性基於「目標特性」而非「執行環境」設計
    學習結果可直接遷移到任何場景
    """
    name: str
    description: str
    
    # === 核心維度 (0.0 - 1.0，任何環境通用) ===
    intensity: float  # 攻擊強度：決定 payload 數量和執行速度
    stealth: float    # 隱蔽性：決定避開偵測的努力程度
    coverage: float   # 技術覆蓋：決定測試技術的廣度
    persistence: float  # 執行韌性：決定失敗後的重試策略
    
    # === 衍生配置（由核心維度計算） ===
    techniques: List[str] = field(default_factory=list)
    payload_count: int = 10
    concurrent_attacks: int = 1
    timing_delay: float = 1.0
    user_agent_rotation: bool = True
    evasion_techniques: List[str] = field(default_factory=list)
    
    # === 元數據 ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 移除 risk_level - 這是環境依賴的，不應在策略中定義
    # 風險由目標敏感度決定，不由策略決定


# === 策略配置 (基於4維度) ===
# 由核心維度衍生，而非硬編碼

def _derive_config_from_dimensions(
    intensity: float, stealth: float, coverage: float, persistence: float
) -> Dict[str, Any]:
    """從4核心維度計算衍生配置
    
    Note: coverage 參數用於計算測試覆蓋範圍配置
    """
    return {
        "payload_count": int(5 + intensity * 45),  # 5-50
        "concurrent_attacks": max(1, int(intensity * 5)),  # 1-5
        "timing_delay": max(0.1, 2.0 - intensity * 1.9),  # 0.1-2.0
        "user_agent_rotation": stealth > 0.3,
        "max_requests_per_minute": int(10 + (1 - stealth) * 90),  # 10-100
        "retry_on_failure": persistence > 0.5,
        "max_retries": int(1 + persistence * 4),  # 1-5
        "test_vector_count": int(3 + coverage * 17),  # 3-20
    }


STRATEGY_PROFILES = {
    "conservative": AttackStrategyProfile(
        name="conservative",
        description="保守策略：高隱蔽、低強度，適合敏感目標或需謹慎的場景",
        intensity=0.3,
        stealth=0.9,
        coverage=0.4,
        persistence=0.3,
        techniques=[
            "basic_sqli",
            "simple_xss",
            "passive_recon",
            "cookie_inspection"
        ],
        payload_count=5,
        concurrent_attacks=1,
        timing_delay=2.0,
        user_agent_rotation=True,
        evasion_techniques=["encoding", "case_variation"],
        metadata={
            "max_requests_per_minute": 10,
            "avoid_techniques": ["time_based_blind", "heavy_fuzzing"]
        }
    ),
    
    "balanced": AttackStrategyProfile(
        name="balanced",
        description="平衡策略：中等強度和隱蔽性，適合大多數測試場景",
        intensity=0.6,
        stealth=0.5,
        coverage=0.7,
        persistence=0.5,
        techniques=[
            "all_sqli",
            "all_xss",
            "csrf",
            "lfi",
            "active_recon",
            "parameter_fuzzing"
        ],
        payload_count=20,
        concurrent_attacks=3,
        timing_delay=0.5,
        user_agent_rotation=True,
        evasion_techniques=["encoding", "case_variation", "comment_injection"],
        metadata={
            "max_requests_per_minute": 30,
            "enable_smart_fuzzing": True
        }
    ),
    
    "aggressive": AttackStrategyProfile(
        name="aggressive",
        description="激進策略：高強度、高覆蓋，適合全面評估或時間緊迫的場景",
        intensity=0.9,
        stealth=0.2,
        coverage=0.95,
        persistence=0.8,
        techniques=[
            "blind_sqli",
            "time_based_sqli",
            "advanced_xss",
            "xxe",
            "ssrf",
            "deserialization",
            "template_injection",
            "comprehensive_fuzzing"
        ],
        payload_count=50,
        concurrent_attacks=5,
        timing_delay=0.1,
        user_agent_rotation=True,
        evasion_techniques=[
            "encoding",
            "fragmentation",
            "polyglot",
            "unicode",
            "double_encoding"
        ],
        metadata={
            "max_requests_per_minute": 100,
            "enable_deep_fuzzing": True,
            "test_error_based": True
        }
    )
}


def get_strategy_profile(variant: int = 0) -> AttackStrategyProfile:
    """根據變體索引獲取策略配置
    
    注意：策略選擇基於「目標敏感度」而非「執行環境」
    這樣學習結果才能直接遷移
    
    Args:
        variant: 策略變體索引 (0=保守, 1=平衡, 2=激進)
        
    Returns:
        對應的策略配置
    """
    strategy_map = {
        0: "conservative",
        1: "balanced",
        2: "aggressive"
    }
    
    strategy_name = strategy_map.get(variant, "balanced")
    return STRATEGY_PROFILES[strategy_name]


def get_strategy_by_target_sensitivity(sensitivity: float) -> AttackStrategyProfile:
    """根據目標敏感度自動選擇策略（環境無關）
    
    這是新的推薦用法，完全基於目標屬性選擇策略
    學習經驗可直接遷移，不存在 sandbox/production 的 domain gap
    
    Args:
        sensitivity: 目標敏感度 (0.0-1.0)
            - 0.0-0.3: 低敏感 (測試環境、已授權靶場)
            - 0.3-0.7: 中敏感 (預生產、有監控的環境)
            - 0.7-1.0: 高敏感 (生產環境、關鍵系統)
    
    Returns:
        根據敏感度自動選擇的策略
    """
    if sensitivity <= 0.3:
        return STRATEGY_PROFILES["aggressive"]
    elif sensitivity <= 0.7:
        return STRATEGY_PROFILES["balanced"]
    else:
        return STRATEGY_PROFILES["conservative"]


def create_custom_strategy(
    intensity: float,
    stealth: float,
    coverage: float,
    persistence: float,
    name: str = "custom"
) -> AttackStrategyProfile:
    """創建自定義策略（基於4核心維度）
    
    允許用戶創建任意組合的策略，所有參數環境無關
    
    Args:
        intensity: 攻擊強度 (0.0-1.0)
        stealth: 隱蔽性 (0.0-1.0)
        coverage: 技術覆蓋 (0.0-1.0)
        persistence: 執行韌性 (0.0-1.0)
        name: 策略名稱
        
    Returns:
        自定義策略配置
    """
    # 根據維度自動計算衍生配置
    config = _derive_config_from_dimensions(intensity, stealth, coverage, persistence)
    
    # 根據覆蓋率選擇技術
    all_techniques = [
        "basic_sqli", "simple_xss", "passive_recon",  # coverage <= 0.3
        "all_sqli", "all_xss", "csrf", "lfi",  # coverage <= 0.6
        "blind_sqli", "time_based_sqli", "xxe", "ssrf", "deserialization"  # coverage > 0.6
    ]
    technique_count = int(len(all_techniques) * coverage)
    techniques = all_techniques[:max(3, technique_count)]
    
    # 根據隱蔽性選擇規避技術
    all_evasions = ["encoding", "case_variation", "comment_injection", 
                   "fragmentation", "polyglot", "unicode", "double_encoding"]
    evasion_count = int(len(all_evasions) * stealth)
    evasion_techniques = all_evasions[:max(2, evasion_count)]
    
    return AttackStrategyProfile(
        name=name,
        description=f"自定義策略 (I={intensity:.1f}, S={stealth:.1f}, C={coverage:.1f}, P={persistence:.1f})",
        intensity=intensity,
        stealth=stealth,
        coverage=coverage,
        persistence=persistence,
        techniques=techniques,
        evasion_techniques=evasion_techniques,
        **config
    )


def get_all_profiles() -> Dict[str, AttackStrategyProfile]:
    """獲取所有策略配置"""
    return STRATEGY_PROFILES.copy()


# === 統一策略選擇指南 (v2.0) ===
STRATEGY_SELECTION_GUIDE = """
統一策略選擇指南 v2.0 - 環境無關設計

核心原則：
- 策略選擇基於「目標敏感度」，而非「執行環境」
- 所有學習經驗可直接遷移，無 Domain Gap
- 沒有 sandbox vs production 的區分

四維度模型：
┌─────────────┬──────────────────────────────────────────┐
│   維度       │ 說明                                     │
├─────────────┼──────────────────────────────────────────┤
│ intensity   │ 攻擊強度：payload 數量、執行速度          │
│ stealth     │ 隱蔽性：避開 WAF/IDS、減少日誌痕跡        │
│ coverage    │ 技術覆蓋：測試漏洞類型的廣度              │
│ persistence │ 執行韌性：遇到錯誤/失敗時的重試策略       │
└─────────────┴──────────────────────────────────────────┘

策略選擇（基於目標敏感度）：

目標敏感度 0.0-0.3 (低敏感)：
  → aggressive 策略
  例：授權靶場、測試環境、CTF

目標敏感度 0.3-0.7 (中敏感)：
  → balanced 策略
  例：預生產環境、有監控的系統

目標敏感度 0.7-1.0 (高敏感)：
  → conservative 策略
  例：生產環境、關鍵基礎設施

使用方式：
  # 方式1：按敏感度自動選擇
  profile = get_strategy_by_target_sensitivity(0.8)
  
  # 方式2：自定義4維度
  profile = create_custom_strategy(
      intensity=0.5, stealth=0.8, coverage=0.6, persistence=0.4
  )

學習遷移保證：
  - 在任何目標上學到的經驗，可直接用於相同敏感度的其他目標
  - 無需考慮「靶場學的能否用於實戰」
  - 因為策略參數與執行環境完全解耦
"""
