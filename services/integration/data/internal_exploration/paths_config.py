"""
AIVA Internal Exploration 路徑配置
==================================
統一管理內部探索模組的所有數據路徑

設計原則:
---------
1. 單一數據源 (SOT): 所有路徑配置集中在此文件
2. 內部 vs 外部分離:
   - Internal: AI Core 模組 (只用 Python)
   - External: Features/Scan 模組 (Python/Go/Rust/TypeScript)
3. 絕對路徑: 避免相對路徑帶來的混亂

最後更新: 2026-01-21
"""

from pathlib import Path

# ==========================================
# 常量定義
# ==========================================

# CLI 命令文件名稱常量
EXTERNAL_COMMANDS_FILENAME = "external_commands.md"

# ==========================================
# 基礎路徑定義
# ==========================================

# 專案根目錄 (AIVA-git)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Integration 模組根目錄
INTEGRATION_ROOT = PROJECT_ROOT / "services" / "integration"

# Internal Exploration 數據根目錄 (SOT)
DATA_ROOT = INTEGRATION_ROOT / "data" / "internal_exploration"

# ==========================================
# 內部模組路徑 (Internal - AI Core Only)
# ==========================================
# 內部模組只分析 Python，因為 AI Core 是純 Python

class InternalPaths:
    """AI Core 內部模組路徑配置
    
    目標: services/core/aiva_core/ (5 大模組)
    - cognitive_core
    - internal_exploration  
    - task_planning
    - core_capabilities
    - service_backbone
    """
    
    # 分析目標
    TARGET_DIR = PROJECT_ROOT / "services" / "core" / "aiva_core"
    
    # 輸出目錄
    OUTPUT_DIR = DATA_ROOT / "analysis_results" / "internal"
    
    # 分類數據 (SOT)
    CLASSIFICATION_FILE = DATA_ROOT / "internal_classification.json"
    CLASSIFICATION_SUMMARY = DATA_ROOT / "internal_classification_summary.md"
    
    # 歷史版本
    HISTORY_DIR = DATA_ROOT / "analysis_history" / "internal"
    
    # CLI 輸出
    CLI_COMMANDS_MD = INTEGRATION_ROOT / "data" / "cli_outputs" / "python" / "internal_commands.md"
    CLI_COMMANDS_JSON = INTEGRATION_ROOT / "data" / "cli_outputs" / "python" / "internal_commands.json"
    
    @classmethod
    def ensure_dirs(cls):
        """確保所有目錄存在"""
        dirs = [
            cls.OUTPUT_DIR,
            cls.HISTORY_DIR,
            cls.CLI_COMMANDS_MD.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

# ==========================================
# 外部模組路徑 (External - Multi-Language)
# ==========================================
# 外部模組支援 Python/Go/Rust/TypeScript

class ExternalPaths:
    """Features/Scan 外部模組路徑配置
    
    目標:
    - services/features/function_* (Python/Rust)
    - services/scan/*_engine (TypeScript)
    - services/features/function_authn_go (Go)
    
    ⚠️ 外部分類器需要讀取 4 個語言的分析結果:
    1. Python: services/features/function_*/analysis_results.json
    2. Rust:   services/features/function_crypto/analysis_results.json
    3. Go:     services/features/function_authn_go/analysis_results.json
    4. TypeScript: services/scan/typescript_engine/analysis_results.json
    """
    
    # 分析目標目錄
    FEATURES_DIR = PROJECT_ROOT / "services" / "features"
    SCAN_DIR = PROJECT_ROOT / "services" / "scan"
    
    # ==========================================
    # 多語言分析結果輸入路徑 (分類器讀取)
    # ==========================================
    # ⚠️ 新架構: 所有分析結果已集中到 integration/data/internal_exploration/analysis_results/external/
    # 分類器從整合目錄讀取所有語言的分析結果
    
    # 整合目錄 (SOT)
    OUTPUT_DIR = DATA_ROOT / "analysis_results" / "external"
    PYTHON_OUTPUT = OUTPUT_DIR / "python"
    GO_OUTPUT = OUTPUT_DIR / "go"
    RUST_OUTPUT = OUTPUT_DIR / "rust"
    TYPESCRIPT_OUTPUT = OUTPUT_DIR / "typescript"
    
    ANALYSIS_INPUTS = {
        "python": [
            # Features 模組 - 中高完成度 (8個)
            PYTHON_OUTPUT / "function_bizlogic.json",      # 業務邏輯 (75%)
            PYTHON_OUTPUT / "function_idor.json",          # IDOR (80%)
            PYTHON_OUTPUT / "function_info_leak.json",     # 信息洩露 (100%)
            PYTHON_OUTPUT / "function_postex.json",        # 後滲透 (50%)
            PYTHON_OUTPUT / "function_sqli.json",          # SQL注入 (95%)
            PYTHON_OUTPUT / "function_ssrf.json",          # SSRF (85%)
            PYTHON_OUTPUT / "function_web_scanner.json",   # Web掃描 (35%)
            PYTHON_OUTPUT / "function_xss.json",           # XSS (90%)
            # Scan 模組 (1個)
            PYTHON_OUTPUT / "python_engine.json",
        ],
        "go": [
            # Features 模組 - 中完成度 (1個)
            GO_OUTPUT / "function_authn_go.json",          # 認證繞過 (70%)
            # Scan 模組 (1個)
            GO_OUTPUT / "go_engine.json",
        ],
        "rust": [
            # Features 模組 - 高完成度 (1個)
            RUST_OUTPUT / "function_crypto.json",          # 密碼學 (95%)
            # Scan 模組 (1個)
            RUST_OUTPUT / "rust_engine.json",
        ],
        "typescript": [
            # Scan 模組 (1個)
            TYPESCRIPT_OUTPUT / "typescript_engine.json",
        ],
    }
    
    # ==========================================
    # 分類器輸出路徑 (執行器讀取)
    # ==========================================
    # 外部執行器只需要讀取這個文件
    
    # 分類數據 (SOT)
    CLASSIFICATION_FILE = DATA_ROOT / "external_classification.json"
    CLASSIFICATION_SUMMARY = DATA_ROOT / "external_classification_summary.md"
    
    # 歷史版本
    HISTORY_DIR = DATA_ROOT / "analysis_history" / "external"
    
    # CLI 輸出
    CLI_COMMANDS = {
        "python": INTEGRATION_ROOT / "data" / "cli_outputs" / "python" / EXTERNAL_COMMANDS_FILENAME,
        "go": INTEGRATION_ROOT / "data" / "cli_outputs" / "go" / EXTERNAL_COMMANDS_FILENAME,
        "rust": INTEGRATION_ROOT / "data" / "cli_outputs" / "rust" / EXTERNAL_COMMANDS_FILENAME,
        "typescript": INTEGRATION_ROOT / "data" / "cli_outputs" / "typescript" / EXTERNAL_COMMANDS_FILENAME,
    }
    
    @classmethod
    def ensure_dirs(cls):
        """確保所有目錄存在"""
        dirs = [
            cls.PYTHON_OUTPUT,
            cls.GO_OUTPUT,
            cls.RUST_OUTPUT,
            cls.TYPESCRIPT_OUTPUT,
            cls.HISTORY_DIR,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # CLI 輸出目錄
        for path in cls.CLI_COMMANDS.values():
            path.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# 統一路徑 (Combined)
# ==========================================

class CombinedPaths:
    """統一的分類數據路徑
    
    合併內部和外部的分類結果，供 AI 使用
    """
    
    # 最新版本指標 (軟連結概念)
    LATEST_CLASSIFICATION = DATA_ROOT / "latest_classification.json"
    
    # 合併的分類數據
    COMBINED_CLASSIFICATION = DATA_ROOT / "classification_data.json"
    COMBINED_SUMMARY = DATA_ROOT / "classification_summary.md"
    
    # 自我修復數據
    SELF_HEALING_DIR = DATA_ROOT / "self_healing"
    DIAGNOSIS_FILE = SELF_HEALING_DIR / "latest_diagnosis.json"
    REPAIR_LOG = SELF_HEALING_DIR / "repair_log.json"

# ==========================================
# 舊路徑映射 (向後相容)
# ==========================================
# 這些路徑可能被舊代碼引用，保留映射以便遷移

LEGACY_PATHS = {
    # 舊路徑 -> 新路徑
    "services_classification_v9_new/classification_data.json": CombinedPaths.COMBINED_CLASSIFICATION,
    "enriched_classification.json": CombinedPaths.COMBINED_CLASSIFICATION,
    "latest_classification.json": CombinedPaths.LATEST_CLASSIFICATION,
}

# ==========================================
# 輔助函數
# ==========================================

def get_project_root() -> Path:
    """獲取專案根目錄"""
    return PROJECT_ROOT

def get_data_root() -> Path:
    """獲取數據根目錄"""
    return DATA_ROOT

def ensure_all_dirs():
    """確保所有數據目錄存在"""
    InternalPaths.ensure_dirs()
    ExternalPaths.ensure_dirs()
    CombinedPaths.SELF_HEALING_DIR.mkdir(parents=True, exist_ok=True)

def resolve_legacy_path(legacy_path: str) -> Path | None:
    """解析舊路徑到新路徑"""
    return LEGACY_PATHS.get(legacy_path)

def print_paths_summary():
    """打印路徑配置摘要"""
    print("=" * 60)
    print("AIVA Internal Exploration 路徑配置")
    print("=" * 60)
    print("\n[專案根目錄]")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    
    print("\n[內部模組路徑] (AI Core - Python Only)")
    print(f"  TARGET_DIR: {InternalPaths.TARGET_DIR}")
    print(f"  OUTPUT_DIR: {InternalPaths.OUTPUT_DIR}")
    print(f"  CLASSIFICATION: {InternalPaths.CLASSIFICATION_FILE}")
    
    print("\n[外部模組路徑] (Features/Scan - Multi-Language)")
    print(f"  FEATURES_DIR: {ExternalPaths.FEATURES_DIR}")
    print(f"  SCAN_DIR: {ExternalPaths.SCAN_DIR}")
    print(f"  CLASSIFICATION: {ExternalPaths.CLASSIFICATION_FILE}")
    
    print("\n[統一路徑]")
    print(f"  LATEST: {CombinedPaths.LATEST_CLASSIFICATION}")
    print(f"  COMBINED: {CombinedPaths.COMBINED_CLASSIFICATION}")
    
    print("=" * 60)

# ==========================================
# 主程式入口
# ==========================================

if __name__ == "__main__":
    print_paths_summary()
    
    # 確保所有目錄存在
    ensure_all_dirs()
    print("\n✅ 所有數據目錄已確認/創建")
