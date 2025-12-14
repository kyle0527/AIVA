"""統一路徑配置 - AIVA 系統輸出路徑管理

此模組集中管理所有 AIVA 服務的輸出路徑，確保數據統一存儲在 integration 模組中。

路徑結構:
    services/integration/data/
    ├── internal_exploration/    # Internal Exploration 分析數據
    │   ├── analysis_results/    # Python/TS/Go/Rust 工具輸出
    │   ├── analysis_history/    # 版本化歷史記錄
    │   └── self_healing/        # 自我診斷報告
    ├── attack_paths/            # 攻擊路徑圖數據
    ├── experiences/             # 經驗記錄數據庫
    └── training/                # 訓練數據集

使用方式:
    from aiva_common.config.paths import (
        INTERNAL_EXPLORATION_DATA,
        ANALYSIS_RESULTS_DIR,
        ensure_directories
    )
    
    # 確保目錄存在
    ensure_directories()
    
    # 使用路徑
    output_file = ANALYSIS_RESULTS_DIR / "my_analysis.json"
"""

import os
from pathlib import Path
from typing import List

# ==================== 專案根路徑 ====================

# 當前文件位置: services/aiva_common/config/paths.py
# parent(0): config/
# parent(1): aiva_common/
# parent(2): services/
# parent(3): 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Services 根目錄
SERVICES_ROOT = PROJECT_ROOT / "services"

# ==================== Integration 模組路徑 ====================

# Integration 模組根目錄
INTEGRATION_ROOT = SERVICES_ROOT / "integration"

# Integration 數據根目錄
INTEGRATION_DATA_ROOT = INTEGRATION_ROOT / "data"

# ==================== Internal Exploration 輸出路徑 ====================

# Internal Exploration 數據根目錄
INTERNAL_EXPLORATION_DATA = INTEGRATION_DATA_ROOT / "internal_exploration"

# 細分路徑
ANALYSIS_RESULTS_DIR = INTERNAL_EXPLORATION_DATA / "analysis_results"      # 當前分析結果
ANALYSIS_HISTORY_DIR = INTERNAL_EXPLORATION_DATA / "analysis_history"      # 版本化歷史
SELF_HEALING_DIR = INTERNAL_EXPLORATION_DATA / "self_healing"              # 自我診斷報告

# ==================== 其他整合模組數據路徑 ====================

# 攻擊路徑圖數據
ATTACK_PATHS_DIR = INTEGRATION_DATA_ROOT / "attack_paths"

# 經驗記錄數據庫
EXPERIENCES_DIR = INTEGRATION_DATA_ROOT / "experiences"

# 訓練數據集
TRAINING_DATA_DIR = INTEGRATION_DATA_ROOT / "training"

# ==================== 環境變量控制 ====================

# 是否使用整合路徑（默認啟用）
USE_INTEGRATED_PATHS = os.getenv("AIVA_USE_INTEGRATED_PATHS", "true").lower() == "true"

# 向後兼容：如果禁用，使用舊路徑
if not USE_INTEGRATED_PATHS:
    # 舊路徑配置（向後兼容）
    CORE_ROOT = SERVICES_ROOT / "core" / "aiva_core"
    ANALYSIS_RESULTS_DIR = CORE_ROOT / "analysis_results"
    ANALYSIS_HISTORY_DIR = CORE_ROOT / "internal_exploration" / "analysis_history"
    SELF_HEALING_DIR = CORE_ROOT / "analysis_results"  # 舊版輸出到同一位置

# ==================== 輔助函數 ====================

def ensure_directories() -> None:
    """確保所有輸出目錄存在
    
    創建所有必要的輸出目錄，如果目錄已存在則不做任何操作。
    建議在任何寫入操作前調用此函數。
    """
    directories = [
        INTERNAL_EXPLORATION_DATA,
        ANALYSIS_RESULTS_DIR,
        ANALYSIS_HISTORY_DIR,
        SELF_HEALING_DIR,
        ATTACK_PATHS_DIR,
        EXPERIENCES_DIR,
        TRAINING_DATA_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_analysis_output_dir(tool_name: str = "default") -> Path:
    """獲取特定工具的輸出目錄
    
    Args:
        tool_name: 工具名稱（python/typescript/go/rust/self_healing）
        
    Returns:
        工具專用輸出目錄路徑
    """
    if tool_name == "self_healing":
        return SELF_HEALING_DIR
    else:
        # 其他工具統一輸出到 analysis_results
        tool_dir = ANALYSIS_RESULTS_DIR / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        return tool_dir


def get_version_dir(version: int) -> Path:
    """獲取指定版本的歷史目錄
    
    Args:
        version: 版本號
        
    Returns:
        版本目錄路徑（例如 analysis_history/v1）
    """
    version_dir = ANALYSIS_HISTORY_DIR / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def get_latest_version_dir() -> Path:
    """獲取最新版本的歷史目錄
    
    Returns:
        最新版本目錄路徑，如果不存在則返回 v1
    """
    if not ANALYSIS_HISTORY_DIR.exists():
        return get_version_dir(1)
    
    # 查找所有版本目錄
    versions = []
    for item in ANALYSIS_HISTORY_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v") and item.name[1:].isdigit():
            versions.append(int(item.name[1:]))
    
    if not versions:
        return get_version_dir(1)
    
    return get_version_dir(max(versions))


def list_all_output_dirs() -> List[Path]:
    """列出所有輸出目錄
    
    Returns:
        所有輸出目錄的路徑列表
    """
    return [
        INTERNAL_EXPLORATION_DATA,
        ANALYSIS_RESULTS_DIR,
        ANALYSIS_HISTORY_DIR,
        SELF_HEALING_DIR,
        ATTACK_PATHS_DIR,
        EXPERIENCES_DIR,
        TRAINING_DATA_DIR,
    ]


def get_config_summary() -> dict:
    """獲取配置摘要
    
    Returns:
        配置摘要字典
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "services_root": str(SERVICES_ROOT),
        "integration_root": str(INTEGRATION_ROOT),
        "use_integrated_paths": USE_INTEGRATED_PATHS,
        "output_directories": {
            "internal_exploration": str(INTERNAL_EXPLORATION_DATA),
            "analysis_results": str(ANALYSIS_RESULTS_DIR),
            "analysis_history": str(ANALYSIS_HISTORY_DIR),
            "self_healing": str(SELF_HEALING_DIR),
            "attack_paths": str(ATTACK_PATHS_DIR),
            "experiences": str(EXPERIENCES_DIR),
            "training": str(TRAINING_DATA_DIR),
        }
    }


# ==================== 路徑導出配置文件 ====================

def export_paths_config(format: str = "json") -> str:
    """導出路徑配置供其他語言使用
    
    Args:
        format: 輸出格式（json/env/typescript/go/rust）
        
    Returns:
        配置文件內容
    """
    import json
    
    config = get_config_summary()
    
    if format == "json":
        return json.dumps(config, indent=2, ensure_ascii=False)
    
    elif format == "env":
        # 環境變量格式
        lines = [
            f"AIVA_PROJECT_ROOT={config['project_root']}",
            f"AIVA_SERVICES_ROOT={config['services_root']}",
            f"AIVA_INTEGRATION_ROOT={config['integration_root']}",
            f"AIVA_USE_INTEGRATED_PATHS={config['use_integrated_paths']}",
        ]
        for key, value in config['output_directories'].items():
            env_key = f"AIVA_{key.upper()}_DIR"
            lines.append(f"{env_key}={value}")
        return "\n".join(lines)
    
    elif format == "typescript":
        # TypeScript 配置格式
        lines = [
            "// Auto-generated paths configuration",
            "export const PATHS = {",
            f"  projectRoot: '{config['project_root']}',",
            f"  servicesRoot: '{config['services_root']}',",
            f"  integrationRoot: '{config['integration_root']}',",
            "  outputDirectories: {",
        ]
        for key, value in config['output_directories'].items():
            lines.append(f"    {key.replace('_', '')}: '{value}',")
        lines.append("  }")
        lines.append("};")
        return "\n".join(lines)
    
    elif format == "go":
        # Go 配置格式
        lines = [
            "// Auto-generated paths configuration",
            "package config",
            "",
            "const (",
            f'    ProjectRoot = "{config["project_root"]}"',
            f'    ServicesRoot = "{config["services_root"]}"',
            f'    IntegrationRoot = "{config["integration_root"]}"',
        ]
        for key, value in config['output_directories'].items():
            const_name = "".join(word.capitalize() for word in key.split("_")) + "Dir"
            lines.append(f'    {const_name} = "{value}"')
        lines.append(")")
        return "\n".join(lines)
    
    elif format == "rust":
        # Rust 配置格式
        lines = [
            "// Auto-generated paths configuration",
            "use std::path::PathBuf;",
            "",
            "pub struct Paths {",
        ]
        for key in config['output_directories'].keys():
            field_name = key.replace('_', '_')
            lines.append(f"    pub {field_name}_dir: PathBuf,")
        lines.append("}")
        lines.append("")
        lines.append("impl Default for Paths {")
        lines.append("    fn default() -> Self {")
        lines.append("        Self {")
        for key, value in config['output_directories'].items():
            field_name = key.replace('_', '_')
            lines.append(f'            {field_name}_dir: PathBuf::from("{value}"),')
        lines.append("        }")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)
    
    else:
        return json.dumps(config, indent=2, ensure_ascii=False)


# ==================== 初始化 ====================

# 模組導出後用戶可手動調用 ensure_directories() 創建目錄

# ==================== 模組導出 ====================

__all__ = [
    # 主要路徑常量
    "PROJECT_ROOT",
    "SERVICES_ROOT",
    "INTEGRATION_ROOT",
    "INTEGRATION_DATA_ROOT",
    "INTERNAL_EXPLORATION_DATA",
    "ANALYSIS_RESULTS_DIR",
    "ANALYSIS_HISTORY_DIR",
    "SELF_HEALING_DIR",
    "ATTACK_PATHS_DIR",
    "EXPERIENCES_DIR",
    "TRAINING_DATA_DIR",
    # 配置選項
    "USE_INTEGRATED_PATHS",
    # 輔助函數
    "ensure_directories",
    "get_analysis_output_dir",
    "get_version_dir",
    "get_latest_version_dir",
    "list_all_output_dirs",
    "get_config_summary",
    "export_paths_config",
]
