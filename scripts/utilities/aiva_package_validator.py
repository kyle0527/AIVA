# -*- coding: utf-8 -*-
"""
AIVA Package Validator (更詳盡版本)

這個腳本用於驗證 AIVA 專案的完整結構和必要文件是否齊全，
包含多語言模組、核心服務、工具和設定檔。
它可以作為 CI/CD 流程的一部分，或在開發環境中手動運行以確保一致性。

執行方式：
在專案根目錄下執行 `python scripts/validation/aiva_package_validator.py`
"""

import os
import sys
from pathlib import Path
import configparser
import json
import yaml
import subprocess # 用於檢查 Go/Rust/Node 環境 (可選)

# --- 設定 ---

# 將專案根目錄添加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# --- 需要檢查的核心目錄 (相對於 PROJECT_ROOT) ---
# (加入了更細緻的 services 子目錄)
REQUIRED_DIRS = [
    Path("api"), Path("api/routers"),
    Path("config"),
    Path("data"), Path("data/ai_commander/knowledge"),
    Path("docs"), Path("docs/ARCHITECTURE"), Path("docs/guides"), Path("docs/plans"),
    Path("docker"), Path("docker/initdb"),
    Path("examples"),
    Path("logs"),
    Path("reports"), Path("reports/security"), Path("reports/connectivity"),
    Path("schemas"), Path("schemas/crosslang"),
    Path("scripts"), Path("scripts/deployment"), Path("scripts/maintenance"), Path("scripts/setup"), Path("scripts/testing"), Path("scripts/validation"),
    Path("services"),
    Path("services/aiva_common"), Path("services/aiva_common/schemas"), Path("services/aiva_common/enums"), Path("services/aiva_common/tools"), Path("services/aiva_common/utils"),
    Path("services/core"), Path("services/core/aiva_core"), Path("services/core/aiva_core/ai_engine"), Path("services/core/aiva_core/messaging"), Path("services/core/aiva_core/planner"), Path("services/core/aiva_core/storage"), Path("services/core/aiva_core/ui_panel"),
    Path("services/scan"), Path("services/scan/aiva_scan"), Path("services/scan/aiva_scan_node"), Path("services/scan/info_gatherer_rust"),
    Path("services/integration"), Path("services/integration/aiva_integration"), Path("services/integration/aiva_integration/analysis"), Path("services/integration/aiva_integration/attack_path_analyzer"), Path("services/integration/aiva_integration/reception"),
    Path("services/features"), Path("services/features/base"), Path("services/features/common"), Path("services/features/function_sqli"), Path("services/features/function_xss"), Path("services/features/function_ssrf"), Path("services/features/function_idor"), Path("services/features/function_sca_go"), Path("services/features/function_sast_rust"), # ... 可根據實際情況添加更多 features
    Path("tests"),
    Path("test_data"), Path("test_data/database"),
    Path("tools"), Path("tools/aiva-contracts-tooling"), Path("tools/aiva-enums-plugin"), Path("tools/aiva-schemas-plugin"), Path("tools/automation"), Path("tools/development"), Path("tools/schema"),
    Path("web"), Path("web/js"),
]

# --- 需要檢查的核心文件 (相對於 PROJECT_ROOT) ---
REQUIRED_FILES = [
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("ruff.toml"),
    Path(".gitignore"),
    Path(".env.example"),
    Path("aiva_launcher.py"),
    Path("config/settings.py"),
    Path("services/aiva_common/schemas/base.py"),
    Path("services/aiva_common/enums/common.py"),
    Path("services/aiva_common/mq.py"), # 核心消息隊列
    Path("services/core/aiva_core/app.py"), # Core 服務入口
    Path("services/core/aiva_core/ai_engine/bio_neuron_core.py"), # AI 核心
    Path("services/scan/unified_scan_engine.py"), # Scan 服務核心
    Path("services/integration/aiva_integration/app.py"), # Integration 服務入口
    Path("services/features/feature_step_executor.py"), # Features 執行器
    Path("docker/docker-compose.yml"),
    Path("api/main.py"), # API 入口
    Path("web/index.html"), # Web UI 入口
]

# --- 需要檢查的多語言建置產物 (相對於 PROJECT_ROOT) ---
# (路徑根據實際情況可能需要調整)
REQUIRED_BUILD_ARTIFACTS = [
    Path("services/features/function_sca_go/worker.exe"),
    Path("services/features/function_ssrf_go/worker.exe"),
    Path("services/features/function_cspm_go/worker.exe"),
    # Path("services/scan/info_gatherer_rust/target/release/info_gatherer_rust"), # Rust 範例，實際路徑依 build 指令而定
    # Path("services/features/function_sast_rust/target/release/function_sast_rust"), # Rust 範例
    Path("services/scan/aiva_scan_node/dist/index.js"), # TypeScript 編譯後範例
]

# --- 需要檢查的 Schema 文件 (相對於 PROJECT_ROOT / schemas) ---
REQUIRED_SCHEMA_FILES = [
    Path("aiva_schemas.json"),
    Path("aiva_schemas.go"),
    Path("aiva_schemas.rs"),
    Path("aiva_schemas.d.ts"),
    Path("enums.ts"),
    Path("crosslang/aiva_crosslang.proto"), # Protobuf 定義
]

# --- Python 包 __init__.py 檢查列表 ---
# (遞迴檢查指定的 Python 包目錄)
PYTHON_PACKAGE_DIRS = [
    Path("api"),
    Path("services"), # services 本身也需要
    Path("services/aiva_common"),
    Path("services/core"),
    Path("services/scan"),
    Path("services/integration"),
    Path("services/features"),
    Path("tests"),
    Path("tools"), # 部分 tools 下的目錄也是 python 包
]

# --- 驗證函數 ---

def check_path(path_to_check: Path, is_dir: bool = False, optional: bool = False) -> bool:
    """檢查指定路徑是否存在以及類型是否正確"""
    full_path = PROJECT_ROOT / path_to_check
    exists = full_path.exists()
    correct_type = full_path.is_dir() if is_dir else full_path.is_file()
    prefix = "[可選] " if optional else ""

    if exists and correct_type:
        print(f"✅ {prefix}[存在] {path_to_check}")
        return True
    elif not exists:
        if optional:
            print(f"⚠️ {prefix}[缺失] {path_to_check} (但不影響驗證結果)")
            return True # 可選項目缺失不算失敗
        else:
            print(f"❌ {prefix}[缺失] {path_to_check}")
            return False
    else:
        type_str = "目錄" if is_dir else "文件"
        actual_type = "文件" if full_path.is_file() else "目錄"
        # 類型錯誤通常比缺失更嚴重，即使是可選的也標記為錯誤
        print(f"❌ {prefix}[類型錯誤] 應為 {type_str}，但找到的是 {actual_type}: {path_to_check}")
        return False

def validate_pyproject() -> bool:
    """驗證 pyproject.toml 是否包含必要的部分 (使用 tomli 增強)"""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        print("❌ [配置錯誤] pyproject.toml 文件不存在")
        return False

    try:
        # 嘗試導入 tomli (Python 3.11+內建 tomllib)
        try:
            import tomli as tomllib
        except ImportError:
            try:
                import tomllib # Python 3.11+
            except ImportError:
                tomllib = None

        if tomllib:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            # 檢查關鍵部分
            if "project" not in data:
                print("❌ [配置錯誤] pyproject.toml 缺少 [project] 部分")
                return False
            if "tool" not in data or "ruff" not in data.get("tool", {}):
                 print("❌ [配置錯誤] pyproject.toml 缺少 [tool.ruff] 部分")
                 return False
            # 可以根據需要添加更多檢查，例如 project.dependencies
            print("✅ [配置] pyproject.toml 結構檢查通過 (使用 toml 解析)")
            return True
        else:
            # Fallback 到簡易檢查
            if pyproject_path.stat().st_size == 0:
                 print("❌ [配置錯誤] pyproject.toml 文件為空")
                 return False
            print("⚠️ [配置] pyproject.toml 結構基本檢查通過 (簡易 - 建議安裝 tomli)")
            return True

    except Exception as e:
        print(f"❌ [配置錯誤] 無法解析 pyproject.toml: {e}")
        return False

def validate_schemas_dir() -> bool:
    """驗證根目錄下的 schemas 文件是否齊全"""
    print("\n--- 正在檢查根目錄 Schema 文件 ---")
    all_present = True
    schema_dir = PROJECT_ROOT / "schemas"
    if not schema_dir.is_dir():
        print(f"❌ [缺失] 根目錄下找不到 'schemas' 目錄")
        return False

    for schema_file in REQUIRED_SCHEMA_FILES:
        # schema_file 已經包含了可能的子目錄 (crosslang)
        if not check_path(Path("schemas") / schema_file, is_dir=False):
             all_present = False
    return all_present

def check_init_recursive(package_dir: Path) -> bool:
    """遞迴檢查目錄及其子目錄是否包含 __init__.py"""
    all_present = True
    base_rel_path = package_dir.relative_to(PROJECT_ROOT)

    # 檢查當前目錄
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        # 根目錄如 services 本身是需要的，子目錄如果沒有 .py 文件則可能不需要
        has_py_files = any(f.suffix == '.py' for f in package_dir.glob('*.py'))
        if package_dir in [PROJECT_ROOT / p for p in PYTHON_PACKAGE_DIRS] or has_py_files:
             print(f"❌ [缺失] Python 包缺少 __init__.py: {base_rel_path}")
             all_present = False
        # else:
        #    print(f"ℹ️ [跳過] 目錄不含 Python 文件，忽略 __init__.py 檢查: {base_rel_path}")
    else:
        print(f"✅ [存在] Python 包 __init__.py: {base_rel_path}")


    # 遞迴檢查子目錄
    for item in package_dir.iterdir():
        # 排除隱藏目錄、特殊目錄 (__pycache__) 和非 Python 相關目錄
        if (item.is_dir() and
            not item.name.startswith(('.', '_')) and
            "node_modules" not in item.parts and
            "target" not in item.parts):
            # 簡易判斷：如果子目錄包含 .py 文件，則遞迴檢查
            # （更精確的判斷可能需要分析 import 關係）
            if any(item.glob('**/*.py')):
                if not check_init_recursive(item):
                    all_present = False
    return all_present


def validate_python_packages() -> bool:
    """檢查指定的 Python 包目錄結構是否正確"""
    print("\n--- 正在遞迴檢查 Python 包 (__init__.py) ---")
    all_valid = True
    for pkg_path in PYTHON_PACKAGE_DIRS:
        full_pkg_path = PROJECT_ROOT / pkg_path
        if not full_pkg_path.is_dir():
            print(f"❌ [嚴重錯誤] 必要的 Python 基礎目錄不存在: {pkg_path}")
            all_valid = False
            continue
        if not check_init_recursive(full_pkg_path):
            all_valid = False
    return all_valid

def check_executables() -> bool:
    """檢查 Go/Rust/Node 等環境是否存在 (可選，但推薦)"""
    print("\n--- 正在檢查外部執行環境 (可選) ---")
    passed = True
    executables = {"go": "version", "rustc": "--version", "node": "--version", "npm": "--version"}
    for cmd, arg in executables.items():
        try:
            # 使用 shell=True 可能有安全風險，但在檢查環境時通常可接受
            # timeout 避免卡住
            result = subprocess.run([cmd, arg], capture_output=True, text=True, check=True, timeout=5, shell=True)
            print(f"✅ [環境] 检测到 {cmd}: {result.stdout.strip().splitlines()[0]}")
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"⚠️ [環境] 未检测到或執行出錯: {cmd} ({e})")
            # passed = False # 可以設為 False，如果希望強制檢查環境
    return passed


# --- 主函數 ---

def main():
    """執行所有驗證檢查"""
    print(f"開始驗證 AIVA 專案結構於: {PROJECT_ROOT}")
    results = {}

    # 檢查外部環境 (可選)
    results["executables"] = check_executables()

    print("\n--- 正在檢查必要目錄 ---")
    results["dirs"] = all(check_path(d, is_dir=True) for d in REQUIRED_DIRS)

    print("\n--- 正在檢查必要文件 ---")
    results["files"] = all(check_path(f, is_dir=False) for f in REQUIRED_FILES)

    print("\n--- 正在檢查多語言建置產物 ---")
    # 將建置產物設為可選，因為它們可能只在特定階段產生
    results["build_artifacts"] = all(check_path(f, is_dir=False, optional=True) for f in REQUIRED_BUILD_ARTIFACTS)

    print("\n--- 正在驗證 pyproject.toml ---")
    results["pyproject"] = validate_pyproject()

    print("\n--- 正在驗證根目錄 schemas 文件 ---")
    results["root_schemas"] = validate_schemas_dir()

    # 將 __init__.py 的檢查放在後面，因為前面檢查了目錄是否存在
    results["python_packages"] = validate_python_packages()


    # --- 總結 ---
    print("\n--- 驗證總結 ---")
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ 通過" if passed else ("❌ 失敗" if check_name != "build_artifacts" else "⚠️ 部分缺失 (可選)") # 對 build_artifacts 給予不同提示
        print(f"{check_name.ljust(15)}: {status}")
        if not passed and check_name != "build_artifacts" and check_name != "executables": # 允許 build artifacts 和 executables 檢查失敗
            all_passed = False

    if all_passed:
        print("\n🎉 AIVA 專案結構驗證通過！ (可能缺少可選的建置產物或外部環境)")
        sys.exit(0)
    else:
        print("\n🔥 AIVA 專案結構驗證失敗。請檢查上面標記為 ❌ 的項目。")
        sys.exit(1)

if __name__ == "__main__":
    main()

