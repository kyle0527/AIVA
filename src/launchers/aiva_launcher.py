# -*- coding: utf-8 -*-
"""
AIVA 平台啟動器 (Launcher)

此腳本是 AIVA 平台的中央入口點。負責：
1. 解析命令列參數。
2. 初始化設定 (日誌、環境變數等)。
3. 啟動 AIVA 核心 AI 服務。
4. (可選) 協調啟動其他必要的微服務 (Scan, Integration, Features 等)。
5. 提供不同的啟動模式 (例如：僅核心、完整系統、測試模式)。

執行方式：
在專案根目錄下執行 `python aiva_launcher.py [選項]`
例如：`python aiva_launcher.py --mode full --target http://example.com`
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv

# --- 設定專案根目錄 ---
# 假設此腳本位於專案根目錄 AIVA-main/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- 載入環境變數 ---
# 優先載入專案根目錄的 .env 檔案
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    print(f"從 {dotenv_path} 載入環境變數")
else:
    print(f"警告：找不到 .env 檔案於 {PROJECT_ROOT}，將依賴系統環境變數。")

# --- 載入核心模組 (延遲載入以避免過早的依賴問題) ---
try:
    from config.settings import get_config  # 載入統一設定
    settings = get_config()
except ImportError as e:
    print(f"錯誤：無法導入必要的 AIVA 模組: {e}")
    print("請確認您已在專案根目錄執行此腳本，且 Python 環境已安裝 requirements.txt 中的依賴。")
    sys.exit(1)

# --- 全域日誌設定 (範例，應使用 aiva_common 中的函數) ---
# (實際應調用 setup_logging())
log_level = settings.get("logging", "level", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
        # 可以添加 FileHandler 將日誌寫入文件
        # logging.FileHandler(PROJECT_ROOT / "logs" / "aiva_launcher.log")
    ]
)
logger = logging.getLogger("AivaLauncher")

# --- 其他服務啟動配置 (範例) ---
# 實際路徑和啟動命令需要根據專案結構確認
SERVICE_CONFIG = {
    "scan_py": { # Python 掃描服務 (範例)
        "command": [sys.executable, str(PROJECT_ROOT / "services/scan/unified_scan_engine.py")],
        "cwd": str(PROJECT_ROOT / "services/scan"),
        "env": os.environ.copy(), # 繼承環境變數
        "process": None, # 用於追蹤子進程
    },
    "scan_node": { # Node.js 掃描服務
        "command": ["node", "dist/index.js"], # 假設已編譯
        "cwd": str(PROJECT_ROOT / "services/scan/aiva_scan_node"),
        "env": os.environ.copy(),
        "process": None,
    },
    "integration": { # Python Integration 服務
        "command": [sys.executable, str(PROJECT_ROOT / "services/integration/aiva_integration/app.py")],
        "cwd": str(PROJECT_ROOT / "services/integration/aiva_integration"),
        "env": os.environ.copy(),
        "process": None,
    },
    "api": { # Python API 服務
        "command": ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"], # 假設使用 FastAPI/Uvicorn
        "cwd": str(PROJECT_ROOT / "api"),
        "env": os.environ.copy(),
        "process": None,
    },
    # --- Core AI Engine Service ---
    "core": {
        "command": [sys.executable, "-m", "uvicorn", "services.core.aiva_core.app:app", "--host", "0.0.0.0", "--port", "8001"], # Core AI Engine with FastAPI
        "cwd": str(PROJECT_ROOT),  # 改為項目根目錄以解決模組導入問題
        "env": os.environ.copy(),
        "process": None,
    },
    # --- 多語言 Feature Workers (範例) ---
    "feature_sca_go": {
        "command": [str(PROJECT_ROOT / "services/features/function_sca_go/worker.exe")], # 假設 Go 已編譯
        "cwd": str(PROJECT_ROOT / "services/features/function_sca_go"),
        "env": os.environ.copy(),
        "process": None,
    },
    "feature_sast_rust": {
        # Rust 需要先編譯，這裡假設已有可執行檔
        "command": [str(PROJECT_ROOT / "services/features/function_sast_rust/target/release/function_sast_worker")], # 假設的 Rust 執行檔路徑
        "cwd": str(PROJECT_ROOT / "services/features/function_sast_rust"),
        "env": os.environ.copy(),
        "process": None,
        "optional": True, # Rust 可能需要手動編譯
    }
    # ... 其他服務 ...
}

# --- 函數定義 ---

def start_service(service_name: str) -> bool:
    """啟動指定的服務"""
    if not _validate_service(service_name):
        return False
    
    if _is_service_already_running(service_name):
        return True
    
    config = SERVICE_CONFIG[service_name]
    _log_service_start_info(service_name, config)
    
    if not _check_executable_exists(service_name, config):
        return False
        
    return _start_service_process(service_name, config)


def _validate_service(service_name: str) -> bool:
    """驗證服務名稱是否有效"""
    if service_name not in SERVICE_CONFIG:
        logger.error(f"未知的服務名稱: {service_name}")
        return False
    return True


def _is_service_already_running(service_name: str) -> bool:
    """檢查服務是否已在運行"""
    service_process = SERVICE_CONFIG[service_name]["process"]
    if service_process is not None and service_process.poll() is None:
        logger.warning(f"服務 {service_name} 似乎已在運行中。")
        return True
    return False


def _log_service_start_info(service_name: str, config: dict) -> None:
    """記錄服務啟動資訊"""
    logger.info(f"正在啟動服務: {service_name}...")
    logger.debug(f"命令: {' '.join(config['command'])}")
    logger.debug(f"工作目錄: {config['cwd']}")


def _check_executable_exists(service_name: str, config: dict) -> bool:
    """檢查可執行文件是否存在"""
    executable_path = Path(config['command'][0])
    
    # 檢查文件是否存在
    if (executable_path.is_file() or 
        os.path.isabs(config['command'][0]) or 
        shutil.which(config['command'][0]) is not None):
        return True
        
    # 檢查相對於工作目錄的路徑
    if (Path(config['cwd']) / config['command'][0]).is_file():
        return True
        
    # 文件不存在的處理
    return _handle_missing_executable(service_name, config)


def _handle_missing_executable(service_name: str, config: dict) -> bool:
    """處理缺失的可執行文件"""
    msg = f"找不到服務 '{service_name}' 的可執行文件: {config['command'][0]}"
    
    if config.get("optional", False):
        logger.warning(f"{msg} (此服務為可選，跳過)")
        return True  # 可選服務找不到不算失敗
    else:
        logger.error(msg)
        return False


def _start_service_process(service_name: str, config: dict) -> bool:
    """啟動服務進程"""
    try:
        process = _create_service_process(config)
        SERVICE_CONFIG[service_name]["process"] = process
        logger.info(f"服務 {service_name} 已啟動 (PID: {process.pid})")
        
        return _verify_service_health(service_name, process)
        
    except Exception as e:
        logger.error(f"啟動服務 {service_name} 時發生錯誤: {e}")
        return False


def _create_service_process(config: dict):
    """創建服務進程"""
    return subprocess.Popen(
        config['command'],
        cwd=config['cwd'],
        env=config['env'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8', errors='ignore'
    )


def _verify_service_health(service_name: str, process) -> bool:
    """驗證服務健康狀態"""
    time.sleep(2)  # 等待 2 秒
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        logger.error(f"服務 {service_name} 啟動後立即退出，返回碼: {process.returncode}")
        logger.error(f"stdout:\n{stdout}")
        logger.error(f"stderr:\n{stderr}")
        SERVICE_CONFIG[service_name]["process"] = None  # 標記為未運行
        return False
    return True

def stop_service(service_name: str):
    """停止指定的服務"""
    if service_name not in SERVICE_CONFIG or SERVICE_CONFIG[service_name]["process"] is None:
        logger.debug(f"服務 {service_name} 未運行或不存在。")
        return

    process = SERVICE_CONFIG[service_name]["process"]
    if process.poll() is None: # 檢查進程是否仍在運行
        logger.info(f"正在停止服務: {service_name} (PID: {process.pid})...")
        try:
            process.terminate() # 發送 SIGTERM
            process.wait(timeout=5) # 等待最多 5 秒
            logger.info(f"服務 {service_name} 已停止。")
        except subprocess.TimeoutExpired:
            logger.warning(f"服務 {service_name} 未能在 5 秒內正常停止，強制終止 (SIGKILL)...")
            process.kill() # 發送 SIGKILL
            process.wait()
            logger.info(f"服務 {service_name} 已被強制終止。")
        except Exception as e:
            logger.error(f"停止服務 {service_name} 時發生錯誤: {e}", exc_info=True)
    else:
        logger.info(f"服務 {service_name} 已停止 (返回碼: {process.returncode})。")

    SERVICE_CONFIG[service_name]["process"] = None

def stop_all_services():
    """停止所有由啟動器管理的服務"""
    logger.info("正在停止所有已啟動的服務...")
    # 反向停止，可能 Core 需要最後停止
    for service_name in reversed(list(SERVICE_CONFIG.keys())):
        stop_service(service_name)
    logger.info("所有服務已停止。")

def monitor_services():
    """監控已啟動的服務狀態，並在主服務退出時停止其他服務"""
    main_process_info = _get_main_process()
    
    if not main_process_info:
        _handle_no_main_process()
        return
    
    main_process_name, main_process = main_process_info
    _monitor_main_process(main_process_name, main_process)


def _get_main_process():
    """取得主進程資訊"""
    if "core" in SERVICE_CONFIG and SERVICE_CONFIG["core"]["process"]:
        return "core", SERVICE_CONFIG["core"]["process"]
    return None


def _handle_no_main_process():
    """處理沒有主進程的情況"""
    logger.warning("未找到核心服務進程進行監控，啟動器將保持運行。請按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(60)  # 保持運行
    except KeyboardInterrupt:
        logger.info("收到退出信號...")


def _monitor_main_process(main_process_name: str, main_process):
    """監控主進程"""
    logger.info(f"正在監控主服務 '{main_process_name}' (PID: {main_process.pid})... 按 Ctrl+C 退出。")
    
    try:
        while main_process.poll() is None:
            _check_service_health()
            time.sleep(5)  # 每5秒檢查一次
            
        logger.info(f"主服務 '{main_process_name}' 已退出。")
        
    except KeyboardInterrupt:
        logger.info("收到退出信號...")


def _check_service_health():
    """檢查所有服務的健康狀態"""
    for name, config in SERVICE_CONFIG.items():
        if config["process"] and config["process"].poll() is not None:
            logger.warning(f"檢測到服務 '{name}' 已意外退出 (返回碼: {config['process'].returncode})。")
            SERVICE_CONFIG[name]["process"] = None  # 標記為停止


# --- 主邏輯 ---

def main(args):
    """主執行函數"""
    _log_startup_info(args)
    
    # 決定要啟動的服務
    services_to_start = _determine_services_to_start(args.mode)
    
    # 啟動核心服務(如果需要)
    core_started = _start_core_service_if_needed(args.mode)
    
    # 啟動其他服務
    services_successfully_started = _start_additional_services(
        services_to_start, args.mode, core_started
    )
    
    # 監控服務或顯示警告
    _monitor_or_warn_services(services_successfully_started)


def _log_startup_info(args):
    """記錄啟動資訊"""
    logger.info("AIVA 平台啟動器開始運行...")
    logger.info(f"啟動模式: {args.mode}")
    if args.target:
        logger.info(f"掃描目標: {args.target}")


def _determine_services_to_start(mode):
    """根據模式決定要啟動的服務"""
    service_configurations = {
        "core_only": ["core"],
        "full": [
            "core",
            "integration", 
            "scan_py",
            "scan_node",
            "api",
            "feature_sca_go",
            "feature_sast_rust",
        ],
        "scan_only": ["scan_py", "scan_node"]
    }
    
    if mode not in service_configurations:
        logger.error(f"未知的啟動模式: {mode}")
        sys.exit(1)
        
    return service_configurations[mode]


def _start_core_service_if_needed(mode):
    """啟動核心服務(如果需要)"""
    if mode not in ["core_only", "full"]:
        return False
        
    try:
        logger.info("正在初始化 AIVA 核心服務...")
        if not start_service("core"):
            raise RuntimeError("核心服務啟動失敗")
        
        logger.info("AIVA 核心服務已作為子進程啟動。")
        return True
        
    except Exception as e:
        logger.critical(f"初始化或啟動 AIVA 核心服務失敗: {e}", exc_info=True)
        sys.exit(1)


def _start_additional_services(services_to_start, mode, core_started):
    """啟動其他微服務"""
    services_successfully_started = []
    
    # 如果核心服務已啟動，添加到成功列表中
    if core_started:
        services_successfully_started.append("core")
    
    if mode == "core_only":
        return services_successfully_started
    
    for service_name in services_to_start:
        if _should_skip_core_service(service_name, core_started):
            # 核心服務已經在上面處理了，跳過
            continue
            
        if _try_start_service(service_name):
            services_successfully_started.append(service_name)
        else:
            _handle_service_start_failure(service_name)
    
    logger.info(f"成功啟動的服務: {', '.join(services_successfully_started)}")
    return services_successfully_started


def _should_skip_core_service(service_name, core_started):
    """檢查是否應該跳過核心服務"""
    return service_name == "core" and core_started


def _try_start_service(service_name):
    """嘗試啟動服務"""
    return start_service(service_name)


def _handle_service_start_failure(service_name):
    """處理服務啟動失敗"""
    if not SERVICE_CONFIG[service_name].get("optional", False):
        logger.error(f"必要服務 '{service_name}' 啟動失敗，中止啟動流程。")
        sys.exit(1)


def _monitor_or_warn_services(services_successfully_started):
    """監控服務或顯示警告"""
    if services_successfully_started:
        monitor_services()
    else:
        logger.warning("沒有成功啟動任何服務。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIVA 平台啟動器")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["core_only", "full", "scan_only"], # 可以擴展更多模式
        default="full",
        help="選擇啟動模式：'core_only' - 僅啟動 AI 核心, 'full' - 啟動所有必要服務, 'scan_only' - 僅啟動掃描相關服務"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="（可選）指定初始掃描目標 URL 或範圍"
    )
    # 可以添加更多參數，如 --config, --log-level 等

    args = parser.parse_args()

    # 使用 try...finally 確保服務能被正確停止
    try:
        main(args)
    except Exception as e:
        logger.critical(f"啟動器發生未處理的異常: {e}", exc_info=True)
    finally:
        stop_all_services() # 無論如何，嘗試停止所有服務
        logger.info("AIVA 平台啟動器已關閉。")
        sys.exit(0) # 正常退出
