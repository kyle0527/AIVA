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
    from config import settings # 載入統一設定
    # from services.core.aiva_core.app import AivaCoreApp # 假設的核心應用類別
    # from services.aiva_common.utils.logging import setup_logging # 假設的日誌設定函數
    # from services.aiva_common.mq import MessageQueueClient # 假設的消息隊列客戶端
except ImportError as e:
    print(f"錯誤：無法導入必要的 AIVA 模組: {e}")
    print("請確認您已在專案根目錄執行此腳本，且 Python 環境已安裝 requirements.txt 中的依賴。")
    sys.exit(1)

# --- 全域日誌設定 (範例，應使用 aiva_common 中的函數) ---
# (實際應調用 setup_logging())
logging.basicConfig(
    level=settings.LOG_LEVEL.upper() if hasattr(settings, 'LOG_LEVEL') else logging.INFO,
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
    if service_name not in SERVICE_CONFIG:
        logger.error(f"未知的服務名稱: {service_name}")
        return False
    if SERVICE_CONFIG[service_name]["process"] is not None and SERVICE_CONFIG[service_name]["process"].poll() is None:
        logger.warning(f"服務 {service_name} 似乎已在運行中。")
        return True # 視為成功

    config = SERVICE_CONFIG[service_name]
    logger.info(f"正在啟動服務: {service_name}...")
    logger.debug(f"命令: {' '.join(config['command'])}")
    logger.debug(f"工作目錄: {config['cwd']}")

    # 檢查可執行文件是否存在
    executable_path = Path(config['command'][0])
    if not executable_path.is_file() and not os.path.isabs(config['command'][0]) and shutil.which(config['command'][0]) is None:
        # 如果不是絕對路徑且不在 PATH 中，嘗試相對於 CWD 檢查
        if not (Path(config['cwd']) / config['command'][0]).is_file():
             msg = f"找不到服務 '{service_name}' 的可執行文件: {config['command'][0]}"
             if config.get("optional", False):
                 logger.warning(f"{msg} (此服務為可選，跳過)")
                 return True # 可選服務找不到不算失敗
             else:
                 logger.error(msg)
                 return False

    try:
        # 使用 Popen 啟動非阻塞子進程
        # 注意：stdout=subprocess.PIPE, stderr=subprocess.PIPE 會緩存輸出，可能導致阻塞
        # 最好重定向到文件或直接顯示在終端
        process = subprocess.Popen(
            config['command'],
            cwd=config['cwd'],
            env=config['env'],
            stdout=subprocess.PIPE, # 為了示範，捕捉輸出。生產環境建議重定向到日誌文件。
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', errors='ignore' # 處理潛在的編碼問題
        )
        SERVICE_CONFIG[service_name]["process"] = process
        logger.info(f"服務 {service_name} 已啟動 (PID: {process.pid})")
        # 簡易健康檢查：等待一小段時間後檢查進程是否意外退出
        time.sleep(2) # 等待 2 秒
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"服務 {service_name} 啟動後立即退出，返回碼: {process.returncode}")
            logger.error(f"stdout:\n{stdout}")
            logger.error(f"stderr:\n{stderr}")
            SERVICE_CONFIG[service_name]["process"] = None # 標記為未運行
            return False
        return True
    except FileNotFoundError:
        msg = f"啟動服務 {service_name} 失敗：找不到命令 {config['command'][0]}。請確保相關環境 (Python, Node, Go, Rust) 已正確安裝並加入 PATH，或提供了正確的可執行文件路徑。"
        if config.get("optional", False):
            logger.warning(f"{msg} (此服務為可選，跳過)")
            return True
        else:
            logger.error(msg)
            return False
    except Exception as e:
        logger.error(f"啟動服務 {service_name} 時發生未預期錯誤: {e}", exc_info=True)
        return False

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
    main_process = None
    # 假設 'core' 是主服務，或者第一個成功啟動的服務
    # 這裡以 core 為例，實際可能需要調整
    if "core" in SERVICE_CONFIG and SERVICE_CONFIG["core"]["process"]:
         main_process_name = "core"
         main_process = SERVICE_CONFIG[main_process_name]["process"]
    # 如果 core 沒啟動，可以選擇第一個啟動的服務作為監控對象
    # 或者讓啟動器保持運行直到手動停止

    if not main_process:
        logger.warning("未找到核心服務進程進行監控，啟動器將保持運行。請按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(60) # 保持運行
        except KeyboardInterrupt:
            logger.info("收到退出信號...")
            return # 觸發 finally 中的 stop_all_services

    logger.info(f"正在監控主服務 '{main_process_name}' (PID: {main_process.pid})... 按 Ctrl+C 退出。")
    try:
        while main_process.poll() is None:
            # 可以定期檢查其他服務的狀態
            for name, config in SERVICE_CONFIG.items():
                if config["process"] and config["process"].poll() is not None:
                     logger.warning(f"檢測到服務 '{name}' 已意外退出 (返回碼: {config['process'].returncode})。")
                     # 可以添加自動重啟邏輯
                     SERVICE_CONFIG[name]["process"] = None # 標記為停止

            time.sleep(5) # 每 5 秒檢查一次
        logger.info(f"主服務 '{main_process_name}' 已退出 (返回碼: {main_process.returncode})。")

    except KeyboardInterrupt:
        logger.info("收到退出信號...")
    except Exception as e:
        logger.error(f"監控服務時發生錯誤: {e}", exc_info=True)


# --- 主邏輯 ---

def main(args):
    """主執行函數"""
    logger.info("AIVA 平台啟動器開始運行...")
    logger.info(f"啟動模式: {args.mode}")
    if args.target:
        logger.info(f"掃描目標: {args.target}") # 這裡只是記錄，實際目標應傳遞給 Core 或 Scan 服務

    core_started = False
    services_to_start = []

    # --- 根據模式決定啟動哪些服務 ---
    if args.mode == "core_only":
        services_to_start = ["core"] # 假設核心服務的 key 是 'core'
    elif args.mode == "full":
        # 依照可能的依賴順序啟動 (例如：MQ -> Core -> Integration -> Scan -> API -> Features)
        # 這裡僅為範例順序，需依實際架構調整
        services_to_start = [
            # "message_queue", # 如果 MQ 是外部服務，則不在此啟動
            "core",          # 假設的核心服務 key
            "integration",
            "scan_py",
            "scan_node",
            "api",
            "feature_sca_go",
            "feature_sast_rust",
            # ... 其他 features ...
        ]
    elif args.mode == "scan_only": # 範例：僅啟動掃描相關服務
        services_to_start = ["scan_py", "scan_node"]
    else:
        logger.error(f"未知的啟動模式: {args.mode}")
        sys.exit(1)

    # --- 啟動核心服務 (如果需要) ---
    # 這裡假設 core 服務是直接在主進程中運行 AivaCoreApp 實例
    # 如果 core 也是一個獨立進程，則使用 start_service('core')
    if args.mode == "core_only" or args.mode == "full":
        try:
            logger.info("正在初始化 AIVA 核心服務...")
            # 假設 AivaCoreApp 有一個 run() 方法
            # aiva_core_app = AivaCoreApp(target=args.target)
            # logger.info("AIVA 核心服務初始化完成。")
            # 為了能管理其他子進程，Core 最好也作為獨立進程啟動
            if not start_service("core"): # 使用 start_service 啟動
                 raise RuntimeError("核心服務啟動失敗")
            core_started = True
            logger.info("AIVA 核心服務已作為子進程啟動。")

        except Exception as e:
            logger.critical(f"初始化或啟動 AIVA 核心服務失敗: {e}", exc_info=True)
            # 確保即使核心啟動失敗，也要嘗試停止已啟動的其他服務
            # finally 區塊會處理這個
            # stop_all_services() # 移至 finally
            sys.exit(1)

    # --- 啟動其他微服務 (如果需要) ---
    services_successfully_started = []
    if args.mode != "core_only":
        for service_name in services_to_start:
             # 跳過核心服務，因為它可能已在上面啟動
            if service_name == "core" and core_started:
                services_successfully_started.append(service_name)
                continue
            if start_service(service_name):
                 services_successfully_started.append(service_name)
            else:
                 # 如果某個非可選服務啟動失敗，記錄錯誤但可以選擇是否中止
                 if not SERVICE_CONFIG[service_name].get("optional", False):
                     logger.error(f"必要服務 '{service_name}' 啟動失敗，中止啟動流程。")
                     # stop_all_services() # 移至 finally
                     sys.exit(1)

    logger.info(f"成功啟動的服務: {', '.join(services_successfully_started)}")

    # --- 監控服務或保持運行 ---
    if services_successfully_started: # 只有成功啟動了服務才需要監控
        monitor_services()
    else:
        logger.warning("沒有成功啟動任何服務。")

    # --- 清理 ---
    # (monitor_services 函數結束後或異常退出時執行)
    # stop_all_services() # 移至 finally

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
