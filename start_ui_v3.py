"""
AIVA v3.0 Modern UI 啟動腳本
快速啟動現代化 Web 界面
"""

import logging
import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """主函數"""
    logger.info("🚀 啟動 AIVA v3.0 Modern UI...")

    try:
        from services.core.aiva_core.ui_panel.server_v3 import start_v3_ui_server

        # 啟動伺服器
        start_v3_ui_server(host="127.0.0.1", port=8080)

    except ImportError as e:
        logger.error(f"❌ 導入錯誤: {e}")
        logger.error("請確保已安裝所需依賴:")
        logger.error("  pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n👋 伺服器已停止")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 啟動失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
