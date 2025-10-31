"""Auto Server - AIVA UI 自動端口伺服器
自動選擇可用端口啟動 UI 面板
"""

import logging
from pathlib import Path
import socket
import sys

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def find_free_port(start_port: int = 8080, max_attempts: int = 100) -> int:
    """尋找可用的端口號.

    Args:
        start_port: 起始端口號
        max_attempts: 最大嘗試次數

    Returns:
        可用的端口號

    Raises:
        RuntimeError: 找不到可用的端口
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    msg = f"無法在 {start_port}-{start_port + max_attempts - 1} 範圍內找到可用端口"
    raise RuntimeError(msg)


def start_auto_server(
    mode: str = "hybrid",
    host: str = "127.0.0.1",
    preferred_ports: list[int] | None = None,
) -> None:
    """啟動自動端口選擇的 UI 伺服器.

    Args:
        mode: 運作模式 (ui/ai/hybrid)
        host: 綁定的主機位址
        preferred_ports: 偏好的端口列表 (優先嘗試)
    """
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        import uvicorn

        from .improved_ui import create_app
    except ImportError as e:
        logger.error(f"匯入錯誤: {e}")
        logger.error("請確保已安裝所需套件: pip install fastapi uvicorn")
        return

    # 建立 FastAPI 應用
    app = create_app(mode=mode)

    # 確定要使用的端口
    port = None

    # 首先嘗試偏好的端口
    if preferred_ports:
        for preferred_port in preferred_ports:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, preferred_port))
                    port = preferred_port
                    logger.info(f"使用偏好端口: {port}")
                    break
            except OSError:
                continue

    # 如果偏好端口都不可用，自動選擇
    if port is None:
        try:
            port = find_free_port()
            logger.info(f"自動選擇端口: {port}")
        except RuntimeError as e:
            logger.error(f"端口選擇失敗: {e}")
            return

    # 啟動伺服器
    logger.info(f"\n{'='*60}")
    logger.info("   🚀 啟動 AIVA UI 面板 (自動端口)")
    logger.info(f"{'='*60}")
    logger.info(f"🌐 位址: http://{host}:{port}")
    logger.info(f"📖 API 文檔: http://{host}:{port}/docs")
    logger.info(f"🔧 模式: {mode}")
    logger.info(f"{'='*60}\n")

    try:
        uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"端口 {port} 被佔用，嘗試其他端口...")
            try:
                new_port = find_free_port(port + 1)
                logger.info(f"使用新端口: {new_port}")
                uvicorn.run(
                    app, host=host, port=new_port, log_level="info", access_log=True
                )
            except RuntimeError as retry_error:
                logger.error(f"重試失敗: {retry_error}")
        else:
            logger.error(f"伺服器啟動失敗: {e}")


def main() -> None:
    """主程式進入點."""
    import argparse

    parser = argparse.ArgumentParser(description="AIVA UI 自動端口伺服器")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["ui", "ai", "hybrid"],
        help="運作模式 (預設: hybrid)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="綁定的主機位址 (預設: 127.0.0.1)"
    )
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        help="偏好的端口列表 (例如: --ports 8080 8081 3000)",
    )

    args = parser.parse_args()

    start_auto_server(mode=args.mode, host=args.host, preferred_ports=args.ports)


if __name__ == "__main__":
    main()
