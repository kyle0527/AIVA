#!/usr/bin/env python3
"""
AIVA gRPC 服務器啟動腳本
=========================

統一通信架構的 gRPC 服務入口點

功能:
- 啟動多服務 gRPC 服務器
- 集成統一 MQ 系統
- 提供健康檢查和監控
- 支援優雅關機
"""

import asyncio
import signal
import logging
import sys
from pathlib import Path
from typing import Optional

# 確保 AIVA 模塊路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.aiva_common.grpc.grpc_server import AIVAGRPCServer
from services.aiva_common.utils.logging import setup_logger

logger = setup_logger(__name__)


class AIVAGRPCServerLauncher:
    """AIVA gRPC 服務器啟動器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 50051):
        self.host = host
        self.port = port
        self.server: Optional[AIVAGRPCServer] = None
        self._shutdown_event = asyncio.Event()
    
    def setup_signal_handlers(self):
        """設置信號處理器"""
        def signal_handler(signum, frame):
            logger.info(f"📡 收到信號 {signum}，開始優雅關機...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_server(self):
        """啟動 gRPC 服務器"""
        try:
            logger.info("🚀 初始化 AIVA gRPC 服務器...")
            
            # 創建服務器實例
            self.server = AIVAGRPCServer(host=self.host, port=self.port)
            
            # 啟動服務器
            await self.server.start()
            
            logger.info("✅ AIVA gRPC 服務器已啟動")
            logger.info(f"📍 監聽地址: {self.host}:{self.port}")
            logger.info("🔧 支援服務:")
            logger.info("   - TaskService (任務管理)")
            logger.info("   - CrossLanguageService (跨語言調用)")
            logger.info("   - HealthCheck (健康檢查)")
            logger.info("📡 統一 MQ 系統已集成")
            logger.info("🎯 準備接收請求...")
            
        except Exception as e:
            logger.error(f"❌ 服務器啟動失敗: {e}")
            raise
    
    async def wait_for_shutdown(self):
        """等待關機信號"""
        await self._shutdown_event.wait()
        
        logger.info("🔄 開始關閉 gRPC 服務器...")
        
        if self.server:
            await self.server.stop()
            logger.info("✅ gRPC 服務器已關閉")
        
        logger.info("👋 AIVA gRPC 服務器已優雅退出")
    
    async def run(self):
        """運行服務器"""
        try:
            # 設置信號處理
            self.setup_signal_handlers()
            
            # 啟動服務器
            await self.start_server()
            
            # 等待關機信號
            await self.wait_for_shutdown()
            
        except KeyboardInterrupt:
            logger.info("🛑 收到鍵盤中斷，關閉服務器...")
        except Exception as e:
            logger.error(f"❌ 服務器運行錯誤: {e}")
            return 1
        
        return 0


async def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AIVA gRPC 統一服務器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python start_grpc_server.py                    # 使用預設配置啟動
  python start_grpc_server.py --host 0.0.0.0    # 監聽所有介面
  python start_grpc_server.py --port 8080       # 使用自訂端口
        """
    )
    
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="服務器監聽地址 (預設: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=50051,
        help="服務器監聽端口 (預設: 50051)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="啟用除錯模式"
    )
    
    args = parser.parse_args()
    
    # 設置日誌級別
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🐛 除錯模式已啟用")
    
    # 顯示啟動資訊
    logger.info("=" * 60)
    logger.info("🤖 AIVA v5.0 統一通信架構")
    logger.info("🚀 gRPC 服務器啟動中...")
    logger.info(f"📍 地址: {args.host}:{args.port}")
    logger.info("=" * 60)
    
    # 創建並運行啟動器
    launcher = AIVAGRPCServerLauncher(host=args.host, port=args.port)
    return await launcher.run()


if __name__ == "__main__":
    try:
        # 運行主函數
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("🛑 用戶中斷，退出程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 程序異常退出: {e}")
        sys.exit(1)