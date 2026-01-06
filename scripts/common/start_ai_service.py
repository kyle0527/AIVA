#!/usr/bin/env python3
"""
AIVA AI 持續運作服務

支持多種運作模式:
1. API 服務模式 - REST API 持續監聽
2. 後台監控模式 - 監控目標並自動掃描
3. 交互式模式 - 命令行交互界面

用法:
    python start_ai_service.py --mode api              # API 服務模式 (推薦)
    python start_ai_service.py --mode monitor          # 後台監控模式
    python start_ai_service.py --mode interactive      # 交互式模式
    python start_ai_service.py --mode daemon           # 守護進程模式
"""

import asyncio
import sys
import os
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# 全局狀態
RUNNING = True
service_mode = None


def signal_handler(signum, frame):
    """處理中斷信號"""
    global RUNNING
    logger.info(f"\n收到信號 {signum}，正在優雅關閉...")
    RUNNING = False


class AIService:
    """AI 持續運作服務"""
    
    def __init__(self, mode: str, config: Dict[str, Any]):
        self.mode = mode
        self.config = config
        self.running = False
        self.start_time = None
        
    async def start(self):
        """啟動服務"""
        self.running = True
        self.start_time = datetime.now()
        
        logger.info(f"🚀 啟動 AIVA AI 服務 - {self.mode} 模式")
        logger.info(f"⏰ 啟動時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.mode == "api":
            await self.run_api_mode()
        elif self.mode == "monitor":
            await self.run_monitor_mode()
        elif self.mode == "interactive":
            await self.run_interactive_mode()
        elif self.mode == "daemon":
            await self.run_daemon_mode()
        else:
            logger.error(f"未知模式: {self.mode}")
            
    async def stop(self):
        """停止服務"""
        self.running = False
        uptime = datetime.now() - self.start_time
        logger.info(f"⏱️  服務運行時間: {uptime}")
        logger.info("[SUCCESS] AIVA AI 服務已停止")
        
    async def run_api_mode(self):
        """API 服務模式 - REST API 持續監聽"""
        import subprocess
        
        host = self.config.get("host", "0.0.0.0")
        port = self.config.get("port", 8000)
        
        logger.info(f"📡 啟動 API 服務: http://{host}:{port}")
        logger.info(f"📚 API 文檔: http://{host}:{port}/docs")
        logger.info("🔐 預設帳號:")
        logger.info("   - Admin: admin / aiva-admin-2025")
        logger.info("   - User: user / aiva-user-2025")
        logger.info("")
        logger.info("按 Ctrl+C 停止服務")
        
        # 使用 subprocess 啟動 uvicorn，避免導入問題
        api_dir = Path(__file__).parent / "api"
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(api_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 實時輸出日誌
            for line in process.stdout:
                print(line.rstrip())
                
            process.wait()
            
        except KeyboardInterrupt:
            logger.info("\n用戶中斷 API 服務")
            if process:
                process.terminate()
                process.wait()
            
    async def run_monitor_mode(self):
        """後台監控模式 - 持續監控並掃描"""
        from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
        
        targets = self.config.get("targets", [])
        interval = self.config.get("interval", 3600)  # 預設 1 小時
        
        if not targets:
            logger.warning("[WARNING] 未指定監控目標，使用預設目標")
            targets = ["http://localhost:3000"]
        
        logger.info(f"🔍 監控模式啟動")
        logger.info(f"📋 監控目標: {', '.join(targets)}")
        logger.info(f"⏱️  掃描間隔: {interval}秒 ({interval/60:.1f}分鐘)")
        
        coordinator = MultiEngineCoordinator()
        scan_count = 0
        
        while RUNNING:
            scan_count += 1
            scan_id = f"scan_monitor_{scan_count:04d}"
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 開始第 {scan_count} 次掃描 [{datetime.now().strftime('%H:%M:%S')}]")
            logger.info(f"{'='*60}")
            
            try:
                # 執行快速掃描
                result = await coordinator.execute_strategy_fast(
                    scan_id=scan_id,
                    targets=targets
                )
                
                logger.info(f"[SUCCESS] 掃描完成: {result.status}")
                logger.info(f"[REPORT] 發現資產: {len(result.assets)}")
                
                if result.assets:
                    logger.info("[INFO] 資產摘要:")
                    for i, asset in enumerate(result.assets[:5], 1):
                        logger.info(f"   [{i}] {asset.type}: {asset.value}")
                    if len(result.assets) > 5:
                        logger.info(f"   ... 還有 {len(result.assets)-5} 個資產")
                        
            except Exception as e:
                logger.error(f"[ERROR] 掃描失敗: {e}")
            
            # 等待下一次掃描
            if RUNNING:
                logger.info(f"\n💤 等待 {interval}秒後進行下一次掃描...")
                logger.info(f"   (按 Ctrl+C 停止服務)")
                
                for remaining in range(interval, 0, -30):
                    if not RUNNING:
                        break
                    if remaining % 300 == 0:  # 每 5 分鐘提示一次
                        logger.info(f"   ⏰ 下次掃描倒計時: {remaining}秒")
                    await asyncio.sleep(min(30, remaining))
                    
    async def run_interactive_mode(self):
        """交互式模式 - AI 對話交互"""
        from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant
        from services.aiva_common.command_center import get_command_center
        
        logger.info("💬 AIVA AI 交互模式啟動")
        logger.info("[AI] 正在初始化 AI 對話助理...")
        
        # 初始化 AI 對話助理和命令中心
        assistant = AIVADialogAssistant()
        command_center = get_command_center()
        
        logger.info("[SUCCESS] AI 對話助理已就緒")
        logger.info("[INFO] 您可以用自然語言與 AI 對話，例如:")
        logger.info("   '掃描 http://localhost:3000'")
        logger.info("   '你會什麼功能？'")
        logger.info("   '系統狀態如何？'")
        logger.info("   輸入 'quit' 退出")
        logger.info("")
        
        while RUNNING:
            try:
                # 非阻塞式輸入
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "AIVA> "
                )
                user_input = user_input.strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["quit", "exit", "q"]:
                    logger.info("👋 再見！")
                    break
                
                # 將用戶輸入交給 AI 對話助理處理
                logger.info(f"[AI] AI 正在思考...")
                response = await assistant.process_user_input(user_input)
                
                # 顯示 AI 回應
                logger.info(f"\n{response.get('message', '無回應')}\n")
                
                # 如果有可執行的任務，交給 CommandCenter 執行
                if response.get('executable') and response.get('task_plan'):
                    logger.info("🚀 AI 開始執行任務...")
                    try:
                        from services.aiva_common.models import AICommand
                        # 將任務計劃轉換為 CommandCenter 可執行的指令
                        command = AICommand(
                            action=response['task_plan'].get('action', 'execute'),
                            params=response['task_plan']
                        )
                        task_result = await command_center.execute(command)
                        
                        if task_result.get('success'):
                            logger.info(f"[SUCCESS] 任務執行成功")
                            if task_result.get('results'):
                                logger.info(f"[REPORT] 結果: {task_result['results']}")
                        else:
                            logger.error(f"[ERROR] 任務執行失敗: {task_result.get('error')}")
                    except Exception as e:
                        logger.error(f"[ERROR] 執行任務時發生錯誤: {e}")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                logger.info("\n用戶中斷")
                break
                
    async def run_daemon_mode(self):
        """守護進程模式 - 結合 API + 監控"""
        logger.info("🔄 守護進程模式啟動")
        logger.info("   同時運行 API 服務和後台監控")
        
        # 創建兩個任務
        api_config = {
            "host": self.config.get("host", "0.0.0.0"),
            "port": self.config.get("port", 8000)
        }
        
        monitor_config = {
            "targets": self.config.get("targets", ["http://localhost:3000"]),
            "interval": self.config.get("interval", 3600)
        }
        
        # 並行運行
        api_service = AIService("api", api_config)
        monitor_service = AIService("monitor", monitor_config)
        
        await asyncio.gather(
            api_service.start(),
            monitor_service.start()
        )


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA AI 持續運作服務",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # API 服務模式 (推薦)
  python start_ai_service.py --mode api
  
  # 後台監控模式
  python start_ai_service.py --mode monitor --targets http://localhost:3000 --interval 1800
  
  # 交互式模式
  python start_ai_service.py --mode interactive
  
  # 守護進程模式 (API + 監控)
  python start_ai_service.py --mode daemon --port 8000 --targets http://localhost:3000
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "monitor", "interactive", "daemon"],
        default="api",
        help="運作模式 (預設: api)"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API 服務綁定地址 (預設: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 服務端口 (預設: 8000)"
    )
    
    parser.add_argument(
        "--targets",
        nargs="+",
        help="監控目標 URL 列表"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="監控掃描間隔（秒）(預設: 3600)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日誌級別 (預設: INFO)"
    )
    
    args = parser.parse_args()
    
    # 設置日誌級別
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 註冊信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 配置
    config = {
        "host": args.host,
        "port": args.port,
        "targets": args.targets or [],
        "interval": args.interval
    }
    
    # 創建並啟動服務
    service = AIService(args.mode, config)
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("\n用戶中斷")
    finally:
        logger.info("服務已停止")


if __name__ == "__main__":
    main()
