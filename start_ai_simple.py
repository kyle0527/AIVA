#!/usr/bin/env python3
"""
AIVA AI 簡化啟動腳本 - 直接啟動協調器服務

持續運行監控模式，定期掃描目標
"""

import asyncio
import sys
import signal
from pathlib import Path
from datetime import datetime
import argparse

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

RUNNING = True


def signal_handler(signum, frame):
    """處理中斷信號"""
    global RUNNING
    logger.info(f"\n收到停止信號，正在優雅關閉...")
    RUNNING = False


async def run_continuous_monitoring(targets, interval=3600):
    """持續監控模式"""
    
    logger.info("=" * 80)
    logger.info("🚀 AIVA AI 持續監控服務")
    logger.info("=" * 80)
    logger.info(f"⏰ 啟動時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📋 監控目標: {', '.join(targets)}")
    logger.info(f"⏱️  掃描間隔: {interval}秒 ({interval/60:.1f}分鐘)")
    logger.info(f"🛑 按 Ctrl+C 停止服務")
    logger.info("")
    
    coordinator = MultiEngineCoordinator()
    scan_count = 0
    start_time = datetime.now()
    
    while RUNNING:
        scan_count += 1
        scan_id = f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scan_count:04d}"
        
        logger.info("=" * 80)
        logger.info(f"🔍 第 {scan_count} 次掃描 [{datetime.now().strftime('%H:%M:%S')}]")
        logger.info("=" * 80)
        
        try:
            # 執行快速掃描
            result = await coordinator.execute_strategy_fast(
                scan_id=scan_id,
                targets=targets
            )
            
            logger.info(f"✅ 掃描完成: {result.status}")
            logger.info(f"📊 發現資產: {len(result.assets)} 個")
            logger.info(f"⏱️  執行時間: {result.execution_time:.2f}秒")
            
            if result.assets:
                logger.info("\n🎯 資產摘要 (前 10 個):")
                for i, asset in enumerate(result.assets[:10], 1):
                    logger.info(f"   [{i:2d}] {asset.type:8s}: {asset.value}")
                if len(result.assets) > 10:
                    logger.info(f"   ... 還有 {len(result.assets)-10} 個資產")
            
            logger.info(f"\n📈 累計統計:")
            logger.info(f"   總掃描次數: {scan_count}")
            uptime = datetime.now() - start_time
            logger.info(f"   運行時間: {uptime}")
            
        except Exception as e:
            logger.error(f"❌ 掃描失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 等待下一次掃描
        if RUNNING:
            logger.info(f"\n💤 等待 {interval}秒後進行下一次掃描...")
            logger.info(f"   下次掃描時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("")
            
            for remaining in range(interval, 0, -30):
                if not RUNNING:
                    break
                if remaining % 300 == 0 or remaining == interval:
                    logger.info(f"   ⏰ 倒計時: {remaining}秒 ({remaining/60:.1f}分鐘)")
                await asyncio.sleep(min(30, remaining))
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ AIVA AI 服務已停止")
    logger.info(f"📊 總掃描次數: {scan_count}")
    logger.info(f"⏱️  總運行時間: {datetime.now() - start_time}")
    logger.info("=" * 80)


async def run_single_scan(targets):
    """單次掃描模式"""
    
    logger.info("=" * 80)
    logger.info("🚀 AIVA AI 單次掃描")
    logger.info("=" * 80)
    logger.info(f"📋 掃描目標: {', '.join(targets)}")
    logger.info("")
    
    coordinator = MultiEngineCoordinator()
    scan_id = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        result = await coordinator.execute_strategy_fast(
            scan_id=scan_id,
            targets=targets
        )
        
        logger.info(f"✅ 掃描完成: {result.status}")
        logger.info(f"📊 發現資產: {len(result.assets)} 個")
        logger.info(f"⏱️  執行時間: {result.execution_time:.2f}秒")
        
        if result.assets:
            logger.info("\n🎯 發現的資產:")
            for i, asset in enumerate(result.assets, 1):
                logger.info(f"   [{i:3d}] {asset.type:8s}: {asset.value}")
        
        logger.info("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"❌ 掃描失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA AI 簡化啟動 - 持續監控服務",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 持續監控模式（預設）
  python start_ai_simple.py http://localhost:3000
  
  # 自定義掃描間隔（每 30 分鐘）
  python start_ai_simple.py http://localhost:3000 --interval 1800
  
  # 監控多個目標
  python start_ai_simple.py http://localhost:3000 http://localhost:3001
  
  # 單次掃描（掃描後立即退出）
  python start_ai_simple.py http://localhost:3000 --once
        """
    )
    
    parser.add_argument(
        "targets",
        nargs="+",
        help="掃描目標 URL"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="掃描間隔（秒），預設 3600 (1小時)"
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="單次掃描模式（掃描後立即退出）"
    )
    
    args = parser.parse_args()
    
    # 註冊信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.once:
            # 單次掃描
            asyncio.run(run_single_scan(args.targets))
        else:
            # 持續監控
            asyncio.run(run_continuous_monitoring(args.targets, args.interval))
    except KeyboardInterrupt:
        logger.info("\n用戶中斷")
    except Exception as e:
        logger.error(f"服務異常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
