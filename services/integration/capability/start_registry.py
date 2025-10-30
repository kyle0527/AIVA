#!/usr/bin/env python3
"""
AIVA 能力註冊中心啟動腳本
整合所有功能的統一入口點

使用範例:
    python start_registry.py                    # 使用預設配置啟動
    python start_registry.py --config custom.yaml  # 使用自訂配置
    python start_registry.py --dev              # 開發模式
    python start_registry.py --discover-only    # 僅執行能力發現
"""

import asyncio
import sys
import argparse
import signal
from pathlib import Path


# 加入 AIVA 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id

from .config import load_config, validate_config, CapabilityRegistryConfig
from .registry import CapabilityRegistry
from .toolkit import CapabilityToolkit
from . import quick_start, get_info

# 設定結構化日誌
logger = get_logger(__name__)


class RegistryService:
    """能力註冊中心服務管理器"""
    
    def __init__(self, config: CapabilityRegistryConfig):
        self.config = config
        self.registry = None
        self.toolkit = None
        self.running = False
        self.trace_id = new_id("trace")
        
        # 設定信號處理
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    async def start(self) -> None:
        """啟動服務"""
        
        logger.info(
            "🚀 AIVA 能力註冊中心啟動中...",
            trace_id=self.trace_id,
            config=self.config.model_dump()
        )
        
        try:
            # 驗證配置
            config_errors = validate_config(self.config)
            if config_errors:
                logger.error("配置驗證失敗", errors=config_errors)
                raise RuntimeError("配置無效")
            
            # 初始化組件
            self.registry = CapabilityRegistry(self.config.database.path)
            self.toolkit = CapabilityToolkit()
            
            # 執行啟動序列
            await self._run_startup_sequence()
            
            self.running = True
            logger.info("✅ 能力註冊中心已成功啟動", trace_id=self.trace_id)
            
            # 如果啟用 API，啟動 FastAPI 服務器
            if self.config.api.docs_enabled:
                await self._start_api_server()
            else:
                # 如果不啟動 API，保持運行狀態
                await self._keep_running()
            
        except Exception as e:
            logger.error(
                "服務啟動失敗",
                error=str(e),
                trace_id=self.trace_id,
                exc_info=True
            )
            raise
    
    async def _run_startup_sequence(self) -> None:
        """執行啟動序列"""
        
        # 1. 系統資訊顯示
        info = get_info()
        logger.info("系統資訊", **info)
        
        # 2. 能力發現
        if self.config.discovery.auto_discovery_enabled:
            logger.info("🔍 開始自動發現能力...")
            discovery_stats = await self.registry.discover_capabilities()
            
            logger.info(
                "能力發現完成",
                discovered=discovery_stats["discovered_count"],
                by_language=discovery_stats.get("languages", {}),
                trace_id=self.trace_id
            )
        
        # 3. 健康檢查
        if self.config.monitoring.health_check_enabled:
            logger.info("💚 執行初始健康檢查...")
            await self._run_health_checks()
        
        # 4. 統計資訊
        stats = await self.registry.get_capability_stats()
        logger.info(
            "系統統計",
            total_capabilities=stats["total_capabilities"],
            health_summary=stats["health_summary"],
            trace_id=self.trace_id
        )
    
    async def _run_health_checks(self) -> None:
        """執行健康檢查"""
        
        capabilities = await self.registry.list_capabilities()
        healthy_count = 0
        failed_count = 0
        
        for capability in capabilities[:5]:  # 檢查前5個能力作為示例
            try:
                evidence = await self.toolkit.test_capability_connectivity(capability)
                if evidence.success:
                    healthy_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.warning(
                    "健康檢查失敗",
                    capability_id=capability.id,
                    error=str(e)
                )
                failed_count += 1
        
        logger.info(
            "健康檢查完成",
            healthy=healthy_count,
            failed=failed_count,
            trace_id=self.trace_id
        )
    
    async def _start_api_server(self) -> None:
        """啟動 API 服務器"""
        
        try:
            import uvicorn
            from .registry import app
            
            logger.info(
                f"🌐 啟動 API 服務器",
                host=self.config.api.host,
                port=self.config.api.port,
                docs_url=f"http://{self.config.api.host}:{self.config.api.port}/docs"
            )
            
            # 配置 uvicorn
            config = uvicorn.Config(
                app,
                host=self.config.api.host,
                port=self.config.api.port,
                log_level=self.config.logging.level.lower(),
                reload=self.config.api.debug
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            logger.error("uvicorn 未安裝，無法啟動 API 服務器")
            await self._keep_running()
        except Exception as e:
            logger.error("API 服務器啟動失敗", error=str(e), exc_info=True)
            raise
    
    async def _keep_running(self) -> None:
        """保持服務運行"""
        
        logger.info("服務運行中，按 Ctrl+C 停止")
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信號")
    
    def _handle_shutdown(self, signum, frame):
        """處理關閉信號"""
        
        logger.info(f"收到關閉信號: {signum}")
        self.running = False
    
    async def stop(self) -> None:
        """停止服務"""
        
        logger.info("🛑 正在停止能力註冊中心...", trace_id=self.trace_id)
        
        self.running = False
        
        # 清理資源
        if self.registry:
            # 這裡可以添加清理邏輯
            pass
        
        logger.info("✅ 能力註冊中心已停止", trace_id=self.trace_id)


async def discover_only(config: CapabilityRegistryConfig) -> None:
    """僅執行能力發現"""
    
    logger.info("🔍 執行能力發現...")
    
    registry = CapabilityRegistry(config.database.path)
    discovery_stats = await registry.discover_capabilities()
    
    print(f"\n📊 發現結果:")
    print(f"   總計: {discovery_stats['discovered_count']} 個能力")
    
    for lang, count in discovery_stats.get('languages', {}).items():
        print(f"   {lang}: {count} 個")
    
    if discovery_stats.get('errors'):
        print(f"\n❌ 錯誤:")
        for error in discovery_stats['errors']:
            print(f"   {error}")
    
    # 顯示統計
    stats = await registry.get_capability_stats()
    print(f"\n📈 系統統計:")
    print(f"   總能力數: {stats['total_capabilities']}")
    print(f"   語言分布: {stats['by_language']}")
    print(f"   健康狀態: {stats['health_summary']}")


async def main():
    """主程式入口"""
    
    parser = argparse.ArgumentParser(
        description="AIVA 能力註冊中心啟動腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  %(prog)s                        # 使用預設配置啟動
  %(prog)s --config custom.yaml   # 使用自訂配置
  %(prog)s --dev                  # 開發模式
  %(prog)s --discover-only        # 僅執行能力發現
  %(prog)s --info                 # 顯示系統資訊
        """
    )
    
    parser.add_argument('--config', '-c', help='配置檔案路徑')
    parser.add_argument('--dev', action='store_true', help='開發模式')
    parser.add_argument('--discover-only', action='store_true', help='僅執行能力發現')
    parser.add_argument('--info', action='store_true', help='顯示系統資訊')
    parser.add_argument('--quick-start', action='store_true', help='快速啟動模式')
    
    args = parser.parse_args()
    
    try:
        # 載入配置
        config = load_config(args.config)
        
        # 開發模式調整
        if args.dev:
            config.api.debug = True
            config.logging.level = "DEBUG"
            config.environment = "development"
        
        # 顯示系統資訊
        if args.info:
            info = get_info()
            print("\n📋 AIVA 能力註冊中心資訊")
            print("=" * 50)
            print(f"名稱: {info['name']}")
            print(f"版本: {info['version']}")
            print(f"描述: {info['description']}")
            print(f"作者: {info['author']}")
            
            print(f"\n🧩 組件:")
            for name, desc in info['components'].items():
                print(f"   {name}: {desc}")
            
            print(f"\n✨ 功能:")
            for feature in info['features']:
                print(f"   • {feature}")
            
            return
        
        # 僅執行發現
        if args.discover_only:
            await discover_only(config)
            return
        
        # 快速啟動
        if args.quick_start:
            result = await quick_start()
            print(f"\n✅ 快速啟動完成")
            print(f"   狀態: {result['status']}")
            print(f"   發現: {result['discovery_stats']['discovered_count']} 個能力")
            print(f"   總計: {result['system_stats']['total_capabilities']} 個已註冊能力")
            return
        
        # 正常啟動服務
        service = RegistryService(config)
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("⚠️  操作已取消")
    except Exception as e:
        logger.error("啟動失敗", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())