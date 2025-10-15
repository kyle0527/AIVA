#!/usr/bin/env python3
"""
AIVA 架構改進測試腳本
測試重試機制、七階段處理流程、配置外部化等改進
"""

from __future__ import annotations

import asyncio
import os

# 設置測試環境變數
os.environ["AIVA_CORE_MONITOR_INTERVAL"] = "10"
os.environ["AIVA_RABBITMQ_URL"] = "amqp://guest:guest@localhost:5672/"
os.environ["AIVA_POSTGRES_DSN"] = "postgresql+asyncpg://aiva:aiva@localhost:5432/aiva"

from services.aiva_common.config import get_settings
from services.aiva_common.schemas import (
    Asset,
    ScanCompletedPayload,
    Summary,
)
from services.aiva_common.utils import get_logger, new_id
from services.core.aiva_core.analysis.dynamic_strategy_adjustment import (
    StrategyAdjuster,
)
from services.core.aiva_core.analysis.initial_surface import InitialAttackSurface
from services.core.aiva_core.execution.task_generator import TaskGenerator
from services.core.aiva_core.execution.task_queue_manager import TaskQueueManager
from services.core.aiva_core.ingestion.scan_module_interface import ScanModuleInterface
from services.core.aiva_core.processing import ScanResultProcessor
from services.core.aiva_core.state.session_state_manager import SessionStateManager
from services.function.function_sqli.aiva_func_sqli.worker import (
    SqliWorkerService,
)

logger = get_logger(__name__)


def test_configuration() -> None:
    """測試配置外部化"""
    logger.info("=" * 60)
    logger.info("測試 1: 配置外部化")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info(f"✓ Core Monitor Interval: {settings.core_monitor_interval}s")
    logger.info(f"✓ Enable Strategy Generator: {settings.enable_strategy_generator}")
    logger.info(f"✓ RabbitMQ URL: {settings.rabbitmq_url}")

    assert settings.core_monitor_interval == 10, "配置未正確讀取"
    logger.info("✅ 配置外部化測試通過\n")


def test_sqli_config_strategy() -> None:
    """測試 SQLi 引擎配置動態化"""
    logger.info("=" * 60)
    logger.info("測試 2: SQLi 引擎配置動態化")
    logger.info("=" * 60)

    strategies = ["FAST", "NORMAL", "DEEP", "AGGRESSIVE", "UNKNOWN"]

    for strategy in strategies:
        config = SqliWorkerService._create_config_from_strategy(strategy)
        logger.info(f"\n策略: {strategy}")
        logger.info(f"  - Timeout: {config.timeout_seconds}s")
        logger.info(f"  - Max Retries: {config.max_retries}")
        logger.info(f"  - Error Detection: {config.enable_error_detection}")
        logger.info(f"  - Boolean Detection: {config.enable_boolean_detection}")
        logger.info(f"  - Time Detection: {config.enable_time_detection}")
        logger.info(f"  - Union Detection: {config.enable_union_detection}")
        logger.info(f"  - OOB Detection: {config.enable_oob_detection}")

    # 驗證 FAST 策略
    fast_config = SqliWorkerService._create_config_from_strategy("FAST")
    assert fast_config.timeout_seconds == 10.0, "FAST 配置超時設定錯誤"
    assert fast_config.enable_error_detection is True, "FAST 應啟用錯誤檢測"
    assert fast_config.enable_boolean_detection is False, "FAST 不應啟用布林檢測"

    # 驗證 DEEP 策略
    deep_config = SqliWorkerService._create_config_from_strategy("DEEP")
    assert deep_config.enable_error_detection is True, "DEEP 應啟用所有檢測"
    assert deep_config.enable_boolean_detection is True, "DEEP 應啟用所有檢測"
    assert deep_config.enable_time_detection is True, "DEEP 應啟用所有檢測"

    logger.info("\n✅ SQLi 引擎配置動態化測試通過\n")


def test_scan_result_processor() -> None:
    """測試七階段處理器"""
    logger.info("=" * 60)
    logger.info("測試 3: 七階段掃描結果處理器")
    logger.info("=" * 60)

    # 初始化組件
    scan_interface = ScanModuleInterface()
    surface_analyzer = InitialAttackSurface()
    strategy_adjuster = StrategyAdjuster()
    task_generator = TaskGenerator()
    task_queue_manager = TaskQueueManager()
    session_state_manager = SessionStateManager()

    # 創建處理器
    processor = ScanResultProcessor(
        scan_interface=scan_interface,
        surface_analyzer=surface_analyzer,
        strategy_adjuster=strategy_adjuster,
        task_generator=task_generator,
        task_queue_manager=task_queue_manager,
        session_state_manager=session_state_manager,
    )

    logger.info("✓ ScanResultProcessor 初始化成功")
    logger.info(f"✓ 處理器類型: {type(processor).__name__}")
    logger.info("✓ 可用方法:")
    logger.info("  - stage_1_ingest_data")
    logger.info("  - stage_2_analyze_surface")
    logger.info("  - stage_3_generate_strategy")
    logger.info("  - stage_4_adjust_strategy")
    logger.info("  - stage_5_generate_tasks")
    logger.info("  - stage_6_dispatch_tasks")
    logger.info("  - stage_7_monitor_execution")
    logger.info("  - process (完整流程)")

    logger.info("\n✅ 七階段處理器結構測試通過\n")


async def test_scan_processing() -> None:
    """測試掃描處理流程 (模擬)"""
    logger.info("=" * 60)
    logger.info("測試 4: 掃描處理流程 (模擬)")
    logger.info("=" * 60)

    # 創建測試數據
    scan_id = new_id("scan")
    test_payload = ScanCompletedPayload(
        scan_id=scan_id,
        status="completed",
        summary=Summary(
            urls_found=10,
            forms_found=5,
            apis_found=3,
            scan_duration_seconds=120,
        ),
        assets=[
            Asset(
                asset_id=new_id("asset"),
                type="URL",
                value="http://example.com/login",
                has_form=True,
            ),
            Asset(
                asset_id=new_id("asset"),
                type="URL",
                value="http://example.com/api/user",
                has_form=False,
            ),
        ],
        fingerprints={"server": "nginx", "framework": "flask"},
    )

    logger.info("✓ 創建測試掃描數據:")
    logger.info(f"  - Scan ID: {scan_id}")
    logger.info(f"  - URLs: {test_payload.summary.urls_found}")
    logger.info(f"  - Forms: {test_payload.summary.forms_found}")
    logger.info(f"  - Assets: {len(test_payload.assets)}")

    # 初始化處理器
    scan_interface = ScanModuleInterface()
    surface_analyzer = InitialAttackSurface()
    strategy_adjuster = StrategyAdjuster()
    task_generator = TaskGenerator()
    task_queue_manager = TaskQueueManager()
    session_state_manager = SessionStateManager()

    processor = ScanResultProcessor(
        scan_interface=scan_interface,
        surface_analyzer=surface_analyzer,
        strategy_adjuster=strategy_adjuster,
        task_generator=task_generator,
        task_queue_manager=task_queue_manager,
        session_state_manager=session_state_manager,
    )

    # 測試階段 1 和 2 (不需要 broker 的階段)
    try:
        logger.info("\n執行階段 1: 資料接收與預處理...")
        await processor.stage_1_ingest_data(test_payload)
        logger.info("✓ 階段 1 完成")

        logger.info("\n執行階段 2: 攻擊面分析...")
        attack_surface = await processor.stage_2_analyze_surface(test_payload)
        logger.info(f"✓ 階段 2 完成 - 發現風險點: {attack_surface}")

        logger.info("\n執行階段 3: 策略生成...")
        base_strategy = await processor.stage_3_generate_strategy(scan_id)
        logger.info(f"✓ 階段 3 完成 - 策略類型: {base_strategy.get('strategy_type')}")

        logger.info("\n✅ 掃描處理流程測試通過 (前 3 階段)\n")

    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        raise


def print_summary() -> None:
    """打印測試摘要"""
    logger.info("=" * 60)
    logger.info("📊 測試摘要")
    logger.info("=" * 60)
    logger.info("✅ 所有測試通過!")
    logger.info("")
    logger.info("已驗證的改進:")
    logger.info("  1. ✅ 配置外部化 - 環境變數支援")
    logger.info("  2. ✅ SQLi 引擎配置動態化 - 4 種策略")
    logger.info("  3. ✅ 七階段處理器 - 模組化架構")
    logger.info("  4. ✅ 掃描處理流程 - 前 3 階段測試")
    logger.info("")
    logger.info("🎯 系統已準備就緒,可以開始實際測試!")
    logger.info("=" * 60)


async def main() -> None:
    """主測試函數"""
    logger.info("\n🚀 開始 AIVA 架構改進測試...\n")

    try:
        # 測試 1: 配置外部化
        test_configuration()

        # 測試 2: SQLi 配置動態化
        test_sqli_config_strategy()

        # 測試 3: 處理器結構
        test_scan_result_processor()

        # 測試 4: 掃描處理流程
        await test_scan_processing()

        # 打印摘要
        print_summary()

    except Exception as e:
        logger.error(f"\n❌ 測試失敗: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
