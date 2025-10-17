#!/usr/bin/env python3
"""
AIVA CLI - 主命令行介面

提供完整的 AIVA 平台命令行操作介面，包括：
- 掃描管理 (scan)
- 漏洞檢測 (detect)
- 報告生成 (report)
- AI 訓練 (ai)
- 系統管理 (system)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.utils import get_logger, new_id
from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    ScanStartPayload,
)
from services.aiva_common.mq import get_broker

logger = get_logger(__name__)


# ============================================================================
# 掃描命令
# ============================================================================

async def cmd_scan_start(args):
    """啟動網站掃描"""
    scan_id = new_id("scan")
    task_id = new_id("task")
    
    logger.info(f"🚀 啟動掃描任務")
    logger.info(f"   掃描 ID: {scan_id}")
    logger.info(f"   任務 ID: {task_id}")
    logger.info(f"   目標 URL: {args.url}")
    logger.info(f"   最大深度: {args.max_depth}")
    
    # 構建掃描請求
    header = MessageHeader(
        message_id=new_id("msg"),
        source_module=ModuleName.CLI,
        target_module=ModuleName.SCAN,
        correlation_id=scan_id,
    )
    
    payload = ScanStartPayload(
        scan_id=scan_id,
        task_id=task_id,
        target_url=args.url,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        scope_domains=[args.url],
    )
    
    message = AivaMessage(
        header=header,
        payload=payload.model_dump(),
    )
    
    # 發送到 RabbitMQ
    broker = await get_broker()
    await broker.publish(
        topic=Topic.TASK_SCAN_START,
        message=message.model_dump_json(),
    )
    
    logger.info("✅ 掃描任務已提交到消息隊列")
    logger.info(f"   訂閱主題: {Topic.TASK_SCAN_START}")
    
    # 如果開啟等待模式，監聽結果
    if args.wait:
        logger.info("⏳ 等待掃描結果...")
        await wait_for_scan_result(scan_id)


async def wait_for_scan_result(scan_id: str):
    """等待掃描結果"""
    broker = await get_broker()
    
    async for mqmsg in broker.subscribe(Topic.RESULTS_SCAN_COMPLETED):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        
        if msg.header.correlation_id == scan_id:
            logger.info("✅ 掃描完成！")
            logger.info(f"   資產數量: {len(msg.payload.get('assets', []))}")
            logger.info(f"   指紋數量: {len(msg.payload.get('fingerprints', {}))}")
            
            # 顯示詳細資訊
            if msg.payload.get('assets'):
                logger.info("\n📦 發現的資產:")
                for asset in msg.payload['assets'][:5]:  # 顯示前5個
                    logger.info(f"   - {asset.get('url', 'N/A')}")
            
            break


# ============================================================================
# 檢測命令
# ============================================================================

async def cmd_detect_sqli(args):
    """啟動 SQL 注入檢測"""
    task_id = new_id("task")
    
    logger.info(f"🔍 啟動 SQL 注入檢測")
    logger.info(f"   任務 ID: {task_id}")
    logger.info(f"   目標 URL: {args.url}")
    logger.info(f"   參數: {args.param}")
    
    # 構建檢測請求
    header = MessageHeader(
        message_id=new_id("msg"),
        source_module=ModuleName.CLI,
        target_module=ModuleName.FUNCTION_SQLI,
        correlation_id=task_id,
    )
    
    payload_data = {
        "task_id": task_id,
        "target_url": args.url,
        "param_name": args.param,
        "method": args.method or "GET",
        "engines": args.engines.split(",") if args.engines else None,
    }
    
    message = AivaMessage(
        header=header,
        payload=payload_data,
    )
    
    broker = await get_broker()
    await broker.publish(
        topic=Topic.TASK_FUNCTION_SQLI,
        message=message.model_dump_json(),
    )
    
    logger.info("✅ SQL 注入檢測任務已提交")
    
    if args.wait:
        await wait_for_detection_result(task_id, "sqli")


async def cmd_detect_xss(args):
    """啟動 XSS 檢測"""
    task_id = new_id("task")
    
    logger.info(f"🔍 啟動 XSS 檢測")
    logger.info(f"   任務 ID: {task_id}")
    logger.info(f"   目標 URL: {args.url}")
    
    header = MessageHeader(
        message_id=new_id("msg"),
        source_module=ModuleName.CLI,
        target_module=ModuleName.FUNCTION_XSS,
        correlation_id=task_id,
    )
    
    payload_data = {
        "task_id": task_id,
        "target_url": args.url,
        "param_name": args.param,
        "xss_type": args.type or "reflected",
    }
    
    message = AivaMessage(header=header, payload=payload_data)
    
    broker = await get_broker()
    await broker.publish(
        topic=Topic.TASK_FUNCTION_XSS,
        message=message.model_dump_json(),
    )
    
    logger.info("✅ XSS 檢測任務已提交")
    
    if args.wait:
        await wait_for_detection_result(task_id, "xss")


async def wait_for_detection_result(task_id: str, vuln_type: str):
    """等待檢測結果"""
    broker = await get_broker()
    
    # 根據漏洞類型訂閱不同的結果主題
    topic_map = {
        "sqli": Topic.RESULTS_FUNCTION_SQLI,
        "xss": Topic.RESULTS_FUNCTION_XSS,
        "ssrf": Topic.RESULTS_FUNCTION_SSRF,
        "idor": Topic.RESULTS_FUNCTION_IDOR,
    }
    
    result_topic = topic_map.get(vuln_type, Topic.RESULTS_FUNCTION_SQLI)
    
    logger.info(f"⏳ 等待 {vuln_type.upper()} 檢測結果...")
    
    async for mqmsg in broker.subscribe(result_topic):
        msg = AivaMessage.model_validate_json(mqmsg.body)
        
        if msg.header.correlation_id == task_id:
            findings = msg.payload.get('findings', [])
            
            if findings:
                logger.info(f"🚨 發現 {len(findings)} 個漏洞！")
                for i, finding in enumerate(findings, 1):
                    logger.info(f"\n漏洞 #{i}:")
                    logger.info(f"   嚴重程度: {finding.get('vulnerability', {}).get('severity', 'N/A')}")
                    logger.info(f"   置信度: {finding.get('vulnerability', {}).get('confidence', 'N/A')}")
                    logger.info(f"   描述: {finding.get('vulnerability', {}).get('description', 'N/A')}")
            else:
                logger.info("✅ 未發現漏洞")
            
            break


# ============================================================================
# AI 訓練命令
# ============================================================================

async def cmd_ai_train(args):
    """啟動 AI 訓練"""
    from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
    from services.core.aiva_core.learning.experience_manager import ExperienceManager
    from services.core.aiva_core.training.training_orchestrator import TrainingOrchestrator
    
    logger.info("🤖 啟動 AI 訓練系統")
    logger.info(f"   訓練模式: {args.mode}")
    logger.info(f"   訓練輪數: {args.epochs}")
    
    # 初始化組件
    bio_net = ScalableBioNet(
        input_dim=512,
        hidden_dims=[1024, 2048, 1024],
        output_dim=256,
    )
    
    logger.info(f"   神經網路參數量: {bio_net.count_params():,}")
    
    exp_manager = ExperienceManager(storage_path=args.storage_path)
    
    orchestrator = TrainingOrchestrator(
        bio_net=bio_net,
        experience_manager=exp_manager,
    )
    
    # 執行訓練
    if args.mode == "realtime":
        logger.info("📡 實時訓練模式：監聽實際任務執行...")
        await orchestrator.train_from_live_tasks(epochs=args.epochs)
    
    elif args.mode == "replay":
        logger.info("📼 回放訓練模式：從歷史經驗學習...")
        await orchestrator.train_from_history(epochs=args.epochs)
    
    elif args.mode == "simulation":
        logger.info("🎮 模擬訓練模式：使用模擬場景...")
        await orchestrator.train_from_simulation(
            num_scenarios=args.scenarios,
            epochs=args.epochs,
        )
    
    logger.info("✅ AI 訓練完成")


async def cmd_ai_status(args):
    """查看 AI 狀態"""
    from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
    from services.core.aiva_core.ai_engine.knowledge_base import KnowledgeBase
    
    logger.info("🤖 AI 系統狀態")
    
    # 載入模型
    bio_net = ScalableBioNet(
        input_dim=512,
        hidden_dims=[1024, 2048, 1024],
        output_dim=256,
    )
    
    logger.info(f"   模型參數量: {bio_net.count_params():,}")
    
    # 知識庫狀態
    kb = KnowledgeBase(storage_path=args.storage_path)
    stats = await kb.get_stats()
    
    logger.info(f"   知識庫條目: {stats.get('total_entries', 0)}")
    logger.info(f"   向量維度: {stats.get('vector_dim', 0)}")
    logger.info(f"   最後更新: {stats.get('last_update', 'N/A')}")


# ============================================================================
# 報告命令
# ============================================================================

async def cmd_report_generate(args):
    """生成報告"""
    logger.info(f"📊 生成報告")
    logger.info(f"   掃描 ID: {args.scan_id}")
    logger.info(f"   格式: {args.format}")
    logger.info(f"   輸出: {args.output}")
    
    # 調用 Integration 模組的報告生成器
    from services.integration.aiva_integration.reporting.report_content_generator import (
        ReportContentGenerator,
    )
    from services.integration.aiva_integration.reporting.formatter_exporter import (
        FormatterExporter,
    )
    
    generator = ReportContentGenerator()
    exporter = FormatterExporter()
    
    # 生成報告內容
    content = await generator.generate_report(
        scan_id=args.scan_id,
        include_findings=not args.no_findings,
        include_stats=True,
    )
    
    # 導出報告
    await exporter.export(
        content=content,
        format=args.format,
        output_path=args.output,
    )
    
    logger.info(f"✅ 報告已生成: {args.output}")


# ============================================================================
# 系統命令
# ============================================================================

async def cmd_system_status(args):
    """查看系統狀態"""
    logger.info("⚙️ AIVA 系統狀態")
    
    broker = await get_broker()
    
    # 檢查各模組狀態
    modules = [
        ModuleName.CORE,
        ModuleName.SCAN,
        ModuleName.FUNCTION_SQLI,
        ModuleName.FUNCTION_XSS,
        ModuleName.INTEGRATION,
    ]
    
    logger.info("\n📡 模組狀態:")
    for module in modules:
        # 發送心跳檢查
        status = "🟢 運行中" if await check_module_alive(module) else "🔴 離線"
        logger.info(f"   {module.value}: {status}")


async def check_module_alive(module: ModuleName) -> bool:
    """檢查模組是否存活"""
    # 簡化版本，實際應該檢查心跳
    return True


# ============================================================================
# 主程序
# ============================================================================

def create_parser():
    """創建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="aiva",
        description="AIVA - AI-powered Vulnerability Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 掃描網站
  aiva scan start https://example.com --max-depth 3
  
  # SQL 注入檢測
  aiva detect sqli https://example.com/login --param username
  
  # XSS 檢測
  aiva detect xss https://example.com/search --param q
  
  # 生成報告
  aiva report generate scan_xxx --format pdf --output report.pdf
  
  # AI 訓練
  aiva ai train --mode realtime --epochs 10
  
  # 查看系統狀態
  aiva system status
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ========== 掃描命令 ==========
    scan_parser = subparsers.add_parser("scan", help="掃描管理")
    scan_sub = scan_parser.add_subparsers(dest="scan_action")
    
    # scan start
    scan_start = scan_sub.add_parser("start", help="啟動掃描")
    scan_start.add_argument("url", help="目標 URL")
    scan_start.add_argument("--max-depth", type=int, default=3, help="最大爬取深度")
    scan_start.add_argument("--max-pages", type=int, default=100, help="最大頁面數")
    scan_start.add_argument("--wait", action="store_true", help="等待掃描完成")
    scan_start.set_defaults(func=cmd_scan_start)
    
    # ========== 檢測命令 ==========
    detect_parser = subparsers.add_parser("detect", help="漏洞檢測")
    detect_sub = detect_parser.add_subparsers(dest="detect_type")
    
    # detect sqli
    sqli_parser = detect_sub.add_parser("sqli", help="SQL 注入檢測")
    sqli_parser.add_argument("url", help="目標 URL")
    sqli_parser.add_argument("--param", required=True, help="測試參數名")
    sqli_parser.add_argument("--method", choices=["GET", "POST"], help="HTTP 方法")
    sqli_parser.add_argument("--engines", help="檢測引擎 (逗號分隔)")
    sqli_parser.add_argument("--wait", action="store_true", help="等待檢測完成")
    sqli_parser.set_defaults(func=cmd_detect_sqli)
    
    # detect xss
    xss_parser = detect_sub.add_parser("xss", help="XSS 檢測")
    xss_parser.add_argument("url", help="目標 URL")
    xss_parser.add_argument("--param", required=True, help="測試參數名")
    xss_parser.add_argument("--type", choices=["reflected", "stored", "dom"], help="XSS 類型")
    xss_parser.add_argument("--wait", action="store_true", help="等待檢測完成")
    xss_parser.set_defaults(func=cmd_detect_xss)
    
    # ========== AI 命令 ==========
    ai_parser = subparsers.add_parser("ai", help="AI 訓練和管理")
    ai_sub = ai_parser.add_subparsers(dest="ai_action")
    
    # ai train
    ai_train = ai_sub.add_parser("train", help="訓練 AI 模型")
    ai_train.add_argument(
        "--mode",
        choices=["realtime", "replay", "simulation"],
        default="realtime",
        help="訓練模式",
    )
    ai_train.add_argument("--epochs", type=int, default=10, help="訓練輪數")
    ai_train.add_argument("--scenarios", type=int, default=100, help="模擬場景數量")
    ai_train.add_argument("--storage-path", default="./data/ai", help="存儲路徑")
    ai_train.set_defaults(func=cmd_ai_train)
    
    # ai status
    ai_status = ai_sub.add_parser("status", help="查看 AI 狀態")
    ai_status.add_argument("--storage-path", default="./data/ai", help="存儲路徑")
    ai_status.set_defaults(func=cmd_ai_status)
    
    # ========== 報告命令 ==========
    report_parser = subparsers.add_parser("report", help="報告生成")
    report_sub = report_parser.add_subparsers(dest="report_action")
    
    # report generate
    report_gen = report_sub.add_parser("generate", help="生成報告")
    report_gen.add_argument("scan_id", help="掃描 ID")
    report_gen.add_argument("--format", choices=["pdf", "html", "json"], default="html", help="報告格式")
    report_gen.add_argument("--output", default="report.html", help="輸出檔案")
    report_gen.add_argument("--no-findings", action="store_true", help="不包含漏洞詳情")
    report_gen.set_defaults(func=cmd_report_generate)
    
    # ========== 系統命令 ==========
    system_parser = subparsers.add_parser("system", help="系統管理")
    system_sub = system_parser.add_subparsers(dest="system_action")
    
    # system status
    system_status = system_sub.add_parser("status", help="查看系統狀態")
    system_status.set_defaults(func=cmd_system_status)
    
    return parser


async def async_main():
    """異步主函數"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    try:
        await args.func(args)
    except KeyboardInterrupt:
        logger.info("\n⚠️ 用戶中斷操作")
    except Exception as e:
        logger.error(f"❌ 錯誤: {e}", exc_info=True)
        sys.exit(1)


def main():
    """主入口點"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
