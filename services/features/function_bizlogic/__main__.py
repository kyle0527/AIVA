"""
AIVA BizLogic CLI Tool
完全獨立的命令行工具，不依賴 MQ，直接執行業務邏輯測試並輸出 JSON。
"""
import argparse
import asyncio
import json
import logging
import sys
import os
import uuid

# --- 環境路徑設定 (確保能讀取到 services) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:

# --- 導入測試器與輔助工具 ---
from services.features.function_bizlogic.price_manipulation_scanner import PriceManipulationScanner as PriceManipulationTester
from services.features.function_bizlogic.race_condition_scanner import RaceConditionScanner as RaceConditionTester
from services.features.function_bizlogic.workflow_bypass_scanner import WorkflowBypassScanner as WorkflowBypassTester
from services.features.function_bizlogic.finding_helper import create_bizlogic_finding

# 設定 Log (輸出到 stderr 以免干擾 stdout 的 JSON 輸出)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BizLogicCLI")


def mk_finding_dict(result, target_url, task_id, scan_id):
    """將測試結果轉換為標準 Finding 格式 (Dict)"""
    evidence_data = {
        "request": {},  # 簡化，實際可從 Tester 獲取更詳細資料
        "response": {},
        "proof": json.dumps(result.get("evidence", ""), ensure_ascii=False),
        "details": result
    }

    payload = create_bizlogic_finding(
        vuln_type=result["type"],
        severity=result["severity"],
        target_url=target_url,
        method="TEST",
        evidence_data=evidence_data,
        task_id=task_id,
        scan_id=scan_id,
        parameter=result.get("endpoint") or result.get("parameter")
    )
    return payload.model_dump()


async def run_price_test(args):
    logger.info(f"啟動價格操控測試: {args.url}")
    tester = PriceManipulationTester(args.url, args.token, args.role)
    results = await tester.run_all_tests(args.endpoint)
    return results


async def run_race_test(args):
    logger.info(f"啟動競態條件測試: {args.url}")
    tester = RaceConditionTester(args.url, args.token)
    
    # 解析 endpoints JSON，若無則使用預設
    endpoints = None
    if args.endpoints:
        try:
            endpoints = json.loads(args.endpoints)
        except json.JSONDecodeError:
            logger.error("Endpoints 參數必須是有效的 JSON 字串")
            return []
            
    results = await tester.run_all_tests(endpoints)
    return results


async def run_workflow_test(args):
    logger.info(f"啟動流程繞過測試: {args.url}")
    tester = WorkflowBypassTester(args.url, args.token)
    
    # 建構 config
    config = {}
    if args.admin_paths:
        config["admin_endpoints"] = args.admin_paths.split(",")
        
    results = await tester.run_all_tests(config)
    return results


async def main():
    parser = argparse.ArgumentParser(description="AIVA Business Logic Vulnerability Scanner")
    
    # 通用參數
    parser.add_argument("--url", required=True, help="目標 URL (例如 http://localhost:3000)")
    parser.add_argument("--token", help="授權 Token (Optional)")
    parser.add_argument("--role", default="customer", help="用戶角色 (customer/merchant/admin)")
    parser.add_argument("--task-id", default=f"task_{uuid.uuid4().hex[:8]}", help="任務 ID")
    parser.add_argument("--scan-id", default=f"scan_{uuid.uuid4().hex[:8]}", help="掃描 ID")

    subparsers = parser.add_subparsers(dest="command", required=True, help="選擇測試模式")

    # 1. 價格操控 (Price Manipulation)
    price_parser = subparsers.add_parser("price", help="價格操控測試")
    price_parser.add_argument("--endpoint", default="/api/checkout", help="主要測試端點")

    # 2. 競態條件 (Race Condition)
    race_parser = subparsers.add_parser("race", help="競態條件測試")
    race_parser.add_argument("--endpoints", help='端點配置 JSON，例如 \'{"coupon": "/api/Recycles"}\'')

    # 3. 流程繞過 (Workflow Bypass)
    flow_parser = subparsers.add_parser("workflow", help="工作流程繞過測試")
    flow_parser.add_argument("--admin-paths", help="自定義管理員路徑，用逗號分隔")

    args = parser.parse_args()

    # 執行對應測試
    raw_results = []
    test_type = "unknown"

    try:
        if args.command == "price":
            test_type = "price_manipulation"
            raw_results = await run_price_test(args)
        elif args.command == "race":
            test_type = "race_condition"
            raw_results = await run_race_test(args)
        elif args.command == "workflow":
            test_type = "workflow_bypass"
            raw_results = await run_workflow_test(args)
    except Exception as e:
        logger.error(f"測試執行發生錯誤: {e}")
        # 出錯時輸出空的 JSON 結構或錯誤訊息，保持 stdout 乾淨
        print(json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False))
        sys.exit(1)

    # 轉換結果為標準格式
    findings = [
        mk_finding_dict(r, args.url, args.task_id, args.scan_id)
        for r in raw_results
    ]

    # 最終輸出 JSON (這是給 CLI 使用者或上層程式讀取的)
    output = {
        "scan_type": test_type,
        "target": args.url,
        "total_findings": len(findings),
        "findings": findings
    }
    
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
