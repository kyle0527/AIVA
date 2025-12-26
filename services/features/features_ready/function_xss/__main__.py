"""
AIVA XSS Module - CLI Entry Point
直接調用 Traditional/Dom/Stored Detector 進行測試，不依賴 MQ。
"""
import argparse
import asyncio
import json
import logging
import sys
import os
import uuid

# --- 環境路徑設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.aiva_common.schemas import FunctionTaskPayload, FunctionTaskTarget, FunctionTaskTestConfig
from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.features.function_xss.dom_xss_detector import DomXssDetector
from services.features.function_xss.payload_generator import XssPayloadGenerator

# 設定 Log
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("XssCLI")


async def run_reflected_test(args):
    """執行反射型 XSS 測試"""
    logger.info(f"啟動反射型 XSS 測試: {args.url} (Param: {args.param})")
    
    # 建立任務物件 (模擬 MQ Payload)
    task = FunctionTaskPayload(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        scan_id=f"scan_{uuid.uuid4().hex[:8]}",
        target=FunctionTaskTarget(
            url=args.url,
            parameter=args.param,
            method=args.method,
            parameter_location=args.location
        ),
        test_config=FunctionTaskTestConfig(timeout=float(args.timeout))
    )

    # 產生 Payloads
    generator = XssPayloadGenerator()
    payloads = generator.generate_basic_payloads()
    logger.info(f"已生成 {len(payloads)} 個測試 Payloads")

    # 執行檢測
    detector = TraditionalXssDetector(task, timeout=float(args.timeout))
    results = await detector.execute(payloads)
    
    return results


async def run_dom_test(args):
    """執行 DOM XSS 測試"""
    logger.info(f"啟動 DOM XSS 測試: {args.url}")
    
    detector = DomXssDetector()
    
    # 模擬測試：通常 DOM 測試需要先抓取網頁內容
    import httpx
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(args.url, timeout=float(args.timeout))
            result = detector.analyze(
                payload="<script>alert(1)</script>", 
                document=resp.text
            )
            return [result] if result else []
        except Exception as e:
            logger.error(f"DOM 檢測請求失敗: {e}")
            return []


async def run_stored_test(args):
    """執行儲存型 XSS 測試"""
    from services.features.function_xss.stored_detector import StoredXssDetector
    
    logger.info(f"啟動儲存型 XSS 測試: {args.url}")
    
    task = FunctionTaskPayload(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        scan_id=f"scan_{uuid.uuid4().hex[:8]}",
        target=FunctionTaskTarget(
            url=args.url,
            parameter=args.param,
            method=args.method,
            parameter_location=args.location
        ),
        test_config=FunctionTaskTestConfig(timeout=float(args.timeout))
    )
    
    generator = XssPayloadGenerator()
    payloads = generator.generate_basic_payloads()[:3]  # 只用前 3 個避免污染
    
    detector = StoredXssDetector(task, timeout=float(args.timeout))
    
    # 如果有提供 view_urls，使用它們
    view_urls = [args.view_url] if args.view_url else None
    
    results = await detector.execute(payloads, view_urls=view_urls)
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="AIVA XSS Vulnerability Scanner CLI")
    
    parser.add_argument("--url", required=True, help="目標 URL")
    parser.add_argument("--type", choices=["reflected", "dom", "stored"], default="reflected", help="檢測類型")
    parser.add_argument("--param", help="測試參數名稱 (Reflected/Stored 必填)", default="q")
    parser.add_argument("--method", choices=["GET", "POST"], default="GET", help="HTTP 方法")
    parser.add_argument("--location", choices=["query", "body", "header"], default="query", help="參數位置")
    parser.add_argument("--timeout", default="30", help="超時秒數")
    parser.add_argument("--view-url", help="查看頁面 URL (Stored XSS 使用)")

    args = parser.parse_args()

    results = []
    try:
        if args.type == "reflected":
            detections = await run_reflected_test(args)
            # 轉換結果為簡單 Dict
            results = [
                {
                    "payload": d.payload,
                    "status": d.response_status,
                    "vulnerable": True,
                    "evidence": d.response_text[:200]  # 只取前 200 字作為證據
                }
                for d in detections
            ]
        elif args.type == "dom":
            # DOM 測試邏輯
            res = await run_dom_test(args)
            results = [
                {
                    "type": "dom",
                    "payload": getattr(r, 'payload', 'unknown'),
                    "vulnerable": True,
                    "detail": str(r)
                } 
                for r in res
            ]
        elif args.type == "stored":
            # Stored XSS 測試
            detections = await run_stored_test(args)
            results = [
                {
                    "payload": d.payload,
                    "status": d.response_status,
                    "vulnerable": True,
                    "evidence": d.response_text[:200]
                }
                for d in detections
            ]
            
    except Exception as e:
        logger.error(f"執行錯誤: {e}", exc_info=True)
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    # 輸出 JSON 結果
    output = {
        "target": args.url,
        "type": args.type,
        "findings_count": len(results),
        "vulnerable": len(results) > 0,
        "findings": results
    }
    
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
