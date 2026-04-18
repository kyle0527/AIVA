"""
AIVA XSS Module - CLI Entry Point
直接調用 XssScanner 進行測試 (Native Mode)。
"""
import argparse
import asyncio
import json
import logging
import sys
import os

# --- 環境路徑設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .scanner import XssScanner

# 設定 Log
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("XssCLI")


async def main():
    parser = argparse.ArgumentParser(description="AIVA XSS Vulnerability Scanner CLI (Native)")
    parser.add_argument("--url", required=True, help="目標 URL (例如 https://example.com)")
    parser.add_argument("--type", default="comprehensive", choices=["reflected", "stored", "dom", "comprehensive"], help="掃描類型")
    parser.add_argument("--timeout", type=int, default=30, help="逾時時間")

    args = parser.parse_args()

    logger.info(f"啟動 XSS {args.type} 測試: {args.url}")

    scanner = XssScanner()

    try:
        # XssScanner 已經實作了 comprehensive 和個別的 scan_type
        # 回傳值是一個 List[FindingPayload] 或 Dict (依賴具體實作)
        result = await scanner.scan(
            target_url=args.url,
            scan_type=args.type,
            options={"timeout": args.timeout}
        )

        # 嘗試將結果轉為 JSON 並印出
        try:
            # 如果是 model，轉成 dict
            if isinstance(result, list):
                out = [r.model_dump() if hasattr(r, "model_dump") else r for r in result]
            else:
                out = result.model_dump() if hasattr(result, "model_dump") else result

            print(json.dumps({
                "scan_type": args.type,
                "target": args.url,
                "status": "success",
                "result": out
            }, indent=2, ensure_ascii=False))
        except TypeError:
            print(json.dumps({
                "scan_type": args.type,
                "target": args.url,
                "status": "success",
                "result": str(result)
            }, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"XSS 掃描發生錯誤: {e}")
        print(json.dumps({
            "scan_type": args.type,
            "target": args.url,
            "status": "failed",
            "error": str(e)
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
