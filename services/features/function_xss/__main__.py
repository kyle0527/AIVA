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

from services.features.function_xss.scanner import XssScanner

# 設定 Log
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("XssCLI")


async def main():
    parser = argparse.ArgumentParser(description="AIVA XSS Vulnerability Scanner CLI (Native)")

    parser.add_argument("--url", required=True, help="目標 URL")
    parser.add_argument("--type", choices=["comprehensive", "reflected", "dom", "stored"], default="comprehensive", help="檢測類型")
    parser.add_argument("--param", help="測試參數名稱 (Reflected/Stored 可選)", default=None)
    parser.add_argument("--method", choices=["GET", "POST"], default="GET", help="HTTP 方法")
    parser.add_argument("--timeout", default="30", help="超時秒數")
    parser.add_argument("--view-url", help="查看頁面 URL (Stored XSS 使用)")

    args = parser.parse_args()

    scanner = XssScanner()

    # 建構選項
    options = {
        "timeout": float(args.timeout),
        "method": args.method,
        "param": args.param,
        "view_url": args.view_url
    }

    try:
        # 執行掃描
        result = await scanner.scan(
            target_url=args.url,
            scan_type=args.type,
            options=options
        )

        # 輸出 JSON 結果
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"執行錯誤: {e}", exc_info=True)
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
