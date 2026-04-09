#!/usr/bin/env python3
"""
XSS CLI - 完全同步版本
AI 控制異步，外部模組只做同步執行
"""
import argparse
import json
import logging
import sys
import os

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:

# 設定 Log
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("XssCLI_Sync")


def run_xss_test(url: str, param: str, method: str = "GET", timeout: int = 30):
    """完全同步的 XSS 測試
    
    不使用 async，直接用 requests 庫（同步）
    """
    import requests
    
    logger.info(f"開始 XSS 測試: {url} (參數: {param})")
    
    # 測試 payloads
    payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "'\"><script>alert(String.fromCharCode(88,83,83))</script>",
        "<svg/onload=alert('XSS')>",
    ]
    
    findings = []
    
    for payload in payloads:
        try:
            # 構建測試 URL
            if method == "GET":
                test_url = f"{url}?{param}={payload}"
                response = requests.get(test_url, timeout=timeout)
            else:
                response = requests.post(url, data={param: payload}, timeout=timeout)
            
            # 檢查是否存在漏洞
            if payload in response.text:
                findings.append({
                    "payload": payload,
                    "status": response.status_code,
                    "vulnerable": True,
                    "evidence": response.text[:200]
                })
                logger.info(f"✓ 發現漏洞: {payload}")
            
        except Exception as e:
            logger.error(f"測試失敗: {payload} - {e}")
    
    return findings


def main():
    """主函數 - 完全同步"""
    parser = argparse.ArgumentParser(description="AIVA XSS Scanner (Sync CLI)")
    
    parser.add_argument("--url", required=True, help="目標 URL")
    parser.add_argument("--param", default="q", help="測試參數名稱")
    parser.add_argument("--method", choices=["GET", "POST"], default="GET", help="HTTP 方法")
    parser.add_argument("--timeout", type=int, default=30, help="超時秒數")
    
    args = parser.parse_args()
    
    try:
        # 直接執行測試（同步）
        findings = run_xss_test(
            url=args.url,
            param=args.param,
            method=args.method,
            timeout=args.timeout
        )
        
        # 輸出 JSON 結果
        output = {
            "target": args.url,
            "findings_count": len(findings),
            "vulnerable": len(findings) > 0,
            "findings": findings
        }
        
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"執行錯誤: {e}", exc_info=True)
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()  # ← 不需要 asyncio.run()！
