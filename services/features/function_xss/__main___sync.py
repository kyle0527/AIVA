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
    sys.path.insert(0, project_root)

# 設定 Log
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("XssCLI_Sync")


def run_xss_test(url: str, param: str, method: str = "GET", timeout: int = 30):
    """完全同步的 XSS 測試
    
    不使用 async，直接用 requests 庫（同步）
    這只是一個簡化的展示。
    """
    import requests
    from .payload_generator import XssPayloadGenerator
    
    payloads = XssPayloadGenerator.generate_all_payloads()
    results = []
    
    for p in payloads:
        try:
            if method.upper() == "GET":
                target_url = f"{url}?{param}={p.payload}"
                r = requests.get(target_url, timeout=timeout)
            else:
                r = requests.post(url, data={param: p.payload}, timeout=timeout)

            if p.payload in r.text:
                results.append({
                    "vulnerable": True,
                    "payload": p.payload,
                    "type": p.context,
                    "method": method
                })
        except requests.RequestException as e:
            logger.debug(f"Request failed: {e}")
            
    return results

def main():
    parser = argparse.ArgumentParser(description="AIVA XSS Vulnerability Scanner CLI (Sync)")
    parser.add_argument("--url", required=True, help="目標 URL")
    parser.add_argument("--param", required=True, help="測試參數")
    parser.add_argument("--method", default="GET", choices=["GET", "POST"])
    
    args = parser.parse_args()

    try:
        findings = run_xss_test(args.url, args.param, args.method)
        print(json.dumps({
            "target": args.url,
            "status": "success",
            "findings_count": len(findings),
            "findings": findings
        }, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(json.dumps({
            "target": args.url,
            "status": "failed",
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()
