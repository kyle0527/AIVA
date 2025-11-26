"""
直接調用 Go 引擎,檢查輸入輸出

驗證:
1. Go 引擎收到的 JSON 格式
2. Go 引擎返回的資產內容
3. 是否真的發送 HTTP 請求
"""

import asyncio
import json
import subprocess
from pathlib import Path


async def test_go_engine_direct():
    """直接調用 Go 引擎"""
    
    go_scanner = Path("C:/D/fold7/AIVA-git/services/scan/engines/go_engine/scanner.exe")
    
    if not go_scanner.exists():
        print(f"❌ Go 掃描器不存在: {go_scanner}")
        return
    
    # 構造符合使用指南的輸入
    scan_input = {
        "scan_id": "direct_test_001",
        "targets": ["http://localhost:3000"],
        "concurrency": 5,
        "timeout": 10
    }
    
    print("=" * 60)
    print("發送給 Go 引擎的參數:")
    print(json.dumps(scan_input, indent=2))
    print()
    
    input_json = json.dumps(scan_input)
    
    # 調用 Go 掃描器
    proc = await asyncio.create_subprocess_exec(
        str(go_scanner),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await proc.communicate(input=input_json.encode('utf-8'))
    
    print("=" * 60)
    print(f"Go 引擎退出碼: {proc.returncode}")
    print()
    
    if stderr:
        print("stderr:")
        print(stderr.decode('utf-8', errors='ignore'))
        print()
    
    if stdout:
        print("stdout:")
        output = stdout.decode('utf-8', errors='ignore')
        print(output)
        print()
        
        # 嘗試解析 JSON
        try:
            result = json.loads(output)
            print("=" * 60)
            print("解析後的結果:")
            print(f"  資產數量: {len(result.get('assets', []))}")
            
            # 顯示前3個資產
            assets = result.get('assets', [])
            for i, asset in enumerate(assets[:3], 1):
                print(f"\n  資產 {i}:")
                print(f"    類型: {asset.get('type')}")
                print(f"    URL: {asset.get('value')}")
                if asset.get('details'):
                    print(f"    詳情: {asset['details']}")
            
            # 檢查是否有 response_headers
            if assets:
                first_asset = assets[0]
                if 'response_headers' in first_asset.get('details', {}):
                    print("\n  ✅ 資產包含 response_headers,證明真的發送了 HTTP 請求!")
                    headers = first_asset['details']['response_headers']
                    print(f"     Server: {headers.get('Server', 'N/A')}")
                    print(f"     Content-Type: {headers.get('Content-Type', 'N/A')}")
                else:
                    print("\n  ⚠️ 資產沒有 response_headers,可能沒有真正發送請求")
        
        except json.JSONDecodeError as e:
            print(f"❌ 無法解析 JSON: {e}")


async def main():
    await test_go_engine_direct()
    
    print("\n" + "=" * 60)
    print("測試完成")
    print("請同時檢查 Juice Shop 訪問日誌,確認是否收到請求")


if __name__ == "__main__":
    asyncio.run(main())
