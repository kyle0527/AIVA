"""啟動所有真實連接器
請根據您的實際檔案結構，修改 services 列表中的模組路徑。
"""
import subprocess
import sys
import time

services = [
    # (模組路徑, 服務名稱, 端口)
    # 請修改這裡的路徑以匹配您專案中真實的 worker 位置
    ("services.function.function_sqli.aiva_func_sqli.smart_sqli_detector", "function_sqli", 8001),
    ("services.function.function_xss.aiva_func_xss.smart_xss_detector", "function_xss", 8002),
    ("services.function.function_idor.aiva_func_idor.smart_idor_detector", "function_idor", 8003),
    # 掃描服務（如果存在）
    # ("services.scan.aiva_scan.worker", "scan_service", 8000),
]

processes = []


def start():
    print("🚀 Starting Real Service Adapters...")
    print("這將啟動 HTTP 服務來包裝您的功能模組...")

    for module, name, port in services:
        print(f"  - Starting {name} on port {port} (Module: {module})...")
        p = subprocess.Popen(
            [sys.executable, "tools/service_adapter.py", "--module", module, "--name", name, "--port", str(port)]
        )
        processes.append(p)

    print("\n✅ All services starting. AI Core can now connect to them.")
    print("Press Ctrl+C to stop all services.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    start()
