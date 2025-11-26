import asyncio
import sys
import os
import traceback

# 將專案根目錄添加到 Python 路徑中，以便進行絕對導入
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 現在我們可以使用與主應用程式相同的導入路徑
try:
    from services.scan.engines.python_engine.core_crawling_engine.http_client_hi import RetryingAsyncClient
    from services.scan.engines.python_engine.header_configuration import HeaderConfiguration
    print("✅ 模組導入成功。")
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    print("請確認您在 AIVA-git 的根目錄下執行此腳本。")
    sys.exit(1)

async def main():
    """
    一個最小化的測試，用於檢查 RetryingAsyncClient 是否可以發送 HTTP 請求。
    """
    print("\n--- 開始 HTTP 客戶端除錯測試 ---")
    target_url = "http://localhost:3000"
    print(f"🎯 測試目標 URL: {target_url}")

    try:
        print("🔧 正在初始化 HeaderConfiguration...")
        header_config = HeaderConfiguration()
        headers = header_config.get_default_headers()
        print("🔧 正在初始化 RetryingAsyncClient...")
        http_client = RetryingAsyncClient(headers=headers)
        
        print(f"🚀 正在嘗試對 {target_url} 發送 GET 請求...")
        
        # 執行請求並獲取回應
        response = await http_client.get(target_url, timeout=10)
        
        if response:
            print(f"✅ 請求成功！")
            print(f"   - 狀態碼: {response.status_code}")
            print(f"   - 回應長度: {len(response.text)} 字元")
            # 打印前 150 個字元以確認內容
            print(f"   - 回應內容預覽: {response.text[:150].replace('\n', ' ')}...")
        else:
            print("❌ 請求返回了 None，沒有收到任何回應。")

    except Exception as e:
        print(f"❌ 請求過程中發生嚴重錯誤！")
        print(f"   - 錯誤類型: {type(e).__name__}")
        print(f"   - 錯誤訊息: {e}")
        print("--- 錯誤追蹤 ---")
        traceback.print_exc()
        print("-----------------")

    finally:
        # 確保客戶端被關閉
        if 'http_client' in locals():
            await http_client.aclose()
            print("🔒 HTTP 客戶端已關閉。")
        print("\n--- 測試結束 ---")


if __name__ == "__main__":
    # 檢查靶場是否在運行
    print("ℹ️  請確保您的 Docker 靶場 (例如 Juice Shop) 正在運行中...")
    asyncio.run(main())
