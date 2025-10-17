"""
啟動 AIVA UI 面板 - 自動端口版本
這個腳本會自動尋找可用的端口來啟動 UI 面板
"""

from pathlib import Path
import sys

# 添加專案根目錄到路徑
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    """主程式進入點."""
    try:
        from services.core.aiva_core.ui_panel import start_auto_server

        print("[START] 啟動 AIVA UI 面板 (自動端口選擇)")
        print("=" * 50)

        # 啟動伺服器，自動選擇端口
        start_auto_server(
            mode="hybrid",  # 混合模式，支援 AI 和 UI 功能
            host="127.0.0.1",
            preferred_ports=[8080, 8081, 3000, 5000, 9000]  # 偏好的端口列表
        )

    except ImportError as e:
        print(f"[FAIL] 匯入錯誤: {e}")
        print("請確保在 AIVA 專案根目錄執行此腳本")
        print("並已安裝所需套件: pip install fastapi uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[U+1F44B] 使用者中斷，正在關閉伺服器...")
        sys.exit(0)
    except Exception as e:
        print(f"[FAIL] 啟動失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
