"""
AIVA UI 面板示範
展示如何使用 UI、AI 或混合模式
"""

import sys


def demo_ui_mode() -> None:
    """示範純 UI 模式."""
    print("\n" + "="*70)
    print("   示範 1: 純 UI 模式")
    print("="*70 + "\n")

    from services.core.aiva_core.ui_panel import Dashboard

    dashboard = Dashboard(mode="ui")

    # 建立掃描任務
    task = dashboard.create_scan_task(
        target_url="https://example.com",
        scan_type="full",
    )
    print(f"\n任務建立成功: {task['task_id']}")
    print(f"建立方式: {task['created_by']}")

    # 執行漏洞檢測
    detection = dashboard.detect_vulnerability(
        vuln_type="xss",
        target="https://example.com/search",
    )
    print(f"\n檢測已啟動: {detection['vuln_type'].upper()}")
    print(f"檢測方式: {detection['method']}")

    # 讀取程式碼
    code_result = dashboard.read_code("services/scan/aiva_scan/worker.py")
    if code_result["status"] == "success":
        lines = len(code_result["content"].splitlines())
        print(f"\n檔案讀取成功: {code_result['path']}")
        print(f"總行數: {lines}")

    # 顯示統計
    stats = dashboard.get_stats()
    print(f"\n統計資訊:")
    print(f"  模式: {stats['mode_display']}")
    print(f"  任務數: {stats['total_tasks']}")
    print(f"  檢測數: {stats['total_detections']}")


def demo_ai_mode() -> None:
    """示範純 AI 模式."""
    print("\n" + "="*70)
    print("   示範 2: 純 AI 模式")
    print("="*70 + "\n")

    from services.core.aiva_core.ui_panel import Dashboard

    dashboard = Dashboard(mode="ai")

    # 建立掃描任務 (AI 自動執行)
    task = dashboard.create_scan_task(
        target_url="https://test.com",
        scan_type="quick",
    )
    print(f"\n任務建立成功: {task['task_id']}")
    print(f"建立方式: {task['created_by']}")
    print(f"AI 信心度: {task['ai_result'].get('confidence', 0):.2%}")

    # 分析程式碼 (AI 執行)
    analysis = dashboard.analyze_code("services/core/aiva_core/app.py")
    if analysis["status"] == "success":
        print(f"\n程式碼分析完成:")
        print(f"  使用工具: {analysis.get('tool_used', 'N/A')}")
        print(f"  信心度: {analysis.get('confidence', 0):.2%}")

    # AI 歷史
    history = dashboard.get_ai_history()
    print(f"\nAI 執行歷史: {len(history)} 筆記錄")


def demo_hybrid_mode() -> None:
    """示範混合模式."""
    print("\n" + "="*70)
    print("   示範 3: 混合模式 (UI + AI)")
    print("="*70 + "\n")

    from services.core.aiva_core.ui_panel import Dashboard

    dashboard = Dashboard(mode="hybrid")

    # 使用 UI 建立任務
    print("\n--- 使用 UI 建立任務 ---")
    task1 = dashboard.create_scan_task(
        target_url="https://ui-target.com",
        scan_type="full",
        use_ai=False,  # 明確指定使用 UI
    )
    print(f"任務 1: {task1['task_id']} (方式: {task1['created_by']})")

    # 使用 AI 建立任務
    print("\n--- 使用 AI 建立任務 ---")
    task2 = dashboard.create_scan_task(
        target_url="https://ai-target.com",
        scan_type="full",
        use_ai=True,  # 明確指定使用 AI
    )
    print(f"任務 2: {task2['task_id']} (方式: {task2['created_by']})")

    # 自動模式 (根據 hybrid 設定)
    print("\n--- 自動模式 (預設使用 AI) ---")
    task3 = dashboard.create_scan_task(
        target_url="https://auto-target.com",
        scan_type="quick",
        # use_ai=None  # 自動決定
    )
    print(f"任務 3: {task3['task_id']} (方式: {task3['created_by']})")

    # 統計
    stats = dashboard.get_stats()
    print(f"\n統計資訊:")
    print(f"  模式: {stats['mode_display']}")
    print(f"  總任務: {stats['total_tasks']}")
    print(f"  AI 知識庫: {stats.get('ai_chunks', 0)} 個片段")


def start_web_server(mode: str = "hybrid") -> None:
    """啟動 Web 伺服器."""
    print("\n" + "="*70)
    print("   啟動 Web UI 伺服器")
    print("="*70 + "\n")

    from services.core.aiva_core.ui_panel import start_ui_server

    start_ui_server(mode=mode, host="127.0.0.1", port=8080)


def main() -> None:
    """主程式."""
    if len(sys.argv) < 2:
        print("AIVA UI 面板示範")
        print("\n用法:")
        print("  python demo_ui_panel.py ui          # 示範純 UI 模式")
        print("  python demo_ui_panel.py ai          # 示範純 AI 模式")
        print("  python demo_ui_panel.py hybrid      # 示範混合模式")
        print("  python demo_ui_panel.py server-ui   # 啟動 UI 模式伺服器")
        print("  python demo_ui_panel.py server-ai   # 啟動 AI 模式伺服器")
        print("  python demo_ui_panel.py server      # 啟動混合模式伺服器")
        return

    mode = sys.argv[1].lower()

    if mode == "ui":
        demo_ui_mode()
    elif mode == "ai":
        demo_ai_mode()
    elif mode == "hybrid":
        demo_hybrid_mode()
    elif mode == "server-ui":
        start_web_server("ui")
    elif mode == "server-ai":
        start_web_server("ai")
    elif mode == "server":
        start_web_server("hybrid")
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    main()
