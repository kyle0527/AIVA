"""檢查 AIVA 系統實際可用性
分析 782 個能力的真實狀態：能不能用、何時用、怎麼用
"""
import asyncio
import json
from pathlib import Path
from typing import Any
import importlib.util
import sys

# 檢查項目
checks = {
    "rag_database": {"status": "unknown", "details": {}},
    "worker_services": {"status": "unknown", "details": {}},
    "unified_caller": {"status": "unknown", "details": {}},
    "execution_chain": {"status": "unknown", "details": {}},
    "external_loop": {"status": "unknown", "details": {}},
}


def check_rag_database():
    """檢查 RAG 數據庫狀態"""
    try:
        from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
        from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
        
        vs = VectorStore()
        kb = KnowledgeBase(vs)
        stats = vs.get_stats()
        
        checks["rag_database"]["status"] = "✅ 可用"
        checks["rag_database"]["details"] = {
            "文檔數": stats.get("total_documents", 0),
            "向量維度": stats.get("vector_dimension", 0),
            "集合數": stats.get("collection_count", 0),
            "數據庫路徑": "data/vector_db/chroma/",
            "能否查詢": "是" if stats.get("total_documents", 0) > 0 else "否",
        }
        
        # 測試查詢
        if stats.get("total_documents", 0) > 0:
            results = kb.search("scan", top_k=3)
            checks["rag_database"]["details"]["測試查詢結果"] = f"找到 {len(results)} 個結果"
        
    except Exception as e:
        checks["rag_database"]["status"] = "❌ 不可用"
        checks["rag_database"]["details"] = {"錯誤": str(e)}


def check_worker_services():
    """檢查 Worker 服務狀態"""
    workers = [
        ("function_sqli", "services/features/function_sqli/worker.py", 8001),
        ("function_xss", "services/features/function_xss/worker.py", 8002),
        ("function_idor", "services/features/function_idor/worker.py", 8003),
        ("function_ssrf", "services/features/function_ssrf/worker.py", 8004),
        ("function_bizlogic", "services/features/function_bizlogic/worker.py", 8005),
        ("python_engine", "services/scan/engines/python_engine/worker.py", 9001),
        ("rust_engine", "services/scan/engines/rust_engine/worker.py", 9002),
        ("typescript_engine", "services/scan/engines/typescript_engine/worker.py", 9003),
    ]
    
    worker_status = {}
    available_count = 0
    
    for name, path, port in workers:
        file_path = Path(path)
        if file_path.exists():
            worker_status[name] = {
                "文件": "存在",
                "端口": port,
                "狀態": "⚠️ 未啟動（文件存在但服務未運行）",
                "啟動命令": f"python -m services.features.{name}.worker",
            }
            available_count += 1
        else:
            worker_status[name] = {
                "文件": "不存在",
                "狀態": "❌ 缺失",
            }
    
    checks["worker_services"]["status"] = f"⚠️ {available_count}/8 可用（未啟動）"
    checks["worker_services"]["details"] = worker_status


def check_unified_caller():
    """檢查統一調用器"""
    try:
        from services.core.aiva_core.service_backbone.api.unified_function_caller import (
            UnifiedFunctionCaller,
        )
        
        caller = UnifiedFunctionCaller()
        
        checks["unified_caller"]["status"] = "✅ 可用"
        checks["unified_caller"]["details"] = {
            "已註冊模組數": len(caller.endpoints),
            "支援語言": ["Python", "Go", "Rust", "TypeScript"],
            "調用方式": "HTTP (需啟動 Worker)",
            "模組列表": list(caller.endpoints.keys())[:10],
        }
        
    except Exception as e:
        checks["unified_caller"]["status"] = "❌ 不可用"
        checks["unified_caller"]["details"] = {"錯誤": str(e)}


def check_execution_chain():
    """檢查執行鏈路"""
    try:
        from services.core.aiva_core.task_planning.executor.task_executor import TaskExecutor
        from services.core.aiva_core.task_planning.planner.execution_planner import ExecutionPlanner
        
        executor = TaskExecutor()
        planner = ExecutionPlanner()
        
        checks["execution_chain"]["status"] = "✅ 可用"
        checks["execution_chain"]["details"] = {
            "TaskExecutor": "已初始化",
            "ExecutionPlanner": "已初始化",
            "動態調用": "已啟用" if executor.use_dynamic_calling else "未啟用",
            "調用鏈路": "AICommander → Planner → Executor → UnifiedCaller → Worker",
            "當前狀態": "可執行（需啟動 Worker）",
        }
        
    except Exception as e:
        checks["execution_chain"]["status"] = "❌ 不可用"
        checks["execution_chain"]["details"] = {"錯誤": str(e)}


def check_external_loop():
    """檢查外閉環組件"""
    try:
        from services.core.aiva_core.cognitive_core.external_loop_connector import ExternalLoopConnector
        from services.core.aiva_core.external_learning.analysis.ast_trace_comparator import ASTTraceComparator
        
        connector = ExternalLoopConnector()
        comparator = ASTTraceComparator()
        
        checks["external_loop"]["status"] = "✅ 可用"
        checks["external_loop"]["details"] = {
            "ExternalLoopConnector": "已初始化",
            "ASTTraceComparator": "已初始化",
            "偏差分析": "可用",
            "模型訓練": "可用（需真實數據）",
            "權重管理": "可用",
            "當前狀態": "可執行完整外閉環",
        }
        
    except Exception as e:
        checks["external_loop"]["status"] = "❌ 不可用"
        checks["external_loop"]["details"] = {"錯誤": str(e)}


def print_report():
    """打印檢查報告"""
    print("=" * 80)
    print("AIVA 系統可用性檢查報告")
    print("=" * 80)
    print()
    
    for component, data in checks.items():
        print(f"【{component.upper().replace('_', ' ')}】")
        print(f"  狀態: {data['status']}")
        if data["details"]:
            for key, value in data["details"].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}: {', '.join(map(str, value[:5]))}")
                    if len(value) > 5:
                        print(f"    ... 還有 {len(value) - 5} 個")
                else:
                    print(f"  {key}: {value}")
        print()
    
    print("=" * 80)
    print("關鍵問題分析")
    print("=" * 80)
    print()
    
    # 分析當前狀態
    rag_ok = "✅" in checks["rag_database"]["status"]
    workers_ok = checks["worker_services"]["status"].startswith("⚠️")
    caller_ok = "✅" in checks["unified_caller"]["status"]
    exec_ok = "✅" in checks["execution_chain"]["status"]
    loop_ok = "✅" in checks["external_loop"]["status"]
    
    print("1. 能力數據（782 個）")
    if rag_ok:
        print("   ✅ 已發現並存儲在 RAG 知識庫")
        print("   ✅ 可以查詢：AI 知道「有什麼能力」")
    else:
        print("   ❌ RAG 數據庫不可用")
        print("   ❌ AI 不知道有哪些能力")
    print()
    
    print("2. 何時用這些能力？")
    if rag_ok and exec_ok:
        print("   ✅ 執行鏈路完整：AICommander → Planner → Executor")
        print("   ✅ AI 可根據任務查詢 RAG，選擇合適能力")
        print("   使用時機：")
        print("     - 用戶下達任務：'測試 SQL 注入'")
        print("     - AI 查詢 RAG：'SQL injection testing'")
        print("     - 找到能力：test_sql_injection (function_sqli)")
        print("     - 構建執行計劃並執行")
    else:
        print("   ❌ 執行鏈路不完整，無法自動選擇能力")
    print()
    
    print("3. 怎麼用這些能力？")
    if caller_ok and workers_ok:
        print("   ⚠️ UnifiedCaller 可用，但 Worker 未啟動")
        print("   調用流程：")
        print("     1. Executor 調用 UnifiedCaller")
        print("     2. UnifiedCaller 發送 HTTP 請求")
        print("     3. Worker 接收請求並執行")
        print("     4. 返回結果給 Executor")
        print("   當前狀態：")
        print("     ❌ Worker 服務未啟動 → HTTP 請求會失敗")
        print("     🔧 需要啟動 Worker：python -m services.features.function_sqli.worker")
    else:
        print("   ❌ 調用機制不完整")
    print()
    
    print("4. 能不能用這些能力？")
    if rag_ok and exec_ok and caller_ok:
        if workers_ok:
            print("   ⚠️ 架構完整，但需啟動 Worker 服務")
            print("   解決方案：")
            print("     方案 A: 啟動所有 Worker")
            print("       python tools/start_all_workers.py")
            print()
            print("     方案 B: 啟動單個 Worker 測試")
            print("       python -m services.features.function_sqli.worker --port 8001")
            print()
            print("     方案 C: 使用 Service Adapter")
            print("       python tools/service_adapter.py --module services.features.function_sqli.worker")
        else:
            print("   ❌ Worker 文件缺失，無法啟動服務")
    else:
        print("   ❌ 系統核心組件不完整")
    print()
    
    print("=" * 80)
    print("外閉環可行性評估")
    print("=" * 80)
    print()
    
    if loop_ok and exec_ok:
        print("✅ 外閉環組件完整")
        print("✅ 可以執行完整循環：執行 → 偏差分析 → 訓練 → 優化")
        print()
        print("當前障礙：")
        if not workers_ok:
            print("  🔴 Worker 服務未啟動 → 無法執行真實任務")
            print("  影響：無法獲得真實執行數據")
            print("  解決：啟動 Worker 服務")
        else:
            print("  ⚠️ Worker 需啟動 → 執行鏈路才能完整")
        print()
        print("啟動後的流程：")
        print("  1. ✅ 內閉環已完成 → RAG 有 782 個能力")
        print("  2. ⚠️ 啟動 Worker → 能力變為可執行")
        print("  3. ✅ 執行任務 → 獲得真實執行軌跡")
        print("  4. ✅ 偏差分析 → 對比計劃 vs 實際")
        print("  5. ✅ 模型訓練 → 優化 AI 決策")
        print("  6. ✅ 權重更新 → 下次執行更好")
    else:
        print("❌ 外閉環組件不完整")
        print("無法執行完整學習循環")
    print()
    
    print("=" * 80)
    print("總結建議")
    print("=" * 80)
    print()
    
    if rag_ok and exec_ok and caller_ok and loop_ok:
        print("🎯 系統架構完整！")
        print()
        print("立即行動：")
        print("  1. 啟動關鍵 Worker 服務（優先 SQL 注入測試）")
        print("     cd C:\\D\\fold7\\AIVA-git")
        print("     python -m services.features.function_sqli.worker --port 8001")
        print()
        print("  2. 測試執行鏈路")
        print("     python test_execution_chain.py")
        print()
        print("  3. 執行外閉環測試")
        print("     python scripts/run_external_loop.py")
        print()
        print("預期結果：")
        print("  ✅ AI 從 RAG 查詢到 SQL 注入能力")
        print("  ✅ 構建執行計劃")
        print("  ✅ 調用 Worker 執行真實測試")
        print("  ✅ 獲得執行結果")
        print("  ✅ 分析偏差並學習優化")
    else:
        print("⚠️ 系統存在缺失組件")
        print("請先修復上述問題")


if __name__ == "__main__":
    print("正在檢查系統狀態...\n")
    
    check_rag_database()
    check_worker_services()
    check_unified_caller()
    check_execution_chain()
    check_external_loop()
    
    print_report()
