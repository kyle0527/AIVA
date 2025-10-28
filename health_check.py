# AIVA 系統健康檢查腳本
import sys
import os
sys.path.insert(0, "services")

def check_schemas():
    """檢查 AIVA Common Schemas 可用性"""
    try:
        from aiva_common.schemas.base import MessageHeader
        from aiva_common.schemas.findings import Target, Vulnerability
        from aiva_common.enums import ModuleName
        
        # 測試建立實例
        header = MessageHeader(
            message_id="health_check_001",
            trace_id="trace_001", 
            source_module=ModuleName.CORE
        )
        return "✅ Schemas OK (完全可用)"
    except ImportError as e:
        return f"❌ Schemas: {e}"
    except Exception as e:
        return f"⚠️ Schemas: 載入成功但測試失敗 - {e}"

def check_tools():
    """檢查專業分析工具可用性"""
    import subprocess
    tools = {}
    
    try:
        result = subprocess.run(["go", "version"], capture_output=True, check=True, text=True)
        tools["Go"] = f"✅ {result.stdout.strip().split()[2]}"
    except:
        tools["Go"] = "❌ 未安裝或不可用"
    
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, check=True, text=True)
        version = result.stdout.strip().split()[1]
        tools["Rust"] = f"✅ {version}"
    except:
        tools["Rust"] = "❌ 未安裝或不可用"
        
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, check=True, text=True)
        tools["Node.js"] = f"✅ {result.stdout.strip()}"
    except:
        tools["Node.js"] = "❌ 未安裝或不可用"
    
    return tools

def check_ai_explorer():
    """檢查 AI 系統探索器可用性"""
    try:
        if os.path.exists("ai_system_explorer_v3.py"):
            return "✅ ai_system_explorer_v3.py 存在"
        else:
            return "❌ ai_system_explorer_v3.py 不存在"
    except Exception as e:
        return f"⚠️ 檢查失敗: {e}"

def check_directories():
    """檢查關鍵目錄結構"""
    critical_dirs = [
        "services/aiva_common/schemas",
        "reports/ai_diagnostics", 
        "logs"
    ]
    
    status = {}
    for dir_path in critical_dirs:
        if os.path.exists(dir_path):
            status[dir_path] = "✅ 存在"
        else:
            status[dir_path] = "❌ 不存在"
            # 嘗試建立目錄
            try:
                os.makedirs(dir_path, exist_ok=True)
                status[dir_path] = "✅ 已建立"
            except:
                status[dir_path] = "❌ 建立失敗"
    
    return status

if __name__ == "__main__":
    print("🔍 AIVA 系統健康檢查")
    print("=" * 50)
    print(f"📂 工作目錄: {os.getcwd()}")
    print(f"🐍 Python 版本: {sys.version.split()[0]}")
    print()
    
    # Schema 檢查
    print(f"🧬 Schema 狀態: {check_schemas()}")
    print()
    
    # 專業工具檢查
    tools = check_tools()
    print("🛠️ 專業工具狀態:")
    for tool, status in tools.items():
        print(f"   {tool}: {status}")
    print()
    
    # AI 探索器檢查
    print(f"🤖 AI 探索器: {check_ai_explorer()}")
    print()
    
    # 目錄結構檢查
    dirs = check_directories()
    print("📁 關鍵目錄:")
    for dir_path, status in dirs.items():
        print(f"   {dir_path}: {status}")
    print()
    
    # 整體健康評估
    schema_ok = "✅" in check_schemas()
    tools_ok = sum(1 for status in tools.values() if "✅" in status) >= 2
    explorer_ok = "✅" in check_ai_explorer()
    dirs_ok = all("✅" in status for status in dirs.values())
    
    if schema_ok and tools_ok and explorer_ok and dirs_ok:
        print("🎉 系統健康狀態: 優秀 (所有組件正常)")
    elif schema_ok and tools_ok:
        print("✅ 系統健康狀態: 良好 (核心功能正常)")
    elif schema_ok:
        print("⚠️ 系統健康狀態: 部分可用 (Schema 正常但工具缺失)")
    else:
        print("❌ 系統健康狀態: 需要修復 (關鍵組件異常)")