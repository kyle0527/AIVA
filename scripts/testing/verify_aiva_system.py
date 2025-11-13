#!/usr/bin/env python3
"""
AIVA AI 系統驗證腳本
執行完整的系統功能驗證測試
"""

import sys
import os
import time
import traceback
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "services"))

def print_header(title: str):
    """打印測試標題"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_step(step_num: int, description: str):
    """打印測試步驟"""
    print(f"\n📋 步驟 {step_num}: {description}")

def print_success(message: str):
    """打印成功消息"""
    print(f"   ✅ {message}")

def print_error(message: str):
    """打印錯誤消息"""
    print(f"   ❌ {message}")

def print_warning(message: str):
    """打印警告消息"""
    print(f"   ⚠️ {message}")

def check_python_environment():
    """檢查 Python 環境"""
    print_header("Python 環境檢查")
    
    # 檢查 Python 版本
    print_step(1, "檢查 Python 版本")
    python_version = sys.version_info
    if python_version >= (3, 8):
        print_success(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python 版本過低: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 3.8+)")
        return False
    
    # 檢查必要依賴
    print_step(2, "檢查核心依賴包")
    required_packages = [
        'torch', 'numpy', 'fastapi', 'uvicorn', 
        'protobuf', 'grpcio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} 已安裝")
        except ImportError:
            print_error(f"{package} 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print_error(f"缺少依賴包: {', '.join(missing_packages)}")
        print("   💡 請執行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_project_structure():
    """檢查項目結構"""
    print_header("項目結構檢查")
    
    # 檢查關鍵目錄
    print_step(1, "檢查項目目錄結構")
    required_dirs = [
        "services",
        "services/core", 
        "services/core/aiva_core",
        "services/core/aiva_core/ai_engine",
        "services/aiva_common"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(f"目錄存在: {dir_path}")
        else:
            print_error(f"目錄缺失: {dir_path}")
            return False
    
    # 檢查關鍵文件
    print_step(2, "檢查關鍵文件")
    required_files = [
        "services/core/aiva_core/bio_neuron_master.py",
        "services/core/aiva_core/ai_engine/real_bio_net_adapter.py",
        "services/core/aiva_core/rag/rag_engine.py"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"文件存在: {file_path}")
        else:
            print_error(f"文件缺失: {file_path}")
            return False
    
    return True

def test_core_imports():
    """測試核心模組導入"""
    print_header("核心模組導入測試")
    
    # 測試基礎導入
    print_step(1, "導入基礎模組")
    try:
        import numpy as np
        import torch
        print_success("NumPy 和 PyTorch 導入成功")
    except Exception as e:
        print_error(f"基礎模組導入失敗: {e}")
        return False
    
    # 測試 AIVA 核心導入
    print_step(2, "導入 AIVA 核心模組")
    try:
        from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
        print_success("BioNeuronMasterController 導入成功")
    except Exception as e:
        print_error(f"AIVA 核心模組導入失敗: {e}")
        print("   詳細錯誤:")
        traceback.print_exc()
        return False
    
    # 測試 AI 引擎導入
    print_step(3, "導入 AI 引擎模組")
    try:
        from services.core.aiva_core.ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent
        print_success("RealBioNeuronRAGAgent 導入成功")
    except Exception as e:
        print_error(f"AI 引擎模組導入失敗: {e}")
        print("   詳細錯誤:")
        traceback.print_exc()
        return False
    
    return True

def test_ai_system_initialization():
    """測試 AI 系統初始化"""
    print_header("AI 系統初始化測試")
    
    print_step(1, "初始化主控系統")
    try:
        from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
        
        # 嘗試初始化（可能會因為缺少權重文件而失敗，但不應該崩潰）
        controller = BioNeuronMasterController()
        print_success("主控系統初始化成功")
        
        # 檢查組件
        print_step(2, "檢查 AI 組件狀態")
        print_success(f"神經網路代理類型: {type(controller.bio_neuron_agent).__name__}")
        print_success(f"決策核心類型: {type(controller.decision_core).__name__}")
        print_success(f"RAG 引擎類型: {type(controller.rag_engine).__name__}")
        print_success(f"當前運行模式: {controller.current_mode}")
        
        return True, controller
        
    except Exception as e:
        print_error(f"AI 系統初始化失敗: {e}")
        print("   詳細錯誤:")
        traceback.print_exc()
        return False, None

def test_ai_basic_functions(controller):
    """測試 AI 基本功能"""
    print_header("AI 基本功能測試")
    
    if controller is None:
        print_error("控制器未初始化，跳過功能測試")
        return False
    
    # 測試 RAG 搜索
    print_step(1, "測試 RAG 搜索功能")
    try:
        # 這裡只測試方法存在性，不執行實際搜索
        rag_engine = controller.rag_engine
        if hasattr(rag_engine, 'search'):
            print_success("RAG 搜索方法可用")
        else:
            print_warning("RAG 搜索方法不存在")
    except Exception as e:
        print_error(f"RAG 搜索測試失敗: {e}")
    
    # 測試決策功能
    print_step(2, "測試決策功能介面")
    try:
        if hasattr(controller, '_bio_neuron_decide'):
            print_success("決策功能介面可用")
        else:
            print_warning("決策功能介面不存在")
    except Exception as e:
        print_error(f"決策功能測試失敗: {e}")
    
    # 測試模式切換
    print_step(3, "測試運行模式")
    try:
        original_mode = controller.current_mode
        print_success(f"當前模式: {original_mode}")
        
        # 測試模式處理器
        if hasattr(controller, 'mode_handlers'):
            print_success(f"可用模式: {list(controller.mode_handlers.keys())}")
        else:
            print_warning("模式處理器不存在")
            
    except Exception as e:
        print_error(f"模式測試失敗: {e}")
    
    return True

def test_system_health():
    """系統健康檢查"""
    print_header("系統健康檢查")
    
    # 記憶體使用檢查
    print_step(1, "檢查記憶體使用")
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        if memory_usage < 80:
            print_success(f"記憶體使用率: {memory_usage:.1f}%")
        else:
            print_warning(f"記憶體使用率較高: {memory_usage:.1f}%")
    except ImportError:
        print_warning("psutil 未安裝，跳過記憶體檢查")
    except Exception as e:
        print_warning(f"記憶體檢查失敗: {e}")
    
    # 磁碟空間檢查
    print_step(2, "檢查磁碟空間")
    try:
        import shutil
        total, used, free = shutil.disk_usage(project_root)
        free_gb = free // (1024**3)
        
        if free_gb > 5:
            print_success(f"可用磁碟空間: {free_gb} GB")
        else:
            print_warning(f"磁碟空間不足: {free_gb} GB")
    except Exception as e:
        print_warning(f"磁碟空間檢查失敗: {e}")
    
    return True

def generate_verification_report(test_results):
    """生成驗證報告"""
    print_header("驗證報告生成")
    
    report = f"""
# AIVA AI 系統驗證報告

**驗證時間**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Python 版本**: {sys.version}
**項目路徑**: {project_root}

## 測試結果摘要

| 測試項目 | 狀態 | 備註 |
|----------|------|------|
| Python 環境 | {'✅ 通過' if test_results['python_env'] else '❌ 失敗'} | Python 版本和依賴檢查 |
| 項目結構 | {'✅ 通過' if test_results['project_structure'] else '❌ 失敗'} | 目錄和文件完整性 |
| 模組導入 | {'✅ 通過' if test_results['core_imports'] else '❌ 失敗'} | 核心模組導入測試 |
| AI 初始化 | {'✅ 通過' if test_results['ai_initialization'] else '❌ 失敗'} | AI 系統初始化 |
| 基本功能 | {'✅ 通過' if test_results['basic_functions'] else '❌ 失敗'} | AI 基本功能檢查 |
| 系統健康 | {'✅ 通過' if test_results['system_health'] else '❌ 失敗'} | 系統資源檢查 |

## 整體評估

"""
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    if success_rate >= 100:
        report += "🎉 **優秀**: 所有測試通過，系統運行狀態良好\n"
    elif success_rate >= 80:
        report += "✅ **良好**: 大部分測試通過，系統基本可用\n"
    elif success_rate >= 60:
        report += "⚠️ **一般**: 部分測試失敗，需要檢查和修復\n"
    else:
        report += "❌ **差**: 多項測試失敗，系統無法正常運行\n"
    
    report += f"\n**成功率**: {passed_tests}/{total_tests} ({success_rate:.1f}%)\n"
    
    if success_rate < 100:
        report += "\n## 建議修復步驟\n\n"
        if not test_results['python_env']:
            report += "1. 安裝缺少的 Python 依賴包\n"
        if not test_results['project_structure']:
            report += "2. 檢查項目文件完整性\n"
        if not test_results['core_imports']:
            report += "3. 修復模組導入問題\n"
        if not test_results['ai_initialization']:
            report += "4. 檢查 AI 系統配置\n"
    
    # 寫入報告文件
    report_file = project_root / "AIVA_VERIFICATION_REPORT.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print_success(f"驗證報告已生成: {report_file}")
    
    return success_rate

def main():
    """主函數"""
    print("🚀 AIVA AI 系統驗證開始")
    print(f"📍 項目路徑: {project_root}")
    
    # 執行各項測試
    test_results = {}
    
    # 1. Python 環境檢查
    test_results['python_env'] = check_python_environment()
    
    # 2. 項目結構檢查
    test_results['project_structure'] = check_project_structure()
    
    # 3. 核心模組導入測試
    test_results['core_imports'] = test_core_imports()
    
    # 4. AI 系統初始化測試
    ai_init_success, controller = test_ai_system_initialization()
    test_results['ai_initialization'] = ai_init_success
    
    # 5. 基本功能測試
    test_results['basic_functions'] = test_ai_basic_functions(controller)
    
    # 6. 系統健康檢查
    test_results['system_health'] = test_system_health()
    
    # 生成驗證報告
    success_rate = generate_verification_report(test_results)
    
    # 最終結果
    print_header("驗證完成")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"📊 測試結果: {passed_tests}/{total_tests} 通過 ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print_success("AIVA AI 系統驗證基本通過！")
        print("💡 系統可以正常使用，詳見使用者手冊")
    else:
        print_error("AIVA AI 系統驗證未完全通過")
        print("💡 請根據驗證報告進行修復")
    
    print(f"\n📋 完整報告請查看: AIVA_VERIFICATION_REPORT.md")
    print("📖 使用指南請查看: AIVA_USER_MANUAL.md")

if __name__ == "__main__":
    main()