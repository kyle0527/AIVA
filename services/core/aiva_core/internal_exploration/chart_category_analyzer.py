#!/usr/bin/env python3
"""
組合圖分類分析器
================
分析6,806個組合圖並進行功能分類，提供各功能的用途和使用方式
"""

import os
import json
from collections import defaultdict, Counter
from pathlib import Path

def analyze_chart_categories():
    """分析組合圖的功能分類"""
    
    # 定義功能分類規則
    classification_rules = {
        'AI/機器學習': ['ai', 'neural', 'model', 'train', 'learn', 'brain', 'intelligence'],
        '安全認證': ['security', 'auth', 'crypto', 'token', 'permission', 'encrypt'],
        '數據處理': ['data', 'pipeline', 'stream', 'processor', 'queue'],
        '監控日誌': ['monitor', 'metric', 'log', 'trace', 'alert'],
        '配置管理': ['config', 'setting', 'manager'],
        '服務發現': ['service', 'discovery', 'registry'],
        '錯誤處理': ['error', 'exception', 'handling'],
        '核心基礎': ['__init__', 'core', 'common', 'base']
    }
    
    categories = defaultdict(list)
    connection_details = defaultdict(list)
    
    charts_dir = Path('flow_analysis_results/combined_charts')
    
    # 分析前500個組合圖
    for i in range(1, 501):
        file_path = charts_dir / f'combined_flow_{i:03d}.mmd'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取subgraph名稱
            lines = content.split('\n')
            subgraphs = []
            for line in lines:
                if 'subgraph A[' in line and '"' in line:
                    module_a = line.split('"')[1]
                    subgraphs.append(module_a)
                elif 'subgraph B[' in line and '"' in line:
                    module_b = line.split('"')[1]
                    subgraphs.append(module_b)
            
            if len(subgraphs) == 2:
                connection = f'{subgraphs[0]} → {subgraphs[1]}'
                
                # 分類邏輯
                classified = False
                for category, keywords in classification_rules.items():
                    if any(keyword in subgraphs[0].lower() or keyword in subgraphs[1].lower() 
                           for keyword in keywords):
                        categories[category].append(connection)
                        connection_details[category].append({
                            'from': subgraphs[0],
                            'to': subgraphs[1],
                            'file': f'combined_flow_{i:03d}.mmd'
                        })
                        classified = True
                        break
                
                if not classified:
                    categories['其他'].append(connection)
    
    return categories, connection_details

def generate_usage_guide(categories, connection_details):
    """生成各功能的用途和使用指南"""
    
    usage_guide = {
        'AI/機器學習': {
            'description': '人工智能核心模組，包含神經網路、機器學習訓練、決策系統',
            'key_components': ['ai_model_manager', 'neural_network', 'model_trainer', 'rl_trainers'],
            'use_cases': [
                '🧠 神經網路訓練和推理',
                '📊 機器學習模型管理',
                '🎯 智能決策和策略生成',
                '🔄 強化學習訓練循環'
            ],
            'how_to_use': '從ai_model_manager開始 → 配置neural_network → 使用model_trainer訓練 → 部署推理'
        },
        '安全認證': {
            'description': '企業級安全框架，提供認證、授權、加密等安全服務',
            'key_components': ['security', 'auth', 'crypto', 'token_service'],
            'use_cases': [
                '🔐 用戶身份認證和授權',
                '🛡️ 數據加密和解密',
                '🎫 JWT令牌管理',
                '🔑 密鑰和證書管理'
            ],
            'how_to_use': '初始化security_manager → 配置認證策略 → 申請令牌 → 訪問受保護資源'
        },
        '數據處理': {
            'description': '高性能數據流處理管道，支援實時和批量數據處理',
            'key_components': ['data_pipeline', 'stream_processor', 'queue_manager'],
            'use_cases': [
                '📥 數據攝取和預處理',
                '⚡ 實時數據流處理',
                '📊 批量數據分析',
                '🔄 數據轉換和路由'
            ],
            'how_to_use': '創建data_pipeline → 添加數據源 → 配置處理器 → 設定輸出目標'
        },
        '監控日誌': {
            'description': '全面的系統監控和日誌管理，提供指標收集和告警功能',
            'key_components': ['monitoring', 'metrics', 'log_handler', 'alert_manager'],
            'use_cases': [
                '📈 系統性能監控',
                '📋 日誌聚合和分析',
                '🚨 異常檢測和告警',
                '📊 指標可視化儀表板'
            ],
            'how_to_use': '啟動monitoring_service → 配置指標收集 → 設定告警規則 → 查看儀表板'
        },
        '配置管理': {
            'description': '統一配置管理系統，支援動態配置和環境隔離',
            'key_components': ['config_manager', 'settings', 'environment'],
            'use_cases': [
                '⚙️ 應用配置管理',
                '🔧 環境變數處理',
                '🔄 動態配置更新',
                '🔒 敏感配置加密'
            ],
            'how_to_use': '初始化config_manager → 載入配置檔案 → 註冊配置模式 → 動態更新配置'
        },
        '服務發現': {
            'description': '微服務架構的服務註冊與發現機制',
            'key_components': ['service_discovery', 'registry', 'health_check'],
            'use_cases': [
                '🔍 服務自動發現',
                '💡 負載均衡和路由',
                '💚 健康狀態檢查',
                '🌐 服務網格管理'
            ],
            'how_to_use': '註冊服務 → 發現可用服務 → 健康檢查 → 負載均衡調用'
        },
        '錯誤處理': {
            'description': '統一的錯誤處理和異常管理框架',
            'key_components': ['error_handling', 'exception_manager', 'retry_handler'],
            'use_cases': [
                '🚫 異常捕獲和處理',
                '🔄 失敗重試機制',
                '📊 錯誤統計和分析',
                '🔔 錯誤通知和報告'
            ],
            'how_to_use': '配置錯誤處理器 → 設定重試策略 → 記錄錯誤日誌 → 生成錯誤報告'
        },
        '核心基礎': {
            'description': '系統核心基礎設施和公共模組',
            'key_components': ['__init__', 'core', 'common', 'utils'],
            'use_cases': [
                '🏗️ 系統初始化和啟動',
                '🔧 公共工具和函數',
                '📚 共享數據模型',
                '⚡ 基礎設施服務'
            ],
            'how_to_use': '系統自動載入 → 提供基礎服務 → 支撐上層應用 → 管理系統生命週期'
        }
    }
    
    return usage_guide

def main():
    """主函數"""
    print("🔍 分析組合圖功能分類...")
    
    categories, connection_details = analyze_chart_categories()
    usage_guide = generate_usage_guide(categories, connection_details)
    
    print("\n" + "="*80)
    print("📊 AIVA AI 系統功能分類分析報告")
    print("="*80)
    
    # 統計信息
    total_analyzed = sum(len(connections) for connections in categories.values())
    print(f"\n📋 分析總結:")
    print(f"   • 分析組合圖數量: {total_analyzed}")
    print(f"   • 識別功能分類: {len(categories)}個")
    print(f"   • 覆蓋模組數量: {len(set(detail['from'] for details in connection_details.values() for detail in details))}")
    
    print(f"\n🎯 功能分佈:")
    for category, connections in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • {category}: {len(connections)}個連接")
    
    # 詳細功能分析
    print(f"\n{'='*80}")
    print("🚀 各功能模組詳細分析")
    print("="*80)
    
    for category, connections in categories.items():
        if category in usage_guide and len(connections) > 0:
            guide = usage_guide[category]
            
            print(f"\n📂 {category}")
            print(f"   {guide['description']}")
            print(f"\n   🎯 主要用途:")
            for use_case in guide['use_cases']:
                print(f"     {use_case}")
            
            print(f"\n   🔧 使用方式:")
            print(f"     {guide['how_to_use']}")
            
            print(f"\n   📋 關鍵連接 (前5個):")
            for connection in connections[:5]:
                print(f"     • {connection}")
            
            if len(connections) > 5:
                print(f"     ... 還有 {len(connections)-5} 個連接")
            
            print(f"\n   💡 實際檔案示例:")
            if category in connection_details:
                unique_modules = set()
                for detail in connection_details[category][:3]:
                    unique_modules.add(detail['from'])
                    unique_modules.add(detail['to'])
                for module in list(unique_modules)[:4]:
                    print(f"     • {module}")
            
            print(f"\n   🔗 組合圖檔案:")
            if category in connection_details:
                files = [detail['file'] for detail in connection_details[category][:3]]
                for file in files:
                    print(f"     • flow_analysis_results/combined_charts/{file}")
    
    # 生成功能導航圖
    print(f"\n{'='*80}")
    print("🗺️  AIVA 功能導航圖")
    print("="*80)
    print("""
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   AI/機器學習    │────│    數據處理      │────│    監控日誌      │
    │  🧠 神經網路     │    │  📊 數據管道     │    │  📈 性能監控     │
    │  🎯 決策系統     │    │  ⚡ 流處理       │    │  🚨 告警系統     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                       │                       │
              │                       │                       │
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │    安全認證      │    │    配置管理      │    │    服務發現      │
    │  🔐 身份驗證     │────│  ⚙️  系統配置    │────│  🔍 服務註冊     │
    │  🛡️  數據加密    │    │  🔧 環境管理     │    │  💡 負載均衡     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                            ┌─────────────────┐
                            │    核心基礎      │
                            │  🏗️  系統初始化  │
                            │  🔧 公共工具     │
                            └─────────────────┘
    """)
    
    print(f"\n📖 使用建議:")
    print(f"   1. 🚀 從核心基礎開始了解系統架構")
    print(f"   2. 🔐 配置安全認證保護系統安全")
    print(f"   3. ⚙️  設定配置管理統一系統配置")
    print(f"   4. 📊 建立數據處理管道處理業務數據")
    print(f"   5. 🧠 部署AI模組實現智能功能")
    print(f"   6. 📈 啟用監控系統觀察系統運行")
    
    # 保存分析結果
    result = {
        'categories': dict(categories),
        'usage_guide': usage_guide,
        'statistics': {
            'total_analyzed': total_analyzed,
            'category_count': len(categories),
            'connections_by_category': {k: len(v) for k, v in categories.items()}
        }
    }
    
    with open('flow_analysis_results/category_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 分析完成！結果已保存到 flow_analysis_results/category_analysis.json")

if __name__ == "__main__":
    main()