#!/usr/bin/env python3
"""
AIVA 數據合約完成度綜合分析工具
分析系統中合約的完成度、覆蓋率、數據填充率
"""

import sqlite3
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

def analyze_database_metrics():
    """分析資料庫中的合約指標"""
    db_path = "logs/contract_metrics.db"
    metrics = {
        'database_exists': os.path.exists(db_path),
        'total_records': 0,
        'latest_metrics': None,
        'alert_count': 0,
        'coverage_rate': 0.0,
        'health_status': 'unknown'
    }
    
    if not metrics['database_exists']:
        return metrics
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # 檢查 contract_metrics 表格
        cur.execute("SELECT COUNT(*) FROM contract_metrics")
        metrics['total_records'] = cur.fetchone()[0]
        
        # 獲取最新指標
        if metrics['total_records'] > 0:
            cur.execute("""
                SELECT usage_coverage, health_status, total_contracts, used_contracts
                FROM contract_metrics 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            result = cur.fetchone()
            if result:
                metrics['coverage_rate'] = result[0] or 0.0
                metrics['health_status'] = result[1] or 'unknown'
                metrics['latest_metrics'] = {
                    'total_contracts': result[2],
                    'used_contracts': result[3]
                }
        
        # 檢查警報數量
        cur.execute("SELECT COUNT(*) FROM alerts WHERE resolved = 0")
        metrics['alert_count'] = cur.fetchone()[0]
        
        conn.close()
        
    except Exception as e:
        print(f"資料庫分析錯誤: {e}")
    
    return metrics

def analyze_schema_definitions():
    """分析 schema 定義完成度"""
    schema_path = Path("services/aiva_common/schemas")
    schema_stats = {
        'total_files': 0,
        'valid_schemas': 0,
        'exported_schemas': 0,
        'categories': {}
    }
    
    if not schema_path.exists():
        return schema_stats
    
    # 統計 schema 檔案
    schema_files = list(schema_path.glob("*.py"))
    schema_stats['total_files'] = len([f for f in schema_files if f.name != '__init__.py'])
    
    # 分析 __init__.py 的導出
    init_file = schema_path / "__init__.py"
    if init_file.exists():
        try:
            content = init_file.read_text(encoding='utf-8')
            # 計算 __all__ 中的項目
            if "__all__" in content:
                # 簡單計算，實際可能需要更複雜的解析
                lines = [line.strip().strip('",\'') for line in content.split('\n')]
                exported_count = len([line for line in lines if line and not line.startswith(('#', '__all__', '[', ']'))])
                schema_stats['exported_schemas'] = exported_count
        except Exception as e:
            print(f"__init__.py 分析錯誤: {e}")
    
    # 分析分類
    for category_dir in schema_path.iterdir():
        if category_dir.is_dir():
            py_files = list(category_dir.glob("*.py"))
            schema_stats['categories'][category_dir.name] = len(py_files)
    
    return schema_stats

def analyze_usage_patterns():
    """分析使用模式"""
    modules_path = Path("services")
    usage_stats = {
        'modules_scanned': 0,
        'files_with_imports': 0,
        'import_patterns': {},
        'coverage_by_module': {}
    }
    
    if not modules_path.exists():
        return usage_stats
    
    # 掃描主要模組
    main_modules = ['features', 'core', 'scan', 'integration']
    
    for module_name in main_modules:
        module_stats = _analyze_single_module(modules_path / module_name)
        if module_stats:
            usage_stats['modules_scanned'] += 1
            usage_stats['coverage_by_module'][module_name] = module_stats
    
    return usage_stats

def _analyze_single_module(module_path):
    """分析單一模組的使用模式，降低認知複雜度"""
    if not module_path.exists():
        return None
        
    py_files = list(module_path.rglob("*.py"))
    files_with_schema_imports = _count_schema_imports(py_files)
    
    return {
        'total_files': len(py_files),
        'files_with_imports': files_with_schema_imports,
        'import_rate': files_with_schema_imports / len(py_files) * 100 if py_files else 0
    }

def _count_schema_imports(py_files):
    """統計包含 schema 導入的檔案數量"""
    count = 0
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            if "aiva_common.schemas" in content:
                count += 1
        except Exception:
            continue
    return count

def calculate_completion_percentage(db_metrics, schema_stats, usage_stats):
    """計算整體完成度百分比"""
    scores = {}
    
    # 1. 資料庫健康度 (25%)
    if db_metrics['database_exists'] and db_metrics['total_records'] > 0:
        db_score = min(db_metrics['coverage_rate'] * 100, 100)
    else:
        db_score = 0
    scores['database_health'] = db_score
    
    # 2. Schema 定義完成度 (25%)
    if schema_stats['total_files'] > 0:
        definition_score = min((schema_stats['exported_schemas'] / max(schema_stats['total_files'], 1)) * 100, 100)
    else:
        definition_score = 0
    scores['schema_definitions'] = definition_score
    
    # 3. 使用覆蓋率 (25%)
    total_import_rate = 0
    if usage_stats['coverage_by_module']:
        rates = [module['import_rate'] for module in usage_stats['coverage_by_module'].values()]
        total_import_rate = sum(rates) / len(rates) if rates else 0
    scores['usage_coverage'] = total_import_rate
    
    # 4. 系統整合度 (25%)
    integration_score = 0
    if usage_stats['modules_scanned'] > 0:
        modules_with_imports = sum(1 for module in usage_stats['coverage_by_module'].values() 
                                 if module['files_with_imports'] > 0)
        integration_score = (modules_with_imports / usage_stats['modules_scanned']) * 100
    scores['system_integration'] = integration_score
    
    # 計算總分
    weights = {'database_health': 0.25, 'schema_definitions': 0.25, 
               'usage_coverage': 0.25, 'system_integration': 0.25}
    
    total_score = sum(scores[key] * weights[key] for key in scores)
    
    return total_score, scores

def print_completion_report():
    """打印完整的完成度報告"""
    print("🎯 AIVA 數據合約完成度綜合分析")
    print("=" * 80)
    
    # 分析各項指標
    print("\n📊 數據收集中...")
    db_metrics = analyze_database_metrics()
    schema_stats = analyze_schema_definitions()
    usage_stats = analyze_usage_patterns()
    
    # 計算完成度
    completion_rate, detailed_scores = calculate_completion_percentage(
        db_metrics, schema_stats, usage_stats
    )
    
    # 打印主要結果
    print(f"\n🏆 整體數據合約完成度: {completion_rate:.1f}%")
    
    # 詳細分析
    print("\n📋 詳細分析:")
    print(f"  💾 資料庫健康度: {detailed_scores['database_health']:.1f}% (權重25%)")
    print(f"     - 資料庫存在: {'✅' if db_metrics['database_exists'] else '❌'}")
    print(f"     - 記錄數量: {db_metrics['total_records']}")
    print(f"     - 覆蓋率: {db_metrics['coverage_rate']:.1f}%")
    print(f"     - 健康狀態: {db_metrics['health_status']}")
    print(f"     - 未解決警報: {db_metrics['alert_count']}")
    
    print(f"\n  📚 Schema定義完成度: {detailed_scores['schema_definitions']:.1f}% (權重25%)")
    print(f"     - Schema檔案總數: {schema_stats['total_files']}")
    print(f"     - 已導出Schema: {schema_stats['exported_schemas']}")
    print(f"     - 分類數量: {len(schema_stats['categories'])}")
    
    print(f"\n  🔗 使用覆蓋率: {detailed_scores['usage_coverage']:.1f}% (權重25%)")
    for module_name, stats in usage_stats['coverage_by_module'].items():
        print(f"     - {module_name}: {stats['import_rate']:.1f}% ({stats['files_with_imports']}/{stats['total_files']})")
    
    print(f"\n  🌐 系統整合度: {detailed_scores['system_integration']:.1f}% (權重25%)")
    print(f"     - 已掃描模組: {usage_stats['modules_scanned']}")
    modules_with_usage = sum(1 for stats in usage_stats['coverage_by_module'].values() 
                           if stats['files_with_imports'] > 0)
    print(f"     - 有使用合約的模組: {modules_with_usage}")
    
    # 健康度評估
    print("\n🏥 健康度評估:")
    if completion_rate >= 80:
        print("  ✅ 優秀: 數據合約系統完成度很高")
    elif completion_rate >= 60:
        print("  ⚠️ 良好: 系統基本完善，有改進空間")
    elif completion_rate >= 40:
        print("  🔶 中等: 需要持續改進")
    else:
        print("  🔴 需要重點關注: 完成度偏低，建議優先改善")
    
    # 改進建議
    print("\n💡 改進建議:")
    if detailed_scores['database_health'] < 50:
        print("  1. 🔧 優先修復資料庫記錄和監控系統")
    if detailed_scores['usage_coverage'] < 60:
        print("  2. 📈 增加各模組對通用合約的使用")
    if detailed_scores['system_integration'] < 70:
        print("  3. 🔗 加強模組間的合約整合")
    
    return completion_rate

if __name__ == "__main__":
    completion_rate = print_completion_report()
    print(f"\n📝 總結: AIVA 數據合約完成度為 {completion_rate:.1f}%")