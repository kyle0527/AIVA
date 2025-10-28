#!/usr/bin/env python3
"""
TODO 8 - 性能優化配置應用器
將優化配置應用到 AI 組件中，實現實際的性能提升
"""

import sys
import json
import yaml
import asyncio
from pathlib import Path
from typing import Dict, Any

# 添加 AIVA 模組路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "services"))

def apply_capability_evaluator_optimizations():
    """應用 CapabilityEvaluator 性能優化"""
    print("🔧 應用 CapabilityEvaluator 性能優化...")
    
    try:
        from aiva_common.ai.performance_config import (
            OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG,
            create_optimized_configs
        )
        
        # 生成優化配置
        config = OPTIMIZED_CAPABILITY_EVALUATOR_CONFIG.to_dict()
        
        # 創建配置文件
        config_dir = project_root / "services/aiva_common/config"
        config_dir.mkdir(exist_ok=True)
        
        # 保存為 YAML 配置文件
        with open(config_dir / "capability_evaluator_performance.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        # 保存為 JSON 配置文件（用於 TypeScript）
        with open(config_dir / "capability_evaluator_performance.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("  ✅ CapabilityEvaluator 配置文件已創建")
        print(f"    - YAML: {config_dir}/capability_evaluator_performance.yaml")
        print(f"    - JSON: {config_dir}/capability_evaluator_performance.json")
        
        # 顯示關鍵優化參數
        print("  📊 關鍵優化參數:")
        print(f"    - 緩存策略: {config['cache_strategy']}")
        print(f"    - 緩存TTL: {config['cache_ttl_seconds']}秒")
        print(f"    - 最大並發: {config['max_concurrent_operations']}")
        print(f"    - 批處理大小: {config['batch_size']}")
        print(f"    - 評估工作者: {config['max_evaluation_workers']}")
        print(f"    - 監控間隔: {config['monitoring_interval_seconds']}秒")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 應用 CapabilityEvaluator 優化失敗: {e}")
        return False

def apply_experience_manager_optimizations():
    """應用 ExperienceManager 性能優化"""
    print("\n🔧 應用 ExperienceManager 性能優化...")
    
    try:
        from aiva_common.ai.performance_config import OPTIMIZED_EXPERIENCE_MANAGER_CONFIG
        
        # 生成優化配置
        config = OPTIMIZED_EXPERIENCE_MANAGER_CONFIG.to_dict()
        
        # 創建配置文件
        config_dir = project_root / "services/aiva_common/config"
        config_dir.mkdir(exist_ok=True)
        
        # 保存為 YAML 配置文件
        with open(config_dir / "experience_manager_performance.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        # 保存為 JSON 配置文件（用於 TypeScript）
        with open(config_dir / "experience_manager_performance.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("  ✅ ExperienceManager 配置文件已創建")
        print(f"    - YAML: {config_dir}/experience_manager_performance.yaml")
        print(f"    - JSON: {config_dir}/experience_manager_performance.json")
        
        # 顯示關鍵優化參數
        print("  📊 關鍵優化參數:")
        print(f"    - 存儲後端: {config['storage_backend']}")
        print(f"    - 批量插入: {config['batch_insert_enabled']}")
        print(f"    - 異步存儲: {config['async_storage_enabled']}")
        print(f"    - 緩冲區大小: {config['storage_buffer_size']}")
        print(f"    - 查詢結果緩存: {config['query_result_cache_enabled']}")
        print(f"    - 會話池: {config['session_pooling_enabled']}")
        print(f"    - 會話緩存大小: {config['session_cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 應用 ExperienceManager 優化失敗: {e}")
        return False

def create_global_performance_config():
    """創建全局性能配置"""
    print("\n🌐 創建全局性能配置...")
    
    try:
        from aiva_common.ai.performance_config import create_optimized_configs, PERFORMANCE_BENCHMARKS
        
        # 生成完整配置
        full_config = create_optimized_configs()
        
        # 創建配置目錄
        config_dir = project_root / "services/aiva_common/config"
        config_dir.mkdir(exist_ok=True)
        
        # 保存全局配置
        with open(config_dir / "ai_performance_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
        
        with open(config_dir / "ai_performance_config.json", "w", encoding="utf-8") as f:
            json.dump(full_config, f, indent=2, ensure_ascii=False)
        
        print("  ✅ 全局性能配置已創建")
        print(f"    - YAML: {config_dir}/ai_performance_config.yaml")  
        print(f"    - JSON: {config_dir}/ai_performance_config.json")
        
        # 顯示性能基準
        print("  🎯 性能基準目標:")
        benchmarks = PERFORMANCE_BENCHMARKS
        for component, metrics in benchmarks.items():
            print(f"    {component}:")
            for metric, value in metrics.items():
                print(f"      - {metric}: {value}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 創建全局配置失敗: {e}")
        return False

def create_typescript_config_files():
    """創建 TypeScript 配置文件"""
    print("\n🔷 創建 TypeScript 性能配置文件...")
    
    try:
        # TypeScript 配置目錄
        ts_config_dir = project_root / "services/features/common/typescript/aiva_common_ts/config"
        ts_config_dir.mkdir(exist_ok=True)
        
        # 從 Python 配置生成 TypeScript 兼容的配置
        from aiva_common.ai.performance_config import create_optimized_configs
        config = create_optimized_configs()
        
        # 保存 TypeScript 配置
        with open(ts_config_dir / "performance-config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 創建 TypeScript 配置加載器
        ts_loader_content = f'''/**
 * AIVA TypeScript 性能配置加載器
 * 自動生成 - 與 Python 配置保持同步
 */

import performanceConfig from './performance-config.json';

export const {{
  capability_evaluator: CAPABILITY_EVALUATOR_CONFIG,
  experience_manager: EXPERIENCE_MANAGER_CONFIG,
  performance_benchmarks: PERFORMANCE_BENCHMARKS,
  global_settings: GLOBAL_SETTINGS
}} = performanceConfig;

export default performanceConfig;
'''
        
        with open(ts_config_dir / "index.ts", "w", encoding="utf-8") as f:
            f.write(ts_loader_content)
        
        print("  ✅ TypeScript 配置文件已創建")
        print(f"    - 配置: {ts_config_dir}/performance-config.json")
        print(f"    - 加載器: {ts_config_dir}/index.ts")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 創建 TypeScript 配置失敗: {e}")
        return False

def create_environment_configs():
    """創建不同環境的配置文件"""
    print("\n🏗️ 創建環境特定配置...")
    
    try:
        from aiva_common.ai.performance_config import (
            CapabilityEvaluatorConfig,
            ExperienceManagerConfig,
            CacheStrategy
        )
        
        # 開發環境配置（較低的資源使用）
        dev_capability_config = CapabilityEvaluatorConfig(
            cache_strategy=CacheStrategy.MEMORY_ONLY,
            cache_ttl_seconds=300,  # 5分鐘
            max_cache_size=100,
            max_concurrent_operations=2,
            max_evaluation_workers=2,
            monitoring_interval_seconds=300.0,
            enable_performance_monitoring=False
        )
        
        dev_experience_config = ExperienceManagerConfig(
            cache_strategy=CacheStrategy.MEMORY_ONLY,
            cache_ttl_seconds=600,  # 10分鐘
            max_cache_size=500,
            max_concurrent_operations=4,
            storage_backend="memory",
            batch_size=50,
            enable_performance_monitoring=False
        )
        
        # 生產環境配置（高性能）
        prod_capability_config = CapabilityEvaluatorConfig(
            cache_strategy=CacheStrategy.HYBRID,
            cache_ttl_seconds=3600,  # 1小時
            max_cache_size=5000,
            max_concurrent_operations=16,
            max_evaluation_workers=8,
            monitoring_interval_seconds=60.0,
            enable_performance_monitoring=True,
            metrics_sampling_rate=0.01  # 1% 採樣
        )
        
        prod_experience_config = ExperienceManagerConfig(
            cache_strategy=CacheStrategy.HYBRID,
            cache_ttl_seconds=7200,  # 2小時
            max_cache_size=10000,
            max_concurrent_operations=20,
            storage_backend="postgresql",
            batch_size=500,
            connection_pool_size=50,
            enable_performance_monitoring=True,
            metrics_sampling_rate=0.01
        )
        
        # 創建環境配置
        environments = {
            "development": {
                "capability_evaluator": dev_capability_config.to_dict(),
                "experience_manager": dev_experience_config.to_dict()
            },
            "production": {
                "capability_evaluator": prod_capability_config.to_dict(), 
                "experience_manager": prod_experience_config.to_dict()
            }
        }
        
        config_dir = project_root / "services/aiva_common/config"
        
        for env_name, env_config in environments.items():
            # YAML 格式
            with open(config_dir / f"ai_performance_{env_name}.yaml", "w", encoding="utf-8") as f:
                yaml.dump(env_config, f, default_flow_style=False, allow_unicode=True)
            
            # JSON 格式
            with open(config_dir / f"ai_performance_{env_name}.json", "w", encoding="utf-8") as f:
                json.dump(env_config, f, indent=2, ensure_ascii=False)
        
        print("  ✅ 環境配置已創建:")
        print("    - development: 低資源開發環境")
        print("    - production: 高性能生產環境")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 創建環境配置失敗: {e}")
        return False

def validate_configurations():
    """驗證配置文件的正確性"""
    print("\n✅ 驗證配置文件...")
    
    config_dir = project_root / "services/aiva_common/config"
    config_files = [
        "ai_performance_config.json",
        "capability_evaluator_performance.json", 
        "experience_manager_performance.json",
        "ai_performance_development.json",
        "ai_performance_production.json"
    ]
    
    validation_results = []
    
    for config_file in config_files:
        file_path = config_dir / config_file
        try:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # 基本驗證
                if isinstance(config_data, dict) and len(config_data) > 0:
                    validation_results.append((config_file, True, "有效"))
                    print(f"  ✅ {config_file}: 驗證通過")
                else:
                    validation_results.append((config_file, False, "配置為空"))
                    print(f"  ❌ {config_file}: 配置為空")
            else:
                validation_results.append((config_file, False, "文件不存在"))
                print(f"  ❌ {config_file}: 文件不存在")
                
        except json.JSONDecodeError as e:
            validation_results.append((config_file, False, f"JSON 格式錯誤: {e}"))
            print(f"  ❌ {config_file}: JSON 格式錯誤")
        except Exception as e:
            validation_results.append((config_file, False, f"驗證錯誤: {e}"))
            print(f"  ❌ {config_file}: 驗證錯誤")
    
    successful_validations = sum(1 for _, success, _ in validation_results if success)
    total_validations = len(validation_results)
    
    print(f"\n📊 驗證結果: {successful_validations}/{total_validations} 個配置文件有效")
    
    return successful_validations == total_validations

def main():
    """主執行函數"""
    print("🚀 開始 TODO 8 - 性能優化配置應用")
    print("=" * 60)
    
    success_count = 0
    total_operations = 6
    
    # 執行優化應用步驟
    operations = [
        ("CapabilityEvaluator 優化", apply_capability_evaluator_optimizations),
        ("ExperienceManager 優化", apply_experience_manager_optimizations), 
        ("全局性能配置", create_global_performance_config),
        ("TypeScript 配置", create_typescript_config_files),
        ("環境配置", create_environment_configs),
        ("配置驗證", validate_configurations)
    ]
    
    for operation_name, operation_func in operations:
        try:
            if operation_func():
                success_count += 1
                print(f"✅ {operation_name} - 完成")
            else:
                print(f"❌ {operation_name} - 失敗")
        except Exception as e:
            print(f"❌ {operation_name} - 錯誤: {e}")
    
    # 生成總結報告
    print("\n" + "=" * 60)
    print("📈 TODO 8 性能優化應用結果")
    print("=" * 60)
    
    success_rate = (success_count / total_operations) * 100
    print(f"✅ 成功操作: {success_count}/{total_operations} ({success_rate:.1f}%)")
    
    if success_count == total_operations:
        print("\n🎉 所有性能優化配置已成功應用！")
        print("\n📁 生成的配置文件:")
        config_dir = project_root / "services/aiva_common/config"
        if config_dir.exists():
            for config_file in config_dir.glob("*.json"):
                print(f"  - {config_file.name}")
        
        print("\n🔧 使用方式:")
        print("  Python: from aiva_common.ai.performance_config import OPTIMIZED_*_CONFIG")
        print("  TypeScript: import { createOptimizedConfigs } from 'aiva_common_ts'")
        
        return 0
    else:
        print(f"\n⚠️ {total_operations - success_count} 個操作失敗，需要檢查和修復")
        return 1

if __name__ == "__main__":
    sys.exit(main())