"""
批量分類現有能力數據 (v1.0)

為現有的 802 個能力添加範圍管理欄位：
- scope (CapabilityScope)
- visibility (CapabilityVisibility)
- access_level (CapabilityAccessLevel)
- cli_maturity (CLIMaturityLevel)
- available_in (List[str])
- service_dependencies (List[str])

遵循 aiva_common v2.0 規範：
✅ 修改現有文件為原則
✅ 使用統一的分類器
✅ 詳細記錄分類結果
✅ 生成統計報告
"""

import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

# 添加 services 路徑
services_path = Path(__file__).parent.parent / "services"
sys.path.insert(0, str(services_path))

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.dual_loop import (
    CapabilityScope,
    CapabilityVisibility,
    CapabilityAccessLevel,
    CLIMaturityLevel
)

# 導入分類器
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "core" / "aiva_core" / "cognitive_core"))
from internal_loop_connector import CapabilityScopeClassifier

logger = get_logger(__name__)


def classify_capabilities(input_file: Path, output_file: Path) -> dict[str, Any]:
    """批量分類能力
    
    Args:
        input_file: 輸入 JSON 文件路徑
        output_file: 輸出 JSON 文件路徑
        
    Returns:
        統計信息
    """
    logger.info("=" * 80)
    logger.info("🚀 開始批量分類現有能力")
    logger.info("=" * 80)
    
    # 載入現有能力數據
    logger.info(f"\n📂 載入能力數據: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        capabilities = json.load(f)
    
    logger.info(f"   ✅ 載入 {len(capabilities)} 個能力")
    
    # 初始化分類器
    classifier = CapabilityScopeClassifier()
    logger.info("\n🔧 初始化分類器完成")
    
    # 統計信息
    stats = {
        "total": len(capabilities),
        "classified": 0,
        "scope_distribution": {},
        "visibility_distribution": {},
        "access_level_distribution": {},
        "cli_maturity_distribution": {},
        "has_cli": 0,
        "has_service_deps": 0,
        "processing_time": 0
    }
    
    start_time = datetime.now()
    
    # 批量分類
    logger.info("\n🔄 開始分類...")
    for i, cap in enumerate(capabilities, 1):
        if i % 100 == 0:
            logger.info(f"   處理中: {i}/{len(capabilities)}")
        
        try:
            file_path = cap.get('file_path', '')
            
            # 1. 分類範圍和可見性
            scope, visibility = classifier.classify_scope(file_path)
            cap['scope'] = scope.value
            cap['visibility'] = visibility.value
            
            # 2. 分類訪問級別
            category = cap.get('module', '').split('/')[-1]  # 取最後一部分作為類別
            access_level = classifier.classify_access_level(category, None)
            cap['access_level'] = access_level.value
            
            # 3. 檢測 CLI 信息
            has_cli, cli_command, cli_maturity = classifier.detect_cli_info(cap)
            cap['has_cli'] = has_cli
            cap['cli_command'] = cli_command if has_cli else None
            cap['cli_maturity'] = cli_maturity.value if has_cli else CLIMaturityLevel.NONE.value
            
            # 4. 檢測可用範圍
            available_in = classifier.detect_available_in(file_path)
            cap['available_in'] = available_in
            
            # 5. 檢測服務依賴
            service_deps = classifier.detect_service_dependencies(cap)
            cap['service_dependencies'] = service_deps
            
            # 更新統計
            stats['classified'] += 1
            stats['scope_distribution'][scope.value] = stats['scope_distribution'].get(scope.value, 0) + 1
            stats['visibility_distribution'][visibility.value] = stats['visibility_distribution'].get(visibility.value, 0) + 1
            stats['access_level_distribution'][access_level.value] = stats['access_level_distribution'].get(access_level.value, 0) + 1
            stats['cli_maturity_distribution'][cli_maturity.value if has_cli else CLIMaturityLevel.NONE.value] = \
                stats['cli_maturity_distribution'].get(cli_maturity.value if has_cli else CLIMaturityLevel.NONE.value, 0) + 1
            
            if has_cli:
                stats['has_cli'] += 1
            if service_deps:
                stats['has_service_deps'] += 1
                
        except Exception as e:
            logger.error(f"❌ 分類失敗 ({cap.get('name', 'unknown')}): {e}")
            continue
    
    end_time = datetime.now()
    stats['processing_time'] = (end_time - start_time).total_seconds()
    
    # 保存分類結果
    logger.info(f"\n💾 保存分類結果: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(capabilities, f, indent=2, ensure_ascii=False)
    
    logger.info("   ✅ 保存完成")
    
    return stats


def generate_report(stats: dict[str, Any], output_dir: Path):
    """生成分類報告
    
    Args:
        stats: 統計信息
        output_dir: 輸出目錄
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"classification_report_{timestamp}.md"
    
    logger.info(f"\n📊 生成分類報告: {report_file}")
    
    report = f"""# 能力分類報告

生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 總覽

| 項目 | 數值 |
|-----|------|
| 總能力數量 | {stats['total']} |
| 成功分類 | {stats['classified']} |
| 分類成功率 | {stats['classified'] / stats['total'] * 100:.1f}% |
| 處理時間 | {stats['processing_time']:.2f} 秒 |
| 平均處理速度 | {stats['total'] / stats['processing_time']:.1f} 個/秒 |

## 🎯 範圍分佈 (Scope)

| 範圍 | 數量 | 百分比 |
|-----|------|--------|
"""
    
    for scope, count in sorted(stats['scope_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total'] * 100
        report += f"| {scope} | {count} | {percentage:.1f}% |\n"
    
    report += """
## 👁️ 可見性分佈 (Visibility)

| 可見性 | 數量 | 百分比 |
|-------|------|--------|
"""
    
    for visibility, count in sorted(stats['visibility_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total'] * 100
        report += f"| {visibility} | {count} | {percentage:.1f}% |\n"
    
    report += """
## 🔐 訪問級別分佈 (Access Level)

| 級別 | 數量 | 百分比 |
|-----|------|--------|
"""
    
    for level, count in sorted(stats['access_level_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total'] * 100
        report += f"| {level} | {count} | {percentage:.1f}% |\n"
    
    report += """
## 🖥️ CLI 成熟度分佈

| 成熟度 | 數量 | 百分比 |
|-------|------|--------|
"""
    
    for maturity, count in sorted(stats['cli_maturity_distribution'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total'] * 100
        report += f"| {maturity} | {count} | {percentage:.1f}% |\n"
    
    report += f"""
## 📈 其他統計

| 項目 | 數值 |
|-----|------|
| 具有 CLI | {stats['has_cli']} ({stats['has_cli'] / stats['total'] * 100:.1f}%) |
| 具有服務依賴 | {stats['has_service_deps']} ({stats['has_service_deps'] / stats['total'] * 100:.1f}%) |

## ✅ 完成

所有能力已成功分類並添加範圍管理欄位！

---

*此報告由 classify_existing_capabilities.py 自動生成*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("   ✅ 報告生成完成")
    
    return report_file


def main():
    """主函數"""
    # 設定路徑
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "analysis_results"
    
    input_file = data_dir / "capabilities_20251125_173734.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"capabilities_classified_{timestamp}.json"
    
    # 檢查輸入文件
    if not input_file.exists():
        logger.error(f"❌ 輸入文件不存在: {input_file}")
        return 1
    
    try:
        # 執行分類
        stats = classify_capabilities(input_file, output_file)
        
        # 生成報告
        report_file = generate_report(stats, data_dir)
        
        # 顯示總結
        logger.info("\n" + "=" * 80)
        logger.info("✅ 分類完成！")
        logger.info("=" * 80)
        logger.info("\n📊 統計總結:")
        logger.info("   總能力數量: %d", stats['total'])
        logger.info("   成功分類: %d", stats['classified'])
        logger.info("   處理時間: %.2f 秒", stats['processing_time'])
        logger.info("\n📁 輸出文件:")
        logger.info(f"   分類結果: {output_file}")
        logger.info(f"   報告: {report_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
