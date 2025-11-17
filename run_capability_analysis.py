"""實際執行多語言能力分析

直接運行完整的能力分析並生成報告
"""

import asyncio
import logging
from pathlib import Path
from collections import Counter
import json
from datetime import datetime

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """執行完整的能力分析"""
    
    # 導入模組
    from services.core.aiva_core.internal_exploration import (
        ModuleExplorer,
        CapabilityAnalyzer
    )
    
    logger.info("=" * 70)
    logger.info("🚀 AIVA 多語言能力分析系統 v2.0 Enhanced")
    logger.info("=" * 70)
    logger.info(f"📅 執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 初始化
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    # 1. 探索模組
    logger.info("🔍 階段 1: 探索模組結構...")
    modules = await explorer.explore_all_modules()
    logger.info(f"   ✅ 發現 {len(modules)} 個模組\n")
    
    # 2. 分析能力
    logger.info("🔍 階段 2: 分析多語言能力...")
    capabilities = await analyzer.analyze_capabilities(modules)
    logger.info(f"   ✅ 提取 {len(capabilities)} 個能力\n")
    
    # 3. 統計分析
    logger.info("=" * 70)
    logger.info("📊 語言分布統計")
    logger.info("=" * 70)
    
    lang_counts = Counter(cap["language"] for cap in capabilities)
    
    # 表格標題
    logger.info(f"\n{'語言':<12} {'能力數':>8}  {'佔比':>8}  {'狀態':>6}")
    logger.info("-" * 45)
    
    for lang, count in lang_counts.most_common():
        percentage = (count / len(capabilities)) * 100
        status = "✅"
        logger.info(f"{lang:<12} {count:>8}  {percentage:>7.1f}%  {status:>6}")
    
    logger.info("-" * 45)
    logger.info(f"{'總計':<12} {len(capabilities):>8}  {100.0:>7.1f}%  ✅\n")
    
    # 4. Rust 詳細分析
    rust_caps = [cap for cap in capabilities if cap["language"] == "rust"]
    if rust_caps:
        logger.info("=" * 70)
        logger.info("🦀 Rust 能力詳細分析")
        logger.info("=" * 70)
        
        methods = [cap for cap in rust_caps if cap.get("is_method")]
        functions = [cap for cap in rust_caps if not cap.get("is_method")]
        
        logger.info(f"\n總計: {len(rust_caps)} 個能力")
        logger.info(f"  📦 結構體方法: {len(methods)}")
        logger.info(f"  📝 頂層函數:   {len(functions)}")
        
        if methods:
            logger.info(f"\n🔝 熱門結構體 (Top 10 方法):")
            
            # 按結構體分組
            by_struct = {}
            for cap in methods:
                struct = cap.get("struct", "Unknown")
                by_struct.setdefault(struct, []).append(cap)
            
            # 排序並顯示
            sorted_structs = sorted(by_struct.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (struct, caps) in enumerate(sorted_structs[:10], 1):
                logger.info(f"   {i:2}. {struct:<30} - {len(caps)} 個方法")
        
        logger.info("")
    
    # 5. 錯誤報告
    analyzer.print_extraction_report()
    
    # 6. 模組統計
    logger.info("=" * 70)
    logger.info("📦 模組能力分布")
    logger.info("=" * 70)
    
    grouped = analyzer.get_capabilities_by_module(capabilities)
    
    logger.info(f"\n{'模組':<25} {'能力數':>8}  {'主要語言':<15}")
    logger.info("-" * 55)
    
    for module, caps in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        main_lang = Counter(c["language"] for c in caps).most_common(1)[0][0]
        logger.info(f"{module:<25} {len(caps):>8}  {main_lang:<15}")
    
    logger.info("")
    
    # 7. 保存結果
    logger.info("=" * 70)
    logger.info("💾 保存分析結果")
    logger.info("=" * 70)
    
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整數據
    output_file = output_dir / f"capabilities_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(capabilities, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✅ 完整數據: {output_file}")
    
    # 保存統計摘要
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_capabilities": len(capabilities),
        "language_distribution": dict(lang_counts),
        "module_distribution": {
            module: len(caps) for module, caps in grouped.items()
        },
        "rust_details": {
            "total": len(rust_caps),
            "methods": len(methods) if rust_caps else 0,
            "functions": len(functions) if rust_caps else 0
        },
        "extraction_report": analyzer.get_extraction_report()
    }
    
    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"   ✅ 統計摘要: {summary_file}")
    
    # 8. 對比分析 (如果有基線)
    baseline_file = output_dir / "baseline.json"
    if baseline_file.exists():
        logger.info("\n📊 與基線對比:")
        
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        baseline_count = baseline.get("total_capabilities", 0)
        diff = len(capabilities) - baseline_count
        diff_pct = (diff / baseline_count * 100) if baseline_count > 0 else 0
        
        if diff > 0:
            logger.info(f"   📈 能力數增加: +{diff} (+{diff_pct:.1f}%)")
        elif diff < 0:
            logger.info(f"   📉 能力數減少: {diff} ({diff_pct:.1f}%)")
        else:
            logger.info(f"   ➡️  能力數不變: {len(capabilities)}")
        
        # 語言對比
        baseline_langs = baseline.get("language_distribution", {})
        for lang in set(list(lang_counts.keys()) + list(baseline_langs.keys())):
            current = lang_counts.get(lang, 0)
            previous = baseline_langs.get(lang, 0)
            if current != previous:
                diff = current - previous
                logger.info(f"   {lang}: {previous} → {current} ({diff:+d})")
    else:
        # 創建基線
        logger.info(f"\n   📝 首次執行，創建基線...")
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ 基線已保存")
    
    logger.info("")
    
    # 9. 完成
    logger.info("=" * 70)
    logger.info("✅ 分析完成！")
    logger.info("=" * 70)
    logger.info(f"\n🎯 核心指標:")
    logger.info(f"   總能力數:   {len(capabilities)}")
    logger.info(f"   成功率:     {analyzer.get_extraction_report()['success_rate']:.1f}%")
    logger.info(f"   覆蓋語言:   {len(lang_counts)}")
    logger.info(f"   覆蓋模組:   {len(grouped)}")
    logger.info("")
    
    return capabilities


if __name__ == "__main__":
    try:
        capabilities = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  分析被用戶中斷")
    except Exception as e:
        print(f"\n\n❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
