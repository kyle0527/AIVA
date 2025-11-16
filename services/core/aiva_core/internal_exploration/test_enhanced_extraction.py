"""測試增強版多語言能力提取

驗證 P0 改進項目:
1. Rust impl 方法提取
2. 錯誤處理和追蹤
3. 統計報告生成
"""

import asyncio
import logging
from pathlib import Path
from .capability_analyzer import CapabilityAnalyzer
from .module_explorer import ModuleExplorer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_rust_extraction():
    """測試 Rust 提取功能"""
    logger.info("=" * 70)
    logger.info("🧪 Testing Enhanced Rust Extraction")
    logger.info("=" * 70)
    
    analyzer = CapabilityAnalyzer()
    
    # 查找 Rust 文件
    rust_files = list(Path("C:/D/fold7/AIVA-git/services").rglob("*.rs"))
    logger.info(f"\n📂 Found {len(rust_files)} Rust files")
    
    # 提取前 5 個 Rust 文件的能力
    for i, rust_file in enumerate(rust_files[:5], 1):
        logger.info(f"\n{i}. Analyzing: {rust_file.name}")
        logger.info(f"   Path: {rust_file.relative_to(Path('C:/D/fold7/AIVA-git'))}")
        
        try:
            caps = await analyzer._extract_capabilities_from_file(rust_file, "test_module")
            
            if caps:
                logger.info(f"   ✅ Found {len(caps)} capabilities:")
                for cap in caps[:3]:  # 顯示前 3 個
                    cap_type = "method" if cap.get("is_method") else "function"
                    logger.info(f"      - {cap['name']} ({cap_type})")
                    if cap.get("struct"):
                        logger.info(f"        Struct: {cap['struct']}")
                    if cap.get("parameters"):
                        params = ", ".join(p["name"] for p in cap["parameters"])
                        logger.info(f"        Params: {params}")
            else:
                logger.info(f"   ⚠️  No capabilities found")
        
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
    
    logger.info("\n" + "=" * 70)


async def test_full_analysis():
    """測試完整的多語言分析"""
    logger.info("=" * 70)
    logger.info("🧪 Testing Full Multi-Language Analysis")
    logger.info("=" * 70)
    
    # 初始化
    explorer = ModuleExplorer()
    analyzer = CapabilityAnalyzer()
    
    # 探索模組
    logger.info("\n📚 Exploring modules...")
    modules = await explorer.explore_all_modules()
    logger.info(f"   Found {len(modules)} modules")
    
    # 分析能力
    logger.info("\n🔍 Analyzing capabilities...")
    capabilities = await analyzer.analyze_capabilities(modules)
    
    # 統計分析
    logger.info("\n📊 Language Distribution:")
    from collections import Counter
    lang_counts = Counter(cap["language"] for cap in capabilities)
    
    for lang, count in lang_counts.most_common():
        percentage = (count / len(capabilities)) * 100
        logger.info(f"   {lang:12} : {count:4} capabilities ({percentage:5.1f}%)")
    
    # 顯示錯誤報告
    logger.info("\n" + "=" * 70)
    analyzer.print_extraction_report()
    
    # Rust 詳細分析
    rust_caps = [cap for cap in capabilities if cap["language"] == "rust"]
    if rust_caps:
        logger.info("\n🦀 Rust Capabilities Details:")
        logger.info(f"   Total: {len(rust_caps)}")
        
        # 統計方法 vs 函數
        methods = [cap for cap in rust_caps if cap.get("is_method")]
        functions = [cap for cap in rust_caps if not cap.get("is_method")]
        
        logger.info(f"   Methods:   {len(methods)}")
        logger.info(f"   Functions: {len(functions)}")
        
        if methods:
            logger.info("\n   Top 10 Methods:")
            for i, cap in enumerate(methods[:10], 1):
                logger.info(f"      {i}. {cap['name']}")
                if cap.get("description") != f"Rust method: {cap['name']}":
                    logger.info(f"         {cap['description'][:60]}...")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Total Capabilities Extracted: {len(capabilities)}")
    logger.info("=" * 70)


async def test_error_handling():
    """測試錯誤處理"""
    logger.info("=" * 70)
    logger.info("🧪 Testing Error Handling")
    logger.info("=" * 70)
    
    analyzer = CapabilityAnalyzer()
    
    # 測試不存在的文件
    logger.info("\n1. Testing non-existent file...")
    await analyzer._extract_capabilities_from_file(
        Path("C:/nonexistent/file.py"),
        "test"
    )
    
    # 測試權限錯誤 (模擬)
    logger.info("\n2. Testing large file skip...")
    # 查找一個實際存在的大文件或跳過
    
    # 顯示錯誤報告
    logger.info("\n📊 Error Report:")
    report = analyzer.get_extraction_report()
    logger.info(f"   Total Errors: {report['total_errors']}")
    
    if report['errors_by_type']:
        logger.info("\n   Errors by Type:")
        for err_type, count in report['errors_by_type'].items():
            logger.info(f"      {err_type}: {count}")
    
    logger.info("\n" + "=" * 70)


async def main():
    """主測試函數"""
    try:
        # 測試 1: Rust 提取
        await test_rust_extraction()
        
        # 測試 2: 錯誤處理
        await test_error_handling()
        
        # 測試 3: 完整分析
        await test_full_analysis()
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
