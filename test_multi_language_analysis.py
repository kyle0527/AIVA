"""測試多語言能力分析整合

驗證 capability_analyzer 使用 language_extractors 正確提取:
- Python 能力 (AST)
- Go 能力 (正則)
- Rust 能力 (正則)
- TypeScript 能力 (正則)

預期結果:
- 成功整合 language_extractors
- 正確識別多語言文件
- 提取所有語言的能力函數
"""

import asyncio
import logging
from pathlib import Path

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from services.core.aiva_core.internal_exploration import ModuleExplorer, CapabilityAnalyzer


async def test_multi_language_analysis():
    """測試多語言能力分析"""
    print("=" * 80)
    print("🚀 測試多語言能力分析整合")
    print("=" * 80)
    
    # 1. 初始化模組探索器
    print("\n1️⃣ 初始化 ModuleExplorer...")
    root_path = Path(__file__).parent / "services"
    explorer = ModuleExplorer(root_path=root_path)
    
    # 2. 掃描所有模組 (包含多語言文件)
    print("\n2️⃣ 掃描模組文件...")
    modules_info = await explorer.explore_all_modules()
    
    # 統計掃描結果
    total_files = sum(m["stats"]["total_files"] for m in modules_info.values())
    by_lang = {}
    for module_data in modules_info.values():
        for lang, count in module_data["stats"]["by_language"].items():
            by_lang[lang] = by_lang.get(lang, 0) + count
    
    print(f"\n   掃描結果:")
    print(f"   - 總模組: {len(modules_info)}")
    print(f"   - 總文件: {total_files}")
    print(f"   - 語言分布:")
    for lang, count in by_lang.items():
        if count > 0:
            print(f"     * {lang}: {count} 個文件")
    
    # 3. 分析能力 (包含多語言)
    print("\n3️⃣ 分析能力 (多語言)...")
    analyzer = CapabilityAnalyzer()
    capabilities = await analyzer.analyze_capabilities(modules_info)
    
    # 統計能力結果
    cap_by_lang = {}
    for cap in capabilities:
        lang = cap.get("language", "python")
        cap_by_lang[lang] = cap_by_lang.get(lang, 0) + 1
    
    print(f"\n   能力分析結果:")
    print(f"   - 總能力: {len(capabilities)}")
    print(f"   - 語言分布:")
    for lang, count in cap_by_lang.items():
        print(f"     * {lang}: {count} 個能力")
    
    # 4. 顯示範例能力 (每種語言各 3 個)
    print("\n4️⃣ 能力範例:")
    for lang in ["python", "go", "rust", "typescript", "javascript"]:
        lang_caps = [c for c in capabilities if c.get("language") == lang]
        if lang_caps:
            print(f"\n   📦 {lang.upper()} 能力:")
            for cap in lang_caps[:3]:
                params = cap.get("parameters", [])
                param_str = ", ".join(p.get("name", "") for p in params) if params else ""
                print(f"     - {cap['name']}({param_str})")
                if cap.get("description"):
                    desc = cap["description"][:60] + "..." if len(cap["description"]) > 60 else cap["description"]
                    print(f"       {desc}")
    
    # 5. 驗證整合狀態
    print("\n5️⃣ 整合驗證:")
    checks = {
        "✅ ModuleExplorer 掃描多語言": total_files > 0 and len(by_lang) > 1,
        "✅ CapabilityAnalyzer 整合 language_extractors": len(cap_by_lang) > 0,
        "✅ Python 能力提取": cap_by_lang.get("python", 0) > 0,
        "✅ Go 能力提取": cap_by_lang.get("go", 0) > 0,
        "✅ Rust 能力提取": cap_by_lang.get("rust", 0) > 0,
        "✅ TypeScript 能力提取": cap_by_lang.get("typescript", 0) > 0,
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    all_passed = all(checks.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 多語言能力分析整合測試通過!")
    else:
        print("⚠️ 部分測試未通過,請檢查實現")
    print("=" * 80)
    
    return capabilities


if __name__ == "__main__":
    asyncio.run(test_multi_language_analysis())
