"""
分析 AIVA Common 的跨語言功能和 AI 模組完備性
"""

from pathlib import Path
import re

# 常量定義
CLASS_PATTERN = r"^class (\w+)\("


def analyze_ai_schemas(aiva_common):
    """分析當前 AI schemas"""
    print("\n[AI] 步驟 1: 分析當前 AI Schemas")
    print("-" * 50)

    ai_file = aiva_common / "schemas" / "ai.py"
    if not ai_file.exists():
        print("[FAIL] ai.py 檔案不存在")
        return [], []

    content = ai_file.read_text(encoding="utf-8")
    ai_classes = re.findall(CLASS_PATTERN, content, re.MULTILINE)

    print(f"[U+1F4C4] ai.py 中的類別 ({len(ai_classes)} 個):")
    for cls in sorted(ai_classes):
        print(f"   - {cls}")

    # 檢查是否有多語言相關的模式
    multilang_patterns = [
        r"language|lang|locale",
        r"i18n|l10n|translation",
        r"cross.?language|multi.?language",
        r"encoding|charset|unicode",
    ]

    multilang_found = []
    for pattern in multilang_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        if matches:
            multilang_found.extend(matches)

    if multilang_found:
        print("\n[U+1F310] 發現多語言相關內容:")
        for match in set(multilang_found):
            print(f"   - {match}")
    else:
        print("\n[WARN]  未發現多語言相關內容")

    return ai_classes, multilang_found


def analyze_cross_language_requirements():
    """分析跨語言支援需求"""
    print("\n[U+1F310] 步驟 2: 跨語言支援需求分析")
    print("-" * 50)

    cross_lang_requirements = {
        "Programming Languages": [
            "Python",
            "JavaScript",
            "Go",
            "Java",
            "C#",
            "Ruby",
            "PHP",
            "Rust",
            "C++",
            "TypeScript",
        ],
        "Web Technologies": [
            "HTML",
            "CSS",
            "SASS",
            "LESS",
            "React",
            "Vue",
            "Angular",
            "Node.js",
        ],
        "Database Languages": [
            "SQL",
            "NoSQL",
            "GraphQL",
            "MongoDB Query",
            "Redis Commands",
        ],
        "Infrastructure": [
            "Docker",
            "Kubernetes",
            "Terraform",
            "CloudFormation",
            "Helm",
        ],
        "Natural Languages": [
            "English",
            "Chinese",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Spanish",
        ],
    }

    print("AIVA 平台應支援的跨語言類別:")
    for category, languages in cross_lang_requirements.items():
        print(f"\n[U+1F4C2] {category}:")
        for lang in languages[:5]:  # 只顯示前5個
            print(f"   - {lang}")
        if len(languages) > 5:
            print(f"   ... 還有 {len(languages) - 5} 個")

    return cross_lang_requirements


def suggest_multilingual_schemas():
    """建議的多語言 Schema 擴展"""
    print("\n[NOTE] 步驟 3: 建議的多語言 Schema 擴展")
    print("-" * 50)

    suggested_schemas = {
        "跨語言分析": [
            "CodeLanguageDetection",
            "MultiLanguageCodeAnalysis",
            "CrossLanguageVulnerability",
            "LanguageSpecificPayload",
        ],
        "國際化支援": [
            "I18nMessage",
            "LocalizedFinding",
            "MultiLanguageReport",
            "CulturalSecurityContext",
        ],
        "AI 多語言能力": [
            "MultiLingualAIModel",
            "LanguageAdaptiveStrategy",
            "CrossLangLearning",
            "CodeTranslationTask",
        ],
    }

    for category, schemas in suggested_schemas.items():
        print(f"\n[CONFIG] {category}:")
        for schema in schemas:
            print(f"   - {schema}")

    return suggested_schemas


def analyze_enums_multilang_support(aiva_common):
    """檢查 Enums 的多語言支援"""
    print("\n[U+1F3F7][U+FE0F]  步驟 4: 檢查 Enums 的多語言支援")
    print("-" * 50)

    enum_files = list((aiva_common / "enums").glob("*.py"))
    all_enums = []

    for enum_file in enum_files:
        if enum_file.name == "__init__.py":
            continue
        try:
            content = enum_file.read_text(encoding="utf-8")
            enums = re.findall(CLASS_PATTERN, content, re.MULTILINE)
            all_enums.extend(enums)
        except OSError:
            continue

    print(f"[STATS] 當前 Enums 總計: {len(all_enums)} 個")

    # 檢查是否有語言相關的枚舉
    language_related = [
        enum
        for enum in all_enums
        if any(
            word in enum.lower() for word in ["lang", "language", "locale", "encoding"]
        )
    ]

    if language_related:
        print("[U+1F310] 語言相關的枚舉:")
        for enum in language_related:
            print(f"   - {enum}")
    else:
        print("[WARN]  缺少語言相關的枚舉")

    return all_enums, language_related


def analyze_ai_capabilities(ai_file):
    """分析 AI 模組完備性"""
    print("\n[BRAIN] 步驟 5: AI 模組完備性檢查")
    print("-" * 50)

    ai_capabilities_needed = {
        "機器學習": ["訓練", "推理", "模型管理", "特徵工程"],
        "深度學習": ["神經網路", "CNN", "RNN", "Transformer"],
        "強化學習": ["獎勵系統", "策略優化", "環境互動", "經驗回放"],
        "自然語言處理": ["文本分析", "語義理解", "代碼理解", "多語言"],
        "安全AI": ["對抗攻擊檢測", "模型魯棒性", "隱私保護", "可解釋性"],
    }

    print("AI 模組應具備的能力:")
    for category, capabilities in ai_capabilities_needed.items():
        print(f"\n[TARGET] {category}:")
        for cap in capabilities:
            print(f"   - {cap}")

    # 檢查當前 AI schemas 是否涵蓋這些能力
    covered_capabilities = []
    if ai_file.exists():
        ai_content = ai_file.read_text(encoding="utf-8")

        for category, capabilities in ai_capabilities_needed.items():
            for cap in capabilities:
                # 簡單的關鍵字匹配
                if any(
                    keyword in ai_content.lower() for keyword in cap.lower().split()
                ):
                    covered_capabilities.append(f"{category}: {cap}")

        print(f"\n[OK] 已涵蓋的能力 ({len(covered_capabilities)} 項):")
        for cap in covered_capabilities[:10]:  # 只顯示前10項
            print(f"   - {cap}")

    return ai_capabilities_needed, covered_capabilities


def analyze_cross_language_and_ai():
    """分析跨語言功能和AI模組完備性"""
    project_root = Path(__file__).parent.parent
    aiva_common = project_root / "services" / "aiva_common"

    print("=" * 100)
    print("[U+1F310] AIVA Common 跨語言功能和 AI 模組完備性分析")
    print("=" * 100)

    # 分析各個部分
    ai_classes, multilang_found = analyze_ai_schemas(aiva_common)
    cross_lang_requirements = analyze_cross_language_requirements()
    suggested_schemas = suggest_multilingual_schemas()
    all_enums, language_related = analyze_enums_multilang_support(aiva_common)
    ai_capabilities, covered_capabilities = analyze_ai_capabilities(
        aiva_common / "schemas" / "ai.py"
    )

    return {
        "ai_classes_count": len(ai_classes) if "ai_classes" in locals() else 0,
        "multilang_support": len(multilang_found) > 0
        if "multilang_found" in locals()
        else False,
        "suggested_enhancements": suggested_schemas,
        "cross_lang_requirements": cross_lang_requirements,
    }


def generate_enhancement_recommendations():
    """生成增強建議"""

    print("\n[START] 步驟 6: 增強建議")
    print("=" * 100)

    recommendations = {
        "立即實施": [
            "添加 ProgrammingLanguage 枚舉",
            "創建 CodeLanguageDetection schema",
            "擴展 AI 模型的多語言支援",
            "添加 LocaleContext 到所有用戶相關的 schema",
        ],
        "短期目標": [
            "實施跨語言代碼分析 schema",
            "添加文化特定的安全上下文",
            "創建多語言報告格式",
            "實施 AI 模型的語言適應性",
        ],
        "長期目標": [
            "完整的國際化框架",
            "跨語言漏洞關聯分析",
            "多語言 AI 訓練管道",
            "全球化安全標準支援",
        ],
    }

    for category, items in recommendations.items():
        print(f"\n[LIST] {category}:")
        for item in items:
            print(f"   - {item}")

    return recommendations


if __name__ == "__main__":
    analysis = analyze_cross_language_and_ai()
    recommendations = generate_enhancement_recommendations()
