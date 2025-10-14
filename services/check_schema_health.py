"""
AIVA Schema 健康檢查腳本
快速驗證 schemas.py 修復狀態
"""
import sys
from pathlib import Path

def main():
    print("🔍 AIVA Schema 健康檢查")
    print("=" * 60)

    errors = []
    warnings = []

    # 測試 1: 模組導入
    print("\n📦 測試模組導入...")
    try:
        from aiva_common.schemas import (
            CVSSv3Metrics,
            SARIFLocation,
            SARIFResult,
            SARIFReport,
            AttackStep,
            AttackPlan,
        )
        print("  ✅ 所有核心類別導入成功")
    except ImportError as e:
        errors.append(f"導入失敗: {e}")
        print(f"  ❌ 導入失敗: {e}")

    # 測試 2: CVSS 計算
    print("\n🧮 測試 CVSS 計算...")
    try:
        from aiva_common.schemas import CVSSv3Metrics
        cvss = CVSSv3Metrics(
            attack_vector='N',
            attack_complexity='L',
            privileges_required='N',
            user_interaction='N',
            scope='C',
            confidentiality='H',
            integrity='H',
            availability='H'
        )
        score = cvss.calculate_base_score()
        if score == 10.0:
            print(f"  ✅ CVSS 計算正確: {score}")
        else:
            warnings.append(f"CVSS 分數異常: {score}")
            print(f"  ⚠️  CVSS 分數異常: {score}")
    except Exception as e:
        errors.append(f"CVSS 計算失敗: {e}")
        print(f"  ❌ CVSS 計算失敗: {e}")

    # 測試 3: SARIF 結構
    print("\n📄 測試 SARIF 結構...")
    try:
        from aiva_common.schemas import SARIFLocation, SARIFResult

        location = SARIFLocation(
            uri="test.py",
            start_line=10,
            start_column=5
        )

        result = SARIFResult(
            rule_id="CWE-89",
            message="SQL 注入測試",
            level="error",
            locations=[location]
        )
        print("  ✅ SARIF 結構創建成功")
    except Exception as e:
        errors.append(f"SARIF 創建失敗: {e}")
        print(f"  ❌ SARIF 創建失敗: {e}")

    # 測試 4: 攻擊計畫
    print("\n⚔️  測試攻擊計畫...")
    try:
        from aiva_common.schemas import AttackStep, AttackPlan

        step = AttackStep(
            step_id="step_001",
            name="SQL注入測試",
            description="測試SQL注入漏洞",
            target="http://test.com",
            mitre_technique_id="T1190"
        )
        print("  ✅ AttackStep 創建成功")
    except Exception as e:
        errors.append(f"AttackStep 創建失敗: {e}")
        print(f"  ❌ AttackStep 創建失敗: {e}")

    # 測試 5: 檢查重複定義
    print("\n🔍 檢查重複類別定義...")
    schema_file = Path(__file__).parent / "aiva_common" / "schemas.py"
    if schema_file.exists():
        content = schema_file.read_text(encoding='utf-8')

        classes_to_check = [
            'CVSSv3Metrics',
            'SARIFLocation',
            'SARIFResult',
            'SARIFReport',
            'AttackStep',
            'AttackPlan',
            'TraceRecord',
        ]

        duplicates_found = False
        for class_name in classes_to_check:
            count = content.count(f'class {class_name}(BaseModel):')
            if count > 1:
                errors.append(f"{class_name} 有 {count} 個定義")
                print(f"  ❌ {class_name} 重複 {count} 次")
                duplicates_found = True

        if not duplicates_found:
            print("  ✅ 無重複類別定義")

    # 總結
    print("\n" + "=" * 60)
    print("📊 檢查總結")
    print("=" * 60)

    if not errors and not warnings:
        print("✨ 所有檢查通過！系統健康狀態良好。")
        return 0
    else:
        if errors:
            print(f"❌ 發現 {len(errors)} 個錯誤:")
            for err in errors:
                print(f"   • {err}")

        if warnings:
            print(f"⚠️  發現 {len(warnings)} 個警告:")
            for warn in warnings:
                print(f"   • {warn}")

        return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
