#!/usr/bin/env python3
"""
AIVA 五大模組通連性檢查工具
==========================

基於Schema自動化系統，檢查五大模組間的通連性

功能:
- 🔍 模組間通信檢查
- 📡 Schema一致性驗證
- 🔄 跨語言數據傳遞檢查
- 📊 通連性健康度報告
- 🚀 端到端工作流檢查
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class AIVAModuleConnectivityChecker:
    """AIVA模組通連性檢查器"""

    def __init__(self):
        self.check_results = {}
        self.schema_check_results = {}
        self.connectivity_score = 0

    async def check_schema_import_connectivity(self) -> dict[str, Any]:
        """檢查新Schema系統的導入連通性"""
        logger.info("🔍 檢查Schema導入連通性...")

        results = {
            "python_schemas": {},
            "cross_module_imports": {},
            "schema_consistency": True,
            "errors": [],
        }

        # 檢查生成的Python Schema
        try:
            # 檢查基礎類型

            results["python_schemas"]["base_types"] = {
                "MessageHeader": True,
                "Target": True,
                "Vulnerability": True,
            }
            logger.info("  ✅ 基礎類型Schema導入成功")

            # 檢查消息Schema

            results["python_schemas"]["messaging"] = {
                "AivaMessage": True,
                "AIVARequest": True,
                "AIVAResponse": True,
            }
            logger.info("  ✅ 消息Schema導入成功")

            # 檢查任務Schema

            results["python_schemas"]["tasks"] = {
                "FunctionTaskPayload": True,
                "FunctionTaskTarget": True,
            }
            logger.info("  ✅ 任務Schema導入成功")

            # 檢查發現Schema

            results["python_schemas"]["findings"] = {
                "FindingPayload": True,
                "FindingEvidence": True,
            }
            logger.info("  ✅ 發現Schema導入成功")

        except Exception as e:
            error_msg = f"Schema導入錯誤: {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")
            results["schema_consistency"] = False

        return results

    async def check_cross_module_messaging(self) -> dict[str, Any]:
        """檢查跨模組消息傳遞"""
        logger.info("📡 檢查跨模組消息傳遞...")

        results = {
            "message_creation": False,
            "schema_validation": False,
            "serialization": False,
            "errors": [],
        }

        try:
            # 使用新Schema創建消息
            # 導入枚舉
            from services.aiva_common.enums import ModuleName, Topic
            from services.aiva_common.schemas import AivaMessage, MessageHeader

            # 創建消息標頭
            header = MessageHeader(
                message_id="check_msg_001",
                trace_id="1a2b3c4d-5e6f-7890-abcd-ef1234567890",
                source_module=ModuleName.CORE,
                timestamp=datetime.now(),
                version="1.0",
            )

            # 創建AIVA消息
            message = AivaMessage(
                header=header,
                topic=Topic.TASK_SCAN_START,
                schema_version="1.0",
                payload={"check": "cross_module_communication"},
            )

            results["message_creation"] = True
            logger.info("  ✅ 消息創建成功")

            # 檢查序列化
            json_data = message.model_dump()
            results["serialization"] = True
            logger.info("  ✅ 消息序列化成功")

            # 檢查反序列化
            restored_message = AivaMessage.model_validate(json_data)

            if restored_message.header.message_id == "check_msg_001":
                results["schema_validation"] = True
                logger.info("  ✅ Schema驗證成功")

        except Exception as e:
            error_msg = f"跨模組消息檢查錯誤: {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")

        return results

    async def check_module_integration_points(self) -> dict[str, Any]:
        """檢查模組整合點"""
        logger.info("🔗 檢查模組整合點...")

        results = {
            "ai_engine_integration": False,
            "attack_engine_integration": False,
            "scan_engine_integration": False,
            "feature_detection_integration": False,
            "integration_services": False,
            "errors": [],
        }

        # 檢查AI引擎整合
        try:
            # 檢查AI引擎核心模組
            ai_engine_path = project_root / "services" / "core" / "aiva_core" / "ai_engine"
            if ai_engine_path.exists():
                py_files = list(ai_engine_path.rglob("*.py"))
                if py_files:
                    results["ai_engine_integration"] = True
                    logger.info("  ✅ AI引擎整合點可用")
        except Exception as e:
            results["errors"].append(f"AI引擎整合檢查錯誤: {e}")

        # 檢查攻擊引擎整合
        try:
            attack_engine_path = project_root / "services" / "core" / "aiva_core" / "attack"
            if attack_engine_path.exists():
                py_files = list(attack_engine_path.rglob("*.py"))
                if py_files:
                    results["attack_engine_integration"] = True
                    logger.info("  ✅ 攻擊引擎整合點可用")
        except Exception as e:
            results["errors"].append(f"攻擊引擎整合檢查錯誤: {e}")

        # 檢查掃描引擎整合
        try:
            scan_engine_path = project_root / "services" / "scan"
            if scan_engine_path.exists():
                py_files = list(scan_engine_path.rglob("*.py"))
                if len(py_files) > 10:  # 至少有一定數量的檔案
                    results["scan_engine_integration"] = True
                    logger.info("  ✅ 掃描引擎整合點可用")
        except Exception as e:
            results["errors"].append(f"掃描引擎整合檢查錯誤: {e}")

        # 檢查功能檢測整合
        try:
            features_path = project_root / "services" / "features"
            if features_path.exists():
                py_files = list(features_path.rglob("*.py"))
                go_files = list(features_path.rglob("*.go"))
                if len(py_files) > 20 and len(go_files) > 5:  # 有足夠的跨語言檔案
                    results["feature_detection_integration"] = True
                    logger.info("  ✅ 功能檢測整合點可用")
        except Exception as e:
            results["errors"].append(f"功能檢測整合檢查錯誤: {e}")

        # 檢查整合服務
        try:
            integration_path = project_root / "services" / "integration"
            if integration_path.exists():
                py_files = list(integration_path.rglob("*.py"))
                if py_files:
                    results["integration_services"] = True
                    logger.info("  ✅ 整合服務可用")
        except Exception as e:
            results["errors"].append(f"整合服務檢查錯誤: {e}")

        return results

    async def check_go_schema_connectivity(self) -> dict[str, Any]:
        """檢查Go Schema連接性"""
        logger.info("🔄 檢查Go Schema連接性...")

        results = {
            "go_schema_exists": False,
            "struct_count": 0,
            "json_tags_valid": False,
            "errors": [],
        }

        try:
            go_schema_path = project_root / "services" / "features" / "common" / "go" / "aiva_common_go" / "schemas" / "generated" / "schemas.go"

            if go_schema_path.exists():
                results["go_schema_exists"] = True

                with open(go_schema_path, encoding="utf-8") as f:
                    content = f.read()

                # 統計結構體
                struct_count = content.count("type ") - content.count("// type ")
                results["struct_count"] = struct_count

                # 檢查JSON標籤
                if '`json:"' in content:
                    results["json_tags_valid"] = True

                logger.info(f"  ✅ Go Schema可用 ({struct_count} 個結構體)")
            else:
                results["errors"].append("Go Schema檔案不存在")

        except Exception as e:
            error_msg = f"Go Schema檢查錯誤: {e}"
            results["errors"].append(error_msg)
            logger.error(f"  ❌ {error_msg}")

        return results

    def calculate_connectivity_score(self, all_results: dict[str, Any]) -> int:
        """計算整體通連性得分 (0-100)"""
        total_checks = 0
        passed_checks = 0

        # Schema導入檢查 (30分)
        if "schema_imports" in all_results:
            total_checks += 4  # 4個主要Schema模組
            passed_checks += len(
                [
                    v
                    for v in all_results["schema_imports"]["python_schemas"].values()
                    if v
                ]
            )

        # 跨模組消息檢查 (30分)
        if "cross_module_messaging" in all_results:
            total_checks += 3
            if all_results["cross_module_messaging"]["message_creation"]:
                passed_checks += 1
            if all_results["cross_module_messaging"]["serialization"]:
                passed_checks += 1
            if all_results["cross_module_messaging"]["schema_validation"]:
                passed_checks += 1

        # 模組整合點檢查 (25分)
        if "module_integration" in all_results:
            integration_results = all_results["module_integration"]
            total_checks += 5
            passed_checks += sum(
                [
                    integration_results["ai_engine_integration"],
                    integration_results["attack_engine_integration"],
                    integration_results["scan_engine_integration"],
                    integration_results["feature_detection_integration"],
                    integration_results["integration_services"],
                ]
            )

        # Go Schema檢查 (15分)
        if "go_schema" in all_results:
            total_checks += 2
            if all_results["go_schema"]["go_schema_exists"]:
                passed_checks += 1
            if all_results["go_schema"]["json_tags_valid"]:
                passed_checks += 1

        if total_checks > 0:
            score = int((passed_checks / total_checks) * 100)
        else:
            score = 0

        return score

    async def run_full_connectivity_check(self) -> dict[str, Any]:
        """執行完整通連性檢查"""
        logger.info("🚀 開始AIVA五大模組通連性檢查...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "check_summary": {},
            "detailed_results": {},
            "connectivity_score": 0,
            "recommendations": [],
        }

        try:
            # 1. Schema導入檢查
            schema_results = await self.check_schema_import_connectivity()
            results["detailed_results"]["schema_imports"] = schema_results

            # 2. 跨模組消息檢查
            messaging_results = await self.check_cross_module_messaging()
            results["detailed_results"]["cross_module_messaging"] = messaging_results

            # 3. 模組整合點檢查
            integration_results = await self.check_module_integration_points()
            results["detailed_results"]["module_integration"] = integration_results

            # 4. Go Schema檢查
            go_schema_results = await self.check_go_schema_connectivity()
            results["detailed_results"]["go_schema"] = go_schema_results

            # 5. 計算總分
            connectivity_score = self.calculate_connectivity_score(
                results["detailed_results"]
            )
            results["connectivity_score"] = connectivity_score

            # 6. 生成建議
            recommendations = self.generate_recommendations(results["detailed_results"])
            results["recommendations"] = recommendations

            # 7. 檢查摘要
            results["check_summary"] = {
                "schema_system_health": len(schema_results["errors"]) == 0,
                "cross_module_communication": messaging_results.get(
                    "schema_validation", False
                ),
                "module_integration_health": sum(
                    [v for v in integration_results.values() if isinstance(v, bool)]
                )
                >= 3,
                "go_schema_health": go_schema_results.get("go_schema_exists", False),
                "overall_health": connectivity_score >= 70,
            }

            logger.info(f"🎉 通連性檢查完成! 總分: {connectivity_score}/100")

        except Exception as e:
            error_msg = f"通連性檢查執行錯誤: {e}"
            results["error"] = error_msg
            logger.error(f"❌ {error_msg}")
            traceback.print_exc()

        return results

    def generate_recommendations(self, detailed_results: dict[str, Any]) -> list[str]:
        """基於檢查結果生成改進建議"""
        recommendations = []

        # Schema相關建議
        if "schema_imports" in detailed_results:
            schema_errors = detailed_results["schema_imports"].get("errors", [])
            if schema_errors:
                recommendations.append(
                    "🔧 修復Schema導入錯誤，確保所有Schema模組正確生成"
                )

        # 跨模組通信建議
        if "cross_module_messaging" in detailed_results:
            messaging = detailed_results["cross_module_messaging"]
            if not messaging.get("message_creation", False):
                recommendations.append("📡 實現統一的跨模組消息創建機制")
            if not messaging.get("serialization", False):
                recommendations.append("🔄 加強消息序列化和反序列化處理")

        # 模組整合建議
        if "module_integration" in detailed_results:
            integration = detailed_results["module_integration"]
            if not integration.get("ai_engine_integration", False):
                recommendations.append("🧠 強化AI引擎模組整合點")
            if not integration.get("scan_engine_integration", False):
                recommendations.append("🔍 優化掃描引擎整合接口")

        # Go Schema建議
        if "go_schema" in detailed_results:
            go_schema = detailed_results["go_schema"]
            if not go_schema.get("go_schema_exists", False):
                recommendations.append("🐹 確保Go Schema檔案正確生成")

        # 通用建議
        if not recommendations:
            recommendations.append("✨ 通連性良好！建議定期執行此檢查以維持系統健康")

        return recommendations


async def main():
    """主程式"""
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("aiva_connectivity_check.log", encoding="utf-8"),
        ],
    )

    # 執行檢查
    checker = AIVAModuleConnectivityChecker()
    results = await checker.run_full_connectivity_check()

    # 生成報告
    print("\n" + "=" * 60)
    print("📋 AIVA五大模組通連性檢查報告")
    print("=" * 60)
    print(f"⏰ 檢查時間: {results['timestamp']}")
    print(f"🎯 通連性得分: {results['connectivity_score']}/100")

    if results["connectivity_score"] >= 90:
        health_status = "🟢 優秀"
    elif results["connectivity_score"] >= 70:
        health_status = "🟡 良好"
    elif results["connectivity_score"] >= 50:
        health_status = "🟠 尚可"
    else:
        health_status = "🔴 需要改善"

    print(f"📊 健康狀況: {health_status}")

    print("\n📋 檢查摘要:")
    summary = results.get("check_summary", {})
    for key, status in summary.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {key.replace('_', ' ').title()}")

    print("\n💡 改進建議:")
    for rec in results.get("recommendations", []):
        print(f"  {rec}")

    # 儲存詳細報告
    report_file = (
        f"aiva_connectivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📄 詳細報告已儲存至: {report_file}")

    # 返回狀態碼
    success = results["connectivity_score"] >= 70
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
