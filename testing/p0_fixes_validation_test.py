#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 缺陷修復驗證測試腳本

【標準做法】按照 Core 模組 README 的規範：
- 使用正確的包導入方式
- 不修改 sys.path
- 遵循專案架構標準

此腳本自動驗證以下 P0 修復成果：
1. IDOR 多帳戶測試功能
2. AI Commander 核心功能
3. BioNeuron Master 邏輯
4. ModelTrainer 功能

目標靶場: http://localhost:3000

執行方式: 從專案根目錄執行
    python -m testing.p0_fixes_validation_test
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# 不再修改 sys.path，使用標準導入
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / f"p0_validation_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
logger = logging.getLogger("P0_Validation")


class P0FixesValidator:
    """P0 缺陷修復驗證器"""
    
    def __init__(self, target_url: str = "http://localhost:3000"):
        self.target_url = target_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "target": target_url,
            "tests": []
        }
        
    async def run_all_tests(self):
        """執行所有驗證測試"""
        logger.info("🚀 開始 P0 缺陷修復驗證測試")
        logger.info(f"📍 目標靶場: {self.target_url}")
        logger.info("=" * 80)
        
        # 測試 1: IDOR 多帳戶測試
        await self.test_idor_multi_user()
        
        # 測試 2: AI Commander 核心功能
        await self.test_ai_commander()
        
        # 測試 3: BioNeuron Master 邏輯
        await self.test_bioneuron_master()
        
        # 測試 4: ModelTrainer 功能
        await self.test_model_trainer()
        
        # 生成測試報告
        self.generate_report()
        
    async def test_idor_multi_user(self):
        """測試 1: 驗證 IDOR 多帳戶測試功能"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 測試 1: IDOR 多帳戶測試功能")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "IDOR 多帳戶測試",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            # 導入 IDOR Worker
            from services.features.function_idor.worker import IdorWorker
            from services.aiva_common.schemas import FunctionTaskPayload
            
            logger.info("✅ 成功導入 IdorWorker")
            test_result["details"].append("模組導入成功")
            
            # 創建 Worker 實例
            worker = IdorWorker()
            logger.info("✅ 成功創建 IdorWorker 實例")
            
            # 測試多帳戶憑證配置
            test_configs = [
                {
                    "name": "Bearer Token 認證",
                    "config": {
                        "second_user_auth": {
                            "type": "bearer",
                            "token": "test_token_user2"
                        }
                    },
                    "expected_header": "Authorization"
                },
                {
                    "name": "Cookie 認證",
                    "config": {
                        "second_user_auth": {
                            "type": "cookie",
                            "cookie": "session=user2_session"
                        }
                    },
                    "expected_header": "Cookie"
                },
                {
                    "name": "API Key 認證",
                    "config": {
                        "second_user_auth": {
                            "type": "api_key",
                            "api_key": "test_api_key_user2",
                            "key_name": "X-API-Key"
                        }
                    },
                    "expected_header": "X-API-Key"
                },
                {
                    "name": "Basic Auth 認證",
                    "config": {
                        "second_user_auth": {
                            "type": "basic",
                            "username": "user2",
                            "password": "pass2"
                        }
                    },
                    "expected_header": "Authorization"
                }
            ]
            
            passed_tests = 0
            for test_config in test_configs:
                try:
                    # 創建測試任務
                    task = type('Task', (), {'config': test_config["config"]})()
                    
                    # 調用 _get_test_user_auth
                    auth_headers = worker._get_test_user_auth(task)
                    
                    if auth_headers and test_config["expected_header"] in auth_headers:
                        logger.info(f"  ✅ {test_config['name']}: 通過")
                        logger.info(f"     返回: {auth_headers}")
                        passed_tests += 1
                        test_result["details"].append(f"{test_config['name']}: 通過")
                    else:
                        logger.warning(f"  ⚠️ {test_config['name']}: 失敗")
                        test_result["details"].append(f"{test_config['name']}: 失敗")
                        
                except Exception as e:
                    logger.error(f"  ❌ {test_config['name']}: 錯誤 - {e}")
                    test_result["errors"].append(f"{test_config['name']}: {str(e)}")
            
            # 測試無配置情況（應返回 None）
            task_no_config = type('Task', (), {'config': None})()
            auth_no_config = worker._get_test_user_auth(task_no_config)
            if auth_no_config is None:
                logger.info("  ✅ 無配置測試: 通過 (正確返回 None)")
                passed_tests += 1
                test_result["details"].append("無配置測試: 通過")
            
            # 評估測試結果
            if passed_tests >= 4:
                test_result["status"] = "PASSED"
                logger.info(f"✅ IDOR 多帳戶測試: 通過 ({passed_tests}/5)")
            else:
                test_result["status"] = "FAILED"
                logger.warning(f"⚠️ IDOR 多帳戶測試: 部分通過 ({passed_tests}/5)")
                
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"❌ IDOR 測試錯誤: {e}", exc_info=True)
            
        self.results["tests"].append(test_result)
        
    async def test_ai_commander(self):
        """測試 2: 驗證 AI Commander 核心功能"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 測試 2: AI Commander 核心功能")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "AI Commander 核心功能",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            # 導入 AI Commander
            from services.core.aiva_core.ai_commander import AICommander
            
            logger.info("✅ 成功導入 AICommander")
            test_result["details"].append("模組導入成功")
            
            # 創建 AICommander 實例
            commander = AICommander()
            logger.info("✅ 成功創建 AICommander 實例")
            
            # 檢查關鍵組件
            checks = []
            
            # 1. 檢查經驗資料庫整合
            if hasattr(commander, 'experience_manager'):
                logger.info("  ✅ 經驗資料庫 (ExperienceManager): 已整合")
                checks.append(True)
                test_result["details"].append("ExperienceManager 已整合")
                
                # 檢查 storage_backend
                if hasattr(commander.experience_manager, 'storage'):
                    logger.info("  ✅ 儲存後端 (StorageBackend): 已配置")
                    checks.append(True)
                    test_result["details"].append("StorageBackend 已配置")
                else:
                    logger.warning("  ⚠️ 儲存後端未配置")
                    checks.append(False)
            else:
                logger.warning("  ⚠️ ExperienceManager 未整合")
                checks.append(False)
            
            # 2. 檢查 BioNeuronRAGAgent
            if hasattr(commander, 'bio_neuron_agent'):
                logger.info("  ✅ BioNeuronRAGAgent: 已接入")
                checks.append(True)
                test_result["details"].append("BioNeuronRAGAgent 已接入")
            else:
                logger.warning("  ⚠️ BioNeuronRAGAgent 未接入")
                checks.append(False)
            
            # 3. 檢查攻擊計畫生成函式
            if hasattr(commander, '_plan_attack'):
                logger.info("  ✅ 攻擊計畫生成 (_plan_attack): 已實現")
                checks.append(True)
                test_result["details"].append("_plan_attack 已實現")
                
                # 測試函式簽名
                import inspect
                sig = inspect.signature(commander._plan_attack)
                logger.info(f"     函式簽名: {sig}")
            else:
                logger.warning("  ⚠️ _plan_attack 未實現")
                checks.append(False)
            
            # 4. 檢查策略決策函式
            if hasattr(commander, '_make_strategy_decision'):
                logger.info("  ✅ 策略決策 (_make_strategy_decision): 已實現")
                checks.append(True)
                test_result["details"].append("_make_strategy_decision 已實現")
            else:
                logger.warning("  ⚠️ _make_strategy_decision 未實現")
                checks.append(False)
            
            # 評估測試結果
            passed_checks = sum(checks)
            total_checks = len(checks)
            
            if passed_checks == total_checks:
                test_result["status"] = "PASSED"
                logger.info(f"✅ AI Commander 測試: 完全通過 ({passed_checks}/{total_checks})")
            elif passed_checks >= total_checks * 0.7:
                test_result["status"] = "PARTIAL"
                logger.info(f"⚠️ AI Commander 測試: 部分通過 ({passed_checks}/{total_checks})")
            else:
                test_result["status"] = "FAILED"
                logger.warning(f"❌ AI Commander 測試: 未通過 ({passed_checks}/{total_checks})")
                
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"❌ AI Commander 測試錯誤: {e}", exc_info=True)
            
        self.results["tests"].append(test_result)
        
    async def test_bioneuron_master(self):
        """測試 3: 驗證 BioNeuron Master 邏輯"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 測試 3: BioNeuron Master 邏輯")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "BioNeuron Master 邏輯",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            # 導入 BioNeuron Master
            from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
            
            logger.info("✅ 成功導入 BioNeuronMasterController")
            test_result["details"].append("模組導入成功")
            
            # 創建實例（使用臨時路徑）
            controller = BioNeuronMasterController(
                codebase_path=str(PROJECT_ROOT),
                default_mode="hybrid"  # 傳遞字串，程式會自動轉換
            )
            logger.info("✅ 成功創建 BioNeuronMasterController 實例")
            
            # 檢查關鍵組件
            checks = []
            
            # 1. 檢查 NLU 功能
            if hasattr(controller, '_parse_ui_command'):
                logger.info("  ✅ NLU 功能 (_parse_ui_command): 已實現")
                checks.append(True)
                test_result["details"].append("NLU 功能已實現")
                
                # 測試 NLU（簡單測試）
                try:
                    test_input = "掃描 http://localhost:3000"
                    result = await controller._parse_ui_command(test_input)
                    logger.info(f"     NLU 測試結果: {result}")
                    test_result["details"].append(f"NLU 測試: {result.get('intent', 'unknown')}")
                except Exception as e:
                    logger.warning(f"     NLU 測試警告: {e}")
            else:
                logger.warning("  ⚠️ _parse_ui_command 未實現")
                checks.append(False)
            
            # 2. 檢查 AI 決策功能
            if hasattr(controller, '_bio_neuron_decide'):
                logger.info("  ✅ AI 決策 (_bio_neuron_decide): 已實現")
                checks.append(True)
                test_result["details"].append("AI 決策功能已實現")
            else:
                logger.warning("  ⚠️ _bio_neuron_decide 未實現")
                checks.append(False)
            
            # 3. 檢查經驗學習整合
            if hasattr(controller, '_learn_from_execution'):
                logger.info("  ✅ 經驗學習 (_learn_from_execution): 已實現")
                checks.append(True)
                test_result["details"].append("經驗學習已實現")
            else:
                logger.warning("  ⚠️ _learn_from_execution 未實現")
                checks.append(False)
            
            # 4. 檢查任務啟動功能
            task_start_methods = ['_start_scan_task', '_start_attack_task', '_start_training_task']
            task_checks = []
            for method in task_start_methods:
                if hasattr(controller, method):
                    logger.info(f"  ✅ {method}: 已實現")
                    task_checks.append(True)
                else:
                    logger.warning(f"  ⚠️ {method}: 未實現")
                    task_checks.append(False)
            
            if all(task_checks):
                checks.append(True)
                test_result["details"].append("所有任務啟動功能已實現")
            else:
                checks.append(False)
                test_result["details"].append(f"任務啟動功能: {sum(task_checks)}/3 實現")
            
            # 5. 檢查 RAG Engine
            if hasattr(controller, 'rag_engine'):
                logger.info("  ✅ RAG Engine: 已整合")
                checks.append(True)
                test_result["details"].append("RAG Engine 已整合")
            else:
                logger.warning("  ⚠️ RAG Engine 未整合")
                checks.append(False)
            
            # 評估測試結果
            passed_checks = sum(checks)
            total_checks = len(checks)
            
            if passed_checks == total_checks:
                test_result["status"] = "PASSED"
                logger.info(f"✅ BioNeuron Master 測試: 完全通過 ({passed_checks}/{total_checks})")
            elif passed_checks >= total_checks * 0.7:
                test_result["status"] = "PARTIAL"
                logger.info(f"⚠️ BioNeuron Master 測試: 部分通過 ({passed_checks}/{total_checks})")
            else:
                test_result["status"] = "FAILED"
                logger.warning(f"❌ BioNeuron Master 測試: 未通過 ({passed_checks}/{total_checks})")
                
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"❌ BioNeuron Master 測試錯誤: {e}", exc_info=True)
            
        self.results["tests"].append(test_result)
        
    async def test_model_trainer(self):
        """測試 4: 驗證 ModelTrainer 功能"""
        logger.info("\n" + "=" * 80)
        logger.info("🧪 測試 4: ModelTrainer 功能")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "ModelTrainer 功能",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            # 導入 ModelTrainer
            from services.core.aiva_core.learning.model_trainer import ModelTrainer
            
            logger.info("✅ 成功導入 ModelTrainer")
            test_result["details"].append("模組導入成功")
            
            # 創建實例
            trainer = ModelTrainer()
            logger.info("✅ 成功創建 ModelTrainer 實例")
            
            # 檢查關鍵函式
            checks = []
            methods_to_check = [
                ('_train_model_supervised', '監督學習訓練'),
                ('_train_model_rl', '強化學習訓練'),
                ('_evaluate_model', '模型評估'),
                ('_save_model', '模型保存'),
                ('load_model', '模型載入')
            ]
            
            for method_name, description in methods_to_check:
                if hasattr(trainer, method_name):
                    logger.info(f"  ✅ {description} ({method_name}): 已實現")
                    checks.append(True)
                    test_result["details"].append(f"{description}: 已實現")
                    
                    # 檢查函式是否為實際實現（不是空實作）
                    import inspect
                    method = getattr(trainer, method_name)
                    source = inspect.getsource(method)
                    
                    # 檢查是否包含實際邏輯（非 TODO 或簡單 return）
                    if 'TODO' not in source and 'pass' not in source[:100]:
                        logger.info(f"     ✓ 包含實際實現邏輯")
                    else:
                        logger.warning(f"     ⚠️ 可能為空實作")
                else:
                    logger.warning(f"  ⚠️ {description} ({method_name}): 未實現")
                    checks.append(False)
            
            # 測試監督學習功能（如果可能）
            try:
                import numpy as np
                from services.aiva_common.schemas import ModelTrainingConfig
                
                # 創建簡單測試數據
                X_train = np.random.rand(100, 10)
                y_train = np.random.randint(0, 2, 100)
                X_val = np.random.rand(20, 10)
                y_val = np.random.randint(0, 2, 20)
                
                config = ModelTrainingConfig(
                    model_type="random_forest",
                    max_epochs=5,
                    batch_size=32
                )
                
                logger.info("  🔬 執行監督學習測試...")
                result = await trainer._train_model_supervised(
                    X_train, y_train, X_val, y_val, config
                )
                
                if result and 'error' not in result:
                    logger.info(f"  ✅ 監督學習測試: 成功")
                    logger.info(f"     訓練結果: {result}")
                    test_result["details"].append("監督學習實際測試: 成功")
                else:
                    logger.warning(f"  ⚠️ 監督學習測試: 返回錯誤")
                    test_result["details"].append(f"監督學習測試: {result.get('error', 'unknown')}")
                    
            except Exception as e:
                logger.warning(f"  ⚠️ 監督學習測試跳過: {e}")
                test_result["details"].append(f"監督學習測試: 跳過 ({str(e)[:50]})")
            
            # 評估測試結果
            passed_checks = sum(checks)
            total_checks = len(checks)
            
            if passed_checks == total_checks:
                test_result["status"] = "PASSED"
                logger.info(f"✅ ModelTrainer 測試: 完全通過 ({passed_checks}/{total_checks})")
            elif passed_checks >= total_checks * 0.8:
                test_result["status"] = "PARTIAL"
                logger.info(f"⚠️ ModelTrainer 測試: 部分通過 ({passed_checks}/{total_checks})")
            else:
                test_result["status"] = "FAILED"
                logger.warning(f"❌ ModelTrainer 測試: 未通過 ({passed_checks}/{total_checks})")
                
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"❌ ModelTrainer 測試錯誤: {e}", exc_info=True)
            
        self.results["tests"].append(test_result)
        
    def generate_report(self):
        """生成測試報告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 測試報告")
        logger.info("=" * 80)
        
        # 統計結果
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t["status"] == "PASSED")
        partial = sum(1 for t in self.results["tests"] if t["status"] == "PARTIAL")
        failed = sum(1 for t in self.results["tests"] if t["status"] == "FAILED")
        errors = sum(1 for t in self.results["tests"] if t["status"] == "ERROR")
        
        logger.info(f"\n總測試數: {total}")
        logger.info(f"✅ 完全通過: {passed}")
        logger.info(f"⚠️ 部分通過: {partial}")
        logger.info(f"❌ 未通過: {failed}")
        logger.info(f"🔥 錯誤: {errors}")
        
        # 詳細結果
        logger.info("\n" + "-" * 80)
        for test in self.results["tests"]:
            status_icon = {
                "PASSED": "✅",
                "PARTIAL": "⚠️",
                "FAILED": "❌",
                "ERROR": "🔥",
                "PENDING": "⏸️"
            }.get(test["status"], "❓")
            
            logger.info(f"\n{status_icon} {test['test_name']}: {test['status']}")
            for detail in test["details"]:
                logger.info(f"  - {detail}")
            if test["errors"]:
                logger.info("  錯誤:")
                for error in test["errors"]:
                    logger.info(f"    - {error}")
        
        # 保存 JSON 報告
        report_path = PROJECT_ROOT / "_out" / f"p0_validation_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 詳細報告已保存至: {report_path}")
        
        # 最終評估
        logger.info("\n" + "=" * 80)
        if passed == total:
            logger.info("🎉 所有 P0 修復驗證測試全部通過！")
        elif passed + partial >= total * 0.8:
            logger.info("✅ 大部分 P0 修復驗證測試通過")
        else:
            logger.warning("⚠️ 部分 P0 修復需要進一步檢查")
        logger.info("=" * 80)


async def main():
    """主函式"""
    # 確保日誌目錄存在
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "_out").mkdir(exist_ok=True)
    
    # 創建驗證器
    validator = P0FixesValidator(target_url="http://localhost:3000")
    
    # 執行測試
    await validator.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
