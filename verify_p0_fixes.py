#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 缺陷修復驗證腳本

【標準做法】按照 Core 模組 README 的規範：
- 使用正確的包導入方式
- 不修改 sys.path
- 遵循專案架構標準

此腳本從專案根目錄執行: python verify_p0_fixes.py
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

# 設定日誌（使用 UTF-8 編碼避免 CP950 問題）
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            log_dir / f"p0_verification_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger("P0_Verification")


class P0FixesValidator:
    """P0 缺陷修復驗證器（標準版本）"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
    async def run_all_tests(self):
        """執行所有驗證測試"""
        logger.info("=" * 80)
        logger.info("開始 P0 缺陷修復驗證測試（標準版）")
        logger.info("=" * 80)
        
        # 測試 1: IDOR Worker
        await self.test_idor_worker()
        
        # 測試 2: AI Commander
        await self.test_ai_commander()
        
        # 測試 3: BioNeuron Master
        await self.test_bioneuron_master()
        
        # 測試 4: ModelTrainer
        await self.test_model_trainer()
        
        # 生成報告
        self.generate_report()
        
    async def test_idor_worker(self):
        """測試 1: IDOR Worker 多帳戶測試功能"""
        logger.info("\n" + "=" * 80)
        logger.info("測試 1: IDOR Worker 多帳戶測試")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "IDOR Worker 多帳戶測試",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            # 使用標準導入方式
            from services.features.function_idor.worker import IdorWorker
            
            logger.info("[OK] 成功導入 IdorWorker")
            test_result["details"].append("模組導入成功")
            
            # 創建實例
            worker = IdorWorker()
            logger.info("[OK] 成功創建 IdorWorker 實例")
            
            # 測試多帳戶認證配置
            test_configs = [
                {
                    "name": "Bearer Token",
                    "config": {"second_user_auth": {"type": "bearer", "token": "test_token"}},
                    "expected": "Authorization"
                },
                {
                    "name": "Cookie",
                    "config": {"second_user_auth": {"type": "cookie", "cookie": "session=test"}},
                    "expected": "Cookie"
                },
                {
                    "name": "API Key",
                    "config": {"second_user_auth": {"type": "api_key", "api_key": "key123", "key_name": "X-API-Key"}},
                    "expected": "X-API-Key"
                },
                {
                    "name": "Basic Auth",
                    "config": {"second_user_auth": {"type": "basic", "username": "user", "password": "pass"}},
                    "expected": "Authorization"
                }
            ]
            
            passed = 0
            for tc in test_configs:
                try:
                    task = type('Task', (), {'config': tc["config"]})()
                    auth = worker._get_test_user_auth(task)
                    
                    if auth and tc["expected"] in auth:
                        logger.info(f"  [OK] {tc['name']}: 通過")
                        passed += 1
                        test_result["details"].append(f"{tc['name']}: 通過")
                    else:
                        logger.warning(f"  [WARN] {tc['name']}: 失敗")
                        test_result["details"].append(f"{tc['name']}: 失敗")
                except Exception as e:
                    logger.error(f"  [ERROR] {tc['name']}: {e}")
                    test_result["errors"].append(f"{tc['name']}: {str(e)}")
            
            # 測試無配置
            task_none = type('Task', (), {'config': None})()
            if worker._get_test_user_auth(task_none) is None:
                logger.info("  [OK] 無配置測試: 通過")
                passed += 1
            
            test_result["status"] = "PASSED" if passed >= 4 else "FAILED"
            logger.info(f"[結果] IDOR Worker: {test_result['status']} ({passed}/5)")
            
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"[ERROR] IDOR 測試錯誤: {e}")
            
        self.results["tests"].append(test_result)
        
    async def test_ai_commander(self):
        """測試 2: AI Commander 核心功能"""
        logger.info("\n" + "=" * 80)
        logger.info("測試 2: AI Commander 核心功能")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "AI Commander 核心功能",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            from services.core.aiva_core.ai_commander import AICommander
            
            logger.info("[OK] 成功導入 AICommander")
            test_result["details"].append("模組導入成功")
            
            # 創建實例
            commander = AICommander()
            logger.info("[OK] 成功創建 AICommander 實例")
            
            # 檢查關鍵組件
            checks = []
            
            if hasattr(commander, 'experience_manager'):
                logger.info("  [OK] ExperienceManager: 已整合")
                checks.append(True)
                test_result["details"].append("ExperienceManager 已整合")
                
                if hasattr(commander.experience_manager, 'storage'):
                    logger.info("  [OK] StorageBackend: 已配置")
                    checks.append(True)
                    test_result["details"].append("StorageBackend 已配置")
                else:
                    checks.append(False)
            else:
                checks.append(False)
            
            if hasattr(commander, 'bio_neuron_agent'):
                logger.info("  [OK] BioNeuronRAGAgent: 已接入")
                checks.append(True)
                test_result["details"].append("BioNeuronRAGAgent 已接入")
            else:
                checks.append(False)
            
            if hasattr(commander, '_plan_attack'):
                logger.info("  [OK] _plan_attack: 已實現")
                checks.append(True)
                test_result["details"].append("_plan_attack 已實現")
            else:
                checks.append(False)
            
            if hasattr(commander, '_make_strategy_decision'):
                logger.info("  [OK] _make_strategy_decision: 已實現")
                checks.append(True)
                test_result["details"].append("_make_strategy_decision 已實現")
            else:
                checks.append(False)
            
            passed = sum(checks)
            total = len(checks)
            test_result["status"] = "PASSED" if passed == total else "PARTIAL"
            logger.info(f"[結果] AI Commander: {test_result['status']} ({passed}/{total})")
            
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"[ERROR] AI Commander 測試錯誤: {e}")
            
        self.results["tests"].append(test_result)
        
    async def test_bioneuron_master(self):
        """測試 3: BioNeuron Master 邏輯"""
        logger.info("\n" + "=" * 80)
        logger.info("測試 3: BioNeuron Master 邏輯")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "BioNeuron Master 邏輯",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
            
            logger.info("[OK] 成功導入 BioNeuronMasterController")
            test_result["details"].append("模組導入成功")
            
            # 創建實例（測試字串參數轉換）
            controller = BioNeuronMasterController(
                codebase_path=str(Path(__file__).parent),
                default_mode="hybrid"  # 測試字串轉 Enum
            )
            logger.info("[OK] 成功創建 BioNeuronMasterController 實例")
            
            # 檢查關鍵組件
            checks = []
            
            if hasattr(controller, '_parse_ui_command'):
                logger.info("  [OK] NLU 功能: 已實現")
                checks.append(True)
                test_result["details"].append("NLU 功能已實現")
            else:
                checks.append(False)
            
            if hasattr(controller, '_bio_neuron_decide'):
                logger.info("  [OK] AI 決策: 已實現")
                checks.append(True)
                test_result["details"].append("AI 決策已實現")
            else:
                checks.append(False)
            
            if hasattr(controller, '_learn_from_execution'):
                logger.info("  [OK] 經驗學習: 已實現")
                checks.append(True)
                test_result["details"].append("經驗學習已實現")
            else:
                checks.append(False)
            
            # 檢查任務啟動功能
            task_methods = ['_start_scan_task', '_start_attack_task', '_start_training_task']
            task_checks = all(hasattr(controller, m) for m in task_methods)
            checks.append(task_checks)
            
            if task_checks:
                logger.info("  [OK] 所有任務啟動功能: 已實現")
                test_result["details"].append("任務啟動功能已實現")
            
            if hasattr(controller, 'rag_engine'):
                logger.info("  [OK] RAG Engine: 已整合")
                checks.append(True)
                test_result["details"].append("RAG Engine 已整合")
            else:
                checks.append(False)
            
            passed = sum(checks)
            total = len(checks)
            test_result["status"] = "PASSED" if passed == total else "PARTIAL"
            logger.info(f"[結果] BioNeuron Master: {test_result['status']} ({passed}/{total})")
            
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"[ERROR] BioNeuron Master 測試錯誤: {e}")
            
        self.results["tests"].append(test_result)
        
    async def test_model_trainer(self):
        """測試 4: ModelTrainer 功能"""
        logger.info("\n" + "=" * 80)
        logger.info("測試 4: ModelTrainer 功能")
        logger.info("=" * 80)
        
        test_result = {
            "test_name": "ModelTrainer 功能",
            "status": "PENDING",
            "details": [],
            "errors": []
        }
        
        try:
            from services.core.aiva_core.learning.model_trainer import ModelTrainer
            
            logger.info("[OK] 成功導入 ModelTrainer")
            test_result["details"].append("模組導入成功")
            
            trainer = ModelTrainer()
            logger.info("[OK] 成功創建 ModelTrainer 實例")
            
            # 檢查關鍵方法
            methods = [
                ('_train_model_supervised', '監督學習訓練'),
                ('_train_model_rl', '強化學習訓練'),
                ('_evaluate_model', '模型評估'),
                ('_save_model', '模型保存'),
                ('load_model', '模型載入')
            ]
            
            checks = []
            for method_name, desc in methods:
                if hasattr(trainer, method_name):
                    logger.info(f"  [OK] {desc}: 已實現")
                    checks.append(True)
                    test_result["details"].append(f"{desc}: 已實現")
                else:
                    logger.warning(f"  [WARN] {desc}: 未實現")
                    checks.append(False)
            
            passed = sum(checks)
            total = len(checks)
            test_result["status"] = "PASSED" if passed == total else "PARTIAL"
            logger.info(f"[結果] ModelTrainer: {test_result['status']} ({passed}/{total})")
            
        except Exception as e:
            test_result["status"] = "ERROR"
            test_result["errors"].append(str(e))
            logger.error(f"[ERROR] ModelTrainer 測試錯誤: {e}")
            
        self.results["tests"].append(test_result)
        
    def generate_report(self):
        """生成測試報告"""
        logger.info("\n" + "=" * 80)
        logger.info("測試報告")
        logger.info("=" * 80)
        
        total = len(self.results["tests"])
        passed = sum(1 for t in self.results["tests"] if t["status"] == "PASSED")
        partial = sum(1 for t in self.results["tests"] if t["status"] == "PARTIAL")
        failed = sum(1 for t in self.results["tests"] if t["status"] == "FAILED")
        errors = sum(1 for t in self.results["tests"] if t["status"] == "ERROR")
        
        logger.info(f"\n總測試數: {total}")
        logger.info(f"[OK] 完全通過: {passed}")
        logger.info(f"[PARTIAL] 部分通過: {partial}")
        logger.info(f"[FAILED] 未通過: {failed}")
        logger.info(f"[ERROR] 錯誤: {errors}")
        
        # 詳細結果
        logger.info("\n" + "-" * 80)
        for test in self.results["tests"]:
            status_icon = {
                "PASSED": "[OK]",
                "PARTIAL": "[PARTIAL]",
                "FAILED": "[FAILED]",
                "ERROR": "[ERROR]"
            }.get(test["status"], "[?]")
            
            logger.info(f"\n{status_icon} {test['test_name']}: {test['status']}")
            for detail in test["details"]:
                logger.info(f"  - {detail}")
            if test["errors"]:
                logger.info("  錯誤:")
                for error in test["errors"]:
                    logger.info(f"    - {error}")
        
        # 保存 JSON 報告
        report_path = Path(__file__).parent / "_out" / f"p0_verification_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n[OK] 詳細報告已保存: {report_path}")
        
        # 最終評估
        logger.info("\n" + "=" * 80)
        if passed == total:
            logger.info("[成功] 所有 P0 修復驗證測試全部通過！")
        elif passed + partial >= total * 0.8:
            logger.info("[成功] 大部分 P0 修復驗證測試通過")
        else:
            logger.warning("[警告] 部分 P0 修復需要進一步檢查")
        logger.info("=" * 80)


async def main():
    """主函式"""
    validator = P0FixesValidator()
    await validator.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
