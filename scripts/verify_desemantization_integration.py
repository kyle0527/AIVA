#!/usr/bin/env python3
"""
HackOne v2.0 去語意化反射引擎整合驗證腳本
Verification Script for De-semanticization Reflex Engine Integration

驗證項目：
1. 模組內部導入是否正確
2. 與功能掃描模組對接是否正常
3. 初級數據讀取是否在整合模組進行
4. RAG 引擎初始化流程
5. 特徵編碼和檢索功能

使用方法：
    python scripts/verify_desemantization_integration.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class IntegrationVerifier:
    """整合驗證器"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
        
    def log_test(self, name: str, passed: bool, message: str = ""):
        """記錄測試結果"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            
        print(f"{status} | {name}")
        if message:
            print(f"     └─ {message}")
    
    async def verify_imports(self):
        """驗證 1: 模組內部導入"""
        print("\n" + "="*60)
        print("驗證 1: 模組內部導入")
        print("="*60)
        
        # 1.1 驗證 CapabilityRecord 擴展
        try:
            from services.integration.capability.models import CapabilityRecord
            
            # 檢查是否有 rag_trigger 欄位
            from pydantic import BaseModel
            fields = CapabilityRecord.model_fields
            has_rag_trigger = 'rag_trigger' in fields
            has_feature_signature = 'feature_signature' in fields
            
            self.log_test(
                "CapabilityRecord 擴展",
                has_rag_trigger and has_feature_signature,
                f"rag_trigger: {has_rag_trigger}, feature_signature: {has_feature_signature}"
            )
        except Exception as e:
            self.log_test("CapabilityRecord 擴展", False, str(e))
        
        # 1.2 驗證 VectorStore 方法
        try:
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            
            methods = [
                '_encode_rag_trigger',
                'add_capability_from_registry',
                'search_by_environment'
            ]
            
            has_methods = all(hasattr(VectorStore, m) for m in methods)
            missing = [m for m in methods if not hasattr(VectorStore, m)]
            
            self.log_test(
                "VectorStore 去語意化方法",
                has_methods,
                f"缺少方法: {missing}" if missing else "所有方法存在"
            )
        except Exception as e:
            self.log_test("VectorStore 去語意化方法", False, str(e))
        
        # 1.3 驗證 RAGEngine 方法
        try:
            from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
            
            methods = [
                'search_capabilities_by_environment',
                'load_capabilities_from_registry'
            ]
            
            has_methods = all(hasattr(RAGEngine, m) for m in methods)
            missing = [m for m in methods if not hasattr(RAGEngine, m)]
            
            self.log_test(
                "RAGEngine 環境檢索方法",
                has_methods,
                f"缺少方法: {missing}" if missing else "所有方法存在"
            )
        except Exception as e:
            self.log_test("RAGEngine 環境檢索方法", False, str(e))
        
        # 1.4 驗證 CapabilityRegistry JSON 載入
        try:
            from services.integration.capability.registry import CapabilityRegistry
            
            methods = [
                'load_capability_from_json',
                'load_capabilities_from_directory'
            ]
            
            has_methods = all(hasattr(CapabilityRegistry, m) for m in methods)
            missing = [m for m in methods if not hasattr(CapabilityRegistry, m)]
            
            self.log_test(
                "CapabilityRegistry JSON 載入",
                has_methods,
                f"缺少方法: {missing}" if missing else "所有方法存在"
            )
        except Exception as e:
            self.log_test("CapabilityRegistry JSON 載入", False, str(e))
    
    async def verify_feature_module_integration(self):
        """驗證 2: 與功能掃描模組對接"""
        print("\n" + "="*60)
        print("驗證 2: 與功能掃描模組對接")
        print("="*60)
        
        # 2.1 檢查功能模組是否使用整合模組
        try:
            from services.features.function_sqli.hackingtool_config import (
                CapabilityRecord
            )
            
            # 確認導入來自整合模組
            import_path = CapabilityRecord.__module__
            is_from_integration = 'integration.capability' in import_path
            
            self.log_test(
                "功能模組使用整合模組數據模型",
                is_from_integration,
                f"導入路徑: {import_path}"
            )
        except Exception as e:
            self.log_test("功能模組使用整合模組數據模型", False, str(e))
        
        # 2.2 檢查整合模組目錄結構
        try:
            integration_path = project_root / "services" / "integration" / "capability"
            
            required_files = [
                "registry.py",
                "models.py",
                "config.py",
                "capabilities"  # 目錄
            ]
            
            existing = [f for f in required_files if (integration_path / f).exists()]
            all_exist = len(existing) == len(required_files)
            
            self.log_test(
                "整合模組目錄結構",
                all_exist,
                f"存在: {len(existing)}/{len(required_files)}"
            )
        except Exception as e:
            self.log_test("整合模組目錄結構", False, str(e))
    
    async def verify_data_loading(self):
        """驗證 3: 初級數據讀取在整合模組"""
        print("\n" + "="*60)
        print("驗證 3: 初級數據讀取在整合模組")
        print("="*60)
        
        # 3.1 驗證 CapabilityRegistry 作為統一入口
        try:
            from services.integration.capability.registry import CapabilityRegistry
            
            registry = CapabilityRegistry(db_path="data/test_capability_registry.db")
            
            # 檢查是否有能力註冊方法
            has_register = hasattr(registry, 'register_capability')
            has_list = hasattr(registry, 'list_capabilities')
            has_get = hasattr(registry, 'get_capability')
            
            all_methods = has_register and has_list and has_get
            
            self.log_test(
                "CapabilityRegistry 作為統一入口",
                all_methods,
                f"register: {has_register}, list: {has_list}, get: {has_get}"
            )
            
            # 清理測試數據庫
            import os
            if os.path.exists("data/test_capability_registry.db"):
                os.remove("data/test_capability_registry.db")
                
        except Exception as e:
            self.log_test("CapabilityRegistry 作為統一入口", False, str(e))
        
        # 3.2 驗證 RAGEngine 從 Registry 載入
        try:
            from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
            
            has_load_method = hasattr(RAGEngine, 'load_capabilities_from_registry')
            
            self.log_test(
                "RAGEngine 從 Registry 載入能力",
                has_load_method,
                "方法存在" if has_load_method else "方法不存在"
            )
        except Exception as e:
            self.log_test("RAGEngine 從 Registry 載入能力", False, str(e))
    
    async def verify_rag_initialization(self):
        """驗證 4: RAG 引擎初始化流程"""
        print("\n" + "="*60)
        print("驗證 4: RAG 引擎初始化流程")
        print("="*60)
        
        # 4.1 驗證 KnowledgeBase → RAGEngine 初始化
        try:
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
            from services.core.aiva_core.cognitive_core.rag.rag_engine import RAGEngine
            
            # 創建測試實例
            vector_store = VectorStore(backend="memory")
            knowledge_base = KnowledgeBase(vector_store)
            rag_engine = RAGEngine(knowledge_base)
            
            has_kb = hasattr(rag_engine, 'knowledge_base')
            kb_has_vs = hasattr(knowledge_base, 'vector_store')
            
            self.log_test(
                "RAG 引擎初始化鏈",
                has_kb and kb_has_vs,
                "VectorStore → KnowledgeBase → RAGEngine"
            )
        except Exception as e:
            self.log_test("RAG 引擎初始化鏈", False, str(e))
        
        # 4.2 驗證 EnhancedDecisionAgent 整合 RAG
        try:
            from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import (
                EnhancedDecisionAgent
            )
            from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            
            # 創建測試實例
            vector_store = VectorStore(backend="memory")
            knowledge_base = KnowledgeBase(vector_store)
            agent = EnhancedDecisionAgent(knowledge_base=knowledge_base)
            
            has_rag = hasattr(agent, 'rag_engine') and agent.rag_engine is not None
            
            self.log_test(
                "EnhancedDecisionAgent 整合 RAG",
                has_rag,
                "rag_engine 已初始化" if has_rag else "rag_engine 未初始化"
            )
        except Exception as e:
            self.log_test("EnhancedDecisionAgent 整合 RAG", False, str(e))
    
    async def verify_feature_encoding(self):
        """驗證 5: 特徵編碼和檢索功能"""
        print("\n" + "="*60)
        print("驗證 5: 特徵編碼和檢索功能")
        print("="*60)
        
        # 5.1 測試特徵編碼
        try:
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            import numpy as np
            
            vector_store = VectorStore(backend="memory")
            
            # 測試權重表編碼
            rag_trigger = {
                'http_403': 1.5,
                'db_error_mysql': 2.5,
                'waf_cloudflare': 1.2
            }
            
            feature_vector = vector_store._encode_rag_trigger(rag_trigger, target_dim=512)
            
            # 驗證向量特性
            is_512_dim = len(feature_vector) == 512
            is_normalized = abs(np.linalg.norm(feature_vector) - 1.0) < 0.01
            is_float32 = feature_vector.dtype == np.float32
            
            all_valid = is_512_dim and is_normalized and is_float32
            
            self.log_test(
                "特徵編碼（權重表 → 512維向量）",
                all_valid,
                f"維度: {len(feature_vector)}, 歸一化: {is_normalized}, 類型: {feature_vector.dtype}"
            )
        except Exception as e:
            self.log_test("特徵編碼（權重表 → 512維向量）", False, str(e))
        
        # 5.2 測試環境檢索
        try:
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            
            vector_store = VectorStore(backend="memory")
            
            # 添加測試能力
            test_capability = {
                'id': 'security.sqli.test',
                'name': 'Test SQLi',
                'module': 'function_sqli',
                'capability_type': 'scanner',
                'rag_trigger': {
                    'http_403': 1.5,
                    'db_error_mysql': 2.5
                },
                'feature_signature': ['http_status', 'database_error']
            }
            
            vector_store.add_capability_from_registry(test_capability)
            
            # 測試檢索
            environment_features = {
                'http_403': 2.0,
                'db_error_mysql': 3.0
            }
            
            results = vector_store.search_by_environment(environment_features, top_k=1)
            
            has_results = len(results) > 0
            if has_results:
                top_result = results[0]
                has_score = 'match_score' in top_result
                has_metadata = 'metadata' in top_result
                
                self.log_test(
                    "環境特徵檢索",
                    has_score and has_metadata,
                    f"找到 {len(results)} 個結果, 最高分數: {top_result.get('match_score', 0):.2f}"
                )
            else:
                self.log_test("環境特徵檢索", False, "未找到匹配結果")
                
        except Exception as e:
            self.log_test("環境特徵檢索", False, str(e))
    
    async def run_all_verifications(self):
        """執行所有驗證"""
        print("\n" + "="*60)
        print("HackOne v2.0 去語意化反射引擎整合驗證")
        print("="*60)
        
        await self.verify_imports()
        await self.verify_feature_module_integration()
        await self.verify_data_loading()
        await self.verify_rag_initialization()
        await self.verify_feature_encoding()
        
        # 輸出總結
        print("\n" + "="*60)
        print("驗證總結")
        print("="*60)
        print(f"✅ 通過: {self.passed}")
        print(f"❌ 失敗: {self.failed}")
        print(f"總計: {self.passed + self.failed}")
        
        success_rate = (self.passed / (self.passed + self.failed) * 100) if (self.passed + self.failed) > 0 else 0
        print(f"成功率: {success_rate:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 所有驗證通過！整合成功！")
            return 0
        else:
            print(f"\n⚠️  有 {self.failed} 個驗證失敗，請檢查上述錯誤")
            return 1


async def main():
    """主函數"""
    verifier = IntegrationVerifier()
    exit_code = await verifier.run_all_verifications()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
