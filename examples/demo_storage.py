#!/usr/bin/env python3
"""
演示 AIVA 存儲系統的使用

展示如何保存和查詢訓練數據
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    ExperienceSample,
    TraceRecord,
)
from pydantic import BaseModel, Field


# 定義缺失的模型類
class AttackResult(BaseModel):
    """攻擊結果"""
    result_id: str
    timestamp: datetime
    success: bool
    vulnerability_confirmed: bool
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    cvss_score: float | None = None
    risk_level: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    """執行追蹤步驟"""
    step_id: str
    timestamp: datetime
    action: str
    payload: str | None = None
    response_status: int | None = None
    response_body: str | None = None
    success: bool
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# 嘗試導入 StorageManager，如果失敗則創建 mock
try:
    from services.core.aiva_core.storage import StorageManager
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("StorageManager not available, creating mock implementation")
    
    class StorageManager:
        """StorageManager 的 mock 實現"""
        def __init__(self, data_root, db_type="hybrid", auto_create_dirs=True):
            self.data_root = Path(data_root)
            self.db_type = db_type
            logger.info(f"Mock StorageManager initialized: {data_root}")
            
        async def save_experience_sample(self, sample):
            logger.info(f"Mock: Would save experience sample {sample.sample_id}")
            return True
            
        async def get_experience_samples(self, limit=100, min_quality=None, vulnerability_type=None):
            logger.info(f"Mock: Would query experience samples (limit={limit})")
            return []
            
        async def save_trace(self, trace):
            logger.info(f"Mock: Would save trace {trace.trace_id}")
            return True
            
        async def save_training_session(self, session_data):
            logger.info(f"Mock: Would save training session {session_data.get('session_id')}")
            return True
            
        async def get_statistics(self):
            return {
                'backend': 'mock',
                'data_root': str(self.data_root),
                'total_experiences': 0,
                'high_quality_experiences': 0,
                'total_traces': 0,
                'total_sessions': 0,
                'total_checkpoints': 0,
                'total_knowledge': 0,
                'database_size': 0,
                'training_size': 0,
                'models_size': 0,
                'knowledge_size': 0,
                'total_size': 0,
            }

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def demo_save_experience():
    """演示保存經驗樣本"""
    logger.info("=== 演示保存經驗樣本 ===")

    # 創建存儲管理器
    storage = StorageManager(
        data_root="/workspaces/AIVA/data", db_type="hybrid", auto_create_dirs=True
    )

    # 創建測試計畫
    plan = AttackPlan(
        plan_id="test-plan-001",
        timestamp=datetime.utcnow(),
        vulnerability_type="sqli",
        target_url="http://testphp.vulnweb.com/artists.php?artist=1",
        steps=[
            {
                "step_id": "step-001",
                "step_type": "scan",
                "description": "掃描 SQL 注入點",
                "payload": "1' OR '1'='1",
                "expected_result": "檢測到注入點",
            }
        ],
        priority=1.0,
        confidence=0.9,
        metadata={"scenario_id": "SQLI-1"},
    )

    # 創建測試結果
    result = AttackResult(
        result_id="result-001",
        timestamp=datetime.utcnow(),
        success=True,
        vulnerability_confirmed=True,
        confidence=0.95,
        evidence=["SQL syntax error detected", "Boolean-based injection successful"],
        cvss_score=7.5,
        risk_level="high",
        metadata={"injection_type": "boolean"},
    )

    # 創建經驗樣本
    sample = ExperienceSample(
        sample_id="sample-001",
        timestamp=datetime.utcnow(),
        plan=plan,
        trace_id="trace-001",
        actual_result=result,
        expected_result=None,
        quality_score=0.85,
        metadata={"training_session": "demo"},
    )

    # 保存樣本
    success = await storage.save_experience_sample(sample)
    logger.info(f"保存經驗樣本: {'成功' if success else '失敗'}")

    # 再保存幾個不同質量的樣本
    for i in range(2, 6):
        sample = ExperienceSample(
            sample_id=f"sample-00{i}",
            timestamp=datetime.utcnow(),
            plan=plan,
            trace_id=f"trace-00{i}",
            actual_result=result,
            expected_result=None,
            quality_score=0.5 + (i * 0.1),  # 質量遞增
            metadata={"training_session": "demo", "batch": i},
        )
        await storage.save_experience_sample(sample)
        logger.info(f"保存樣本 {i}: 質量分數 {sample.quality_score}")


async def demo_query_experiences():
    """演示查詢經驗樣本"""
    logger.info("\n=== 演示查詢經驗樣本 ===")

    storage = StorageManager(
        data_root="/workspaces/AIVA/data", db_type="hybrid", auto_create_dirs=False
    )

    # 查詢所有樣本
    all_samples = await storage.get_experience_samples(limit=100)
    logger.info(f"總共有 {len(all_samples)} 個經驗樣本")

    # 查詢高質量樣本
    high_quality = await storage.get_experience_samples(min_quality=0.7)
    logger.info(f"高質量樣本 (>= 0.7): {len(high_quality)} 個")

    # 查詢特定類型
    sqli_samples = await storage.get_experience_samples(vulnerability_type="sqli")
    logger.info(f"SQL 注入樣本: {len(sqli_samples)} 個")

    # 顯示詳細信息
    if high_quality:
        sample = high_quality[0]
        logger.info("\n最佳樣本:")
        logger.info(f"  ID: {sample.sample_id}")
        logger.info(f"  質量分數: {sample.quality_score}")
        logger.info(f"  類型: {sample.plan.vulnerability_type}")
        logger.info(f"  成功: {sample.actual_result.success}")
        logger.info(f"  CVSS: {sample.actual_result.cvss_score}")


async def demo_save_trace():
    """演示保存追蹤記錄"""
    logger.info("\n=== 演示保存追蹤記錄 ===")

    storage = StorageManager(
        data_root="/workspaces/AIVA/data", db_type="hybrid", auto_create_dirs=False
    )

    # 創建測試計畫
    plan = AttackPlan(
        plan_id="test-plan-trace",
        timestamp=datetime.utcnow(),
        vulnerability_type="xss",
        target_url="http://testhtml5.vulnweb.com/",
        steps=[
            {
                "step_id": "step-001",
                "step_type": "scan",
                "description": "掃描 XSS",
                "payload": "<script>alert('XSS')</script>",
            }
        ],
        priority=1.0,
        confidence=0.9,
    )

    # 創建追蹤步驟
    steps = [
        TraceStep(
            step_id="step-001",
            timestamp=datetime.utcnow(),
            action="scan",
            payload="<script>alert('XSS')</script>",
            response_status=200,
            response_body="<html>...<script>alert('XSS')</script>...</html>",
            success=True,
            duration_ms=150,
            metadata={"reflected": True},
        ),
        TraceStep(
            step_id="step-002",
            timestamp=datetime.utcnow(),
            action="verify",
            payload="<img src=x onerror=alert('XSS')>",
            response_status=200,
            success=True,
            duration_ms=120,
            metadata={"confirmed": True},
        ),
    ]

    # 創建追蹤記錄
    trace = TraceRecord(
        trace_id="trace-xss-001",
        session_id="session-demo",
        timestamp=datetime.utcnow(),
        plan=plan,
        steps=steps,
        total_steps=2,
        successful_steps=2,
        failed_steps=0,
        duration_seconds=0.27,
        final_result=AttackResult(
            result_id="result-xss-001",
            timestamp=datetime.utcnow(),
            success=True,
            vulnerability_confirmed=True,
            confidence=0.99,
            evidence=["Reflected XSS confirmed"],
            cvss_score=6.1,
            risk_level="medium",
        ),
        metadata={"xss_type": "reflected"},
    )

    # 保存追蹤
    success = await storage.save_trace(trace)
    logger.info(f"保存追蹤記錄: {'成功' if success else '失敗'}")


async def demo_training_session():
    """演示保存訓練會話"""
    logger.info("\n=== 演示保存訓練會話 ===")

    storage = StorageManager(
        data_root="/workspaces/AIVA/data", db_type="hybrid", auto_create_dirs=False
    )

    # 訓練會話數據
    session_data = {
        "session_id": "training-demo-001",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "session_type": "batch",
        "scenario_id": "SQLI-1",
        "config": {
            "episodes": 100,
            "learning_rate": 0.001,
            "gamma": 0.99,
        },
        "total_episodes": 50,  # 進行中
        "successful_episodes": 42,
        "total_samples": 150,
        "high_quality_samples": 89,
        "avg_reward": 12.5,
        "avg_quality": 0.72,
        "best_reward": 18.9,
        "status": "running",
        "metadata": {"trainer": "demo", "notes": "測試訓練"},
    }

    # 保存會話
    success = await storage.save_training_session(session_data)
    logger.info(f"保存訓練會話: {'成功' if success else '失敗'}")


async def demo_statistics():
    """演示獲取統計信息"""
    logger.info("\n=== 演示統計信息 ===")

    storage = StorageManager(
        data_root="/workspaces/AIVA/data", db_type="hybrid", auto_create_dirs=False
    )

    stats = await storage.get_statistics()

    print("\n📊 數據統計:")
    print(f"  後端類型: {stats['backend']}")
    print(f"  數據根目錄: {stats['data_root']}")
    print("\n📝 數據庫統計:")
    print(f"  總經驗樣本: {stats.get('total_experiences', 0)}")
    print(f"  高質量樣本: {stats.get('high_quality_experiences', 0)}")
    print(f"  追蹤記錄: {stats.get('total_traces', 0)}")
    print(f"  訓練會話: {stats.get('total_sessions', 0)}")
    print(f"  模型檢查點: {stats.get('total_checkpoints', 0)}")
    print(f"  知識條目: {stats.get('total_knowledge', 0)}")

    # 按類型統計
    if "experiences_by_type" in stats:
        print("\n📋 樣本類型分布:")
        for vtype, count in stats["experiences_by_type"].items():
            print(f"  {vtype}: {count}")

    # 存儲大小
    print("\n💾 存儲大小:")
    for key in [
        "database_size",
        "training_size",
        "models_size",
        "knowledge_size",
        "total_size",
    ]:
        if key in stats:
            size_mb = stats[key] / (1024 * 1024) if stats[key] else 0
            print(f"  {key}: {size_mb:.2f} MB")


async def main():
    """主函數"""
    logger.info("🚀 AIVA 存儲系統演示\n")

    # 1. 保存經驗樣本
    await demo_save_experience()

    # 2. 查詢經驗樣本
    await demo_query_experiences()

    # 3. 保存追蹤記錄
    await demo_save_trace()

    # 4. 保存訓練會話
    await demo_training_session()

    # 5. 查看統計
    await demo_statistics()

    logger.info("\n✅ 演示完成!")
    logger.info("\n💡 提示: 數據已保存到 /workspaces/AIVA/data/")
    logger.info("   - 數據庫: data/database/aiva.db")
    logger.info("   - JSONL: data/training/experiences/")


if __name__ == "__main__":
    asyncio.run(main())
