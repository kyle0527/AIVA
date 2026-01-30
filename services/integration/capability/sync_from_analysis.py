"""從 internal_exploration 同步能力到 CapabilityRegistry

此腳本負責：
1. 讀取 services/integration/data/internal_exploration/latest_classification.json（唯一數據源）
2. 將流程轉換為 CapabilityRecord
3. 註冊到 integration.CapabilityRegistry（單一數據源）
4. 自動同步到 RAG 向量數據庫

⚠️ 架構更新 (2026-01-03):
- 簡化路徑：移除 analysis_data 中間層
- 直接讀取 internal_exploration Pipeline 輸出
- 五大模組架構（external_learning → cognitive_core.learning_system）

遵循 aiva_common 規範：
- 單一數據來源 (SOT) 原則
- 使用 aiva_common 的枚舉和 Schema
- 完整的類型註解
- 結構化日誌記錄
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from aiva_common.utils.logging import get_logger
from aiva_common.enums.modules import ProgrammingLanguage
from services.integration.capability.registry import registry as global_registry
from services.integration.capability.models import (
    CapabilityRecord,
    CapabilityType,
    CapabilityStatus
)

logger = get_logger(__name__)

# 路徑常量 (2026-01-03 簡化：唯一數據源)
INTERNAL_EXPLORATION_DATA = Path("services/integration/data/internal_exploration")
CORE_CAPABILITIES_FILE = INTERNAL_EXPLORATION_DATA / "latest_classification.json"

# 向後相容：舊路徑別名
ANALYSIS_DATA_ROOT = INTERNAL_EXPLORATION_DATA  # deprecated


class CapabilitySyncer:
    """能力同步器 - 從 analysis_data 同步到 Registry"""
    
    def __init__(self):
        self.registry = global_registry
        self.synced_count = 0
        self.failed_count = 0
        self.errors: List[str] = []
    
    async def sync_from_analysis_data(self, module: str = "core") -> Dict[str, Any]:
        """從 internal_exploration 同步能力到 Registry
        
        架構更新 (2026-01-04):
        - 統一使用 latest_classification.json 作為唯一數據源
        - 支援按 module 過濾（從 primary_module 欄位）
        
        Args:
            module: 模組名稱過濾 (core, features, scan, integration, 或 all)
            
        Returns:
            {
                "synced_count": int,
                "failed_count": int,
                "errors": List[str]
            }
        """
        logger.info(f"🔄 開始同步能力... (過濾: {module})")
        
        # 讀取唯一數據源: latest_classification.json
        capabilities_file = CORE_CAPABILITIES_FILE
        
        if not capabilities_file.exists():
            error_msg = f"找不到能力數據文件: {capabilities_file}"
            logger.error(error_msg)
            return {
                "synced_count": 0,
                "failed_count": 0,
                "errors": [error_msg]
            }
        
        logger.info(f"   讀取: {capabilities_file}")
        
        with open(capabilities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flows = data.get('flows', [])
        
        # 按模組過濾
        if module != "all":
            flows = [f for f in flows if f.get('primary_module', '').lower() == module.lower()]
        
        total_flows = len(flows)
        logger.info(f"   發現 {total_flows} 個流程 (過濾後)")
        
        # 轉換並註冊每個流程
        for i, flow in enumerate(flows, 1):
            try:
                await self._sync_single_flow(flow, module)
                self.synced_count += 1
                
                if i % 100 == 0:
                    logger.info(f"   進度: {i}/{total_flows} ({i*100//total_flows}%)")
                    
            except Exception as e:
                self.failed_count += 1
                error_msg = f"流程 {flow.get('id')} 同步失敗: {e}"
                self.errors.append(error_msg)
                logger.warning(f"   {error_msg}")
        
        logger.info(f"✅ 同步完成:")
        logger.info(f"   - 成功: {self.synced_count}")
        logger.info(f"   - 失敗: {self.failed_count}")
        
        if self.errors:
            logger.info(f"   - 前 5 個錯誤:")
            for error in self.errors[:5]:
                logger.info(f"     {error}")
        
        return {
            "synced_count": self.synced_count,
            "failed_count": self.failed_count,
            "errors": self.errors
        }
    
    async def _sync_single_flow(self, flow: Dict[str, Any], module: str):
        """同步單個流程到 Registry
        
        Args:
            flow: 流程數據字典
            module: 模組名稱
        """
        # 提取流程信息
        flow_id = flow.get('id')
        path = flow.get('path', [])
        full_path = flow.get('full_path', [])
        primary_module = flow.get('primary_module', 'unknown')
        classifications_raw = flow.get('classifications', [])
        
        # 提取分類描述（classifications 是字典列表）
        classifications = []
        for cls in classifications_raw:
            if isinstance(cls, dict):
                classifications.append(cls.get('description', cls.get('script', 'unknown')))
            elif isinstance(cls, str):
                classifications.append(cls)
        
        # 構建能力 ID（使用路徑作為唯一標識）
        capability_id = f"{module}.flow_{flow_id}"
        
        # 構建能力名稱
        capability_name = " → ".join(path) if len(path) <= 3 else f"{path[0]} → ... → {path[-1]}"
        
        # 構建描述
        description = f"數據流程 (長度: {flow.get('length', 0)} 步)"
        if classifications:
            description += f" - {classifications[0]}"
        
        # 確定入口點（使用第一個文件路徑）
        entrypoint = full_path[0] if full_path else "unknown"
        
        # 確定能力類型（基於分類）
        capability_type = self._determine_capability_type(classifications)
        
        # 創建 CapabilityRecord
        # CapabilityStatus 可用值: HEALTHY, DEGRADED, FAILED, UNKNOWN
        capability = CapabilityRecord(
            id=capability_id,
            name=capability_name,
            description=description,
            version="1.0.0",
            module=primary_module,
            language=ProgrammingLanguage.PYTHON,
            entrypoint=entrypoint,
            capability_type=capability_type,
            status=CapabilityStatus.HEALTHY,  # 初始狀態為 HEALTHY
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config={
                "flow_id": flow_id,
                "path": path,
                "full_path": full_path,
                "length": flow.get('length', 0),
                "classifications": classifications,
                "component_type": flow.get('component_type', 'unknown'),
                "category": flow.get('category', 'unknown'),
                "sub_category": flow.get('sub_category', 'unknown')
            },
            tags=classifications[:10] if classifications else [],
            category=flow.get('category', 'unknown'),
            priority=flow.get('priority', 50)
        )
        
        # 註冊到 Registry（會自動同步到 RAG）
        await self.registry.register_capability(capability)
    
    def _determine_capability_type(self, classifications: List[str]) -> CapabilityType:
        """根據分類確定能力類型
        
        CapabilityType 可用枚舉值:
        - SCANNER: 掃描、探測類
        - DETECTOR: 檢測、驗證類
        - ANALYZER: 分析、處理類
        - REPORTER: 報告、輸出類
        - UTILITY: 工具、通用類（默認）
        
        Args:
            classifications: 分類列表（已提取的字符串）
            
        Returns:
            CapabilityType 枚舉
        """
        if not classifications:
            return CapabilityType.UTILITY
            
        classifications_str = " ".join(classifications).lower()
        
        # 優先匹配更具體的類型
        if any(keyword in classifications_str for keyword in ['scan', 'explore', 'search', 'crawler', '掃描', '探索', '搜索', '爬蟲']):
            return CapabilityType.SCANNER
        elif any(keyword in classifications_str for keyword in ['detect', 'check', 'verify', 'validate', '檢測', '驗證', '檢查']):
            return CapabilityType.DETECTOR
        elif any(keyword in classifications_str for keyword in ['analyze', 'parse', 'process', 'classify', '分析', '處理', '解析', '分類']):
            return CapabilityType.ANALYZER
        elif any(keyword in classifications_str for keyword in ['report', 'output', 'format', 'log', '報告', '輸出', '紀錄']):
            return CapabilityType.REPORTER
        else:
            # 默認為 UTILITY（包括工具、配置、整合等）
            return CapabilityType.UTILITY


async def main():
    """主函數 - 執行同步"""
    logger.info("=" * 60)
    logger.info("📦 AIVA 能力同步工具")
    logger.info("   從 analysis_data 同步到 CapabilityRegistry")
    logger.info("=" * 60)
    
    syncer = CapabilitySyncer()
    
    # 同步 core 模組
    result = await syncer.sync_from_analysis_data(module="core")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 同步摘要")
    logger.info(f"   ✅ 成功同步: {result['synced_count']} 個能力")
    logger.info(f"   ❌ 失敗: {result['failed_count']} 個")
    logger.info("=" * 60)
    
    return result['synced_count'] > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

