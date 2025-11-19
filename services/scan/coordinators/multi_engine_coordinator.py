"""
多引擎協調器 - 實現 Python、TypeScript、Rust 三引擎協同掃描

設計目標 (參考 Nmap 和 OWASP 最佳實踐):
1. 階段式掃描: Rust 快速發現 -> AI 決策 -> 三引擎組合 -> 深度分析
2. 模式化設計: Fast Discovery / Deep Analysis / Focused Verification
3. 符合五大模組: Scan 執行掃描, Core 下令, Integration 協調結果
4. 三引擎協同: 充分發揮各引擎優勢

掃描流程 (4 階段):
Phase 0: Rust 快速發現 (Fast Discovery)
  - 大範圍多目標掃描
  - 識別技術棧 (PHP/Java/Node.js/.NET)
  - 標記敏感特徵 (API端點/管理介面/配置檔)
  - 輸出: 目標基礎資訊

Phase 1: AI 決策編排 (Core 模組)
  - 分析 Rust 發現的資訊
  - 生成三引擎組合策略:
    * Python: 靜態內容抓取
    * TypeScript: 動態渲染 (SPA/React/Vue)
    * Rust: 敏感資訊深度掃描

Phase 2: 三引擎並行執行 (Scan 模組)
  - Python 爬蟲引擎
  - TypeScript Playwright 動態渲染
  - Rust 深度分析 + 密鑰驗證

Phase 3: 結果聚合與分析 (Integration 模組)
  - 整合三引擎掃描結果
  - 去重和關聯分析
  - 為功能模組提供測試目標

Phase 4: 實際攻擊測試 (Feature 模組)
  - XSS/SQLi/SSRF 等實際測試
  - 基於掃描結果動態調整策略
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from services.aiva_common.schemas import (
    Asset,
    ScanStartPayload,
    ScanCompletedPayload,
    Summary
)
from services.aiva_common.enums.assets import AssetType
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class EngineType(str, Enum):
    """引擎類型"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"


class ScanPhase(str, Enum):
    """掃描階段 - 基於 OWASP 和 Nmap 最佳實踐"""
    RUST_FAST_DISCOVERY = "rust_fast_discovery"      # Phase 0: Rust 快速發現
    AI_DECISION = "ai_decision"                      # Phase 1: AI 決策編排 (Core 模組)
    MULTI_ENGINE_SCAN = "multi_engine_scan"          # Phase 2: 三引擎並行執行
    RESULT_AGGREGATION = "result_aggregation"        # Phase 3: 結果聚合 (Integration 模組)
    # 新增標準掃描階段
    DISCOVERY = "discovery"                          # 標準發現階段
    DEEP_SCAN = "deep_scan"                          # 深度掃描階段
    SENSITIVE = "sensitive"                          # 敏感資訊掃描階段
    # Phase 4 (Feature 模組實際攻擊測試) 不在 Scan 模組範圍內


@dataclass
class EngineResult:
    """單個引擎的掃描結果"""
    engine: EngineType
    phase: ScanPhase
    assets: List[Asset] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class CoordinatedScanResult:
    """協同掃描結果"""
    scan_id: str
    total_assets: int
    assets_by_type: Dict[str, int]
    engine_results: List[EngineResult]
    coordination_strategy: str
    total_time: float
    quality_metrics: Dict[str, Any]
    
    def get_asset_coverage(self) -> float:
        """計算資產覆蓋率 (vs Burp Pro baseline)"""
        # Burp Pro baseline: ~100 assets per 1000 pages
        baseline = 100
        coverage = min(self.total_assets / baseline, 1.5)  # 最高150%
        return coverage
    
    def get_depth_score(self) -> float:
        """計算深度分數 (基於敏感資訊發現)"""
        sensitive_count = self.assets_by_type.get("sensitive_info", 0)
        # 每發現10個敏感資訊得1分，最高10分
        return min(sensitive_count / 10, 10.0)


class MultiEngineCoordinator:
    """
    多引擎協調器
    
    核心策略:
    1. 階段式掃描: Discovery -> Deep Scan -> Sensitive -> Analysis
    2. 並行執行: Python + TypeScript 並行，Rust 後置
    3. 結果融合: 去重、分類、優先級排序
    4. 智能調度: 根據目標特性選擇引擎組合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # 引擎可用性
        self.available_engines: Set[EngineType] = set()
        self._check_engine_availability()
        
        # 協調策略
        self.coordination_strategy = self._determine_strategy()
        
    def _check_engine_availability(self):
        """檢查引擎可用性"""
        try:
            # 檢查 Python 引擎
            from ..engines.python_engine.scan_orchestrator import ScanOrchestrator
            self.available_engines.add(EngineType.PYTHON)
            self.logger.info("✓ Python 引擎可用")
        except ImportError as e:
            self.logger.warning(f"✗ Python 引擎不可用: {e}")
        
        try:
            # 檢查 TypeScript 引擎 (通過檢查 Node.js 服務)
            # TODO: 實際檢查 aiva_scan_node 服務狀態
            self.available_engines.add(EngineType.TYPESCRIPT)
            self.logger.info("✓ TypeScript 引擎可用")
        except Exception as e:
            self.logger.warning(f"✗ TypeScript 引擎不可用: {e}")
        
        try:
            # 檢查 Rust 引擎
            from ..engines.rust_engine.python_bridge import RustScanner
            self.available_engines.add(EngineType.RUST)
            self.logger.info("✓ Rust 引擎可用")
        except ImportError as e:
            self.logger.warning(f"✗ Rust 引擎不可用: {e}")
    
    def _determine_strategy(self) -> str:
        """確定協調策略"""
        available_count = len(self.available_engines)
        
        if available_count == 3:
            return "full_coordination"  # 三引擎全協同
        elif available_count == 2:
            return "partial_coordination"  # 兩引擎協同
        elif available_count == 1:
            return "single_engine"  # 單引擎降級
        else:
            return "no_engine"  # 無可用引擎
    
    async def execute_coordinated_scan(
        self,
        request: ScanStartPayload
    ) -> CoordinatedScanResult:
        """
        執行協同掃描 - 新的 4 階段流程
        
        Phase 0: Rust 快速發現 (Fast Discovery)
          - 大範圍快速掃描多目標
          - 識別技術棧和敏感特徵
          
        Phase 1: AI 決策編排 (Core 模組負責)
          - 注意: 這裡只是記錄,實際 AI 決策在 Core 模組
          - Scan 模組根據 AI 下令的策略執行
          
        Phase 2: 三引擎並行執行
          - Python: 靜態內容爬取
          - TypeScript: 動態渲染分析
          - Rust: 深度分析 + 密鑰驗證
          
        Phase 3: 結果聚合 (本模組完成,交給 Integration)
          - 整合三引擎結果
          - 去重和關聯分析
        """
        start_time = time.time()
        scan_id = request.scan_id
        
        self.logger.info(f"🎯 開始協同掃描: {scan_id}")
        self.logger.info(f"📊 協調策略: {self.coordination_strategy}")
        self.logger.info(f"🔧 可用引擎: {', '.join(e.value for e in self.available_engines)}")
        
        # 存儲各階段結果
        phase_results: List[EngineResult] = []
        all_assets: List[Asset] = []
        
        # ===== Phase 0: Rust 快速發現 =====
        self.logger.info("🚀 Phase 0: Rust 快速發現...")
        if EngineType.RUST in self.available_engines:
            rust_discovery = await self._phase_0_rust_fast_discovery(request)
            if rust_discovery:
                phase_results.append(rust_discovery)
                if rust_discovery.success:
                    all_assets.extend(rust_discovery.assets)
                self.logger.info(
                    f"✅ Phase 0 完成: 發現 {len(rust_discovery.assets)} 個基礎資訊"
                )
        
        # ===== Phase 1: AI 決策編排 =====
        # 注意: 實際 AI 決策在 Core 模組,這裡只是記錄
        self.logger.info("🧠 Phase 1: AI 決策編排 (由 Core 模組負責)")
        # Core 模組會分析 Phase 0 結果並生成掃描策略
        # 這裡 Scan 模組只負責執行 AI 下令的策略
        
        # ===== Phase 2: 三引擎並行執行 =====
        self.logger.info("⚙️ Phase 2: 三引擎並行執行...")
        if self.coordination_strategy in ["full_coordination", "partial_coordination"]:
            multi_engine_results = await self._phase_2_multi_engine_scan(request, all_assets)
            phase_results.extend(multi_engine_results)
            for result in multi_engine_results:
                if result.success:
                    all_assets.extend(result.assets)
            
            self.logger.info(
                f"✅ Phase 2 完成: 新增 {sum(len(r.assets) for r in multi_engine_results)} 個資產"
            )
        
        # ===== Phase 3: 結果聚合 =====
        self.logger.info("📊 Phase 3: 結果聚合與分析...")
        final_assets, quality_metrics = await self._phase_3_result_aggregation(all_assets)
        
        total_time = time.time() - start_time
        
        # 統計資產類型
        assets_by_type = self._count_assets_by_type(final_assets)
        
        result = CoordinatedScanResult(
            scan_id=scan_id,
            total_assets=len(final_assets),
            assets_by_type=assets_by_type,
            engine_results=phase_results,
            coordination_strategy=self.coordination_strategy,
            total_time=total_time,
            quality_metrics=quality_metrics
        )
        
        self._log_summary(result)
        
        return result
    
    async def _phase_0_rust_fast_discovery(
        self,
        request: ScanStartPayload
    ) -> Optional[EngineResult]:
        """
        Phase 0: Rust 快速發現
        
        使用 Rust Mode 1 (Fast Discovery):
        - 大範圍快速掃描
        - 識別技術棧 (PHP/Java/Node.js/.NET)
        - 標記 API 端點/管理介面/配置檔
        - 輸出基礎資訊給 AI 決策
        """
        if EngineType.RUST not in self.available_engines:
            return None
        
        start_time = time.time()
        
        try:
            self.logger.info("  🦀 Rust 引擎: Mode 1 (Fast Discovery)")
            
            # TODO: 實現 Rust 快速發現調用
            # 設置環境變數: RUST_SCAN_MODE=fast_discovery
            # 調用 info_gatherer_rust 服務
            await asyncio.sleep(0)  # 保持異步語義
            
            result = EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.RUST_FAST_DISCOVERY,
                assets=[],  # TODO: Rust 快速發現結果
                metadata={
                    "mode": "fast_discovery",
                    "target_count": 1,
                },
                execution_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rust 快速發現失敗: {e}")
            return EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.RUST_FAST_DISCOVERY,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _phase_2_multi_engine_scan(
        self,
        request: ScanStartPayload,
        discovered_assets: List[Asset]
    ) -> List[EngineResult]:
        """
        Phase 2: 三引擎並行執行
        
        根據 Phase 0 和 AI 決策結果:
        - Python: 靜態內容爬取
        - TypeScript: Playwright 動態渲染
        - Rust: Mode 2 深度分析 + 密鑰驗證
        """
        results: List[EngineResult] = []
        tasks = []
        
        # Python 引擎任務
        if EngineType.PYTHON in self.available_engines:
            tasks.append(self._run_python_engine(request))
        
        # TypeScript 引擎任務
        if EngineType.TYPESCRIPT in self.available_engines:
            tasks.append(self._run_typescript_engine(request))
        
        # Rust 引擎任務 (Mode 2: Deep Analysis)
        if EngineType.RUST in self.available_engines:
            tasks.append(self._run_rust_deep_analysis(discovered_assets))
        
        # 並行執行
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results = [r for r in results if isinstance(r, EngineResult)]
        
        return results
    
    async def _phase_3_result_aggregation(
        self,
        all_assets: List[Asset]
    ) -> tuple[List[Asset], Dict[str, Any]]:
        """
        Phase 3: 結果聚合與分析
        
        這是 Scan 模組的最後階段:
        - 去重和關聯分析
        - 為 Integration 模組準備結果
        - Integration 模組會進一步協調給 Feature 模組
        """
        # 去重
        await asyncio.sleep(0)  # 保持異步語義
        unique_assets = self._deduplicate_assets(all_assets)
        
        # 計算質量指標
        quality_metrics = {
            "total_discovered": len(all_assets),
            "after_dedup": len(unique_assets),
            "dedup_rate": 1 - (len(unique_assets) / len(all_assets)) if all_assets else 0,
        }
        
        return unique_assets, quality_metrics
    
    async def _run_python_engine(self, request: ScanStartPayload) -> EngineResult:
        """運行 Python 爬蟲引擎"""
        start_time = time.time()
        try:
            self.logger.info("  🐍 Python 引擎: 開始掃描")
            from ..engines.python_engine.scan_orchestrator import ScanOrchestrator
            
            # 實際調用 Python Engine
            orchestrator = ScanOrchestrator()
            scan_result = await orchestrator.execute_scan(request)
            
            execution_time = time.time() - start_time
            self.logger.info(f"  🐍 Python 引擎完成: {len(scan_result.assets)} 個資產, {execution_time:.1f}s")
            
            return EngineResult(
                engine=EngineType.PYTHON,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                assets=scan_result.assets,
                metadata={
                    "urls_found": scan_result.summary.urls_found,
                    "forms_found": scan_result.summary.forms_found,
                    "scan_duration": scan_result.summary.scan_duration_seconds
                },
                execution_time=execution_time
            )
        except Exception as e:
            self.logger.error(f"  ❌ Python 引擎錯誤: {e}")
            return EngineResult(
                engine=EngineType.PYTHON,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _run_typescript_engine(self, request: ScanStartPayload) -> EngineResult:
        """運行 TypeScript Playwright 引擎
        
        當前狀態: TypeScript Worker 尚未實現
        未來計劃: 創建獨立的 TypeScript Worker 訂閱 RabbitMQ
        """
        start_time = time.time()
        try:
            self.logger.info("  📜 TypeScript 引擎: 當前未實現，跳過")
            
            # TypeScript Worker 尚未實現，優雅降級
            await asyncio.sleep(0)  # 保持異步語義
            
            return EngineResult(
                engine=EngineType.TYPESCRIPT,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                assets=[],
                metadata={"status": "not_implemented", "note": "TypeScript Worker pending"},
                execution_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"  ❌ TypeScript 引擎錯誤: {e}")
            return EngineResult(
                engine=EngineType.TYPESCRIPT,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _run_rust_deep_analysis(self, _assets: List[Asset]) -> EngineResult:
        """運行 Rust Mode 2 深度分析
        
        當前狀態: Rust Python Bridge 可能未完全集成
        未來計劃: 通過 Python Bridge 或 RabbitMQ Worker 調用
        """
        start_time = time.time()
        try:
            self.logger.info("  🦀 Rust 引擎: 當前未實現，跳過")
            
            # Rust Bridge 尚未完全集成，優雅降級
            await asyncio.sleep(0)  # 保持異步語義
            
            return EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                assets=[],
                metadata={"mode": "deep_analysis", "status": "not_implemented"},
                execution_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"  ❌ Rust 深度分析錯誤: {e}")
            return EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.MULTI_ENGINE_SCAN,
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _phase_1_discovery(
        self,
        request: ScanStartPayload
    ) -> List[EngineResult]:
        """
        Phase 1: 資產發現階段
        
        策略: Python + TypeScript 並行執行
        - Python: 基礎 HTTP 爬蟲 (快速廣度優先)
        - TypeScript: 動態頁面發現 (JavaScript 渲染)
        """
        results: List[EngineResult] = []
        tasks = []
        
        # Python 引擎任務
        if EngineType.PYTHON in self.available_engines:
            tasks.append(self._run_python_engine(request))
        
        # TypeScript 引擎任務
        if EngineType.TYPESCRIPT in self.available_engines:
            tasks.append(self._run_typescript_engine(request))
        
        # 並行執行
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed_results:
                if isinstance(result, EngineResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Discovery 階段錯誤: {result}")
        
        return results
    
    async def _phase_2_deep_scan(
        self,
        request: ScanStartPayload,
        discovered_assets: List[Asset]
    ) -> Optional[EngineResult]:
        """
        Phase 2: 深度掃描階段
        
        重點: TypeScript 引擎深度分析
        - SPA 路由發現
        - AJAX 端點捕獲
        - WebSocket 檢測
        - 動態內容提取
        """
        start_time = time.time()
        
        try:
            self.logger.info("📘 TypeScript 引擎: 開始深度掃描...")
            
            # TODO: 實現 TypeScript 深度掃描
            # 1. 分析已發現的資產
            # 2. 識別需要深度分析的目標 (SPA、AJAX heavy)
            # 3. 執行 Playwright 深度渲染
            
            await asyncio.sleep(2)  # 模擬掃描時間
            
            execution_time = time.time() - start_time
            
            return EngineResult(
                engine=EngineType.TYPESCRIPT,
                phase=ScanPhase.DEEP_SCAN,
                assets=[],  # TODO: 深度掃描發現的新資產
                metadata={
                    "deep_scan_targets": len(discovered_assets),
                    "new_endpoints": 0
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"深度掃描錯誤: {e}")
            return None
    
    async def _phase_3_sensitive_scan(
        self,
        all_assets: List[Asset]
    ) -> Optional[EngineResult]:
        """
        Phase 3: 敏感資訊掃描
        
        重點: Rust 引擎高速並行掃描
        - API Keys (AWS, GitHub, etc.)
        - Secrets (JWT, Tokens)
        - Credentials (Passwords, DB Strings)
        - Personal Info (Email, IP)
        """
        start_time = time.time()
        
        try:
            self.logger.info("🦀 Rust 引擎: 開始敏感資訊掃描...")
            
            # TODO: 實現 Rust 引擎調用
            # 1. 提取所有文本內容
            # 2. 調用 Rust 掃描器並行掃描
            # 3. 收集敏感資訊發現
            
            await asyncio.sleep(0.5)  # Rust 很快
            
            execution_time = time.time() - start_time
            
            return EngineResult(
                engine=EngineType.RUST,
                phase=ScanPhase.SENSITIVE,
                assets=[],  # TODO: 敏感資訊資產
                metadata={
                    "scanned_assets": len(all_assets),
                    "patterns_matched": 0
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"敏感資訊掃描錯誤: {e}")
            return None
    
    async def _phase_4_analysis(
        self,
        all_assets: List[Asset]
    ) -> tuple[List[Asset], Dict[str, Any]]:
        """
        Phase 4: 結果分析與整合
        
        處理:
        1. 去重 (URL 正規化)
        2. 分類 (按資產類型)
        3. 優先級排序 (高價值資產優先)
        4. 質量評估
        """
        self.logger.info("📊 開始結果分析與整合...")
        await asyncio.sleep(0)  # 保持異步語義
        
        # 去重
        unique_assets = self._deduplicate_assets(all_assets)
        
        # 質量評估
        quality_metrics = {
            "total_raw_assets": len(all_assets),
            "unique_assets": len(unique_assets),
            "deduplication_rate": 1 - (len(unique_assets) / len(all_assets)) if all_assets else 0,
            "coverage_score": self._calculate_coverage_score(unique_assets),
            "depth_score": self._calculate_depth_score(unique_assets)
        }
        
        self.logger.info(f"✅ 分析完成: {len(unique_assets)} 個唯一資產")
        
        return unique_assets, quality_metrics
    
    def _deduplicate_assets(self, assets: List[Asset]) -> List[Asset]:
        """資產去重"""
        seen = set()
        unique = []
        
        for asset in assets:
            # 使用 asset.value 作為去重鍵
            key = (asset.type, asset.value)
            if key not in seen:
                seen.add(key)
                unique.append(asset)
        
        return unique
    
    def _calculate_coverage_score(self, assets: List[Asset]) -> float:
        """計算覆蓋率分數 (0-100)"""
        # 簡單實現: 基於資產數量
        # Burp Pro baseline: 100 資產/1000頁面
        score = min(len(assets) / 100 * 100, 100)
        return round(score, 2)
    
    def _calculate_depth_score(self, assets: List[Asset]) -> float:
        """計算深度分數 (0-100)"""
        # 簡單實現: 基於資產類型多樣性
        asset_types = {a.type for a in assets}
        score = min(len(asset_types) / 5 * 100, 100)  # 假設5種類型為滿分
        return round(score, 2)
    
    def _count_assets_by_type(self, assets: List[Asset]) -> Dict[str, int]:
        """統計各類型資產數量"""
        counts: Dict[str, int] = {}
        for asset in assets:
            asset_type = str(asset.type)
            counts[asset_type] = counts.get(asset_type, 0) + 1
        return counts
    
    def _log_summary(self, result: CoordinatedScanResult):
        """記錄摘要日誌"""
        self.logger.info("=" * 60)
        self.logger.info("📊 協同掃描完成摘要")
        self.logger.info("=" * 60)
        self.logger.info(f"掃描 ID: {result.scan_id}")
        self.logger.info(f"協調策略: {result.coordination_strategy}")
        self.logger.info(f"總資產數: {result.total_assets}")
        self.logger.info(f"總耗時: {result.total_time:.2f}s")
        
        self.logger.info("\n資產類型分布:")
        for asset_type, count in result.assets_by_type.items():
            self.logger.info(f"  - {asset_type}: {count}")
        
        self.logger.info("\n引擎執行情況:")
        for engine_result in result.engine_results:
            status = "✓" if engine_result.success else "✗"
            self.logger.info(
                f"  {status} {engine_result.engine.value} "
                f"({engine_result.phase.value}): "
                f"{len(engine_result.assets)} 資產, "
                f"{engine_result.execution_time:.2f}s"
            )
        
        self.logger.info("\n質量指標:")
        for metric, value in result.quality_metrics.items():
            self.logger.info(f"  - {metric}: {value}")
        
        self.logger.info("=" * 60)


# 便利函數
async def coordinate_scan(request: ScanStartPayload) -> CoordinatedScanResult:
    """執行協同掃描的便利函數"""
    coordinator = MultiEngineCoordinator()
    return await coordinator.execute_coordinated_scan(request)
