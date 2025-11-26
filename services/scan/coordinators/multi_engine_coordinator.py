"""
多引擎協調器 - 實現 Python、TypeScript、Rust、Go 四引擎協同掃描

設計目標 (參考 Nmap 和 OWASP 最佳實踐):
1. 階段式掃描: Rust 快速發現 -> AI 決策 -> 多引擎組合 -> 深度分析
2. 模式化設計: Fast Discovery / Deep Analysis / Focused Verification
3. 符合五大模組: Scan 執行掃描, Core 下令, Integration 協調結果
4. 四引擎協同: 充分發揮各引擎優勢 (Python靜態/TypeScript動態/Rust敏感/Go並發)

掃描流程 (4 階段):
Phase 0: Rust 快速發現 (Fast Discovery)
  - 大範圍多目標掃描
  - 識別技術棧 (PHP/Java/Node.js/.NET)
  - 標記敏感特徵 (API端點/管理介面/配置檔)
  - 輸出: 目標基礎資訊

Phase 1: AI 決策編排 (Core 模組)
  - 分析 Rust 發現的資訊
  - 生成多引擎組合策略:
    * Python: 靜態內容抓取
    * TypeScript: 動態渲染 (SPA/React/Vue)
    * Rust: 敏感資訊深度掃描
    * Go: 高並發掃描和服務發現

Phase 2: 多引擎並行執行 (Scan 模組)
  - Python 爬蟲引擎
  - TypeScript Playwright 動態渲染
  - Rust 深度分析 + 密鑰驗證

Phase 3: 結果聚合與分析 (Integration 模組)
  - 整合多引擎掃描結果
  - 去重和關聯分析

執行模式:
1. Sequential (順序執行): 用於穩定驗證，每個階段完全完成後再進入下一階段
2. Coordinated (協同執行): 根據 AI 決策，在不同階段使用不同引擎組合

Phase 4: 實際攻擊測試 (Feature 模組)
  - XSS/SQLi/SSRF 等實際測試
  - 基於掃描結果動態調整策略
"""

import asyncio
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class EngineType(str, Enum):
    """掃描引擎類型枚舉 - 模組特定"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"


class ScanPhase(str, Enum):
    """掃描階段 - 基於 OWASP 和 Nmap 最佳實踐 - 模組特定"""
    # Phase 0: 快速發現
    RUST_FAST_DISCOVERY = "rust_fast_discovery"      # Phase 0: Rust 快速發現
    
    # Phase 1: AI 決策與多引擎掃描
    AI_DECISION = "ai_decision"                      # Phase 1: AI 決策編排
    MULTI_ENGINE_SCAN = "multi_engine_scan"          # Phase 1: 多引擎並行執行
    PYTHON_STATIC_ANALYSIS = "python_static_analysis"  # Phase 1a: Python 靜態分析
    TYPESCRIPT_DYNAMIC = "typescript_dynamic"       # Phase 1b: TypeScript 動態渲染
    RUST_DEEP_ANALYSIS = "rust_deep_analysis"       # Phase 1c: Rust 深度分析
    
    # Phase 2: 特殊掃描
    SENSITIVE_DATA_SCAN = "sensitive_data_scan"      # Phase 2: 敏感資料掃描
    
    # Phase 3: 結果處理
    RESULT_AGGREGATION = "result_aggregation"        # Phase 3: 結果聚合
    AI_STRATEGY_DECISION = "ai_strategy_decision"    # Phase 3: AI 策略決策

from services.aiva_common.schemas import (
    Asset,
    Fingerprints,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
    Phase0CompletedPayload,
    Phase1CompletedPayload,
)
from services.aiva_common.enums.assets import AssetType
from services.aiva_common.utils import get_logger




class MultiEngineCoordinator:
    """
    多引擎協調器 - 使用適配器模式（重構後）
    
    核心職責：
    1. 編排四引擎並行執行（不關心引擎具體實現）
    2. 聚合和去重掃描結果
    3. 生成標準化的 Phase1CompletedPayload
    
    設計模式：
    - 適配器模式：統一四引擎接口
    - 依賴注入：可替換的適配器實例
    - 開放封閉原則：新增引擎不修改 Coordinator
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 初始化 logger
        self.logger = get_logger("MultiEngineCoordinator")
        
        # ✅ 適配器模式：初始化四引擎適配器
        from .engines import PythonAdapter, TypeScriptAdapter, RustAdapter, GoAdapter
        
        self.adapters = {
            "python": PythonAdapter(),
            "typescript": TypeScriptAdapter(),
            "rust": RustAdapter(),
            "go": GoAdapter(),
        }
        
        # 引擎可用性（保留向後兼容）
        self.available_engines: Set[EngineType] = set()
        
        # 協調策略
        self.coordination_strategy = "adapter_based"  # 新架構標記
        
        # 初始化標記
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化並檢查所有引擎的可用性
        
        這個方法必須在使用協調器之前調用，用於：
        1. 檢查每個引擎適配器是否可用
        2. 更新 available_engines 集合
        3. 記錄引擎可用性狀態
        
        範例:
            coordinator = MultiEngineCoordinator()
            await coordinator.initialize()  # 必須先初始化
            result = await coordinator.execute_phase1(...)
        """
        if self._initialized:
            self.logger.debug("協調器已經初始化，跳過")
            return
        
        self.logger.info("🔍 開始檢查引擎可用性...")
        
        for engine_name, adapter in self.adapters.items():
            try:
                is_available = await adapter.is_available()
                if is_available:
                    self.available_engines.add(EngineType(engine_name))
                    self.logger.info(f"  ✅ {engine_name.upper()} 引擎可用")
                else:
                    self.logger.warning(f"  ❌ {engine_name.upper()} 引擎不可用")
            except Exception as e:
                self.logger.error(f"  ❌ {engine_name.upper()} 引擎檢查失敗: {e}")
        
        self._initialized = True
        
        if not self.available_engines:
            self.logger.error("⚠️  警告：沒有任何引擎可用！")
        else:
            available_list = [e.value for e in self.available_engines]
            self.logger.info(f"✅ 引擎初始化完成，可用引擎: {', '.join(available_list)}")
    
    # ========== 已移除的舊接口（有嚴重缺陷，請勿使用）==========
    # 
    # execute_coordinated_scan() - 已刪除
    #   問題：調用不存在的 _phase_0_rust_fast_discovery() 和 _phase_2_multi_engine_scan()
    #
    # 所有 _phase_X 和 _run_X 私有方法 - 已刪除
    #   問題：實現有缺陷、同步調用阻塞、引擎未實現
    #
    # ✅ 請改用新接口：
    #   1. execute_phase0() - Rust 快速發現
    #   2. execute_phase1() - 多引擎深度掃描（完整支援四引擎）
    #   3. 或使用預設策略：execute_strategy_fast/balanced/comprehensive/aggressive/smart()
    #
    # ==========================================================
    
    
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
    
    # ==================== 預設策略接口（簡化 AI 決策）====================
    
    async def execute_strategy_fast(
        self,
        scan_id: str,
        targets: List[str],
        max_depth: int = 2
    ) -> Phase1CompletedPayload:
        """快速掃描策略 - 僅使用 Python 引擎
        
        適用場景：
        - 快速驗證目標可達性
        - 基礎資產發現
        - 開發測試環境
        
        引擎組合：Python
        預期時間：< 30秒
        """
        # 🚀 執行快速掃描策略
        return await self.execute_phase1(
            scan_id=scan_id,
            targets=targets,
            selected_engines=["python"],
            max_depth=max_depth,
            max_urls=100
        )
    
    async def execute_strategy_balanced(
        self,
        scan_id: str,
        targets: List[str],
        max_depth: int = 5
    ) -> Phase1CompletedPayload:
        """均衡掃描策略 - Python + Rust 組合
        
        適用場景：
        - 一般 Web 應用掃描
        - 包含靜態和敏感信息掃描
        - 生產環境常規掃描
        
        引擎組合：Python (爬取) + Rust (敏感信息)
        預期時間：1-3分鐘
        """
        # ⚡️ 執行均衡掃描策略
        return await self.execute_phase1(
            scan_id=scan_id,
            targets=targets,
            selected_engines=["python", "rust"],
            max_depth=max_depth,
            max_urls=500
        )
    
    async def execute_strategy_comprehensive(
        self,
        scan_id: str,
        targets: List[str],
        max_depth: int = 5
    ) -> Phase1CompletedPayload:
        """全面掃描策略 - Python + TypeScript + Rust 組合
        
        適用場景：
        - SPA 應用（React/Vue/Angular）
        - 需要 JavaScript 渲染的頁面
        - 深度安全審計
        
        引擎組合：Python (靜態) + TypeScript (動態) + Rust (敏感)
        預期時間：3-5分鐘
        """
        # 🔍 執行全面掃描策略
        return await self.execute_phase1(
            scan_id=scan_id,
            targets=targets,
            selected_engines=["python", "typescript", "rust"],
            max_depth=max_depth,
            max_urls=1000
        )
    
    async def execute_strategy_aggressive(
        self,
        scan_id: str,
        targets: List[str],
        max_depth: int = 7
    ) -> Phase1CompletedPayload:
        """激進掃描策略 - 四引擎全開
        
        適用場景：
        - 大型應用全面掃描
        - 需要服務發現（SSRF/CSPM）
        - 完整安全評估
        
        引擎組合：Python + TypeScript + Rust + Go (全部)
        預期時間：5-10分鐘
        """
        # 💥 執行激進掃描策略
        
        # 確保所有引擎可用
        available = [e.value for e in self.available_engines]
        if len(available) < 4:
            self.logger.warning(
                f"⚠️ 激進策略需要四引擎，但僅 {len(available)} 個可用: {available}"
            )
        
        return await self.execute_phase1(
            scan_id=scan_id,
            targets=targets,
            selected_engines=["python", "typescript", "rust", "go"],
            max_depth=max_depth,
            max_urls=2000
        )
    
    async def execute_strategy_smart(
        self,
        scan_id: str,
        targets: List[str]
    ) -> Phase1CompletedPayload:
        """智能掃描策略 - 基於 Phase 0 結果自動選擇引擎
        
        流程：
        1. 執行 Phase 0 (Rust 快速發現)
        2. 分析技術棧和特徵
        3. 自動選擇最佳引擎組合
        4. 執行 Phase 1 深度掃描
        
        適用場景：
        - AI 不確定如何選擇引擎
        - 需要自動優化掃描策略
        - 未知目標類型
        """
        # 🧠 執行智能掃描策略
        
        # Step 1: Phase 0 快速發現
        phase0_result = await self.execute_phase0(
            scan_id=scan_id,
            targets=targets,
            max_depth=2,
            timeout=60
        )
        
        if phase0_result.status != "success":
            # ⚠️ Phase 0 失敗，降級為快速掃描
            return await self.execute_strategy_fast(scan_id, targets)
        
        # Step 2: 分析建議的引擎
        recommendations = phase0_result.recommendations
        suggested_engines = recommendations.get("suggested_engines", [])
        
        if not suggested_engines:
            # ℹ️ 無引擎建議，使用均衡策略
            suggested_engines = ["python", "rust"]
        
        # 📊 智能策略選擇引擎完成
        
        # Step 3: 根據建議執行 Phase 1
        return await self.execute_phase1(
            scan_id=scan_id,
            targets=targets,
            selected_engines=suggested_engines,
            max_depth=5,
            max_urls=1000,
            phase0_result=phase0_result.model_dump()
        )
    
    # ==================== AI 直接指揮接口 ====================
    
    async def execute_phase0(
        self,
        scan_id: str,
        targets: List[str],
        max_depth: int = 3,
        timeout: int = 600
    ) -> "Phase0CompletedPayload":
        """執行 Phase 0 快速偵察 - AI 直接調用接口
        
        這是供 AI 命令中心直接調用的接口，無需 RabbitMQ。
        
        Args:
            scan_id: 掃描 ID
            targets: 目標 URL 列表
            max_depth: 最大爬取深度
            timeout: 超時時間（秒）
            
        Returns:
            Phase 0 完成結果
            
        Example:
            result = await coordinator.execute_phase0(
                scan_id="scan_abc123",
                targets=["https://example.com"],
                max_depth=3,
                timeout=600
            )
        """
        from services.aiva_common.schemas import (
            Phase0CompletedPayload,
            Summary,
            Fingerprints
        )
        
        start_time = time.time()
        
        self.logger.info(
            f"🦀 Phase 0 開始: {scan_id} "
            f"(目標: {len(targets)}個, 深度: {max_depth})"
        )
        
        try:
            # 檢查 Rust 引擎是否可用
            if EngineType.RUST not in self.available_engines:
                error_msg = "Rust 引擎不可用，無法執行 Phase 0"
                # 錯誤紀錄
                return Phase0CompletedPayload(
                    scan_id=scan_id,
                    status="failed",
                    execution_time=time.time() - start_time,
                    assets=[],
                    fingerprints=None,
                    summary=Summary(),
                    recommendations={},
                    error_info=error_msg
                )
            
            # 調用 Rust 引擎執行快速偵察
            from ..engines.rust_engine.python_bridge import rust_info_gatherer
            
            # 執行掃描 (同步)
            raw_results = []
            for target in targets:
                try:
                    result = rust_info_gatherer.scan_target(
                        target,
                        {"mode": "deep_analysis", "timeout": int(timeout / len(targets))}
                    )
                    raw_results.append(result)
                except Exception as e:
                    # ✱️ 目標掃描失敗
                    raw_results.append({"success": False, "error": str(e)})
            
            # 解析結果
            assets = []
            fingerprints_data = {}
            summary_data = {"urls_found": 0, "forms_found": 0, "apis_found": 0}
            
            for result in raw_results:
                # 解析資產
                if "assets" in result:
                    assets.extend([
                        Asset(
                            asset_id=f"{scan_id}_{i}",
                            type=asset.get("type", "unknown"),
                            value=asset.get("value", ""),
                            parameters=asset.get("parameters"),
                            has_form=asset.get("has_form", False)
                        )
                        for i, asset in enumerate(result["assets"])
                    ])
                
                # 解析指紋
                if "fingerprints" in result:
                    for key, value in result["fingerprints"].items():
                        if key not in fingerprints_data:
                            fingerprints_data[key] = {}
                        fingerprints_data[key].update(value or {})
                
                # 累加統計
                if "summary" in result:
                    summary_data["urls_found"] += result["summary"].get("urls_found", 0)
                    summary_data["forms_found"] += result["summary"].get("forms_found", 0)
                    summary_data["apis_found"] += result["summary"].get("apis_found", 0)
            
            # 構建 Phase 0 結果
            execution_time = time.time() - start_time
            
            # 生成 AI 決策建議
            fingerprints_str = str(fingerprints_data).lower()
            recommendations = {
                "needs_js_engine": any([
                    "react" in fingerprints_str,
                    "vue" in fingerprints_str,
                    "angular" in fingerprints_str
                ]),
                "needs_form_testing": summary_data["forms_found"] > 0,
                "needs_api_testing": summary_data["apis_found"] > 0,
                "suggested_engines": []
            }
            
            # 引擎建議
            if recommendations["needs_js_engine"]:
                recommendations["suggested_engines"].append("typescript")
            if summary_data["forms_found"] > 0:
                recommendations["suggested_engines"].append("python")
            if len(assets) > 50:
                recommendations["suggested_engines"].append("go")
            
            # 總是建議 Rust 深度掃描
            recommendations["suggested_engines"].append("rust")
            
            self.logger.info(
                f"✅ Phase 0 完成: {scan_id} "
                f"(資產: {len(assets)}, 耗時: {execution_time:.2f}s)"
            )
            
            return Phase0CompletedPayload(
                scan_id=scan_id,
                status="success",
                execution_time=execution_time,
                assets=assets,
                fingerprints=Fingerprints(**fingerprints_data) if fingerprints_data else None,
                summary=Summary(**summary_data),
                recommendations=recommendations,
                error_info=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Phase 0 執行異常: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            return Phase0CompletedPayload(
                scan_id=scan_id,
                status="failed",
                execution_time=execution_time,
                assets=[],
                fingerprints=None,
                summary=Summary(),
                recommendations={},
                error_info=error_msg
            )
    
    async def execute_phase1(
        self,
        scan_id: str,
        targets: List[str],
        selected_engines: List[str],
        max_depth: int = 5,
        max_urls: int = 1000,
        phase0_result: Optional[Dict[str, Any]] = None
    ) -> "Phase1CompletedPayload":
        """執行 Phase 1 深度掃描 - 使用適配器模式（重構後）
        
        核心改進：
        1. 從 171 複雜度降至 < 30
        2. 使用適配器統一接口，不再關心引擎細節
        3. 清晰的錯誤隔離和狀態管理
        
        Args:
            scan_id: 掃描 ID
            targets: 目標 URL 列表
            selected_engines: AI 選擇的引擎列表 (python/typescript/rust/go)
            max_depth: 最大爬取深度
            max_urls: 最大 URL 數量
            phase0_result: Phase 0 結果（可選）
            
        Returns:
            Phase 1 完成結果
        """
        from services.aiva_common.schemas import (
            Phase1CompletedPayload,
            Summary
        )
        
        start_time = time.time()
        
        self.logger.info(
            f"🚀 Phase 1 開始: {scan_id} "
            f"(引擎: {', '.join(selected_engines)}, 目標: {len(targets)}個)"
        )
        
        try:
            # 1. 準備 AI 簡化指令 - 協調器負責轉換為各引擎的具體格式
            # AI 只需下達簡單的高層次指令,協調器確保正確執行
            ai_command = {
                "scan_id": scan_id,
                "targets": targets,  # AI 指定的目標列表
                "depth": max_depth,   # AI 指定的深度 (1-5)
                "timeout": 10,        # 統一超時
            }
            
            # 2. 任務分發 - 協調器為每個引擎轉換命令格式
            tasks = []
            active_engines = []
            
            for engine_name in selected_engines:
                adapter = self.adapters.get(engine_name)
                
                if not adapter or not await adapter.is_available():
                    continue
                
                # 協調器根據引擎類型轉換命令
                engine_options = self._translate_command_for_engine(
                    engine_name, ai_command, max_urls
                )
                
                active_engines.append(engine_name)
                tasks.append(adapter.scan(targets, engine_options))
                # (已在上方處理)
            
            if not tasks:
                # 沒有可用的引擎
                return self._create_empty_result(scan_id, "failed", "沒有可用的引擎")
            
            # 3. 並行執行 - return_exceptions=True 確保單引擎失敗不拖垮整體
            # 並行執行引擎
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 4. 結果聚合 - 錯誤隔離與資產收集
            all_assets = []
            engine_status = {}
            failed_engines = []
            
            for engine_name, result in zip(active_engines, results):
                if isinstance(result, Exception):
                    # 錯誤隔離：單引擎失敗被記錄，但不影響其他引擎
                    error_msg = str(result)
                    # 引擎失敗
                    engine_status[engine_name] = {"status": "failed", "error": error_msg}
                    failed_engines.append(engine_name)
                elif isinstance(result, dict):
                    # 適配器保證返回標準格式 {"assets": [...], "metadata": {...}}
                    assets = result.get("assets", [])
                    
                    # 🔧 修正: 統一轉換為 Asset 對象 (處理不同引擎的格式差異)
                    normalized_assets = []
                    for asset in assets:
                        if isinstance(asset, Asset):
                            # 已經是 Asset 對象 (Python Engine)
                            normalized_assets.append(asset)
                        elif isinstance(asset, dict):
                            # 需要轉換的 dict (Go/Rust/TypeScript Engine)
                            try:
                                # 檢查資產類型並提取 value
                                asset_type = asset.get("type", "unknown")
                                
                                # 不同引擎的 value 提取邏輯
                                if asset_type == "web_vulnerability":
                                    # Go 引擎的漏洞格式: 使用 affected_url 或 name 作為 value
                                    value = (
                                        asset.get("details", {}).get("affected_url") or
                                        asset.get("name", "") or
                                        str(asset.get("details", {}).get("finding_id", ""))
                                    )
                                else:
                                    # 標準格式: 直接使用 value 欄位
                                    value = asset.get("value", "")
                                
                                # 如果還是沒有 value,跳過這個資產
                                if not value:
                                    self.logger.warning(f"資產缺少 value,跳過: type={asset_type}")
                                    continue
                                
                                normalized_assets.append(Asset(
                                    asset_id=asset.get("asset_id", f"{scan_id}_{len(all_assets)}"),
                                    type=asset_type,
                                    value=value,
                                    parameters=None,  # 🔧 修正: Asset.parameters 期望 List[str],暫不傳遞複雜物件
                                    has_form=asset.get("has_form", False)
                                ))
                            except Exception as e:
                                self.logger.warning(f"資產轉換失敗: {e}, 跳過資產: {asset.get('type', 'unknown')}")
                    
                    all_assets.extend(normalized_assets)
                    engine_status[engine_name] = {
                        "status": "completed",
                        "assets_count": len(normalized_assets)
                    }
                    # 引擎成功
            
            # 5. 去重
            unique_assets = self._deduplicate_assets(all_assets)
            
            # 6. 構建統計信息
            summary = Summary(
                urls_found=len([a for a in unique_assets if a.type == "url"]),
                forms_found=len([a for a in unique_assets if a.has_form]),
                apis_found=len([a for a in unique_assets if a.type == "api"]),
                scan_duration_seconds=int(time.time() - start_time)
            )
            
            # 7. 判斷最終狀態
            execution_time = time.time() - start_time
            
            if not engine_status or all(s["status"] == "failed" for s in engine_status.values()):
                final_status = "failed"
                error_msg = f"所有引擎失敗: {', '.join(failed_engines)}"
            elif failed_engines:
                final_status = "partial_success"
                error_msg = f"部分引擎失敗: {', '.join(failed_engines)}"
            else:
                final_status = "completed"
                error_msg = None
            
            self.logger.info(
                f"{'✅' if final_status == 'completed' else '⚠️'} Phase 1 {final_status}: {scan_id} "
                f"(資產: {len(unique_assets)}, 成功: {len(active_engines) - len(failed_engines)}/{len(active_engines)}, "
                f"耗時: {execution_time:.2f}s)"
            )
            
            return Phase1CompletedPayload(
                scan_id=scan_id,
                status=final_status,
                execution_time=execution_time,
                summary=summary,
                fingerprints=None,
                assets=unique_assets,
                engine_results=engine_status,
                phase0_summary=phase0_result if phase0_result else {},
                error_info=error_msg
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Phase 1 執行異常: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            
            return self._create_empty_result(scan_id, "failed", error_msg)
    
    def _create_empty_result(
        self,
        scan_id: str,
        status: str,
        error_msg: str
    ) -> "Phase1CompletedPayload":
        """創建空結果(錯誤情況)"""
        from services.aiva_common.schemas import Phase1CompletedPayload, Summary
        
        return Phase1CompletedPayload(
            scan_id=scan_id,
            status=status,
            execution_time=0,
            summary=Summary(),
            fingerprints=None,
            assets=[],
            engine_results={},
            phase0_summary={},
            error_info=error_msg
        )
    
    def _translate_command_for_engine(
        self, engine_name: str, ai_command: dict, max_urls: int
    ) -> dict:
        """協調器: 將 AI 簡化指令轉換為引擎特定格式
        
        降低 AI 指令複雜度 - AI 只需提供:
        - scan_id: 掃描ID
        - targets: 目標列表
        - depth: 深度(1-5)
        - timeout: 超時
        
        協調器負責轉換為各引擎使用指南要求的格式
        """
        scan_id = ai_command["scan_id"]
        depth = ai_command["depth"]
        timeout = ai_command["timeout"]
        
        if engine_name == "go":
            # Go 使用指南: 只需 scan_id, targets, concurrency, timeout
            return {
                "scan_id": scan_id,
                "concurrency": 5,  # Go 最佳並發數
                "timeout": timeout,
            }
        
        elif engine_name == "rust":
            # Rust 使用指南: --url, --mode, --timeout
            return {
                "scan_id": scan_id,
                "mode": "deep" if depth >= 3 else "fast",
                "timeout": timeout,
            }
        
        elif engine_name == "python":
            # Python 引擎: strategy, max_depth, max_pages
            return {
                "scan_id": scan_id,
                "strategy": "deep" if depth >= 3 else "normal",
                "max_depth": depth,
                "max_pages": max_urls,
                "timeout": timeout,
            }
        
        elif engine_name == "typescript":
            # TypeScript 引擎配置
            return {
                "scan_id": scan_id,
                "max_depth": depth,
                "timeout": timeout,
            }
        
        else:
            # 未知引擎,返回基本配置
            return {
                "scan_id": scan_id,
                "timeout": timeout,
            }


# ==================== 便利函數 ====================

async def quick_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload:
    """快速掃描 - 使用 Python 引擎"""
    coordinator = MultiEngineCoordinator()
    return await coordinator.execute_strategy_fast(scan_id, targets)

async def smart_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload:
    """智能掃描 - 自動選擇引擎組合"""
    coordinator = MultiEngineCoordinator()
    return await coordinator.execute_strategy_smart(scan_id, targets)

async def full_scan(scan_id: str, targets: List[str]) -> Phase1CompletedPayload:
    """全面掃描 - 四引擎全開"""
    coordinator = MultiEngineCoordinator()
    return await coordinator.execute_strategy_aggressive(scan_id, targets)
