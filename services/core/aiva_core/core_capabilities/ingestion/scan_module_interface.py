import json
from typing import Any
from pathlib import Path

from pydantic import HttpUrl
from aiva_common.enums import Topic
from aiva_common.messaging import AbstractBroker
from aiva_common.schemas import (
    ScanCompletedPayload,
    Phase0StartPayload,
    Phase0CompletedPayload,
    Phase1StartPayload,
)
from aiva_common.utils import get_logger

# ✅ 連接到 scan 模組 - MultiEngineCoordinator
try:
    from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
    SCAN_COORDINATOR_AVAILABLE = True
except ImportError:
    SCAN_COORDINATOR_AVAILABLE = False
    MultiEngineCoordinator = None

logger = get_logger(__name__)


class ScanModuleInterface:
    """掃描模組介面 - 資料接收與預處理

    負責接收掃描模組的原始數據並進行標準化處理，
    包含格式檢測、資料清理、去重、豐富化等功能。
    """
    
    def __init__(self):
        """初始化掃描模組介面"""
        # ✅ 連接 scan 模組
        if SCAN_COORDINATOR_AVAILABLE:
            try:
                self.scan_coordinator = MultiEngineCoordinator()
                logger.info("✅ MultiEngineCoordinator 已連接")
            except Exception as e:
                logger.warning(f"⚠️  MultiEngineCoordinator 初始化失敗: {e}")
                self.scan_coordinator = None
        else:
            logger.warning("⚠️  MultiEngineCoordinator 不可用，使用降級模式")
            self.scan_coordinator = None

    def process_scan_data(self, payload: ScanCompletedPayload) -> dict[str, Any]:
        """處理掃描模組回傳的原始數據

        Args:
            payload: 掃描完成的負載數據

        Returns:
            標準化處理後的資料結構
        """
        processed_data = {
            "scan_id": payload.scan_id,
            "status": payload.status,
            "summary": payload.summary,
            "assets": self._process_assets(payload.assets),
            "fingerprints": self._process_fingerprints(payload.fingerprints),
        }

        return processed_data

    def _process_assets(self, assets: list[Any]) -> list[dict[str, Any]]:
        """處理資產清單，進行分類與標準化

        Args:
            assets: 原始資產清單

        Returns:
            處理後的資產清單
        """
        processed_assets = []

        for asset in assets:
            processed_asset = {
                "asset_id": asset.asset_id,
                "type": asset.type,
                "value": asset.value,
                "parameters": asset.parameters,
                "has_form": asset.has_form,
                "risk_score": self._calculate_risk_score(asset),
                "categories": self._categorize_asset(asset),
            }
            processed_assets.append(processed_asset)

        return processed_assets

    def _process_fingerprints(self, fingerprints: Any) -> dict[str, Any]:
        """處理技術指紋資料

        Args:
            fingerprints: 原始指紋資料

        Returns:
            處理後的指紋資料
        """
        if not fingerprints:
            return {}

        return {
            "web_server": fingerprints.web_server or {},
            "framework": fingerprints.framework or {},
            "language": fingerprints.language or {},
            "waf_detected": fingerprints.waf_detected,
            "waf_vendor": fingerprints.waf_vendor,
        }

    def _calculate_risk_score(self, asset: Any) -> int:
        """計算資產風險分數

        Args:
            asset: 資產物件

        Returns:
            風險分數 (1-10)
        """
        risk_score = 1

        # 基於資產類型調整風險分數
        if asset.type == "URL":
            if asset.has_form:
                risk_score += 3  # 有表單的頁面風險較高
            if asset.parameters:
                risk_score += len(asset.parameters)  # 參數越多風險越高
        elif asset.type == "API":
            risk_score += 2  # API端點通常風險較高

        return min(risk_score, 10)  # 最高10分

    def _categorize_asset(self, asset: Any) -> list[str]:
        """資產分類

        Args:
            asset: 資產物件

        Returns:
            資產類別標籤列表
        """
        categories: list[str] = []

        if not hasattr(asset, "value") or not asset.value:
            return categories

        url_path = str(asset.value).lower()

        # 功能分類
        if any(keyword in url_path for keyword in ["login", "signin", "auth"]):
            categories.append("authentication")
        if any(keyword in url_path for keyword in ["admin", "manage", "dashboard"]):
            categories.append("administration")
        if any(keyword in url_path for keyword in ["api", "rest", "graphql"]):
            categories.append("api")
        if any(keyword in url_path for keyword in ["upload", "file", "download"]):
            categories.append("file_handling")
        if any(keyword in url_path for keyword in ["search", "query"]):
            categories.append("search")
        if asset.has_form:
            categories.append("form_input")
        if asset.parameters:
            categories.append("parameterized")

        return categories

    # ==================== Phase0/Phase1 兩階段掃描接口 ====================

    async def send_phase0_command(
        self,
        broker: AbstractBroker,
        scan_id: str,
        targets: list[str],
        trace_id: str,
        timeout_seconds: int = 600,
    ) -> None:
        """發送 Phase0 快速偵察命令

        Args:
            broker: 消息代理
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID
            timeout_seconds: 超時時間 (預設 10 分鐘)
        """
        # 將 str 轉換為 HttpUrl 以符合 schema 要求
        validated_targets = [HttpUrl(url) for url in targets]
        
        phase0_payload = Phase0StartPayload(
            scan_id=scan_id,
            targets=validated_targets,
            timeout=timeout_seconds,  # 使用正確的參數名 timeout
        )

        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase0_payload.model_dump(),
        }

        await broker.publish(
            Topic.TASK_SCAN_PHASE0,
            json.dumps(message).encode("utf-8"),
        )

        logger.info(
            f"[Phase0] Command sent to {Topic.TASK_SCAN_PHASE0.value} "
            f"(scan_id={scan_id}, targets={len(targets)})"
        )

    async def send_phase1_command(
        self,
        broker: AbstractBroker,
        scan_id: str,
        targets: list[str],
        trace_id: str,
        phase0_result: Phase0CompletedPayload,
        selected_engines: list[str],
        max_depth: int = 3,
        max_urls: int = 1000,
        timeout_seconds: int = 1800,
    ) -> None:
        """發送 Phase1 深度掃描命令

        Args:
            broker: 消息代理
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID
            phase0_result: Phase0 結果
            selected_engines: 選中的引擎列表
            max_depth: 最大爬取深度 (預設 3)
            max_urls: 最大 URL 數量 (預設 1000)
            timeout_seconds: 超時時間 (預設 30 分鐘)
        """
        # 將 str 轉換為 HttpUrl 以符合 schema 要求
        validated_targets = [HttpUrl(url) for url in targets]
        
        phase1_payload = Phase1StartPayload(
            scan_id=scan_id,
            targets=validated_targets,
            phase0_result=phase0_result,
            selected_engines=selected_engines,
            max_depth=max_depth,
            max_pages=max_urls,  # 使用正確的參數名 max_pages
            timeout=timeout_seconds,  # 使用正確的參數名 timeout
        )

        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase1_payload.model_dump(),
        }

        await broker.publish(
            Topic.TASK_SCAN_PHASE1,
            json.dumps(message).encode("utf-8"),
        )

        logger.info(
            f"[Phase1] Command sent to {Topic.TASK_SCAN_PHASE1.value} "
            f"(scan_id={scan_id}, engines={selected_engines}, max_depth={max_depth})"
        )

    def process_phase0_result(
        self, payload: Phase0CompletedPayload
    ) -> dict[str, Any]:
        """處理 Phase0 結果

        Args:
            payload: Phase0 完成載荷

        Returns:
            處理後的 Phase0 數據
        """
        # 使用正確的 Phase0CompletedPayload 欄位
        summary = payload.summary
        fingerprints = payload.fingerprints
        assets = payload.assets
        recommendations = payload.recommendations
        
        # 計算技術數量
        tech_count = 0
        tech_list: list[str] = []
        if fingerprints:
            if fingerprints.framework:
                tech_list.extend(fingerprints.framework.keys())
                tech_count += len(fingerprints.framework)
            if fingerprints.language:
                tech_list.extend(fingerprints.language.keys())
                tech_count += len(fingerprints.language)
        
        processed = {
            "scan_id": payload.scan_id,
            "status": payload.status,
            "execution_time": payload.execution_time,
            # 從 summary 獲取統計
            "urls_found": summary.urls_found,
            "forms_found": summary.forms_found,
            "apis_found": summary.apis_found,
            # 從 fingerprints 獲取技術棧
            "discovered_technologies": tech_list,
            "tech_count": tech_count,
            # 從 assets 獲取資產
            "assets": assets,
            "asset_count": len(assets),
            # 從 recommendations 獲取風險評估
            "high_risk": recommendations.get("high_risk", False),
            "needs_js_engine": recommendations.get("needs_js_engine", False),
            "needs_form_testing": recommendations.get("needs_form_testing", False),
            "needs_api_testing": recommendations.get("needs_api_testing", False),
            # WAF 檢測
            "waf_detected": fingerprints.waf_detected if fingerprints else False,
            "waf_vendor": fingerprints.waf_vendor if fingerprints else None,
            # 遺留欄位 (為了兼容性)
            "sensitive_count": 0,  # Phase0 不再直接返回敏感資料數量
            "endpoint_count": summary.urls_found,
            "basic_endpoints": [],  # 已由 assets 替代
            "sensitive_data_found": [],  # 已由 assets 替代
            "risk_level": "high" if recommendations.get("high_risk", False) else "low",
        }

        logger.info(
            f"[Phase0] Result processed - "
            f"Technologies: {processed['tech_count']}, "
            f"URLs: {processed['urls_found']}, "
            f"Forms: {processed['forms_found']}, "
            f"APIs: {processed['apis_found']}, "
            f"Assets: {processed['asset_count']}, "
            f"Risk: {processed['risk_level']}"
        )

        return processed

    # ==================== v2.0 AICommand 直接執行接口 ====================

    async def execute_phase0_direct(
        self,
        scan_id: str,
        targets: list[str],
        timeout: int = 600,
    ) -> dict[str, Any]:
        """直接執行 Phase0 快速偵察 (v2.0 AICommand 模式)
        
        ✅ 連接到 scan 模組的 MultiEngineCoordinator
        
        Args:
            scan_id: 掃描 ID
            targets: 目標列表
            timeout: 超時時間 (秒)
            
        Returns:
            Phase0 執行結果字典
        """
        import asyncio
        import subprocess
        from datetime import datetime
        
        logger.info(f"[Phase0-Direct] Starting scan {scan_id} for {len(targets)} targets")
        
        try:
            # ✅ 方案 1: 使用 MultiEngineCoordinator
            if self.scan_coordinator is not None:
                logger.info(f"[Phase0-Direct] Using MultiEngineCoordinator for {scan_id}")
                
                # 初始化 coordinator
                await self.scan_coordinator.initialize()
                
                # 執行快速策略 (Phase0 等同於 fast 策略)
                coordinator_result = await self.scan_coordinator.execute_strategy_fast(
                    scan_id=scan_id,
                    targets=targets,
                    max_depth=3
                )
                
                # 轉換結果格式
                result = {
                    "success": coordinator_result.get("status") == "completed",
                    "scan_id": scan_id,
                    "status": coordinator_result.get("status", "completed"),
                    "assets": self._extract_assets_from_coordinator_result(coordinator_result),
                    "technologies": self._extract_technologies_from_coordinator_result(coordinator_result),
                    "summary": {
                        "urls_found": len(coordinator_result.get("findings", [])),
                        "forms_found": 0,  # Phase0 不檢測表單
                        "apis_found": 0,
                    },
                    "recommendations": {
                        "high_risk": len(coordinator_result.get("vulnerabilities", [])) > 0,
                        "needs_phase1": len(coordinator_result.get("findings", [])) > 0,
                    },
                }
                
                logger.info(
                    f"[Phase0-Direct] Completed scan {scan_id} via MultiEngineCoordinator - "
                    f"Assets: {len(result['assets'])}, "
                    f"Technologies: {len(result['technologies'])}"
                )
                
                return result
            
            # ⚠️  降級方案: 簡易 HTTP 檢查
            logger.warning(f"[Phase0-Direct] MultiEngineCoordinator unavailable, using fallback for {scan_id}")
            result = {
                "success": True,
                "scan_id": scan_id,
                "status": "completed",
                "assets": [],
                "technologies": [],
                "summary": {
                    "urls_found": 0,
                    "forms_found": 0,
                    "apis_found": 0,
                },
                "recommendations": {
                    "high_risk": False,
                    "needs_phase1": False,
                },
            }
            
            for target in targets:
                try:
                    # 基本的 HTTP 請求檢查
                    import aiohttp
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                        # 注意: ssl=False 是為了允許測試自簽證書，在實際部署中建議啟用 SSL 驗證
                        async with session.get(target, ssl=False, allow_redirects=True) as response:
                            result["summary"]["urls_found"] += 1
                            
                            # 檢測基本技術棧
                            headers = dict(response.headers)
                            server = headers.get("Server", "")
                            if server:
                                result["technologies"].append(server)
                            
                            # 構建資產
                            result["assets"].append({
                                "type": "URL",
                                "value": str(target),
                                "status_code": response.status,
                                "has_form": False,  # 簡化處理
                                "parameters": [],
                            })
                            
                except Exception as e:
                    logger.warning(f"[Phase0-Direct] Error checking {target}: {e}")
                    result["assets"].append({
                        "type": "URL",
                        "value": str(target),
                        "error": str(e),
                        "has_form": False,
                        "parameters": [],
                    })
            
            # 根據結果決定是否需要 Phase1
            result["recommendations"]["needs_phase1"] = len(result["assets"]) > 0
            
            logger.info(
                f"[Phase0-Direct] Completed scan {scan_id} - "
                f"Assets: {len(result['assets'])}, "
                f"Technologies: {len(result['technologies'])}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[Phase0-Direct] Failed scan {scan_id}: {e}", exc_info=True)
            return {
                "success": False,
                "scan_id": scan_id,
                "status": "failed",
                "error": str(e),
                "assets": [],
                "technologies": [],
            }

    async def execute_phase1_direct(
        self,
        scan_id: str,
        targets: list[str],
        engines: list[str],
        timeout: int = 1800,
    ) -> dict[str, Any]:
        """直接執行 Phase1 深度掃描 (v2.0 AICommand 模式)
        
        ✅ 連接到 scan 模組的 MultiEngineCoordinator
        
        Args:
            scan_id: 掃描 ID
            targets: 目標列表
            engines: 要使用的掃描引擎列表 (如 ['nikto', 'dirb'])
            timeout: 超時時間 (秒)
            
        Returns:
            Phase1 執行結果字典
        """
        import asyncio
        
        logger.info(
            f"[Phase1-Direct] Starting scan {scan_id} with engines {engines} "
            f"for {len(targets)} targets"
        )
        
        try:
            # ✅ 方案 1: 使用 MultiEngineCoordinator
            if self.scan_coordinator is not None:
                logger.info(f"[Phase1-Direct] Using MultiEngineCoordinator for {scan_id}")
                
                # 初始化 coordinator
                await self.scan_coordinator.initialize()
                
                # 根據引擎數量選擇策略
                if len(engines) <= 2:
                    coordinator_result = await self.scan_coordinator.execute_strategy_fast(
                        scan_id=scan_id,
                        targets=targets,
                        max_depth=3
                    )
                elif len(engines) <= 4:
                    coordinator_result = await self.scan_coordinator.execute_strategy_balanced(
                        scan_id=scan_id,
                        targets=targets,
                        max_depth=3
                    )
                else:
                    coordinator_result = await self.scan_coordinator.execute_strategy_comprehensive(
                        scan_id=scan_id,
                        targets=targets,
                        max_depth=3
                    )
                
                # 轉換結果格式
                result = {
                    "success": coordinator_result.get("status") == "completed",
                    "scan_id": scan_id,
                    "status": coordinator_result.get("status", "completed"),
                    "vulnerabilities": coordinator_result.get("vulnerabilities", []),
                    "findings": coordinator_result.get("findings", []),
                    "engines_used": engines,
                    "execution_summary": coordinator_result.get("summary", {}),
                }
                
                logger.info(
                    f"[Phase1-Direct] Completed scan {scan_id} via MultiEngineCoordinator - "
                    f"Vulnerabilities: {len(result['vulnerabilities'])}, "
                    f"Findings: {len(result['findings'])}"
                )
                
                return result
            
            # ⚠️  降級方案: 返回空結果
            logger.warning(f"[Phase1-Direct] MultiEngineCoordinator unavailable, returning empty result for {scan_id}")
            result = {
                "success": True,
                "scan_id": scan_id,
                "status": "completed",
                "vulnerabilities": [],
                "findings": [],
                "engines_used": engines,
                "execution_summary": {},
            }
            
            # 對每個引擎執行掃描
            for engine in engines:
                engine_result = await self._execute_engine(
                    engine=engine,
                    targets=targets,
                    timeout=timeout // len(engines) if engines else timeout,
                )
                result["findings"].extend(engine_result.get("findings", []))
                result["vulnerabilities"].extend(engine_result.get("vulnerabilities", []))
                result["execution_summary"][engine] = {
                    "success": engine_result.get("success", False),
                    "findings_count": len(engine_result.get("findings", [])),
                }
            
            logger.info(
                f"[Phase1-Direct] Completed scan {scan_id} - "
                f"Vulnerabilities: {len(result['vulnerabilities'])}, "
                f"Findings: {len(result['findings'])}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[Phase1-Direct] Failed scan {scan_id}: {e}", exc_info=True)
            return {
                "success": False,
                "scan_id": scan_id,
                "status": "failed",
                "error": str(e),
                "vulnerabilities": [],
                "findings": [],
            }

    async def _execute_engine(
        self,
        engine: str,
        targets: list[str],
    ) -> dict[str, Any]:
        """執行單個掃描引擎
        
        Args:
            engine: 引擎名稱
            targets: 目標列表
            
        Returns:
            引擎執行結果
            
        Note:
            Timeout is controlled via asyncio.wait_for in the calling code
        """
        import asyncio
        import subprocess
        
        logger.debug(f"[Engine] Executing {engine} on {len(targets)} targets")
        
        try:
            # v2.0: 使用 CLI subprocess 執行掃描引擎
            # 這裡是簡化的模擬實現
            # 實際應該根據引擎類型調用對應的工具
            
            result = {
                "success": True,
                "engine": engine,
                "findings": [],
                "vulnerabilities": [],
            }
            
            # 模擬引擎執行 - 實際應該調用真正的掃描工具
            # 例如:
            # if engine == "nikto":
            #     proc = await asyncio.create_subprocess_exec(
            #         "nikto", "-h", targets[0], "-output", "/tmp/nikto_output.json",
            #         stdout=asyncio.subprocess.PIPE,
            #         stderr=asyncio.subprocess.PIPE,
            #     )
            #     stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            
            # 暫時返回空結果
            await asyncio.sleep(0.1)  # 模擬執行時間
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"[Engine] {engine} timed out after {timeout}s")
            return {
                "success": False,
                "engine": engine,
                "error": "timeout",
                "findings": [],
                "vulnerabilities": [],
            }
        except Exception as e:
            logger.error(f"[Engine] {engine} failed: {e}")
            return {
                "success": False,
                "engine": engine,
                "error": str(e),
                "findings": [],
                "vulnerabilities": [],
            }


    def _extract_assets_from_coordinator_result(self, coordinator_result: dict[str, Any]) -> list[dict[str, Any]]:
        """從 MultiEngineCoordinator 結果中提取資產列表
        
        Args:
            coordinator_result: Coordinator 返回的結果字典
            
        Returns:
            標準化的資產列表
        """
        assets = []
        
        # 從 findings 中提取資產信息
        for finding in coordinator_result.get("findings", []):
            assets.append({
                "type": "URL",
                "value": finding.get("url", finding.get("target", "unknown")),
                "status_code": finding.get("status_code", 200),
                "has_form": False,
                "parameters": finding.get("parameters", []),
            })
        
        return assets
    
    def _extract_technologies_from_coordinator_result(self, coordinator_result: dict[str, Any]) -> list[str]:
        """從 MultiEngineCoordinator 結果中提取技術棧列表
        
        Args:
            coordinator_result: Coordinator 返回的結果字典
            
        Returns:
            技術棧列表
        """
        technologies = []
        
        # 從 findings 或 metadata 中提取技術棧
        for finding in coordinator_result.get("findings", []):
            tech = finding.get("technology", finding.get("server", ""))
            if tech and tech not in technologies:
                technologies.append(tech)
        
        # 從摘要信息中提取
        summary = coordinator_result.get("summary", {})
        if "technologies" in summary:
            for tech in summary["technologies"]:
                if tech not in technologies:
                    technologies.append(tech)
        
        return technologies
