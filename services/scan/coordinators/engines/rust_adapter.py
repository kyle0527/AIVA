"""
Rust 掃描引擎適配器

封裝 Rust FFI Bridge 的調用邏輯，實現非阻塞的異步包裝。

關鍵特性：
- 線程池包裝同步 FFI 調用（避免阻塞事件循環）
- 修復 targets[0] Bug，循環處理所有目標
- 健壯的錯誤處理和資產提取

優勢：
- 高性能敏感信息掃描（正則匹配）
- 快速網絡層發現
- 低內存開銷的大規模掃描

適用場景：
- 敏感數據檢測（API Keys, Tokens, 密碼）
- 快速端點發現（Phase 0）
- 技術棧識別
"""

import asyncio
import concurrent.futures
from typing import List, Dict, Any
from services.aiva_common.schemas import Asset
from .base import BaseScannerAdapter


class RustAdapter(BaseScannerAdapter):
    """Rust 引擎適配器
    
    使用線程池包裝同步 Rust FFI 調用，避免阻塞 asyncio 事件循環。
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.rust_scanner = None
    
    async def is_available(self) -> bool:
        """檢查 Rust 引擎是否可用"""
        try:
            # 嘗試導入 Rust Bridge
            from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer
            self.rust_scanner = rust_info_gatherer
            return True
        except ImportError as e:
            self.logger.warning(f"Rust 引擎不可用: {e}")
            return False
        except Exception as e:
            self.logger.warning(f"Rust Bridge 加載失敗: {e}")
            return False
    
    async def scan(
        self,
        targets: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行 Rust 引擎掃描
        
        使用線程池執行同步 FFI 調用，避免阻塞事件循環。
        
        Args:
            targets: 目標 URL 列表
            options: 掃描配置
        
        Returns:
            包含 assets 和 metadata 的字典
        """
        try:
            if self.rust_scanner is None:
                if not await self.is_available():
                    return {
                        "assets": [],
                        "metadata": {"engine": "rust", "targets_scanned": 0},
                        "error": "Rust 引擎不可用"
                    }
            
            self.logger.info(f"🦀 Rust 引擎開始掃描: {len(targets)} 個目標")
            
            # 獲取事件循環
            loop = asyncio.get_event_loop()
            
            # ✅ 關鍵架構: 在線程池中執行同步 FFI 調用，釋放 GIL 並避免阻塞
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                # 定義同步的掃描函數
                def _sync_scan_all_targets():
                    all_rust_assets = []
                    
                    # ✅ Bug 修復: 循環處理所有目標，而非只掃描 targets[0]
                    for target in targets:
                        try:
                            # 調用 Rust FFI(嚴格符合 USAGE_GUIDE.md 要求)
                            # Rust 使用指南只要求: --url, --mode, --timeout
                            result = self.rust_scanner.scan_target(target, {
                                "mode": options.get("mode", "fast"),  # fast/deep
                                "timeout": options.get("timeout", 10)  # 使用指南建議值
                            })
                            
                            # 驗證並提取結果
                            if result and result.get("success") and "results" in result:
                                results_data = result["results"]
                                
                                # 提取資產（可能是 endpoints 或 assets 字段）
                                raw_assets = results_data.get("assets", [])
                                if not raw_assets:
                                    # 嘗試從 endpoints 字段提取
                                    endpoints = results_data.get("endpoints", [])
                                    for idx, endpoint in enumerate(endpoints):
                                        raw_assets.append({
                                            "asset_id": f"rust_{target}_{idx}",
                                            "type": "url",
                                            "value": endpoint,
                                            "parameters": [],
                                            "has_form": False
                                        })
                                
                                all_rust_assets.extend(raw_assets)
                                
                        except Exception as e:
                            # 記錄錯誤但不中斷整個批次
                            self.logger.warning(f"  Rust 掃描目標失敗 {target}: {e}")
                    
                    return all_rust_assets
                
                # 在線程池中執行
                raw_assets = await loop.run_in_executor(pool, _sync_scan_all_targets)
            
            # 標準化資產
            standardized_assets = self._standardize_assets(raw_assets, "rust")
            
            self.logger.info(f"✅ Rust 引擎完成: {len(standardized_assets)} 個資產")
            
            return {
                "assets": standardized_assets,
                "metadata": {
                    "engine": "rust",
                    "targets_scanned": len(targets),
                    "assets_found": len(standardized_assets)
                },
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Rust 引擎錯誤: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "assets": [],
                "metadata": {
                    "engine": "rust",
                    "targets_scanned": len(targets)
                },
                "error": error_msg
            }
