"""
TypeScript 掃描引擎適配器

封裝 Node.js Playwright 引擎的調用邏輯，實現健壯的跨進程通信。

關鍵特性：
- 異步子進程調用（asyncio.create_subprocess_exec）
- 健壯的 JSON 解析（應對 console.log 污染）
- 移除硬編碼的 [:5] 目標限制

優勢：
- 動態內容渲染（SPA, React, Vue, Angular）
- JavaScript 執行和事件觸發
- 瀏覽器 API 模擬（localStorage, WebSocket 等）

適用場景：
- 單頁應用（SPA）深度掃描
- AJAX 密集型頁面
- 需要 JavaScript 執行的動態內容
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from services.aiva_common.schemas import Asset
from .base import BaseScannerAdapter


class TypeScriptAdapter(BaseScannerAdapter):
    """TypeScript 引擎適配器
    
    封裝 Node.js 子進程調用，實現健壯的 IPC 和 JSON 解析。
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.ts_scanner_path = None
    
    async def is_available(self) -> bool:
        """檢查 TypeScript 引擎是否可用"""
        try:
            # 檢查編譯後的 TypeScript 掃描器是否存在
            scanner_path = Path(__file__).parent.parent / "engines" / "typescript_engine" / "dist" / "index.js"
            
            if not scanner_path.exists():
                self.logger.warning(f"TypeScript 掃描器不存在: {scanner_path}")
                return False
            
            # 檢查 Node.js 是否可用
            try:
                proc = await asyncio.create_subprocess_exec(
                    "node", "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                
                if proc.returncode != 0:
                    self.logger.warning("Node.js 不可用")
                    return False
                
            except FileNotFoundError:
                self.logger.warning("Node.js 未安裝")
                return False
            
            self.ts_scanner_path = scanner_path
            return True
            
        except Exception as e:
            self.logger.warning(f"TypeScript 引擎檢查失敗: {e}")
            return False
    
    async def scan(
        self,
        targets: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行 TypeScript 引擎掃描
        
        Args:
            targets: 目標 URL 列表
            options: 掃描配置
        
        Returns:
            包含 assets 和 metadata 的字典
        """
        all_assets = []
        
        try:
            if self.ts_scanner_path is None:
                if not await self.is_available():
                    return {
                        "assets": [],
                        "metadata": {"engine": "typescript", "targets_scanned": 0},
                        "error": "TypeScript 引擎不可用"
                    }
            
            self.logger.info(f"📜 TypeScript 引擎開始掃描: {len(targets)} 個目標")
            
            # ✅ Bug 修復: 處理所有目標，而非 [:5] 硬編碼限制
            for idx, target in enumerate(targets, 1):
                try:
                    self.logger.debug(f"  掃描目標 {idx}/{len(targets)}: {target}")
                    
                    # 調用 Node.js 掃描器
                    timeout_value = options.get("timeout", 10)  # 使用協調器傳入的值
                    proc = await asyncio.create_subprocess_exec(
                        "node",
                        str(self.ts_scanner_path),
                        target,
                        "--max-depth", str(options.get("max_depth", 3)),
                        "--timeout", str(timeout_value),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # 等待執行完成
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout_value + 10  # 額外 10 秒緩衝
                    )
                    
                    if proc.returncode != 0:
                        self.logger.warning(
                            f"  目標 {target} 掃描失敗 (退出碼 {proc.returncode}): "
                            f"{stderr.decode('utf-8', errors='ignore')}"
                        )
                        continue
                    
                    # ✅ Bug 修復: 健壯的 JSON 解析（應對 console.log 污染）
                    ts_result = self._robust_parse_json(stdout.decode('utf-8', errors='ignore'))
                    
                    # 提取資產
                    if "assets" in ts_result and isinstance(ts_result["assets"], list):
                        for asset_data in ts_result["assets"]:
                            try:
                                asset = Asset(
                                    asset_id=asset_data.get("asset_id", f"ts_{idx}_{len(all_assets)}"),
                                    type=asset_data.get("type", "url"),
                                    value=asset_data.get("value", ""),
                                    parameters=asset_data.get("parameters", []),
                                    has_form=asset_data.get("has_form", False)
                                )
                                all_assets.append(asset)
                            except Exception as e:
                                self.logger.warning(f"  資產解析失敗: {e}")
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"  目標 {target} 掃描超時")
                except Exception as e:
                    self.logger.error(f"  目標 {target} 掃描錯誤: {e}")
            
            self.logger.info(f"✅ TypeScript 引擎完成: {len(all_assets)} 個資產")
            
            return {
                "assets": all_assets,
                "metadata": {
                    "engine": "typescript",
                    "targets_scanned": len(targets),
                    "assets_found": len(all_assets)
                },
                "error": None
            }
            
        except Exception as e:
            error_msg = f"TypeScript 引擎錯誤: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "assets": all_assets,  # 返回已收集的資產
                "metadata": {
                    "engine": "typescript",
                    "targets_scanned": len(targets)
                },
                "error": error_msg
            }
    
    def _robust_parse_json(self, stdout_str: str) -> Dict:
        """健壯的 JSON 解析器，能夠從被日誌污染的輸出中提取有效 JSON
        
        解析策略（按優先級）：
        1. 嘗試解析完整輸出（最快路徑）
        2. 嘗試解析最後一行（通常結構化輸出在最後）
        3. 使用 Regex 尋找最大的 JSON 物件（最暴力但最有效）
        
        Args:
            stdout_str: Node.js 進程的 stdout 輸出
        
        Returns:
            解析後的 JSON 字典，失敗時返回空資產列表
        """
        stdout_str = stdout_str.strip()
        
        if not stdout_str:
            return {"assets": []}
        
        # 策略 1: 嘗試解析完整輸出（最快路徑）
        try:
            return json.loads(stdout_str)
        except json.JSONDecodeError:
            pass  # 繼續嘗試其他策略
        
        # 策略 2: 嘗試解析最後一行（通常結構化輸出會在最後打印）
        lines = stdout_str.split('\n')
        if lines:
            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        
        # 策略 3: 使用 Regex 尋找最大的 JSON 物件
        # 匹配以 { 開頭，以 } 結尾，且包含 "assets" 的最長字串
        try:
            # re.DOTALL 讓 . 可以匹配換行符
            match = re.search(r'(\{.*"assets":.*\})', stdout_str, re.DOTALL)
            if match:
                json_candidate = match.group(1)
                return json.loads(json_candidate)
        except Exception as e:
            self.logger.debug(f"Regex 解析失敗: {e}")
        
        # 所有策略失敗，返回空結果
        self.logger.warning("無法從 TypeScript 輸出中解析 JSON，返回空資產列表")
        return {"assets": []}
