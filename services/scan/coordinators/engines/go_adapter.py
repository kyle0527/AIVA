"""
Go 掃描引擎適配器 - v2.1

實現與 Go 二進制掃描器的進程間通信，支持高並發掃描。

通信協議:
- 輸入: JSON 格式 ScanRequest (via stdin)
- 輸出: JSON 格式 ScanResult (via stdout)
- 錯誤: stderr

優勢:
- 極高的並發性能（Goroutines）
- 低延遲網絡請求
- 適合大規模掃描
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from services.aiva_common.schemas import Asset
from .base import BaseScannerAdapter


class GoAdapter(BaseScannerAdapter):
    """Go 引擎適配器
    
    通過子進程調用 Go 編譯的掃描器，實現高並發掃描。
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.go_scanner_path = None
    
    async def is_available(self) -> bool:
        """檢查 Go 引擎是否可用"""
        try:
            # 檢查 Go 掃描器二進制文件是否存在
            # 可能的位置：
            # 1. services/scan/engines/go_engine/scanner
            # 2. services/scan/engines/go_engine/scanner.exe (Windows)
            
            # 從 coordinators/engines/go_adapter.py 到 scan/engines/go_engine
            # 需要: ../../engines/go_engine
            base_path = Path(__file__).parent.parent.parent / "engines" / "go_engine"
            
            possible_paths = [
                base_path / "scanner",
                base_path / "scanner.exe",
                base_path / "bin" / "scanner",
                base_path / "bin" / "scanner.exe"
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.go_scanner_path = path
                    self.logger.info(f"找到 Go 掃描器: {path}")
                    return True
            
            self.logger.warning(f"Go 掃描器二進制文件不存在，檢查路徑: {base_path}")
            return False
            
        except Exception as e:
            self.logger.warning(f"Go 引擎檢查失敗: {e}")
            return False
    
    async def scan(
        self,
        targets: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行 Go 引擎掃描
        
        Args:
            targets: 目標 URL 列表
            options: 掃描配置
        
        Returns:
            包含 assets 和 metadata 的字典
        """
        try:
            if self.go_scanner_path is None:
                if not await self.is_available():
                    return {
                        "assets": [],
                        "metadata": {"engine": "go", "targets_scanned": 0},
                        "error": "Go 引擎不可用"
                    }
            
            self.logger.info(f"🚀 Go 引擎開始掃描: {len(targets)} 個目標")
            
            # 準備輸入數據(嚴格符合 Go USAGE_GUIDE.md 要求)
            # Go 使用指南只要求: scan_id, targets, concurrency, timeout
            scan_input = {
                "scan_id": options.get("scan_id", f"scan_{int(time.time())}"),
                "targets": targets,
                "concurrency": options.get("concurrency", 5),  # 使用指南建議值
                "timeout": options.get("timeout", 10),  # 使用指南建議值
            }
            
            input_json = json.dumps(scan_input)
            
            # 調用 Go 掃描器
            proc = await asyncio.create_subprocess_exec(
                str(self.go_scanner_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 發送配置並等待結果
            timeout_value = options.get("timeout", 10)
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json.encode('utf-8')),
                timeout=timeout_value + 10  # 額外 10 秒緩衝
            )
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"Go 掃描器執行失敗: {error_msg}")
                return {
                    "assets": [],
                    "metadata": {"engine": "go", "targets_scanned": len(targets)},
                    "error": error_msg
                }
            
            # 解析 JSON 輸出（符合 Go Scanner 的 ScanResult 格式）
            try:
                result = json.loads(stdout.decode('utf-8', errors='ignore'))
            except json.JSONDecodeError as e:
                self.logger.error(f"Go 輸出解析失敗: {e}")
                return {
                    "assets": [],
                    "metadata": {"engine": "go", "targets_scanned": len(targets)},
                    "error": f"JSON 解析失敗: {str(e)}"
                }
            
            # 提取資產（已經是標準 Asset 格式）
            assets = result.get("assets", [])
            
            # 提取元數據
            metadata = {
                "engine": "go",
                "scanner_type": result.get("scanner_type", "ssrf"),
                "scan_id": result.get("scan_id", ""),
                "status": result.get("status", "unknown"),
                "targets_scanned": result.get("targets_scanned", len(targets)),
                "assets_found": len(assets),
                "execution_time": result.get("execution_time", 0),
                "requests_made": result.get("requests_made", 0),
                "success_count": result.get("success_count", 0),
                "failure_count": result.get("failure_count", 0)
            }
            
            # 提取錯誤和警告
            errors = result.get("errors", [])
            warnings = result.get("warnings", [])
            
            if errors:
                self.logger.warning(f"Go 掃描器報告 {len(errors)} 個錯誤")
            if warnings:
                self.logger.info(f"Go 掃描器報告 {len(warnings)} 個警告")
            
            self.logger.info(
                f"✅ Go 引擎完成: {len(assets)} 個資產, "
                f"{metadata.get('execution_time', 0):.1f}s, "
                f"成功 {metadata.get('success_count', 0)}/{metadata.get('targets_scanned', 0)}"
            )
            
            return {
                "assets": assets,
                "metadata": metadata,
                "error": None,
                "errors": errors,
                "warnings": warnings
            }
            
        except asyncio.TimeoutError:
            error_msg = "Go 掃描器執行超時"
            self.logger.error(error_msg)
            return {
                "assets": [],
                "metadata": {"engine": "go", "targets_scanned": len(targets)},
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Go 引擎錯誤: {str(e)}"
            self.logger.error(error_msg)
            return {
                "assets": [],
                "metadata": {"engine": "go", "targets_scanned": len(targets)},
                "error": error_msg
            }
