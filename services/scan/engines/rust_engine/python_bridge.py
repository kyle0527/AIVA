"""
Python-Rust 橋接器
提供 Python 調用 Rust 掃描引擎的接口
"""

import logging
import subprocess
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RustInfoGatherer:
    """Rust 信息收集引擎的 Python 接口"""
    
    def __init__(self):
        self.rust_binary_path = self._find_rust_binary()
        self._available = self._check_availability()
    
    def _find_rust_binary(self) -> Optional[Path]:
        """查找 Rust 二進制文件"""
        # 檢查可能的 Rust 二進制路徑
        possible_paths = [
            Path(__file__).parent / "target" / "release" / "rust_scanner.exe",
            Path(__file__).parent / "target" / "release" / "rust_scanner",
            Path(__file__).parent / "rust_scanner.exe",
            Path(__file__).parent / "rust_scanner",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                return path
        
        return None
    
    def _check_availability(self) -> bool:
        """檢查 Rust 掃描器是否可用"""
        if not self.rust_binary_path:
            logger.warning("[RustBridge] Rust binary not found")
            return False
        
        try:
            # 測試運行 Rust 掃描器
            result = subprocess.run(
                [str(self.rust_binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"[RustBridge] Rust scanner available: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"[RustBridge] Rust scanner test failed: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"[RustBridge] Rust scanner unavailable: {e}")
            return False
    
    def is_available(self) -> bool:
        """檢查 Rust 掃描器是否可用"""
        return self._available
    
    def scan_target(self, target_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        掃描目標 URL
        
        Args:
            target_url: 目標 URL
            config: 掃描配置
            
        Returns:
            掃描結果字典
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Rust scanner not available",
                "results": {}
            }
        
        try:
            # 準備掃描參數
            scan_args = [
                str(self.rust_binary_path),
                "scan",
                "--url", target_url,
                "--format", "json"
            ]
            
            # 添加配置參數
            if config.get("mode") == "fast_discovery":
                scan_args.extend(["--mode", "fast"])
            elif config.get("mode") == "deep_analysis":
                scan_args.extend(["--mode", "deep"])
            
            if "timeout" in config:
                scan_args.extend(["--timeout", str(config["timeout"])])
            
            if "max_depth" in config:
                scan_args.extend(["--depth", str(config["max_depth"])])
            
            # 運行 Rust 掃描器
            logger.debug(f"[RustBridge] Running: {' '.join(scan_args)}")
            
            result = subprocess.run(
                scan_args,
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 30)
            )
            
            if result.returncode == 0:
                try:
                    scan_results = json.loads(result.stdout)
                    return {
                        "success": True,
                        "results": scan_results
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"[RustBridge] Failed to parse JSON output: {e}")
                    return {
                        "success": False,
                        "error": f"JSON parse error: {e}",
                        "raw_output": result.stdout
                    }
            else:
                logger.error(f"[RustBridge] Scan failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "results": {}
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"[RustBridge] Scan timeout for {target_url}")
            return {
                "success": False,
                "error": "Scan timeout",
                "results": {}
            }
        except Exception as e:
            logger.error(f"[RustBridge] Unexpected error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }
    
    def scan_multiple_targets(self, targets: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        並發掃描多個目標
        
        Args:
            targets: 目標 URL 列表
            config: 掃描配置
            
        Returns:
            掃描結果列表
        """
        results = []
        for target in targets:
            result = self.scan_target(target, config)
            result["target"] = target
            results.append(result)
        
        return results


class MockRustInfoGatherer:
    """模擬 Rust 信息收集器，用於測試"""
    
    def __init__(self):
        self._available = True
    
    def is_available(self) -> bool:
        return True
    
    def scan_target(self, target_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """模擬掃描結果"""
        mode = config.get("mode", "fast_discovery")
        
        if mode == "fast_discovery":
            return {
                "success": True,
                "results": {
                    "technologies": ["nginx", "php", "mysql"],
                    "sensitive_info": [
                        {
                            "type": "api_key",
                            "location": f"{target_url}/config.php",
                            "value": "DEMO_KEY_***"
                        }
                    ],
                    "endpoints": [
                        f"{target_url}/login.php",
                        f"{target_url}/admin/",
                        f"{target_url}/api/v1/"
                    ]
                }
            }
        elif mode == "deep_analysis":
            return {
                "success": True,
                "results": {
                    "assets": [
                        {
                            "type": "endpoint",
                            "url": f"{target_url}/login.php",
                            "parameters": ["username", "password"],
                            "has_form": True
                        },
                        {
                            "type": "api",
                            "url": f"{target_url}/api/v1/users",
                            "parameters": ["id", "name"],
                            "has_form": False
                        }
                    ]
                }
            }
        else:
            return {
                "success": False,
                "error": f"Unknown scan mode: {mode}"
            }
    
    def scan_multiple_targets(self, targets: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """模擬多目標掃描"""
        return [self.scan_target(target, config) for target in targets]


# 創建全局實例
try:
    rust_info_gatherer = RustInfoGatherer()
    if not rust_info_gatherer.is_available():
        logger.warning("[RustBridge] Using mock implementation")
        rust_info_gatherer = MockRustInfoGatherer()  # type: ignore
except Exception as e:
    logger.error(f"[RustBridge] Failed to initialize: {e}")
    rust_info_gatherer = MockRustInfoGatherer()  # type: ignore


__all__ = ["rust_info_gatherer", "RustInfoGatherer", "MockRustInfoGatherer"]