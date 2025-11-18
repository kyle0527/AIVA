"""
Rust Info Gatherer Python Bridge

This module provides Python interface for the Rust-based info gatherer.
It bridges between Python modules and the Rust scanner implementation.
"""

import logging
import subprocess
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RustInfoGatherer:
    """
    Python bridge to Rust Info Gatherer
    
    This class provides a Python interface to communicate with the
    Rust-based information gathering scanner.
    """
    
    def __init__(self):
        self.rust_binary_path = self._find_rust_binary()
        self.initialized = self.rust_binary_path is not None
        
        if self.initialized:
            logger.info("Rust Info Gatherer bridge initialized successfully")
        else:
            logger.warning("Rust Info Gatherer binary not found - functionality will be limited")
    
    def _find_rust_binary(self) -> Optional[Path]:
        """Find the compiled Rust binary"""
        # Look for the binary in the target directory
        base_dir = Path(__file__).parent.parent
        
        # Try release build first, then debug build
        possible_paths = [
            base_dir / "target" / "release" / "info_gatherer_rust.exe",
            base_dir / "target" / "release" / "info_gatherer_rust",
            base_dir / "target" / "debug" / "info_gatherer_rust.exe", 
            base_dir / "target" / "debug" / "info_gatherer_rust",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return None
    
    def is_available(self) -> bool:
        """Check if the Rust scanner is available"""
        return self.initialized
    
    def scan_target(
        self, 
        target_url: str,
        scan_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scan a target using the Rust info gatherer
        
        Args:
            target_url: The URL to scan
            scan_config: Optional configuration for the scan
            
        Returns:
            Scan results as a dictionary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "Rust Info Gatherer not available",
                "results": []
            }
        
        try:
            # Prepare command arguments
            cmd = [str(self.rust_binary_path), "--url", target_url]
            
            if scan_config:
                # Add configuration as JSON
                config_json = json.dumps(scan_config)
                cmd.extend(["--config", config_json])
            
            # Execute the Rust scanner
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Parse JSON output from Rust scanner
                try:
                    scan_results = json.loads(result.stdout)
                    return {
                        "success": True,
                        "results": scan_results,
                        "stderr": result.stderr
                    }
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Rust scanner output: {e}")
                    return {
                        "success": False,
                        "error": f"Invalid JSON output: {e}",
                        "raw_output": result.stdout
                    }
            else:
                logger.error(f"Rust scanner failed with code {result.returncode}")
                return {
                    "success": False,
                    "error": f"Scanner failed: {result.stderr}",
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Rust scanner timeout")
            return {
                "success": False,
                "error": "Scanner timeout after 5 minutes"
            }
        except Exception as e:
            logger.error(f"Unexpected error running Rust scanner: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def get_scanner_info(self) -> Dict[str, Any]:
        """Get information about the Rust scanner"""
        if not self.initialized:
            return {
                "available": False,
                "error": "Scanner not available"
            }
        
        try:
            result = subprocess.run(
                [str(self.rust_binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "available": True,
                "binary_path": str(self.rust_binary_path),
                "version_info": result.stdout.strip() if result.returncode == 0 else "Unknown"
            }
        except Exception as e:
            return {
                "available": True,
                "binary_path": str(self.rust_binary_path),
                "error": f"Could not get version: {str(e)}"
            }

# Create a default instance for easy importing
rust_info_gatherer = RustInfoGatherer()

# Export main classes and functions
__all__ = ['RustInfoGatherer', 'rust_info_gatherer']