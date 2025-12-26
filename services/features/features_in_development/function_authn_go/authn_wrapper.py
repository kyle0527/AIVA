"""
function_authn_go - Go 認證測試引擎的 Python 包裝器

根據 SIMPLE_ARCHITECTURE.md 核心理念：
- Go 模組就用 Go，不需要 Python 回退
- 未編譯就報錯，提供明確的編譯指引
- 有問題就修復，不繞過問題

Usage:
    from services.features.features_in_development.function_authn_go.authn_wrapper import scan_authentication
    
    result = scan_authentication(
        target="https://example.com/login",
        options={"username": "admin"}
    )
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# 日誌
try:
    from services.aiva_common.utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def _find_go_binary() -> Optional[Path]:
    """尋找編譯好的 Go 二進制文件"""
    current_dir = Path(__file__).parent
    
    # 可能的路徑
    possible_paths = [
        current_dir / "bin" / "authn-worker.exe",  # Windows
        current_dir / "bin" / "authn-worker",      # Linux/Mac
        current_dir / "cmd" / "worker" / "main.exe",
        current_dir / "cmd" / "worker" / "main",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            logger.info(f"✅ 找到 Go 二進制文件: {path}")
            return path
    
    logger.error("❌ 未找到編譯好的 Go 二進制文件")
    return None


def _check_go_availability() -> bool:
    """檢查 Go 引擎是否可用"""
    go_binary = _find_go_binary()
    
    if go_binary and go_binary.exists():
        logger.info("✅ Go 認證測試引擎可用")
        return True
    else:
        logger.error(
            "❌ Go 認證測試引擎不可用\n"
            "請編譯 Go 代碼：\n"
            "  cd services/features/features_in_development/function_authn_go\n"
            "  go build -o bin/authn-worker.exe cmd/worker/main.go\n"
            "\n"
            "或參考 BUILD_GUIDE.md 文檔"
        )
        return False


def scan_authentication(
    target: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    執行認證漏洞掃描（同步接口）
    
    這是功能模組的主要接口，提供同步調用方式。
    AI Commander 將使用 asyncio.to_thread() 來包裝此方法。
    
    Args:
        target: 登入端點 URL
        options: 掃描選項（可選）
            - username: str - 測試用戶名（默認: "admin"）
            - test_types: List[str] - 測試類型列表
                * "weak_password" - 弱密碼測試
                * "default_credentials" - 默認憑證測試
                * "password_spray" - 密碼噴灑測試
                * "credential_stuffing" - 憑證填充測試
            - timeout: int - 超時時間（秒，默認: 30）
            - wordlist: str - 密碼字典文件路徑
            
    Returns:
        Dict: 掃描結果
            {
                "vulnerabilities": [
                    {
                        "type": "weak_password",
                        "severity": "high",
                        "confidence": "high",
                        "username": "admin",
                        "password": "admin123",
                        "description": "發現弱密碼",
                        "impact": "攻擊者可以使用此憑證登入系統",
                        "remediation": "使用強密碼策略"
                    }
                ],
                "scan_info": {
                    "target": "https://example.com/login",
                    "tests_performed": ["weak_password", "default_credentials"],
                    "duration": 12.5,
                    "timestamp": "2025-12-17T10:00:00",
                    "go_engine_version": "1.0.0"
                },
                "total_vulnerabilities": 1
            }
            
    Raises:
        RuntimeError: 當 Go 引擎未編譯時
        subprocess.CalledProcessError: 當 Go 程序執行失敗時
        
    Example:
        >>> result = scan_authentication(
        ...     "https://example.com/login",
        ...     {"username": "admin", "test_types": ["weak_password"]}
        ... )
        >>> print(f"Found {result['total_vulnerabilities']} vulnerabilities")
    """
    start_time = datetime.now()
    
    # 默認選項
    if options is None:
        options = {}
    
    # 檢查 Go 引擎是否可用
    go_binary = _find_go_binary()
    if not go_binary:
        raise RuntimeError(
            "Go 認證測試引擎未編譯！\n"
            "請執行以下命令編譯：\n"
            "  cd services/features/features_in_development/function_authn_go\n"
            "  go build -o bin/authn-worker.exe cmd/worker/main.go\n"
            "\n"
            "或參考 BUILD_GUIDE.md 文檔獲取詳細指引"
        )
    
    # 提取參數
    username = options.get("username", "admin")
    test_types = options.get("test_types", ["weak_password", "default_credentials"])
    timeout = options.get("timeout", 30)
    wordlist = options.get("wordlist", "")
    
    # 構建命令
    cmd = [
        str(go_binary),
        "scan",
        "--target", target,
        "--username", username,
        "--timeout", str(timeout),
        "--output-format", "json"
    ]
    
    # 添加測試類型
    for test_type in test_types:
        cmd.extend(["--test-type", test_type])
    
    # 添加字典文件（如果指定）
    if wordlist:
        cmd.extend(["--wordlist", wordlist])
    
    logger.info(f"🎯 執行認證測試: {target}")
    logger.debug(f"命令: {' '.join(cmd)}")
    
    try:
        # 執行 Go 程序
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,  # 額外 10 秒緩衝
            check=True
        )
        
        # 解析 JSON 輸出
        try:
            scan_result = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"解析 Go 輸出失敗: {e}")
            logger.error(f"原始輸出: {result.stdout}")
            raise ValueError(f"Go 引擎輸出格式錯誤: {e}")
        
        # 添加執行時間
        duration = (datetime.now() - start_time).total_seconds()
        if "scan_info" not in scan_result:
            scan_result["scan_info"] = {}
        scan_result["scan_info"]["duration"] = duration
        scan_result["scan_info"]["timestamp"] = start_time.isoformat()
        
        # 計算漏洞總數
        vulnerabilities = scan_result.get("vulnerabilities", [])
        scan_result["total_vulnerabilities"] = len(vulnerabilities)
        
        logger.info(
            f"✅ 認證測試完成: 發現 {len(vulnerabilities)} 個漏洞 "
            f"(執行時間: {duration:.2f}s)"
        )
        
        return scan_result
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ 認證測試超時: {timeout}秒")
        raise RuntimeError(f"認證測試超時（{timeout}秒）")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Go 程序執行失敗: {e}")
        logger.error(f"標準輸出: {e.stdout}")
        logger.error(f"標準錯誤: {e.stderr}")
        
        # 嘗試從錯誤輸出中提取有用信息
        error_msg = e.stderr or e.stdout or str(e)
        raise RuntimeError(f"Go 認證測試引擎執行失敗: {error_msg}")
    
    except Exception as e:
        logger.error(f"❌ 認證測試失敗: {e}", exc_info=True)
        raise


def get_engine_info() -> Dict[str, Any]:
    """
    獲取 Go 引擎信息
    
    Returns:
        Dict: 引擎信息
            {
                "available": bool,
                "binary_path": str or None,
                "version": str or None
            }
    """
    go_binary = _find_go_binary()
    
    info = {
        "available": go_binary is not None,
        "binary_path": str(go_binary) if go_binary else None,
        "version": None
    }
    
    # 嘗試獲取版本
    if go_binary:
        try:
            result = subprocess.run(
                [str(go_binary), "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()
        except Exception as e:
            logger.warning(f"無法獲取 Go 引擎版本: {e}")
    
    return info


# 測試函數
if __name__ == "__main__":
    print("🔍 檢查 Go 引擎狀態...")
    info = get_engine_info()
    print(f"✅ 可用: {info['available']}")
    print(f"📁 路徑: {info['binary_path']}")
    print(f"📌 版本: {info['version']}")
    
    if info['available']:
        print("\n🎯 執行測試掃描...")
        try:
            result = scan_authentication(
                target="https://httpbin.org/basic-auth/user/pass",
                options={"username": "user", "test_types": ["weak_password"]}
            )
            print(f"✅ 測試完成: 發現 {result['total_vulnerabilities']} 個漏洞")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ 測試失敗: {e}")
    else:
        print("\n❌ Go 引擎不可用，請先編譯")
        print("參考 BUILD_GUIDE.md 獲取編譯指引")
