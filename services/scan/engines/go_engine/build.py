#!/usr/bin/env python3
"""
Go 掃描器構建腳本

根據 AIVA Common README 規範，提供統一的 Go 掃描器構建機制。
支持 Windows、Linux 和 macOS 跨平台構建。
"""

import asyncio
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

# 設置日誌記錄器 (簡化版，避免模組依賴)
def get_logger(name: str) -> logging.Logger:
    """取得日誌記錄器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# 常量定義 (依照 aiva_common README 規範)
SCANNER_DIRS = ["ssrf_scanner", "cspm_scanner", "sca_scanner"]
BUILD_OUTPUT_DIR = "bin"
WINDOWS_EXT = ".exe"
GO_MODULE_PATH = "aiva/scan/go_scanners"

# 支援的目標平台
BUILD_TARGETS = {
    "windows": {"GOOS": "windows", "GOARCH": "amd64", "ext": ".exe"},
    "linux": {"GOOS": "linux", "GOARCH": "amd64", "ext": ""},
    "darwin": {"GOOS": "darwin", "GOARCH": "amd64", "ext": ""},
    "arm64": {"GOOS": "linux", "GOARCH": "arm64", "ext": ""}
}


class GoScannerBuilder:
    """Go 掃描器構建器"""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """初始化構建器
        
        Args:
            root_dir: Go 引擎根目錄，預設為當前腳本所在目錄
        """
        self.root_dir = root_dir or Path(__file__).parent
        self.build_dir = self.root_dir / BUILD_OUTPUT_DIR
        logger.info(f"[Build] Initializing Go scanner builder in: {self.root_dir}")
        
    async def check_prerequisites(self) -> bool:
        """檢查構建前置條件"""
        try:
            # 檢查 Go 是否已安裝
            result = await self._run_command(["go", "version"])
            if result.returncode != 0:
                logger.error("[Build] Go is not installed or not in PATH")
                return False
            
            logger.info(f"[Build] Go version: {result.stdout.strip()}")
            
            # 檢查 go.mod 是否存在
            go_mod = self.root_dir / "go.mod"
            if not go_mod.exists():
                logger.error(f"[Build] go.mod not found in {self.root_dir}")
                return False
            
            # 檢查掃描器目錄
            missing_dirs = []
            for scanner_dir in SCANNER_DIRS:
                scanner_path = self.root_dir / scanner_dir
                if not scanner_path.exists():
                    missing_dirs.append(scanner_dir)
            
            if missing_dirs:
                logger.warning(f"[Build] Missing scanner directories: {missing_dirs}")
            
            logger.info("[Build] Prerequisites check completed successfully")
            return True
            
        except Exception as exc:
            logger.error(f"[Build] Prerequisites check failed: {exc}")
            return False
    
    async def build_all_scanners(
        self, 
        target_platform: str = "auto",
        clean_first: bool = True
    ) -> bool:
        """構建所有掃描器
        
        Args:
            target_platform: 目標平台 (auto/windows/linux/darwin/arm64)
            clean_first: 是否先清理構建目錄
            
        Returns:
            是否所有掃描器都構建成功
        """
        logger.info(f"[Build] Starting build for platform: {target_platform}")
        
        if not await self.check_prerequisites():
            return False
        
        if clean_first:
            self.clean_build_dir()
        
        # 確定構建目標
        if target_platform == "auto":
            target_platform = self._detect_platform()
        
        if target_platform not in BUILD_TARGETS:
            logger.error(f"[Build] Unsupported platform: {target_platform}")
            return False
        
        # 設置環境變數
        build_env = self._setup_build_env(target_platform)
        
        # 創建構建目錄
        self.build_dir.mkdir(exist_ok=True)
        
        # 依次構建每個掃描器
        success_count = 0
        for scanner_name in SCANNER_DIRS:
            try:
                if await self._build_scanner(scanner_name, build_env):
                    success_count += 1
                    logger.info(f"[Build] ✅ {scanner_name} built successfully")
                else:
                    logger.error(f"[Build] ❌ {scanner_name} build failed")
            except Exception as exc:
                logger.error(f"[Build] ❌ {scanner_name} build error: {exc}")
        
        total_scanners = len(SCANNER_DIRS)
        logger.info(f"[Build] Build completed: {success_count}/{total_scanners} scanners built successfully")
        
        return success_count == total_scanners
    
    async def build_scanner(self, scanner_name: str, target_platform: str = "auto") -> bool:
        """構建單一掃描器
        
        Args:
            scanner_name: 掃描器名稱 (ssrf_scanner/cspm_scanner/sca_scanner)
            target_platform: 目標平台
            
        Returns:
            是否構建成功
        """
        if scanner_name not in SCANNER_DIRS:
            logger.error(f"[Build] Unknown scanner: {scanner_name}")
            return False
        
        if not await self.check_prerequisites():
            return False
        
        if target_platform == "auto":
            target_platform = self._detect_platform()
        
        build_env = self._setup_build_env(target_platform)
        self.build_dir.mkdir(exist_ok=True)
        
        return await self._build_scanner(scanner_name, build_env)
    
    def clean_build_dir(self) -> None:
        """清理構建目錄"""
        if self.build_dir.exists():
            import shutil
            shutil.rmtree(self.build_dir)
            logger.info(f"[Build] Cleaned build directory: {self.build_dir}")
        
        # 同時清理各掃描器目錄下的可執行檔
        for scanner_name in SCANNER_DIRS:
            scanner_dir = self.root_dir / scanner_name
            if scanner_dir.exists():
                for exe_file in scanner_dir.glob("worker*"):
                    if exe_file.is_file() and (exe_file.suffix == ".exe" or exe_file.stat().st_mode & 0o111):
                        exe_file.unlink()
                        logger.debug(f"[Build] Removed old executable: {exe_file}")
    
    async def _build_scanner(self, scanner_name: str, build_env: dict) -> bool:
        """構建單一掃描器的內部實現"""
        scanner_dir = self.root_dir / scanner_name
        if not scanner_dir.exists():
            logger.warning(f"[Build] Scanner directory not found: {scanner_dir}")
            return False
        
        # 檢查 main.go 是否存在
        main_go = scanner_dir / "main.go"
        if not main_go.exists():
            logger.warning(f"[Build] main.go not found in {scanner_dir}")
            return False
        
        # 確定輸出檔名
        target_platform = build_env.get("GOOS", "linux")
        ext = BUILD_TARGETS[target_platform]["ext"] if target_platform in BUILD_TARGETS else ""
        if not ext and target_platform == "windows":
            ext = WINDOWS_EXT
        
        output_name = f"worker{ext}"
        output_path = scanner_dir / output_name
        
        # 構建命令
        build_cmd = [
            "go", "build",
            "-o", str(output_path),
            "-ldflags", "-s -w",  # 減少二進制大小
            "./main.go"
        ]
        
        logger.info(f"[Build] Building {scanner_name}...")
        logger.debug(f"[Build] Command: {' '.join(build_cmd)}")
        logger.debug(f"[Build] Working dir: {scanner_dir}")
        logger.debug(f"[Build] Environment: {build_env}")
        
        try:
            # 執行構建
            result = await self._run_command(
                build_cmd, 
                cwd=scanner_dir, 
                env=build_env
            )
            
            if result.returncode == 0:
                logger.info(f"[Build] ✅ {scanner_name} built successfully: {output_path}")
                
                # 驗證可執行檔是否存在且有執行權限
                if output_path.exists():
                    if platform.system() != "Windows":
                        # Unix-like 系統設置執行權限
                        os.chmod(output_path, 0o755)
                    
                    file_size = output_path.stat().st_size
                    logger.debug(f"[Build] Output file size: {file_size:,} bytes")
                    return True
                else:
                    logger.error(f"[Build] Build succeeded but output file not found: {output_path}")
                    return False
            else:
                logger.error(f"[Build] ❌ {scanner_name} build failed:")
                logger.error(f"[Build] stdout: {result.stdout}")
                logger.error(f"[Build] stderr: {result.stderr}")
                return False
                
        except Exception as exc:
            logger.error(f"[Build] ❌ {scanner_name} build exception: {exc}")
            return False
    
    async def _run_command(
        self, 
        command: list[str], 
        cwd: Optional[Path] = None, 
        env: Optional[dict] = None
    ) -> subprocess.CompletedProcess:
        """異步執行系統命令"""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode or 0,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace')
        )
    
    def _detect_platform(self) -> str:
        """自動檢測當前平台"""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "darwin"
        elif system == "linux":
            return "linux"
        else:
            logger.warning(f"[Build] Unknown platform: {system}, defaulting to linux")
            return "linux"
    
    def _setup_build_env(self, target_platform: str) -> dict:
        """設置構建環境變數"""
        build_config = BUILD_TARGETS[target_platform]
        env = {
            "GOOS": build_config["GOOS"],
            "GOARCH": build_config["GOARCH"],
            "CGO_ENABLED": "0",  # 靜態鏈接，提高可移植性
        }
        
        logger.debug(f"[Build] Build environment: {env}")
        return env


async def main():
    """主函數 - 命令行介面"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Go Scanner Builder")
    parser.add_argument(
        "--scanner", 
        choices=SCANNER_DIRS + ["all"], 
        default="all",
        help="要構建的掃描器 (預設: all)"
    )
    parser.add_argument(
        "--platform", 
        choices=["auto"] + list(BUILD_TARGETS.keys()), 
        default="auto",
        help="目標平台 (預設: auto)"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="構建前清理舊檔案"
    )
    parser.add_argument(
        "--check", 
        action="store_true",
        help="只檢查前置條件，不執行構建"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="啟用詳細輸出"
    )
    
    args = parser.parse_args()
    
    # 設置日誌級別
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    builder = GoScannerBuilder()
    
    try:
        if args.check:
            # 只檢查前置條件
            success = await builder.check_prerequisites()
            sys.exit(0 if success else 1)
        
        if args.scanner == "all":
            # 構建所有掃描器
            success = await builder.build_all_scanners(
                target_platform=args.platform,
                clean_first=args.clean
            )
        else:
            # 構建單一掃描器
            if args.clean:
                builder.clean_build_dir()
            
            success = await builder.build_scanner(
                scanner_name=args.scanner,
                target_platform=args.platform
            )
        
        if success:
            logger.info("[Build] 🎉 Build completed successfully!")
            sys.exit(0)
        else:
            logger.error("[Build] 💥 Build failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("[Build] Build cancelled by user")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"[Build] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())