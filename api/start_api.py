# -*- coding: utf-8 -*-
"""
AIVA API 啟動腳本

快速啟動 AIVA Security Platform API 服務的便利腳本。
支援開發和生產環境的不同配置。
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='AIVA Security Platform API Launcher')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (default: 1)')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], 
                       help='Log level (default: info)')
    parser.add_argument('--install-deps', action='store_true', help='Install required dependencies')
    
    args = parser.parse_args()
    
    # 檢查並安裝依賴
    if args.install_deps or not check_dependencies():
        print("🔧 Installing API dependencies...")
        install_dependencies()
    
    # 設置環境變數
    setup_environment()
    
    # 構建 uvicorn 命令
    cmd = [
        sys.executable, '-m', 'uvicorn',
        'main:app',
        '--host', args.host,
        '--port', str(args.port),
        '--log-level', args.log_level
    ]
    
    if args.reload:
        cmd.append('--reload')
    
    if args.workers > 1 and not args.reload:
        cmd.extend(['--workers', str(args.workers)])
    
    # 輸出啟動信息
    print("🚀 Starting AIVA Security Platform API")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔒 Default credentials:")
    print(f"   - Admin: admin / aiva-admin-2025")
    print(f"   - User: user / aiva-user-2025")
    print(f"   - Viewer: viewer / aiva-viewer-2025")
    print("-" * 50)
    
    # 切換到 API 目錄
    api_dir = Path(__file__).parent
    os.chdir(api_dir)
    
    # 啟動服務
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⛔ API server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

def check_dependencies():
    """檢查必需的依賴是否已安裝"""
    required_packages = ['fastapi', 'uvicorn', 'PyJWT']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            return False
    
    return True

def install_dependencies():
    """安裝 API 依賴"""
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if requirements_file.exists():
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        try:
            subprocess.run(cmd, check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        print("⚠️ requirements.txt not found, installing basic dependencies...")
        basic_deps = ['fastapi', 'uvicorn[standard]', 'PyJWT', 'httpx']
        cmd = [sys.executable, '-m', 'pip', 'install'] + basic_deps
        try:
            subprocess.run(cmd, check=True)
            print("✅ Basic dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install basic dependencies: {e}")
            sys.exit(1)

def setup_environment():
    """設置環境變數"""
    # 確保 Python 路徑包含項目根目錄
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 設置環境變數
    os.environ.setdefault('AIVA_API_ENV', 'development')
    os.environ.setdefault('LOG_LEVEL', 'INFO')

if __name__ == '__main__':
    main()