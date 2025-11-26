#!/usr/bin/env python3
"""
AIVA 離線環境配置器
為 AI 組件提供基本的離線運行支援
"""
import os
import sys
from pathlib import Path

def setup_offline_environment():
    """設置離線環境配置"""
    print("🔧 設置 AIVA 離線環境...")
    
    # 設置基本環境變數 (簡化版)
    env_vars = {
        # 核心配置
        "ENVIRONMENT": "offline",
        "LOG_LEVEL": "INFO",
        
        # RabbitMQ 配置 (離線模式)
        "RABBITMQ_URL": "memory://localhost",
        "RABBITMQ_USER": "offline",
        "RABBITMQ_PASSWORD": "offline",
        
        # 資料庫配置
        "POSTGRES_HOST": "localhost",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   ✅ {key}={value}")
        
    # 創建 .env 檔案
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write("# AIVA 離線環境配置\n")
        f.write("# 支援基本 AI 功能而不需要外部服務\n\n")
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
            
    print(f"✅ 離線環境配置已保存到 {env_file}")
    return True

def patch_rabbitmq_dependency():
    """修補 RabbitMQ 依賴問題"""
    print("🔧 修補 RabbitMQ 依賴...")
    
    # 修改統一配置以支援離線模式
    config_file = Path(__file__).parent / "services" / "aiva_common" / "config" / "unified_config.py"
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 檢查是否已經修補
        if "ENVIRONMENT" not in content or "offline" not in content:
            # 在 _get_rabbitmq_url 方法中添加離線模式支援
            old_method = '''    def _get_rabbitmq_url(self):
        """獲取 RabbitMQ URL"""
        url = os.getenv("RABBITMQ_URL")
        if url:
            return url
            
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = os.getenv("RABBITMQ_PORT", "5672")
        user = os.getenv("RABBITMQ_USER")
        password = os.getenv("RABBITMQ_PASSWORD")
        vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        if not user or not password:
            raise ValueError("RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set")
            
        return f"amqp://{user}:{password}@{host}:{port}{vhost}"'''
            
            new_method = '''    def _get_rabbitmq_url(self):
        """獲取 RabbitMQ URL"""
        # 檢查是否為離線模式
        if os.getenv("ENVIRONMENT") == "offline":
            return "memory://localhost"
            
        url = os.getenv("RABBITMQ_URL")
        if url:
            return url
            
        host = os.getenv("RABBITMQ_HOST", "localhost")
        port = os.getenv("RABBITMQ_PORT", "5672")
        user = os.getenv("RABBITMQ_USER")
        password = os.getenv("RABBITMQ_PASSWORD")
        vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        if not user or not password:
            # 離線模式回退
            if os.getenv("ENVIRONMENT") == "offline":
                return "memory://localhost"
            raise ValueError("RABBITMQ_URL or RABBITMQ_USER/RABBITMQ_PASSWORD must be set")
            
        return f"amqp://{user}:{password}@{host}:{port}{vhost}"'''
            
            if old_method in content:
                content = content.replace(old_method, new_method)
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                print("✅ 統一配置已修補，支援離線模式")
            else:
                print("⚠️ 無法找到預期的配置方法，跳過修補")
        else:
            print("✅ 配置已經支援離線模式")
    else:
        print("❌ 找不到統一配置檔案")
        
def create_offline_launcher():
    """創建離線啟動器"""
    print("🚀 創建離線啟動器...")
    
    launcher_content = '''#!/usr/bin/env python3
"""
AIVA 離線模式啟動器
"""
import os
import sys
from pathlib import Path

# 設置項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_offline_env():
    """設置離線環境"""
    env_vars = {
        "RABBITMQ_URL": "memory://localhost",
        "RABBITMQ_USER": "offline",
        "RABBITMQ_PASSWORD": "offline",
        "ENVIRONMENT": "offline",
        "LOG_LEVEL": "INFO",
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🔧 離線環境已設置")

def main():
    setup_offline_env()
    
    print("🚀 AIVA 離線模式啟動")
    print("=" * 40)
    print("✅ 環境變數已設置")
    print("📋 可用功能:")
    print("  - AI 組件探索")
    print("  - 學習成效分析")  
    print("  - 基礎安全掃描")
    print("  - 系統健康檢查")
    print()
    print("🔧 建議的測試命令:")
    print("  python health_check.py")
    print("  python ai_component_explorer.py")
    print("  python ai_system_explorer_v3.py --help")
    print()

if __name__ == "__main__":
    main()
'''
    
    launcher_file = Path(__file__).parent / "launch_offline_mode.py"
    with open(launcher_file, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
        
    print(f"✅ 離線啟動器已創建: {launcher_file}")

def main():
    """主要修復流程"""
    print("🔧 AIVA 環境依賴修復工具")
    print("=" * 50)
    
    # 1. 設置離線環境
    setup_offline_environment()
    print()
    
    # 2. 修補 RabbitMQ 依賴
    patch_rabbitmq_dependency()
    print()
    
    # 3. 創建離線啟動器
    create_offline_launcher()
    print()
    
    print("🎉 環境修復完成！")
    print("=" * 50)
    print("📋 使用說明:")
    print("1. 運行離線模式: python launch_offline_mode.py")
    print("2. 測試健康狀態: python health_check.py")
    print("3. 探索 AI 組件: python ai_component_explorer.py")
    print("4. 執行學習分析: python ai_system_explorer_v3.py --detailed")
    print()
    print("⚠️ 注意: 離線模式下部分功能可能受限")
    print("💡 建議: 後續可配置完整的 Docker 環境獲得全部功能")

if __name__ == "__main__":
    main()