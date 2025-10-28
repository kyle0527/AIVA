#!/usr/bin/env python3
"""
AIVA 環境依賴修復工具
自動設置所需的環境變數和服務依賴
"""
import os
import sys
import subprocess
import time
import requests
from pathlib import Path

class EnvironmentFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.docker_compose_file = self.project_root / "docker" / "docker-compose.yml"
        
    def setup_environment_variables(self):
        """設置必要的環境變數"""
        print("🔧 設置環境變數...")
        
        env_vars = {
            # RabbitMQ 配置
            "AIVA_RABBITMQ_URL": "amqp://guest:guest@localhost:5672/",
            "AIVA_RABBITMQ_USER": "guest",
            "AIVA_RABBITMQ_PASSWORD": "guest",
            "AIVA_RABBITMQ_HOST": "localhost",
            "AIVA_RABBITMQ_PORT": "5672",
            "AIVA_RABBITMQ_VHOST": "/",
            
            # Redis 配置
            "AIVA_REDIS_URL": "redis://localhost:6379/0",
            
            # PostgreSQL 配置
            "AIVA_POSTGRES_HOST": "localhost",
            "AIVA_POSTGRES_PORT": "5432",
            "AIVA_POSTGRES_USER": "aiva",
            "AIVA_POSTGRES_PASSWORD": "aiva",
            "AIVA_POSTGRES_DB": "aiva",
            
            # Neo4j 配置
            "AIVA_NEO4J_URL": "bolt://localhost:7687",
            "AIVA_NEO4J_USER": "neo4j",
            "AIVA_NEO4J_PASSWORD": "password",
            
            # AIVA 特定配置
            "AIVA_MQ_EXCHANGE": "aiva.topic",
            "AIVA_MQ_DLX": "aiva.dlx",
            "AIVA_LOG_LEVEL": "INFO",
            "AIVA_ENVIRONMENT": "development"
        }
        
        # 設置環境變數
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"   ✅ {key}={value}")
            
        # 創建 .env 檔案
        with open(self.env_file, 'w', encoding='utf-8') as f:
            f.write("# AIVA 環境配置\n")
            f.write("# 自動生成於環境修復過程\n\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
                
        print(f"✅ 環境變數已保存到 {self.env_file}")
        
    def check_docker_installed(self):
        """檢查 Docker 是否已安裝"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Docker 已安裝: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker 未安裝或無法訪問")
            return False
            
    def check_docker_compose_installed(self):
        """檢查 Docker Compose 是否已安裝"""
        try:
            result = subprocess.run(["docker", "compose", "version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Docker Compose 已安裝: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker Compose 未安裝或無法訪問")
            return False
            
    def start_docker_services(self):
        """啟動 Docker 服務"""
        if not self.check_docker_installed() or not self.check_docker_compose_installed():
            print("⚠️ Docker 環境不完整，跳過服務啟動")
            return False
            
        print("🚀 啟動 Docker 服務...")
        try:
            # 切換到 docker 目錄
            docker_dir = self.project_root / "docker"
            
            # 啟動服務
            result = subprocess.run(
                ["docker", "compose", "up", "-d"], 
                cwd=docker_dir,
                capture_output=True, 
                text=True, 
                check=True
            )
            
            print("✅ Docker 服務啟動成功")
            print("等待服務就緒...")
            time.sleep(10)  # 等待服務啟動
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker 服務啟動失敗: {e}")
            print(f"錯誤輸出: {e.stderr}")
            return False
            
    def check_service_health(self):
        """檢查服務健康狀態"""
        print("🏥 檢查服務健康狀態...")
        
        services = {
            "RabbitMQ Management": "http://localhost:15672",
            "RabbitMQ AMQP": "amqp://localhost:5672",
            "Redis": "redis://localhost:6379",
            "PostgreSQL": "postgresql://localhost:5432",
            "Neo4j": "http://localhost:7474"
        }
        
        for service_name, url in services.items():
            if service_name == "RabbitMQ Management":
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ {service_name}: 正常")
                    else:
                        print(f"   ⚠️ {service_name}: 異常 (狀態碼 {response.status_code})")
                except Exception as e:
                    print(f"   ❌ {service_name}: 無法連接 - {e}")
            elif service_name == "Neo4j":
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ {service_name}: 正常")
                    else:
                        print(f"   ⚠️ {service_name}: 異常")
                except Exception as e:
                    print(f"   ❌ {service_name}: 無法連接 - {e}")
            else:
                print(f"   🔍 {service_name}: 待測試")
                
    def test_ai_components(self):
        """測試 AI 組件是否可以正常工作"""
        print("🧠 測試 AI 組件...")
        
        try:
            # 測試基礎健康檢查
            result = subprocess.run([
                sys.executable, "health_check.py"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("✅ 健康檢查通過")
                print(result.stdout[-500:])  # 顯示最後500字符
            else:
                print("⚠️ 健康檢查有警告")
                print(result.stderr[-300:] if result.stderr else result.stdout[-300:])
                
        except Exception as e:
            print(f"❌ AI 組件測試失敗: {e}")
            
    def test_ai_functionality(self):
        """測試 AI 功能是否正常"""
        print("🎯 測試 AI 功能...")
        
        try:
            # 測試 AI 功能驗證器
            result = subprocess.run([
                sys.executable, "ai_functionality_validator.py", "--quick"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)
            
            if result.returncode == 0:
                print("✅ AI 功能測試通過")
            else:
                print("⚠️ AI 功能測試有問題")
                print(result.stderr[-200:] if result.stderr else result.stdout[-200:])
                
        except subprocess.TimeoutExpired:
            print("⏰ AI 功能測試超時")
        except Exception as e:
            print(f"❌ AI 功能測試失敗: {e}")
            
    def create_startup_script(self):
        """創建啟動腳本"""
        startup_script = self.project_root / "start_aiva_environment.py"
        
        script_content = '''#!/usr/bin/env python3
"""
AIVA 快速環境啟動腳本
"""
import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 設置環境變數
env_vars = {
    "AIVA_RABBITMQ_URL": "amqp://guest:guest@localhost:5672/",
    "AIVA_RABBITMQ_USER": "guest",
    "AIVA_RABBITMQ_PASSWORD": "guest",
    "AIVA_REDIS_URL": "redis://localhost:6379/0",
    "AIVA_POSTGRES_HOST": "localhost",
    "AIVA_POSTGRES_PORT": "5432",
    "AIVA_POSTGRES_USER": "aiva",
    "AIVA_POSTGRES_PASSWORD": "aiva",
    "AIVA_POSTGRES_DB": "aiva",
    "AIVA_NEO4J_URL": "bolt://localhost:7687",
    "AIVA_NEO4J_USER": "neo4j",
    "AIVA_NEO4J_PASSWORD": "password",
    "AIVA_LOG_LEVEL": "INFO",
    "AIVA_ENVIRONMENT": "development"
}

for key, value in env_vars.items():
    os.environ[key] = value

print("🚀 AIVA 環境已設置完成")
print("現在可以運行 AI 組件了！")

if __name__ == "__main__":
    print("環境變數設置完成，可以執行以下命令測試：")
    print("python health_check.py")
    print("python ai_functionality_validator.py --quick")
    print("python ai_security_test.py --target http://localhost:3000")
'''
        
        with open(startup_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
            
        print(f"✅ 啟動腳本已創建: {startup_script}")
        
    def run_complete_fix(self):
        """執行完整的環境修復流程"""
        print("🔧 開始 AIVA 環境依賴修復...")
        print("=" * 60)
        
        # 1. 設置環境變數
        self.setup_environment_variables()
        print()
        
        # 2. 啟動 Docker 服務
        docker_success = self.start_docker_services()
        print()
        
        # 3. 檢查服務健康狀態
        if docker_success:
            self.check_service_health()
        print()
        
        # 4. 測試 AI 組件
        self.test_ai_components()
        print()
        
        # 5. 測試 AI 功能
        self.test_ai_functionality()
        print()
        
        # 6. 創建啟動腳本
        self.create_startup_script()
        print()
        
        print("🎉 環境修復完成！")
        print("=" * 60)
        print("📋 後續步驟:")
        print("1. 確保 Docker 服務正在運行")
        print("2. 運行 'python start_aiva_environment.py' 設置環境")
        print("3. 測試 AI 組件: 'python health_check.py'")
        print("4. 執行 AI 測試: 'python ai_security_test.py --target http://localhost:3000'")
        
if __name__ == "__main__":
    fixer = EnvironmentFixer()
    fixer.run_complete_fix()