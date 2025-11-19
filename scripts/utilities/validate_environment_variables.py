#!/usr/bin/env python3
"""
AIVA 環境變數驗證工具
確保所有服務使用一致的環境變數配置
以 Docker Compose Production 版本為標準
"""

import os
import sys
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvironmentStandard:
    """環境變數標準定義"""
    name: str
    required: bool
    default_value: Optional[str]
    description: str
    production_value: Optional[str] = None
    docker_value: Optional[str] = None


# AIVA 標準環境變數定義
AIVA_ENV_STANDARDS = {
    # 資料庫配置
    "DATABASE_URL": EnvironmentStandard(
        name="DATABASE_URL",
        required=True,
        default_value=None,
        description="資料庫連接 URL",
        production_value="postgresql://aiva:aiva_secure_password@postgres:5432/aiva",
        docker_value="postgresql://aiva:aiva_secure_password@postgres:5432/aiva"
    ),
    "POSTGRES_HOST": EnvironmentStandard(
        name="POSTGRES_HOST",
        required=False,
        default_value="localhost",
        description="PostgreSQL 主機",
        production_value="postgres",
        docker_value="postgres"
    ),
    "POSTGRES_PORT": EnvironmentStandard(
        name="POSTGRES_PORT",
        required=False,
        default_value="5432",
        description="PostgreSQL 端口",
        production_value="5432",
        docker_value="5432"
    ),
    "POSTGRES_DB": EnvironmentStandard(
        name="POSTGRES_DB",
        required=False,
        default_value="aiva",
        description="PostgreSQL 資料庫名稱",
        production_value="aiva",
        docker_value="aiva"
    ),
    "POSTGRES_USER": EnvironmentStandard(
        name="POSTGRES_USER",
        required=False,
        default_value="aiva",
        description="PostgreSQL 用戶名",
        production_value="aiva",
        docker_value="aiva"
    ),
    "POSTGRES_PASSWORD": EnvironmentStandard(
        name="POSTGRES_PASSWORD",
        required=False,
        default_value="aiva_secure_password",
        description="PostgreSQL 密碼",
        production_value="aiva_secure_password",
        docker_value="aiva_secure_password"
    ),
    
    # 消息隊列配置
    "RABBITMQ_URL": EnvironmentStandard(
        name="RABBITMQ_URL",
        required=True,
        default_value=None,
        description="RabbitMQ 連接 URL",
        production_value="amqp://aiva:aiva_mq_password@rabbitmq:5672/aiva",
        docker_value="amqp://aiva:aiva_mq_password@rabbitmq:5672/aiva"
    ),
    "RABBITMQ_HOST": EnvironmentStandard(
        name="RABBITMQ_HOST",
        required=False,
        default_value="localhost",
        description="RabbitMQ 主機",
        production_value="rabbitmq",
        docker_value="rabbitmq"
    ),
    "RABBITMQ_PORT": EnvironmentStandard(
        name="RABBITMQ_PORT",
        required=False,
        default_value="5672",
        description="RabbitMQ 端口",
        production_value="5672",
        docker_value="5672"
    ),
    "RABBITMQ_USER": EnvironmentStandard(
        name="RABBITMQ_USER",
        required=False,
        default_value=None,
        description="RabbitMQ 用戶名",
        production_value="aiva",
        docker_value="aiva"
    ),
    "RABBITMQ_PASSWORD": EnvironmentStandard(
        name="RABBITMQ_PASSWORD",
        required=False,
        default_value=None,
        description="RabbitMQ 密碼",
        production_value="aiva_mq_password",
        docker_value="aiva_mq_password"
    ),
    "RABBITMQ_VHOST": EnvironmentStandard(
        name="RABBITMQ_VHOST",
        required=False,
        default_value="/",
        description="RabbitMQ Virtual Host",
        production_value="aiva",
        docker_value="aiva"
    ),
    
    # Redis 配置
    "AIVA_REDIS_URL": EnvironmentStandard(
        name="AIVA_REDIS_URL",
        required=False,
        default_value="redis://localhost:6379/0",
        description="Redis 連接 URL",
        production_value="redis://:aiva_redis_password@redis:6379/0",
        docker_value="redis://:aiva_redis_password@redis:6379/0"
    ),
    
    # Neo4j 配置
    "AIVA_NEO4J_URL": EnvironmentStandard(
        name="AIVA_NEO4J_URL",
        required=False,
        default_value="bolt://localhost:7687",
        description="Neo4j 連接 URL",
        production_value="bolt://neo4j:password@neo4j:7687",
        docker_value="bolt://neo4j:password@neo4j:7687"
    ),
    "AIVA_NEO4J_USER": EnvironmentStandard(
        name="AIVA_NEO4J_USER",
        required=False,
        default_value="neo4j",
        description="Neo4j 用戶名",
        production_value="neo4j",
        docker_value="neo4j"
    ),
    "AIVA_NEO4J_PASSWORD": EnvironmentStandard(
        name="AIVA_NEO4J_PASSWORD",
        required=False,
        default_value="password",
        description="Neo4j 密碼",
        production_value="password",
        docker_value="password"
    ),
    
    # 安全配置
    "API_KEY": EnvironmentStandard(
        name="API_KEY",
        required=False,
        default_value=None,
        description="API 主密鑰",
        production_value="production_api_key_change_me",
        docker_value="dev_api_key_for_docker_testing"
    ),
    "INTEGRATION_TOKEN": EnvironmentStandard(
        name="INTEGRATION_TOKEN",
        required=False,
        default_value=None,
        description="Integration 模組認證令牌",
        production_value="integration_secure_token",
        docker_value="docker_integration_token"
    ),
    
    # 其他配置
    "LOG_LEVEL": EnvironmentStandard(
        name="LOG_LEVEL",
        required=False,
        default_value="INFO",
        description="日誌級別",
        production_value="INFO",
        docker_value="INFO"
    ),
    "AUTO_MIGRATE": EnvironmentStandard(
        name="AUTO_MIGRATE",
        required=False,
        default_value="1",
        description="自動遷移資料庫",
        production_value="1",
        docker_value="1"
    ),
}


class EnvironmentValidator:
    """環境變數驗證器"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
    
    def validate_current_environment(self) -> bool:
        """驗證當前環境變數"""
        print("🔍 驗證當前環境變數...")
        
        for env_name, standard in AIVA_ENV_STANDARDS.items():
            current_value = os.getenv(env_name)
            
            if standard.required and current_value is None:
                self.errors.append(f"❌ 必需環境變數 {env_name} 未設置")
            elif current_value is None and standard.default_value:
                self.info.append(f"ℹ️  環境變數 {env_name} 未設置，將使用預設值: {standard.default_value}")
            elif current_value:
                self.info.append(f"✅ 環境變數 {env_name} = {current_value}")
        
        return len(self.errors) == 0
    
    def validate_file_consistency(self, file_path: Path) -> bool:
        """驗證配置文件一致性"""
        if not file_path.exists():
            self.warnings.append(f"⚠️  配置文件不存在: {file_path}")
            return True
        
        print(f"🔍 檢查文件: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            inconsistencies = []
            for env_name, standard in AIVA_ENV_STANDARDS.items():
                if env_name in content:
                    # 檢查是否有過時的命名或值
                    if env_name.startswith("AIVA_POSTGRES") and "POSTGRES_" in content:
                        inconsistencies.append(f"發現過時的 POSTGRES_ 前綴，應使用 {env_name}")
                    elif env_name.startswith("AIVA_RABBITMQ") and "RABBITMQ_" in content:
                        inconsistencies.append(f"發現過時的 RABBITMQ_ 前綴，應使用 {env_name}")
            
            if inconsistencies:
                self.warnings.extend([f"⚠️  {file_path}: {issue}" for issue in inconsistencies])
                return False
            else:
                self.info.append(f"✅ {file_path}: 配置一致")
                return True
                
        except Exception as e:
            self.errors.append(f"❌ 讀取文件 {file_path} 時出錯: {e}")
            return False
    
    def generate_standard_env_file(self, target: str = "docker") -> str:
        """生成標準環境變數文件"""
        lines = [
            "# AIVA 標準環境變數配置",
            f"# 目標環境: {target}",
            "# 由 validate_environment_variables.py 自動生成",
            "",
        ]
        
        categories = {
            "資料庫配置": ["AIVA_DATABASE_URL", "POSTGRES_HOST", "POSTGRES_PORT", 
                         "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
            "消息隊列配置": ["RABBITMQ_URL", "RABBITMQ_HOST", "RABBITMQ_PORT",
                           "RABBITMQ_USER", "RABBITMQ_PASSWORD", "RABBITMQ_VHOST"],
            "安全配置": ["API_KEY", "INTEGRATION_TOKEN"],
            "其他配置": ["LOG_LEVEL", "ENVIRONMENT", "AUTO_MIGRATE"],
        }
        
        for category, env_vars in categories.items():
            lines.append(f"# ================================")
            lines.append(f"# {category}")
            lines.append(f"# ================================")
            
            for env_name in env_vars:
                if env_name in AIVA_ENV_STANDARDS:
                    standard = AIVA_ENV_STANDARDS[env_name]
                    value = getattr(standard, f"{target}_value") or standard.default_value
                    if value:
                        lines.append(f"{env_name}={value}")
                    lines.append(f"# {standard.description}")
                    lines.append("")
            
            lines.append("")
        
        return "\\n".join(lines)
    
    def print_report(self):
        """打印驗證報告"""
        print("\\n" + "="*60)
        print("📋 AIVA 環境變數驗證報告")
        print("="*60)
        
        if self.errors:
            print("\\n❌ 錯誤:")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print("\\n⚠️  警告:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if self.info:
            print("\\nℹ️  資訊:")
            for info in self.info:
                print(f"  {info}")
        
        print(f"\\n📊 總結:")
        print(f"  - 錯誤: {len(self.errors)}")
        print(f"  - 警告: {len(self.warnings)}")
        print(f"  - 資訊: {len(self.info)}")
        
        if len(self.errors) == 0:
            print("\\n✅ 環境變數驗證通過！")
        else:
            print("\\n❌ 環境變數驗證失敗！")


def main():
    """主函數"""
    validator = EnvironmentValidator()
    
    # 驗證當前環境
    current_valid = validator.validate_current_environment()
    
    # 驗證關鍵配置文件
    project_root = Path(__file__).parent.parent.parent
    key_files = [
        project_root / ".env.docker",
        project_root / ".env.example",
        project_root / "docker/compose/docker-compose.yml",
        project_root / "docker/compose/docker-compose.production.yml",
        project_root / "services/aiva_common/config/unified_config.py",
    ]
    
    all_files_valid = True
    for file_path in key_files:
        if not validator.validate_file_consistency(file_path):
            all_files_valid = False
    
    # 打印報告
    validator.print_report()
    
    # 如果需要，生成標準配置文件
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        target = sys.argv[2] if len(sys.argv) > 2 else "docker"
        print(f"\\n📄 生成 {target} 環境標準配置:")
        print("-" * 40)
        print(validator.generate_standard_env_file(target))
    
    # 返回退出碼
    sys.exit(0 if current_valid and all_files_valid else 1)


if __name__ == "__main__":
    main()