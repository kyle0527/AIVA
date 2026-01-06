"""創建能力元數據數據庫

執行 PostgreSQL 數據庫 schema 創建

Usage:
    python scripts/migrations/create_capability_db.py
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 數據庫 URL (使用環境變量或默認值)
DATABASE_URL = os.getenv(
    "AIVA_CAPABILITY_DB_URL",
    "postgresql://aiva_user:aiva_password@localhost:5432/aiva_capabilities"
)


def create_database_if_not_exists():
    """創建數據庫 (如果不存在)"""
    # 連接到 postgres 默認數據庫
    default_url = DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    engine = create_engine(default_url, isolation_level="AUTOCOMMIT")
    
    db_name = DATABASE_URL.split("/")[-1]
    
    with engine.connect() as conn:
        # 檢查數據庫是否存在
        result = conn.execute(
            text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        )
        
        if not result.fetchone():
            logger.info(f"Creating database: {db_name}")
            conn.execute(text(f"CREATE DATABASE {db_name}"))
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")


def create_tables():
    """創建所有表"""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # 1. 能力主表
        logger.info("Creating table: capability_records")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS capability_records (
                id SERIAL PRIMARY KEY,
                key VARCHAR(500) UNIQUE NOT NULL,
                name VARCHAR(200) NOT NULL,
                module VARCHAR(200) NOT NULL,
                language VARCHAR(50) NOT NULL,
                file_path TEXT,
                
                version INTEGER NOT NULL DEFAULT 1,
                content_hash VARCHAR(64) NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                
                metadata_json TEXT NOT NULL,
                
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """))
        
        # 索引
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cap_name ON capability_records(name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cap_module ON capability_records(module)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cap_hash ON capability_records(content_hash)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_cap_active ON capability_records(is_active)"))
        
        # 2. 版本歷史表
        logger.info("Creating table: capability_versions")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS capability_versions (
                id SERIAL PRIMARY KEY,
                capability_key VARCHAR(500) NOT NULL,
                version INTEGER NOT NULL,
                
                content_hash VARCHAR(64) NOT NULL,
                metadata_json TEXT NOT NULL,
                
                change_type VARCHAR(20) NOT NULL,
                change_summary TEXT,
                
                created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                
                UNIQUE(capability_key, version),
                FOREIGN KEY (capability_key) REFERENCES capability_records(key)
            )
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ver_change_type ON capability_versions(change_type)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ver_created_at ON capability_versions(created_at)"))
        
        # 3. 變更日誌表
        logger.info("Creating table: capability_change_logs")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS capability_change_logs (
                id SERIAL PRIMARY KEY,
                scan_id VARCHAR(36) NOT NULL,
                scan_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                
                added_count INTEGER NOT NULL DEFAULT 0,
                modified_count INTEGER NOT NULL DEFAULT 0,
                deleted_count INTEGER NOT NULL DEFAULT 0,
                unchanged_count INTEGER NOT NULL DEFAULT 0,
                
                total_capabilities INTEGER NOT NULL,
                
                details_json TEXT
            )
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_log_timestamp ON capability_change_logs(scan_timestamp)"))
        
        # 4. 調用統計表
        logger.info("Creating table: capability_invocation_stats")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS capability_invocation_stats (
                id SERIAL PRIMARY KEY,
                capability_key VARCHAR(500) NOT NULL,
                
                invocation_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                
                avg_duration_ms FLOAT,
                last_invoked_at TIMESTAMP WITH TIME ZONE,
                
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                
                FOREIGN KEY (capability_key) REFERENCES capability_records(key)
            )
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_stats_key ON capability_invocation_stats(capability_key)"))
        
        conn.commit()
        
    logger.info("All tables created successfully")


def verify_tables():
    """驗證表是否正確創建"""
    engine = create_engine(DATABASE_URL)
    
    expected_tables = [
        "capability_records",
        "capability_versions",
        "capability_change_logs",
        "capability_invocation_stats"
    ]
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """))
        
        existing_tables = [row[0] for row in result]
        
        logger.info(f"Existing tables: {existing_tables}")
        
        for table in expected_tables:
            if table in existing_tables:
                logger.info(f"✓ Table '{table}' exists")
            else:
                logger.error(f"✗ Table '{table}' missing!")
                return False
        
        return True


def main():
    """主函數"""
    try:
        logger.info("=" * 60)
        logger.info("AIVA Capability Metadata Database Setup")
        logger.info("=" * 60)
        
        # Step 1: 創建數據庫
        logger.info("\nStep 1: Creating database if not exists...")
        create_database_if_not_exists()
        
        # Step 2: 創建表
        logger.info("\nStep 2: Creating tables...")
        create_tables()
        
        # Step 3: 驗證
        logger.info("\nStep 3: Verifying tables...")
        if verify_tables():
            logger.info("\n✓ Database setup completed successfully!")
            logger.info(f"Database URL: {DATABASE_URL}")
        else:
            logger.error("\n✗ Database setup failed!")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during database setup: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
