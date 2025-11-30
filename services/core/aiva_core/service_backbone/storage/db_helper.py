"""資料庫連接輔助函數

統一管理所有資料庫連接，從配置文件讀取設定
"""
import asyncpg
from urllib.parse import urlparse
from services.aiva_common.config.settings import get_settings
from services.aiva_common.utils.logging import get_logger

logger = get_logger(__name__)

_db_conn_pool = None


async def get_db_connection():
    """獲取資料庫連接（從統一配置讀取）
    
    Returns:
        asyncpg.Connection: 資料庫連接
    """
    settings = get_settings()
    db_url = settings.get_database_url()
    
    # 解析資料庫 URL
    parsed = urlparse(db_url)
    
    logger.debug(f"連接資料庫: {parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}")
    
    conn = await asyncpg.connect(
        host=parsed.hostname or 'localhost',
        port=parsed.port or 5432,
        user=parsed.username or 'postgres',
        password=parsed.password or 'postgres',
        database=parsed.path.lstrip('/') or 'aiva_db'
    )
    
    return conn


async def get_db_pool():
    """獲取資料庫連接池（單例模式）
    
    Returns:
        asyncpg.Pool: 資料庫連接池
    """
    global _db_conn_pool
    
    if _db_conn_pool is None:
        settings = get_settings()
        db_url = settings.get_database_url()
        parsed = urlparse(db_url)
        
        _db_conn_pool = await asyncpg.create_pool(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            user=parsed.username or 'postgres',
            password=parsed.password or 'postgres',
            database=parsed.path.lstrip('/') or 'aiva_db',
            min_size=settings.database.pool_size,
            max_size=settings.database.pool_size + settings.database.max_overflow
        )
        
        logger.info(f"資料庫連接池已創建: {parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}")
    
    return _db_conn_pool


def get_db_config() -> dict:
    """獲取資料庫配置字典
    
    Returns:
        dict: 包含 host, port, user, password, database 的字典
    """
    settings = get_settings()
    db_url = settings.get_database_url()
    parsed = urlparse(db_url)
    
    return {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 5432,
        'user': parsed.username or 'postgres',
        'password': parsed.password or 'postgres',
        'database': parsed.path.lstrip('/') or 'aiva_db'
    }
