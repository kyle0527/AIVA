#!/bin/bash
set -e

echo "[entrypoint] Starting AIVA Integration Module..."
echo "[entrypoint] Waiting for database connection..."

# 等待資料庫啟動
ATTEMPTS=30
SLEEP=2
DB_READY=false

for i in $(seq 1 $ATTEMPTS); do
    echo "  Attempt $i/$ATTEMPTS - Testing database connection..."
    
    # 使用 Python 測試資料庫連接
    if python3 -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ.get('AIVA_DATABASE_URL', ''))
    conn.close()
    print('Database connection successful!')
    exit(0)
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
" 2>/dev/null; then
        DB_READY=true
        break
    else
        echo "    Database not ready, waiting ${SLEEP}s..."
        sleep $SLEEP
    fi
done

if [ "$DB_READY" != "true" ]; then
    echo "[entrypoint] ERROR: Database connection failed after $ATTEMPTS attempts"
    exit 1
fi

# 執行資料庫遷移 (如果啟用)
if [ "${AUTO_MIGRATE:-0}" = "1" ]; then
    echo "[entrypoint] Running database migrations..."
    if [ -f "services/integration/alembic.ini" ]; then
        alembic -c services/integration/alembic.ini upgrade head || {
            echo "[entrypoint] ERROR: Database migration failed"
            exit 1
        }
        echo "[entrypoint] Database migrations completed successfully"
    else
        echo "[entrypoint] No alembic.ini found, skipping migrations"
    fi
fi

# 設置預設環境變數
export AIVA_LOG_LEVEL=${AIVA_LOG_LEVEL:-INFO}
export AIVA_CORS_ORIGINS=${AIVA_CORS_ORIGINS:-"*"}
export AIVA_RATE_LIMIT_RPS=${AIVA_RATE_LIMIT_RPS:-20}
export AIVA_RATE_LIMIT_BURST=${AIVA_RATE_LIMIT_BURST:-60}

echo "[entrypoint] Starting uvicorn server..."
echo "  - Host: 0.0.0.0"
echo "  - Port: 8083"
echo "  - Log Level: $AIVA_LOG_LEVEL"
echo "  - Workers: 1"

# 啟動應用
exec uvicorn services.integration.aiva_integration.app:app \
    --host 0.0.0.0 \
    --port 8083 \
    --log-level $(echo $AIVA_LOG_LEVEL | tr '[:upper:]' '[:lower:]') \
    --access-log \
    --no-use-colors