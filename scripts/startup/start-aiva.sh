#!/bin/bash
# AIVA 快速啟動腳本 (Linux/macOS)
# 用於本地 Docker Compose 環境

set -e

ACTION="${1:-core}"
BUILD_FLAG="${2:-}"

echo "🚀 AIVA 微服務啟動器"
echo "============================================================"

show_status() {
    echo ""
    echo "📊 當前服務狀態:"
    docker-compose ps
    
    echo ""
    echo "🔍 核心服務健康檢查:"
    if curl -f http://localhost:8000/health 2>/dev/null; then
        echo "✅ AIVA Core: 健康"
    else
        echo "❌ AIVA Core: 不可用"
    fi
}

start_core() {
    echo ""
    echo "🏗️ 啟動核心服務和基礎設施..."
    
    if [ "$BUILD_FLAG" == "--build" ]; then
        echo "📦 構建 Docker 鏡像..."
        docker-compose build aiva-core
    fi
    
    docker-compose up -d
    
    echo ""
    echo "⏳ 等待服務啟動（60秒）..."
    sleep 60
    
    show_status
    
    echo ""
    echo "✅ 核心服務已啟動！"
    echo "   API: http://localhost:8000"
    echo "   Admin: http://localhost:8001"
    echo "   RabbitMQ UI: http://localhost:15672 (guest/guest)"
    echo "   Neo4j Browser: http://localhost:7474 (neo4j/aiva123)"
}

start_components() {
    local profile=$1
    
    echo ""
    echo "🔧 啟動組件: $profile"
    
    if [ "$BUILD_FLAG" == "--build" ]; then
        echo "📦 構建組件鏡像..."
        docker-compose build
    fi
    
    docker-compose --profile "$profile" up -d
    
    echo ""
    echo "⏳ 等待組件啟動（30秒）..."
    sleep 30
    
    show_status
    
    echo ""
    echo "✅ 組件已啟動！"
}

stop_all() {
    echo ""
    echo "🛑 停止所有服務..."
    docker-compose down
    echo "✅ 所有服務已停止"
}

show_logs() {
    echo ""
    echo "📜 顯示實時日誌..."
    docker-compose logs -f --tail=100
}

# 主邏輯
case "$ACTION" in
    core)
        start_core
        ;;
    scanners)
        start_components "scanners"
        ;;
    testing)
        start_components "testing"
        ;;
    explorers)
        start_components "explorers"
        ;;
    validators)
        start_components "validators"
        ;;
    pentest)
        start_components "pentest"
        ;;
    all)
        start_components "all"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "使用方法: $0 {core|scanners|testing|explorers|validators|pentest|all|stop|status|logs} [--build]"
        exit 1
        ;;
esac

echo ""
