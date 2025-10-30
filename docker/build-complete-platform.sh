#!/bin/bash
# AIVA 完整平台 Docker 映像檔建立腳本
# 建立完整的 AIVA 系統，包含所有服務與資料庫

set -e

# =====================================
# 配置參數
# =====================================

IMAGE_NAME="aiva-complete"
IMAGE_TAG="${1:-latest}"
BUILD_CONTEXT="../"
DOCKERFILE="docker/Dockerfile.complete"
DOCKER_COMPOSE_FILE="docker/docker-compose.complete.yml"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =====================================
# 輔助函數
# =====================================

print_header() {
    echo -e "\n${PURPLE}=================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=================================${NC}\n"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_docker() {
    print_info "檢查 Docker 環境..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安裝"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon 未運行"
        exit 1
    fi
    
    print_success "Docker 環境正常"
}

check_compose() {
    print_info "檢查 Docker Compose..."
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose 未安裝"
        exit 1
    fi
    print_success "Docker Compose 可用"
}

cleanup_old_containers() {
    print_info "清理舊容器..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans || true
    docker system prune -f --volumes || true
    print_success "清理完成"
}

build_image() {
    print_info "開始建立 AIVA 完整映像檔..."
    print_info "映像檔名稱: $IMAGE_NAME:$IMAGE_TAG"
    print_info "建立上下文: $BUILD_CONTEXT"
    print_info "Dockerfile: $DOCKERFILE"
    
    local start_time=$(date +%s)
    
    # 建立映像檔
    if docker build \
        --tag "$IMAGE_NAME:$IMAGE_TAG" \
        --file "$DOCKERFILE" \
        --progress=plain \
        --no-cache \
        "$BUILD_CONTEXT"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "映像檔建立成功！耗時: ${duration}s"
        
        # 顯示映像檔資訊
        docker images "$IMAGE_NAME:$IMAGE_TAG"
        
    else
        print_error "映像檔建立失敗"
        exit 1
    fi
}

test_image() {
    print_info "測試映像檔..."
    
    # 基本測試 - 檢查 Python 版本
    if docker run --rm "$IMAGE_NAME:$IMAGE_TAG" python --version; then
        print_success "Python 測試通過"
    else
        print_error "Python 測試失敗"
        return 1
    fi
    
    # 檢查 Rust 組件
    if docker run --rm "$IMAGE_NAME:$IMAGE_TAG" which function_sast_rust; then
        print_success "Rust 組件測試通過"
    else
        print_warning "Rust 組件可能未正確安裝"
    fi
    
    # 檢查 Go 組件
    if docker run --rm "$IMAGE_NAME:$IMAGE_TAG" which function_authn_go; then
        print_success "Go 組件測試通過"
    else
        print_warning "Go 組件可能未正確安裝"
    fi
    
    print_success "映像檔測試完成"
}

start_complete_system() {
    print_info "啟動完整 AIVA 系統..."
    
    # 使用 Docker Compose 啟動完整系統
    if docker-compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        print_success "系統啟動成功"
        print_info "等待服務初始化..."
        sleep 30
        
        print_info "服務狀態："
        docker-compose -f "$DOCKER_COMPOSE_FILE" ps
        
        print_info "AIVA 服務端點："
        echo -e "${CYAN}  主 API:        http://localhost:8000${NC}"
        echo -e "${CYAN}  管理 API:      http://localhost:8001${NC}"
        echo -e "${CYAN}  WebSocket:     ws://localhost:8002${NC}"
        echo -e "${CYAN}  儀表板:        http://localhost:8080${NC}"
        echo -e "${CYAN}  RabbitMQ UI:   http://localhost:15672 (aiva/aiva_mq_password)${NC}"
        echo -e "${CYAN}  Neo4j UI:      http://localhost:7474 (neo4j/password)${NC}"
        
    else
        print_error "系統啟動失敗"
        return 1
    fi
}

health_check() {
    print_info "執行健康檢查..."
    
    local retries=12
    local count=0
    
    while [ $count -lt $retries ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_success "AIVA 主服務健康檢查通過"
            return 0
        fi
        
        count=$((count + 1))
        print_info "等待服務啟動... ($count/$retries)"
        sleep 10
    done
    
    print_warning "健康檢查超時，請手動驗證服務狀態"
    return 1
}

show_logs() {
    print_info "顯示服務日誌..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" logs --tail=50
}

create_backup() {
    print_info "創建系統備份..."
    
    local backup_dir="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # 備份資料卷
    docker run --rm \
        -v aiva-complete-platform_postgres_data:/source:ro \
        -v "$(pwd)/$backup_dir":/backup \
        alpine tar czf /backup/postgres_data.tar.gz -C /source .
    
    docker run --rm \
        -v aiva-complete-platform_aiva_data:/source:ro \
        -v "$(pwd)/$backup_dir":/backup \
        alpine tar czf /backup/aiva_data.tar.gz -C /source .
    
    print_success "備份創建完成: $backup_dir"
}

show_usage() {
    echo -e "${PURPLE}AIVA 完整平台 Docker 建立腳本${NC}"
    echo ""
    echo "用法: $0 [選項] [映像檔標籤]"
    echo ""
    echo "選項:"
    echo "  build     建立映像檔 (預設)"
    echo "  start     建立並啟動完整系統"
    echo "  test      測試映像檔"
    echo "  health    執行健康檢查"
    echo "  logs      顯示服務日誌"
    echo "  stop      停止所有服務"
    echo "  restart   重啟所有服務"
    echo "  backup    創建系統備份"
    echo "  clean     清理所有資源"
    echo "  help      顯示此幫助"
    echo ""
    echo "範例:"
    echo "  $0 start v2.0          # 建立 v2.0 版本並啟動"
    echo "  $0 build latest        # 僅建立 latest 版本"
    echo "  $0 health              # 檢查服務健康狀態"
}

# =====================================
# 主要執行流程
# =====================================

main() {
    local action="${1:-build}"
    
    case "$action" in
        "build")
            print_header "AIVA 完整平台映像檔建立"
            check_docker
            build_image
            test_image
            print_success "建立流程完成！"
            print_info "使用 '$0 start' 啟動完整系統"
            ;;
            
        "start")
            print_header "AIVA 完整平台系統啟動"
            check_docker
            check_compose
            build_image
            test_image
            start_complete_system
            health_check
            print_success "完整系統已啟動！"
            ;;
            
        "test")
            print_header "AIVA 映像檔測試"
            check_docker
            test_image
            ;;
            
        "health")
            print_header "AIVA 系統健康檢查"
            health_check
            ;;
            
        "logs")
            print_header "AIVA 系統日誌"
            show_logs
            ;;
            
        "stop")
            print_header "停止 AIVA 系統"
            docker-compose -f "$DOCKER_COMPOSE_FILE" down
            print_success "系統已停止"
            ;;
            
        "restart")
            print_header "重啟 AIVA 系統"
            docker-compose -f "$DOCKER_COMPOSE_FILE" restart
            print_success "系統已重啟"
            ;;
            
        "backup")
            print_header "AIVA 系統備份"
            create_backup
            ;;
            
        "clean")
            print_header "清理 AIVA 資源"
            cleanup_old_containers
            docker rmi "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null || true
            print_success "清理完成"
            ;;
            
        "help"|"-h"|"--help")
            show_usage
            ;;
            
        *)
            print_error "未知選項: $action"
            show_usage
            exit 1
            ;;
    esac
}

# 執行主函數
main "$@"