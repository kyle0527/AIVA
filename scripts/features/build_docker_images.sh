#!/bin/bash
# AIVA Features Supplement v2 - Docker Build Script
# 構建所有補充功能模組的Docker映像

set -e

echo "🚀 Building AIVA Features Supplement v2 Docker Images..."
echo "=================================================="

# 設置顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函數：構建並標記映像
build_image() {
    local service_name=$1
    local dockerfile_path=$2
    local image_tag="aiva/${service_name}:latest"
    
    echo -e "${YELLOW}Building ${service_name}...${NC}"
    
    if docker build -f "${dockerfile_path}" -t "${image_tag}" .; then
        echo -e "${GREEN}✅ Successfully built ${image_tag}${NC}"
    else
        echo -e "${RED}❌ Failed to build ${image_tag}${NC}"
        exit 1
    fi
}

# 切換到專案根目錄
cd "$(dirname "$0")/../.."

echo "📍 Current directory: $(pwd)"
echo ""

# 構建SSRF Worker
echo "1. Building SSRF Worker..."
build_image "ssrf_worker" "services/features/function_ssrf/Dockerfile"

echo ""

# 構建IDOR Worker  
echo "2. Building IDOR Worker..."
build_image "idor_worker" "services/features/function_idor/Dockerfile"

echo ""

# 構建AUTHN GO Worker
echo "3. Building AUTHN GO Worker..."
build_image "authn_go_worker" "services/features/function_authn_go/Dockerfile"

echo ""
echo "🎉 All images built successfully!"
echo ""

# 顯示構建的映像
echo "📦 Built Images:"
docker images | grep "aiva/" | grep -E "(ssrf_worker|idor_worker|authn_go_worker)"

echo ""
echo "🔧 Next Steps:"
echo "1. Run: docker-compose -f docker-compose.features.yml up -d"
echo "2. Check logs: docker-compose -f docker-compose.features.yml logs -f"
echo "3. Validate: ./scripts/features/test_workers.sh"