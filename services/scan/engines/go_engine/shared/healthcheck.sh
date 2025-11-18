#!/bin/sh
# 健康檢查腳本 - 檢查各掃描器狀態

# 檢查 SSRF Scanner (預設端口 8081)
if curl -s -f http://localhost:8081/health > /dev/null 2>&1; then
    echo "SSRF Scanner: ✅ 健康"
else
    echo "SSRF Scanner: ❌ 異常"
    exit 1
fi

# 檢查 CSPM Scanner (預設端口 8082)  
if curl -s -f http://localhost:8082/health > /dev/null 2>&1; then
    echo "CSPM Scanner: ✅ 健康"
else
    echo "CSPM Scanner: ❌ 異常"
    exit 1
fi

# 檢查 SCA Scanner (預設端口 8083)
if curl -s -f http://localhost:8083/health > /dev/null 2>&1; then
    echo "SCA Scanner: ✅ 健康"
else
    echo "SCA Scanner: ❌ 異常"
    exit 1
fi

echo "所有 GO 掃描器運行正常 🎉"
exit 0