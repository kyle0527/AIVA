# GO Scanner 掃描器統一建置映像檔
FROM golang:1.21-alpine AS builder

WORKDIR /app

# 安裝建置依賴
RUN apk add --no-cache git ca-certificates tzdata

# 複製共用組件
COPY shared/ ./shared/

# 複製所有掃描器模組
COPY ssrf_scanner/ ./ssrf_scanner/
COPY csmp_scanner/ ./cspm_scanner/
COPY sca_scanner/ ./sca_scanner/

# 複製通用 GO 依賴
COPY ../../../features/common/go/ ./common/go/

# 建置所有掃描器二進位檔案
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o bin/ssrf_scanner ./ssrf_scanner/
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o bin/cspm_scanner ./cspm_scanner/
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o bin/sca_scanner ./sca_scanner/

# 階段2: 執行環境
FROM alpine:latest

# 安裝執行時依賴
RUN apk --no-cache add ca-certificates tzdata curl

WORKDIR /root/

# 從建置階段複製二進位檔案
COPY --from=builder /app/bin/ ./bin/

# 建立必要目錄
RUN mkdir -p logs reports config

# 設定時區
ENV TZ=UTC

# 健康檢查端點
COPY healthcheck.sh ./
RUN chmod +x healthcheck.sh

# 公開端口 (各掃描器可使用不同端口)
EXPOSE 8081 8082 8083

# 預設執行所有掃描器 (可透過環境變數覆蓋)
CMD ["sh", "-c", "./bin/ssrf_scanner & ./bin/cspm_scanner & ./bin/sca_scanner & wait"]