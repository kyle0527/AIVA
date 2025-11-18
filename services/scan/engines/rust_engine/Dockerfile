# Rust 敏感資訊掃描引擎 Dockerfile
# 支持三種掃描模式: Fast Discovery / Deep Analysis / Focused Verification

FROM rust:1.75-slim AS builder

WORKDIR /build

# 安裝依賴
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製 Cargo 文件
COPY Cargo.toml Cargo.lock ./

# 複製 aiva_common_rust 依賴
COPY ../../features/common/rust/aiva_common_rust /build/aiva_common_rust

# 複製源代碼
COPY src ./src

# 構建 Release 版本
RUN cargo build --release

# 運行時鏡像
FROM debian:bookworm-slim

WORKDIR /app

# 安裝運行時依賴
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# 從 builder 複製二進制文件
COPY --from=builder /build/target/release/aiva-info-gatherer /app/

# 創建日誌目錄
RUN mkdir -p /var/log/aiva/metrics

# 環境變數配置
ENV RUST_LOG=info
ENV RUST_SCAN_MODE=deep_analysis
ENV AIVA_METRICS_OUTPUT_FILE=/var/log/aiva/metrics/info_gatherer_rust.jsonl
ENV AIVA_METRICS_INTERVAL=60

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD pgrep -f aiva-info-gatherer || exit 1

# 運行
CMD ["./aiva-info-gatherer"]
