# 06｜部署、測試與驗收（成熟版）
**日期**：2025-11-06  

## 1. 建置
- 先生成合約（如需）：`scripts/gen_contracts.sh`
- 建置 CRYPTO Rust 引擎 + Python 封裝：`scripts/build_crypto_engine.sh`
- 建置 Docker：`scripts/build_docker_crypto.sh`、`scripts/build_docker_postex.sh`

## 2. 執行（單機）
- 啟動 AMQP（RabbitMQ）與依賴後，執行：
  - `scripts/run_crypto_worker.sh`
  - `scripts/run_postex_worker.sh`

## 3. Compose Overlay
- 以 `compose_overlay/docker-compose.crypto_postex.yml` 疊加現有 Compose：  
  `docker compose -f docker-compose.yml -f compose_overlay/docker-compose.crypto_postex.yml up -d`

## 4. 測試
- 單元：`scripts/run_tests.sh`（pytest）  
- E2E：於 Compose 內發送 Task → 收集 Finding / 狀態。

## 5. 驗收
- 指標達成（準確率/覆蓋/效能）。  
- SARIF 與 AMQP 封包通過驗證；日誌與指標可觀測。

## 6. 故障排除
- 檢查環境變數 `AIVA_AMQP_URL`、Rust 編譯鏈、Python venv。
