# 01｜CRYPTO + POSTEX 急需實現報告
**版本**：v1.1（成熟版）  
**日期**：2025-11-06  
**範圍**：CRYPTO（密碼學弱點）＋ POSTEX（後滲透行為）  
**合約**：以 `services/aiva_common` 統一資料合約為唯一事實來源（SSoT）；產出 SARIF、AMQP 封包一致。

## 結論（摘要）
- 兩模組目前空白（0/4：Worker/Detector/Engine/Config 未落地），**需立即實作**。
- 本報告給出可直接整合的成熟實作（含 Docker、測試、腳本、Compose overlay）。
- 依「四件標準 + AMQP + SARIF + Docker」對齊整體架構；支援經驗庫回寫。

## 可交付物
- 目錄 `services/features/function_crypto`、`services/features/function_postex` 完整程式與 Docker。
- overlay：`compose_overlay/docker-compose.crypto_postex.yml` 可一鍵啟動兩服務。
- 腳本：build / run / test / 合約生成等共 10+ 支。

> 細節請見 02~06 文件。
