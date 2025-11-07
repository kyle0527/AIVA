# 03｜POSTEX 設計規格（成熟版）
**日期**：2025-11-06  

## 1. 目標與範圍
- 在**授權**環境內模擬：權限提升、橫向移動、持久化；預設 `safe_mode=true`，所有行為可審計/可回滾。
- 接收 `PostExTaskPayload`，輸出 `FindingPayload` 與 SARIF。

## 2. 架構
- **Engines**：`privilege_engine.py` / `lateral_engine.py` / `persistence_engine.py`，皆支援 dry-run 與審計。
- **Detector（Python）**：彙整為分類化 `Vulnerability` 與建議。
- **Worker（Python, async）**：AMQP 訂閱 `Topic.TASK_FUNCTION_POSTEX`，回報狀態與 Finding。
- **Config（Pydantic）**：白名單、允許操作列表、dry-run 策略。

## 3. 安全
- 所有敏感操作需帶 `authorization_token`；若無 → 強制 safe-mode。
- 全程 structured log（who/when/what/where/result）。

## 4. 測試與驗收
- 覆蓋率 ≥ 85%；E2E 於容器內完成；所有動作可回滾。
- 成功案例寫入經驗庫。

## 5. 目錄
- 參見 `services/features/function_postex/*` 與《06》部署文件。
