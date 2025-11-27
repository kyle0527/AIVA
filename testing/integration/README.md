# 🧪 整合測試套件

## 目錄

- [目錄說明](#目錄說明)
- [測試腳本](#測試腳本)
  - [1. test_ai_command_scan.py - AI 命令中心整合](#1-test_ai_command_scanpy---ai-命令中心整合)
  - [2. test_dual_loop_juice_shop.py - 雙閉環系統實戰](#2-test_dual_loop_juice_shoppy---雙閉環系統實戰)
  - [3. test_two_phase_scan.py - 兩階段掃描編排](#3-test_two_phase_scanpy---兩階段掃描編排)
  - [4. test_multi_language_analysis.py - 多語言能力分析](#4-test_multi_language_analysispy---多語言能力分析)
- [測試對比](#測試對比)
- [運行所有整合測試](#運行所有整合測試)
- [依賴服務檢查](#依賴服務檢查)
- [開發建議](#開發建議)
- [常見問題](#常見問題)
- [後續擴展](#後續擴展)

---

## 目錄說明

此目錄包含 AIVA 系統的**整合測試**，用於驗證多個模組間的協作和複雜業務流程。

與根目錄的測試工具不同：
- **根目錄測試** (`quick_test.py`, `diagnose.py`, `aiva_test.py`) - 基礎功能驗證
- **整合測試** (`tests/integration/`) - 複雜業務流程和架構驗證

---

## 測試腳本

### 1. 🎯 `test_ai_command_scan.py` - AI 命令中心整合

**測試目標**: AI 命令中心與 Scan 模組的整合

**測試內容**:
- ✅ AI 命令中心初始化
- ✅ Scan 命令處理器註冊
- ✅ Phase 0 快速偵察命令
- ✅ Phase 1 深度掃描命令
- ✅ Phase 2 漏洞驗證命令
- ✅ 數據合約正確傳遞

**運行方式**:
```bash
python tests/integration/test_ai_command_scan.py
```

**適用場景**:
- 驗證命令中心架構
- 測試模組間通訊
- 驗證掃描階段切換

---

### 2. 🔄 `test_dual_loop_juice_shop.py` - 雙閉環系統實戰

**測試目標**: 完整的雙閉環架構（內循環優化 + 外循環報告）

**測試內容**:
- ✅ Features 執行（XSS 掃描）
- ✅ Coordinator 收集和聚合
- ✅ 內循環優化（payload 優化）
- ✅ 外循環報告（Markdown 報告生成）
- ✅ 統計數據收集
- ✅ 性能指標追蹤

**運行方式**:
```bash
# 確保 Juice Shop 在運行
docker ps | grep juice-shop

# 運行測試
python tests/integration/test_dual_loop_juice_shop.py
```

**適用場景**:
- 驗證雙閉環架構設計
- 測試協調器邏輯
- 驗證報告生成流程

**測試目標**: 
- http://localhost:3000 (Juice Shop)

---

### 3. 📊 `test_two_phase_scan.py` - 兩階段掃描編排

**測試目標**: TwoPhaseScanOrchestrator 的編排邏輯

**測試內容**:
- ✅ Phase 0 快速偵察（5-10 分鐘）
- ✅ Phase 1 深度掃描（10-30 分鐘）
- ✅ 多靶場並行掃描
- ✅ 結果聚合和報告
- ✅ RabbitMQ 消息傳遞

**運行方式**:
```bash
# 確保 RabbitMQ 和靶場都在運行
docker ps | grep rabbitmq
docker ps | grep juice-shop

# 運行測試
python tests/integration/test_two_phase_scan.py
```

**依賴服務**:
- RabbitMQ: localhost:5672
- Juice Shop: localhost:3000, 3001, 3003
- WebGoat: localhost:8080

**適用場景**:
- 驗證掃描編排邏輯
- 測試多階段掃描流程
- 測試消息隊列整合

---

### 4. 🌐 `test_multi_language_analysis.py` - 多語言能力分析

**測試目標**: 內部探索系統的多語言能力提取

**測試內容**:
- ✅ Python 能力提取（AST）
- ✅ Go 能力提取（正則）
- ✅ Rust 能力提取（正則）
- ✅ TypeScript 能力提取（正則）
- ✅ ModuleExplorer 整合
- ✅ CapabilityAnalyzer 整合

**運行方式**:
```bash
python tests/integration/test_multi_language_analysis.py
```

**適用場景**:
- 驗證多語言支持
- 測試內部探索功能
- 測試能力提取邏輯

---

## 測試對比

| 測試類型 | 位置 | 執行時間 | 依賴服務 |
|---------|------|---------|---------|
| **基礎測試** | 根目錄 | < 10 秒 | Docker 靶場 |
| **整合測試** | tests/integration | 5-30 分鐘 | Docker + RabbitMQ |

---

## 運行所有整合測試

```bash
# 1. 確保所有服務運行
docker-compose up -d

# 2. 運行所有整合測試
cd tests/integration
python test_ai_command_scan.py
python test_dual_loop_juice_shop.py
python test_two_phase_scan.py
python test_multi_language_analysis.py
```

---

## 依賴服務檢查

### 檢查 Docker 靶場
```bash
docker ps | grep -E "juice-shop|webgoat"
```

### 檢查 RabbitMQ
```bash
docker ps | grep rabbitmq
# 或訪問管理界面: http://localhost:15672
```

### 快速診斷
```bash
# 使用根目錄的診斷工具
python diagnose.py docker
```

---

## 開發建議

### 何時運行整合測試？

✅ **應該運行**:
- 修改了多模組協作邏輯
- 修改了掃描編排流程
- 修改了命令中心架構
- 修改了雙閉環邏輯
- 重大功能更新前

❌ **不需要運行**:
- 只修改單一模組的內部邏輯
- 修改配置文件
- 修改文檔
- 快速測試功能（用 `quick_test.py`）

### 測試策略

```
日常開發:
  快速測試 (quick_test.py) → 功能測試 (aiva_test.py) → 整合測試 (integration/)

發布前:
  完整測試套件 (aiva_test.py full) + 所有整合測試
```

---

## 常見問題

### Q: 為什麼整合測試這麼慢？
**A**: 整合測試會執行完整的業務流程，包括實際的掃描和報告生成，所以需要較長時間。

### Q: 整合測試失敗怎麼辦？
**A**: 
1. 先運行 `python diagnose.py` 檢查基礎設施
2. 確認 Docker 和 RabbitMQ 正常運行
3. 檢查測試腳本的輸出日誌
4. 參考各腳本的文檔字符串

### Q: 可以跳過整合測試嗎？
**A**: 日常開發可以只運行快速測試。但在以下情況必須運行整合測試：
- Pull Request 提交前
- 發布新版本前
- 修改核心架構後

---

## 後續擴展

可以添加的整合測試：
- [ ] 完整的 SQLi 檢測流程測試
- [ ] 完整的 LFI/RCE 檢測流程測試
- [ ] 多模組協同測試（Scan + Report + Verify）
- [ ] 負載測試和性能測試
- [ ] 錯誤恢復和重試機制測試

---

最後更新: 2025-11-22
