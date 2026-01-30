# AIVA 外部模塊系統狀態報告

**日期**: 2026-01-14  
**版本**: v1.0  
**狀態**: 生產就緒（部分模塊）

---

## 📊 系統概覽

### 模塊統計
- **總流程數**: 210
- **總模塊數**: 8
- **支持語言**: Python, Go, TypeScript
- **CLI 就緒**: 1 個模塊（XSS）
- **Worker 需求**: 7 個模塊

### 分類數據
```json
{
  "python": 203 flows (5 modules),
  "go": 4 flows (1 module),
  "typescript": 3 flows (1 module)
}
```

---

## 🎯 模塊狀態詳情

### Python 模塊（5個）

#### ✅ function_xss (生產就緒)
- **流程數**: 195
- **CLI 接口**: ✅ 完整
- **直接執行**: `python -m services.features.function_xss`
- **參數支持**: 7 個（url, type, param, method, location, timeout, view-url）
- **測試狀態**: ✅ 已通過實戰測試
- **靶場驗證**: Juice Shop, WebGoat
- **攻擊模式**: Reflected, DOM, Stored
- **Worker 需求**: ❌ 不需要

#### ⚠️ function_ssrf
- **流程數**: 2
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Python Worker
- **測試狀態**: 未測試

#### ⚠️ function_sqli
- **流程數**: 2
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Python Worker
- **測試狀態**: 未測試

#### ⚠️ function_idor
- **流程數**: 2
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Python Worker
- **測試狀態**: 未測試

#### ⚠️ function_bizlogic
- **流程數**: 2
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Python Worker
- **測試狀態**: 未測試

---

### Go 模塊（1個）

#### ⚠️ function_authn_go
- **流程數**: 4
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Go Worker
- **測試狀態**: 未測試
- **Go 版本**: 1.25.0 ✅ 已安裝
- **模組結構**: cmd/worker/, internal/

---

### TypeScript 模塊（1個）

#### ⚠️ typescript_engine
- **流程數**: 3
- **CLI 接口**: ❌ 無
- **執行方式**: Worker-based
- **Worker 需求**: ✅ RabbitMQ + Node.js Worker
- **測試狀態**: 未測試
- **路徑狀態**: 需要確認

---

## 🛠️ 基礎設施需求

### 已完成
- ✅ Python 3.13 環境
- ✅ Go 1.25.0 環境
- ✅ 分類數據生成器
- ✅ 統一執行器框架
- ✅ 交互式選單系統
- ✅ CLI 命令數據庫（210條）
- ✅ 文檔生成系統

### 待部署
- ⏳ RabbitMQ 消息隊列
- ⏳ Python Worker 服務
- ⏳ Go Worker 服務
- ⏳ TypeScript Worker 服務
- ⏳ Worker 管理系統
- ⏳ 任務調度系統

---

## 📈 測試覆蓋率

### Python XSS 模塊
| 測試項 | 狀態 | 覆蓋率 |
|--------|------|--------|
| Reflected XSS | ✅ | 100% |
| DOM XSS | ✅ | 100% |
| Stored XSS | ✅ | 100% |
| GET Method | ✅ | 100% |
| POST Method | ✅ | 100% |
| Query Location | ✅ | 100% |
| Body Location | ✅ | 100% |
| Header Location | ❌ | 0% |
| Juice Shop | ✅ | 4 endpoints |
| WebGoat | ✅ | 2 endpoints |

### 其他模塊
| 模塊 | CLI 測試 | Worker 測試 | 靶場測試 |
|------|----------|-------------|----------|
| SSRF | ❌ | ❌ | ❌ |
| SQLi | ❌ | ❌ | ❌ |
| IDOR | ❌ | ❌ | ❌ |
| BizLogic | ❌ | ❌ | ❌ |
| Authn (Go) | ❌ | ❌ | ❌ |
| TypeScript Engine | ❌ | ❌ | ❌ |

---

## 🚀 執行能力矩陣

| 模塊 | 語言 | 直接CLI | Worker | 測試通過 | 生產就緒 |
|------|------|---------|--------|----------|----------|
| function_xss | Python | ✅ | ❌ | ✅ | ✅ |
| function_ssrf | Python | ❌ | ✅ | ⏳ | ⏳ |
| function_sqli | Python | ❌ | ✅ | ⏳ | ⏳ |
| function_idor | Python | ❌ | ✅ | ⏳ | ⏳ |
| function_bizlogic | Python | ❌ | ✅ | ⏳ | ⏳ |
| function_authn_go | Go | ❌ | ✅ | ⏳ | ⏳ |
| typescript_engine | TypeScript | ❌ | ✅ | ⏳ | ⏳ |

---

## 📂 關鍵檔案位置

### 執行器與分類器
```
services/core/aiva_core/internal_exploration/
├── aiva_external_executor.py        # 統一執行器
├── aiva_external_classifier.py      # 分類數據生成器
├── classification_data.json         # 210個flow的分類數據
└── InteractiveMenu (class)          # 4層交互式選單
```

### 模塊實現
```
services/features/
├── function_xss/
│   ├── __main__.py                  # ✅ CLI 入口
│   ├── detector.py                  # XSS 檢測核心
│   └── payloads.py                  # Payload 生成
├── function_ssrf/
│   ├── worker.py                    # ⏳ Worker 入口
│   └── detector.py
├── function_sqli/
│   ├── worker.py                    # ⏳ Worker 入口
│   └── detector.py
└── ...
```

### 文檔與腳本
```
docs/
├── MULTI_LANGUAGE_CAPABILITY_TESTING_REPORT.md  # 測試報告
├── EXTERNAL_CLI_VERIFICATION_REPORT.md          # CLI驗證報告
└── EXTERNAL_MODULE_EXECUTION_GUIDE.md           # 執行指南

features_classification/
├── EXTERNAL_CLI_COMMANDS_REFERENCE.md           # CLI命令參考
└── external_cli_commands_db.json                # CLI命令數據庫

(root)/
├── test_multi_capabilities.ps1                   # 測試腳本
└── 啟動外部能力選單.bat                          # 選單啟動
```

---

## 🎯 使用指南

### 快速啟動 XSS 測試
```bash
# 1. Reflected XSS
python -m services.features.function_xss \
    --url "http://localhost:3000" \
    --param q \
    --type reflected

# 2. DOM XSS
python -m services.features.function_xss \
    --url "http://localhost:3000/#/search" \
    --type dom

# 3. Stored XSS
python -m services.features.function_xss \
    --url "http://localhost:3000/api/Feedbacks" \
    --param comment \
    --type stored \
    --view-url "http://localhost:3000/#/about"
```

### 啟動交互式選單
```bash
.\啟動外部能力選單.bat
```
或
```bash
cd services/core/aiva_core/internal_exploration
python aiva_external_executor.py --menu
```

### 查看所有可用命令
```bash
cd services/core/aiva_core/internal_exploration
python aiva_external_executor.py --list
```

### 生成文檔
```bash
cd services/core/aiva_core/internal_exploration
python aiva_external_executor.py --generate-doc md
python aiva_external_executor.py --generate-doc json
```

---

## 📊 性能指標

### XSS 模塊性能
- **平均響應時間**: ~2-3秒/端點
- **Payload 生成**: 3個/請求
- **HTTP 請求**: 1-6個/測試
- **超時設置**: 10-30秒可調
- **內存佔用**: <100MB
- **CPU 使用**: <10%

### 系統資源
- **Python 進程**: 1個
- **內存使用**: ~150MB（包含依賴）
- **硬盤空間**: ~500MB（代碼+數據）
- **網路帶寬**: ~10KB/請求

---

## 🔧 故障排除

### 常見問題

#### 1. ModuleNotFoundError
```
問題: No module named services.features.function_xxx.__main__
解決: 該模塊需要 Worker，無法直接 CLI 執行
```

#### 2. 靶場無回應
```
問題: Connection refused
解決: 確認靶場服務已啟動
     - Juice Shop: docker ps | grep juice-shop
     - WebGoat: docker ps | grep webgoat
```

#### 3. Worker 模塊無法執行
```
問題: 需要 RabbitMQ
解決: 部署 Worker 基礎設施（見下一階段計劃）
```

---

## 🚀 下階段計劃

### Phase 1: Worker 系統部署（1週）
- [ ] 安裝配置 RabbitMQ
- [ ] 啟動 Python Worker 服務
- [ ] 測試 SSRF/SQLi/IDOR/BizLogic
- [ ] 監控系統部署

### Phase 2: Go/TypeScript 集成（1週）
- [ ] 啟動 Go Worker 服務
- [ ] 測試 Authentication 模塊
- [ ] 配置 TypeScript 環境
- [ ] 測試前端分析引擎

### Phase 3: 統一接口（2週）
- [ ] 完善 aiva_external_executor.py
- [ ] 整合所有模塊到選單
- [ ] 實現自動化報告生成
- [ ] 性能優化

### Phase 4: 生產部署（1週）
- [ ] Docker 容器化
- [ ] CI/CD 流程
- [ ] 雲端部署
- [ ] 文檔完善

---

## 📝 更新日誌

### 2026-01-14
- ✅ 完成 Python XSS 模塊實戰測試
- ✅ 驗證 15+ HTTP 請求成功發送
- ✅ 測試 Reflected/DOM/Stored 三種模式
- ✅ 創建自動化測試腳本
- ✅ 生成詳細測試報告
- ✅ 更新 README.md 進度

### 2026-01-13
- ✅ 完成 210 個 flow 分類
- ✅ 生成統一分類數據
- ✅ 實現交互式選單系統
- ✅ 創建 CLI 命令數據庫

### 2026-01-12
- ✅ 架構簡化完成
- ✅ 移除冗餘服務
- ✅ 保留核心 API

---

**報告生成時間**: 2026-01-14  
**下次更新**: Worker 系統部署後  
**維護者**: AIVA 開發團隊
