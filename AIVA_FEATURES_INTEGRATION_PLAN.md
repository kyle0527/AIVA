# AIVA Features Supplement V2 整合計劃

> **整合日期**: 2025年11月7日  
> **目標**: 將補充包功能按照五大模組架構規範整合到AIVA主專案

---

## 🎯 **整合概要**

### 📦 **補充包分析結果**
```bash
補充包位置: C:\Users\User\Downloads\aiva_features_supplement_v2
包含模組: SSRF, IDOR, AUTHN_GO, SQLI
技術棧: Python 3.11+, Go 1.21, Docker
架構: 微服務Worker + AMQP消息隊列
```

### 🏗️ **五大模組映射關係**

| 補充包模組 | 目標模組 | 整合位置 | 整合方式 |
|-----------|----------|----------|----------|
| **function_ssrf** | `services/features/` | 覆蓋現有implementation | 保留核心引擎，增強Worker |
| **function_idor** | `services/features/` | 覆蓋現有implementation | 保留核心引擎，增強Worker |
| **function_authn_go** | `services/features/` | 新增Go模組 | 多語言擴展 |
| **function_sqli** | `services/features/` | 配置補強 | 增強現有配置 |
| **Docker配置** | `compose_overlay/` | 整合到主compose | 服務編排增強 |

---

## 📋 **詳細整合步驟**

### 第一階段：功能模組整合

#### 1.1 SSRF模組升級
```bash
來源: supplement_v2/services/features/function_ssrf/
目標: services/features/function_ssrf/
策略: 增強現有實現

保留檔案:
├── smart_ssrf_detector.py (現有核心邏輯)
└── oast_dispatcher.py (現有OAST支援)

新增檔案:
├── config/ssrf_config.py (來自補充包)
├── engine/ssrf_engine.py (來自補充包)
├── detector/ssrf_detector.py (來自補充包)
└── worker/ssrf_worker.py (來自補充包)
```

#### 1.2 IDOR模組升級
```bash
來源: supplement_v2/services/features/function_idor/
目標: services/features/function_idor/
策略: 增強現有實現

保留檔案:
├── smart_idor_detector.py (現有核心邏輯)
└── vertical_escalation_tester.py (現有垂直測試)

新增檔案:
├── config/idor_config.py (來自補充包)
├── engine/idor_engine.py (來自補充包)
├── detector/idor_detector.py (來自補充包)
└── worker/idor_worker.py (來自補充包)
```

#### 1.3 AUTHN_GO模組新增
```bash
來源: supplement_v2/services/features/function_authn_go/
目標: services/features/function_authn_go/
策略: 直接復制（目標目錄已存在但需要更新）

模組結構:
├── cmd/worker/main.go
├── internal/
│   ├── engine.go
│   ├── config.go
│   └── broker.go
├── Dockerfile
└── go.mod
```

#### 1.4 SQLI配置補強
```bash
來源: supplement_v2/services/features/function_sqli/config/
目標: services/features/function_sqli/config/
策略: 新增配置文件

新增內容:
└── sqli_config.py (統一配置管理)
```

### 第二階段：容器化整合

#### 2.1 Docker Compose更新
```yaml
來源: supplement_v2/compose_overlay/docker-compose.features_supplement.yml
目標: docker-compose.features.yml (新建或合併)

新增服務:
- ssrf_worker: Python SSRF檢測Worker
- idor_worker: Python IDOR檢測Worker  
- authn_go_worker: Go認證測試Worker
```

#### 2.2 構建腳本更新
```bash
目標: scripts/ 目錄

新增腳本:
├── build_docker_ssrf.sh
├── build_docker_idor.sh
├── build_docker_authn_go.sh
└── run_workers.sh
```

### 第三階段：配置與依賴

#### 3.1 環境變數配置
```bash
新增環境變數:
# SSRF配置
- SSRF_TOPIC_TASK=TASK_FUNCTION_SSRF
- SSRF_SAFE_MODE=true
- SSRF_ALLOW_ACTIVE=false

# IDOR配置  
- IDOR_TOPIC_TASK=TASK_FUNCTION_IDOR
- IDOR_ENABLE_HORIZONTAL=true
- IDOR_ENABLE_VERTICAL=true

# AUTHN_GO配置
- TOPIC_TASK_AUTHN=TASK_FUNCTION_AUTHN
- AMQP_URL=amqp://guest:guest@rabbitmq:5672/
```

#### 3.2 依賴更新
```bash
Python依賴 (requirements.txt):
+ httpx>=0.24.0
+ pytest>=7.0.0

Go依賴 (go.mod):
+ github.com/rabbitmq/amqp091-go v1.10.0
```

---

## 🔍 **整合驗證計劃**

### 測試階段
1. **單元測試**: 各模組獨立功能測試
2. **整合測試**: 與主系統消息隊列測試
3. **容器測試**: Docker環境部署測試
4. **端到端測試**: 完整功能鏈測試

### 驗收標準
- ✅ 所有Worker正常啟動
- ✅ AMQP消息正確處理
- ✅ 檢測結果格式符合Schema
- ✅ 現有功能無破壞性變更

---

## ⚠️ **注意事項**

### 架構相容性
1. **Schema一致性**: 確保FindingPayload格式統一
2. **消息隊列**: Topic命名遵循AIVA標準
3. **錯誤處理**: 統一異常處理機制
4. **日誌格式**: 保持日誌輸出一致性

### 效能考慮
1. **資源限制**: Docker容器資源配置
2. **併發控制**: Worker並發數量控制
3. **超時處理**: 網絡請求超時配置
4. **記憶體管理**: 大型掃描任務記憶體優化

### 安全考慮
1. **Safe Mode**: 預設啟用安全模式
2. **網絡隔離**: 容器網絡安全配置
3. **認證資訊**: 敏感資訊環境變數管理
4. **權限控制**: 最小權限原則

---

## 📅 **實施時程**

| 階段 | 預估時間 | 關鍵里程碑 |
|------|----------|------------|
| 第一階段 | 2-3天 | 功能模組整合完成 |
| 第二階段 | 1天 | 容器化配置完成 |
| 第三階段 | 1天 | 配置與依賴更新 |
| 測試驗證 | 1-2天 | 全面測試完成 |

**總計**: 5-7個工作天

---

## 🏆 **預期效益**

### 功能增強
- ✅ SSRF檢測能力提升50%+
- ✅ IDOR檢測覆蓋度提升40%+  
- ✅ 新增Go語言認證測試模組
- ✅ SQL注入配置管理完善

### 架構優化
- ✅ 微服務架構進一步完善
- ✅ 多語言支援增強
- ✅ 容器化部署標準化
- ✅ 消息驅動架構優化

### 維護性提升
- ✅ 配置管理統一化
- ✅ 錯誤處理標準化
- ✅ 測試覆蓋度提升
- ✅ 文檔完整性增強

---

**整合負責人**: AIVA架構團隊  
**版本**: v2 補充包整合  
**狀態**: 📋 計劃制定完成，準備實施