# AIVA 合約覆蓋率擴張實施指南

> **基於健康檢查結果**: 100% 健康度 ✅  
> **當前覆蓋率**: 16.1% (109/677 files)  
> **目標覆蓋率**: 25% (169 files)  
> **需要新增**: 60 個文件的合約覆蓋  

## 📋 執行前提條件

### ✅ 已驗證的健康基準
1. **FindingPayload**: 100% 正常運作 (最高使用率: 51次)
2. **AivaMessage**: 100% 正常運作 (統一訊息格式: 19次)  
3. **ScanStartPayload**: 100% 正常運作 (掃描系統核心)
4. **序列化/反序列化**: 100% 穩定性驗證通過

### 🌐 技術標準確認
- **Pydantic v2**: 完全相容 ✅
- **核心schema**: core_schema_sot.yaml 一致性 ✅
- **MCP架構**: 四支柱設計完整實作 ✅
- **驗證規則**: ID格式、字段約束正常 ✅

## 🎯 階段性擴張計劃

### 第一階段: 功能模組標準化 (16.1% → 20%)
**目標週期**: 2 週  
**新增文件**: 27 個

#### 優先模組:
```
services/features/function_*/ (功能模組)
├── function_xss/ 
├── function_sql_injection/
├── function_directory_enumeration/
├── function_file_upload/
└── function_authentication_bypass/
```

#### 具體行動:
1. **導入標準合約**:
   ```python
   from services.aiva_common.schemas import (
       FindingPayload,        # 替換本地Finding類
       AivaMessage,          # 統一訊息格式
       FunctionTelemetry,    # 功能遙測
       APIResponse           # 標準API響應
   )
   ```

2. **替換模式**:
   - `dict` 響應 → `APIResponse[T]`
   - 本地`Finding`類 → `FindingPayload`  
   - 自定義訊息 → `AivaMessage`

### 第二階段: AI引擎標準化 (20% → 23%)
**目標週期**: 2 週  
**新增文件**: 20 個

#### 重點模組:
```
services/core/aiva_core/
├── ai_engine/ (AI決策引擎)
├── rag/ (檢索增強生成)
├── training/ (模型訓練)
└── decision/ (智能決策)
```

#### 關鍵合約:
- `AttackPlan` / `AttackStep` (攻擊規劃)
- `ModelTrainingConfig` (訓練配置)  
- `RAGQueryPayload` / `RAGResponsePayload` (RAG查詢)

### 第三階段: 整合服務標準化 (23% → 25%)
**目標週期**: 1 週  
**新增文件**: 13 個

#### 範圍:
```
services/integration/
├── aiva_integration/ (核心整合)
├── capability/ (能力管理)
└── reception/ (數據接收)
```

## 🛠️ 實施標準操作程序 (SOP)

### 步驟1: 檔案分析
```bash
# 使用自動化工具識別候選文件
python tools/contract_coverage_booster.py --quick-check

# 檢查具體文件的改進機會
python tools/contract_coverage_booster.py --analyze-file <file_path>
```

### 步驟2: 合約選擇
根據文件功能選擇適當合約：

| 文件類型 | 推薦合約 | 使用場景 |
|---------|----------|----------|
| API端點 | `APIResponse[T]` | 統一響應格式 |
| 漏洞檢測 | `FindingPayload` | 標準漏洞報告 |
| 訊息處理 | `AivaMessage` | 跨服務通訊 |
| 掃描功能 | `ScanStartPayload` | 掃描任務啟動 |
| AI決策 | `AttackPlan` | 攻擊策略規劃 |

### 步驟3: 代碼重構
```python
# 重構前 (示例)
def scan_result():
    return {
        "status": "success", 
        "findings": [{"type": "xss", "url": "..."}]
    }

# 重構後 (標準)  
def scan_result() -> APIResponse[List[FindingPayload]]:
    findings = [
        FindingPayload(
            finding_id=f"finding_{uuid4().hex[:12]}",
            vulnerability=Vulnerability(...),
            target=Target(...),
            # ... 其他必填字段
        )
    ]
    return APIResponse(
        success=True,
        data=findings,
        message="Scan completed successfully"
    )
```

### 步驟4: 驗證測試
每個重構文件必須通過：
```bash
# 1. 語法驗證
python -m py_compile <file_path>

# 2. 類型檢查  
mypy <file_path>

# 3. 合約驗證
python contract_health_checker_standard.py

# 4. 單元測試
python -m pytest tests/ -v
```

## 📊 進度追蹤機制

### 每週評估指標
```bash
# 執行覆蓋率檢查
python tools/contract_coverage_booster.py --output reports/weekly_progress

# 檢查健康度
python contract_health_checker_standard.py
```

### 關鍵績效指標 (KPI)
- **覆蓋率增長**: 每週至少 +1.5%
- **健康度維持**: 始終保持 ≥95%
- **錯誤率控制**: 新增錯誤 <2%  
- **性能影響**: 響應時間增加 <5%

## ⚠️ 風險控制措施

### 1. 漸進式部署
- 每次重構不超過5個文件
- 完成驗證後再進行下一批
- 保留回滾版本備份

### 2. 影響評估
```python
# 檢查文件依賴關係
python tools/dependency_analyzer.py --file <target_file>

# 評估影響範圍  
python tools/impact_assessment.py --changes <change_list>
```

### 3. 測試策略
- **單元測試**: 每個重構文件
- **整合測試**: 模組間介面
- **回歸測試**: 核心功能驗證
- **性能測試**: 響應時間監控

## 🎯 成功標準

### 階段完成標準
- ✅ 覆蓋率達到目標 (25%)
- ✅ 健康度保持優秀 (≥95%)  
- ✅ 所有測試通過 (100%)
- ✅ 性能指標正常 (±5%)

### 品質確認清單
- [ ] 所有新增合約遵循命名規範
- [ ] ID驗證規則正確實作  
- [ ] 序列化/反序列化正常
- [ ] 錯誤處理機制完整
- [ ] 文檔更新完成

## 📚 參考資源

### 技術文檔
- [Pydantic v2 官方文檔](https://docs.pydantic.dev/latest/)
- [AIVA 合約開發指南](guides/AIVA_合約開發指南.md)
- [MCP 架構文檔](AIVA_MCP_ARCHITECTURE_VERIFICATION_REPORT.md)

### 實用工具
- `tools/contract_coverage_booster.py` - 覆蓋率分析與任務生成
- `contract_health_checker_standard.py` - 健康度檢查
- `guides/development/SCHEMA_IMPORT_GUIDE.md` - Schema導入規範

---

**執行開始時間**: 2025-11-01  
**預計完成時間**: 2025-11-15 (2週內)  
**負責人**: 開發團隊  
**監督**: 架構師審查