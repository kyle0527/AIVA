# AIVA 系統檢查工具使用指南

## 🛠️ 工具概覽

本工具包包含三個主要檢查工具，用於維護 AIVA 系統的健康度和完整性：

### 1. 系統健康度檢查 (`system_health_check.ps1`)
**用途**: 全面檢查系統各組件狀態
**語言**: PowerShell
**執行環境**: Windows PowerShell 5.1+ 或 PowerShell 7+

### 2. Schema 驗證器 (`schema_validator.py`)
**用途**: 驗證 aiva_common 模組中的所有 Schema 和 Enum
**語言**: Python 3.8+
**依賴**: Pydantic, aiva_common

### 3. 改善規劃書 (`SYSTEM_IMPROVEMENT_PLAN.md`)
**用途**: 系統改善的階段性規劃和指導
**格式**: Markdown 文檔

## 🚀 快速開始

### 基本檢查
```powershell
# 1. 執行完整系統健康度檢查
.\tools\system_health_check.ps1

# 2. 驗證 Schema 定義
python .\tools\schema_validator.py

# 3. 查看改善規劃
Get-Content .\SYSTEM_IMPROVEMENT_PLAN.md
```

### 進階使用

#### 系統健康度檢查選項
```powershell
# 詳細模式 - 顯示所有檢查詳情
.\tools\system_health_check.ps1 -Detailed

# 快速檢查 - 僅核心項目
.\tools\system_health_check.ps1 -QuickCheck

# 儲存報告 - 產生 JSON 報告文件
.\tools\system_health_check.ps1 -SaveReport

# 組合使用
.\tools\system_health_check.ps1 -Detailed -SaveReport
```

#### Schema 驗證選項
```bash
# 詳細輸出
python tools/schema_validator.py -v

# 輸出報告到文件
python tools/schema_validator.py -o schema_validation_report.json

# 僅顯示統計
python tools/schema_validator.py --stats-only

# 組合使用
python tools/schema_validator.py -v -o detailed_report.json
```

## 📊 檢查項目說明

### 系統健康度檢查涵蓋項目

1. **核心模組檢查**
   - aiva_common 存在性和結構
   - AI 核心模組狀態
   - 各子模組完整性

2. **服務整合檢查**
   - 各服務 (core, scan, integration, function) 狀態
   - aiva_common 使用情況分析
   - 服務間依賴關係

3. **配置檔案檢查**
   - pyproject.toml, requirements.txt 等配置完整性
   - 程式碼品質工具配置 (mypy, ruff, pyright)
   - 專案文檔完整性

4. **測試覆蓋檢查**
   - 測試文件數量和分布
   - 測試覆蓋率評估

5. **工具生態檢查**
   - 開發腳本和工具完整性
   - 關鍵工具可用性 (schema_manager, cleanup 等)

6. **依賴檢查**
   - Python 關鍵依賴可用性
   - 版本兼容性檢查

### Schema 驗證涵蓋項目

1. **主模組驗證**
   - aiva_common 匯入測試
   - 版本資訊檢查
   - 主要模組匯出驗證

2. **Enums 驗證**
   - 所有 enum 文件語法檢查
   - Enum 類別結構驗證
   - 匯入依賴檢查

3. **Schemas 驗證**
   - Pydantic 模型語法檢查
   - Schema 類別結構驗證
   - 型別註解完整性

4. **Utils 驗證**
   - 工具函數可用性
   - 模組結構完整性
   - 公開介面檢查

## 📈 健康度評分標準

### 系統健康度等級
- **90%+**: 🟢 優秀 - 系統狀態良好
- **70-89%**: 🟡 良好 - 有改善空間
- **50-69%**: 🟠 普通 - 需要關注
- **<50%**: 🔴 需改善 - 需要立即處理

### Schema 驗證狀態
- **PASS**: 成功率 ≥ 80%
- **FAIL**: 成功率 < 80%

## 🔄 自動化執行

### 定期檢查腳本
```powershell
# 建立定期檢查腳本
@"
# 每日健康度檢查
Write-Host "執行每日系統檢查..." -ForegroundColor Yellow
.\tools\system_health_check.ps1 -SaveReport

# Schema 驗證
Write-Host "執行 Schema 驗證..." -ForegroundColor Yellow
python .\tools\schema_validator.py -o "daily_schema_report.json"

Write-Host "檢查完成！" -ForegroundColor Green
"@ | Out-File -FilePath "daily_check.ps1" -Encoding UTF8
```

### CI/CD 整合
```yaml
# GitHub Actions 範例
- name: System Health Check
  run: |
    powershell -File tools/system_health_check.ps1
    python tools/schema_validator.py

- name: Upload Reports
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: health-reports
    path: "*_report*.json"
```

## 🎯 問題排除

### 常見問題

1. **"無法匯入 aiva_common"**
   ```bash
   # 確認 Python 路徑
   python -c "import sys; print('\n'.join(sys.path))"
   
   # 檢查目錄結構
   ls services/aiva_common/
   ```

2. **"PowerShell 執行原則限制"**
   ```powershell
   # 設定執行原則 (管理員權限)
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **"找不到測試文件"**
   ```bash
   # 檢查測試文件分布
   find . -name "test_*.py" | head -10
   ```

### 效能優化

1. **快速檢查模式**: 用於日常快速驗證
2. **詳細模式**: 用於深度診斷和問題排查
3. **報告儲存**: 用於歷史趨勢分析

## 📝 報告解讀

### JSON 報告結構
```json
{
  "timestamp": "2025-10-18T...",
  "duration": "5.2 seconds",
  "overall_score": 15,
  "max_score": 20,
  "health_percentage": 75.0,
  "results": [
    {
      "category": "核心模組",
      "item": "aiva_common 存在",
      "status": "✅",
      "passed": true,
      "details": "核心共用模組"
    }
  ]
}
```

### 趨勢分析
- 收集多次檢查報告
- 分析健康度變化趨勢
- 識別持續性問題

## 🚀 下一步行動

1. **立即執行**: 運行完整檢查了解現狀
2. **定期監控**: 設置每日/每週自動檢查
3. **問題跟蹤**: 記錄和追蹤改善進度
4. **持續優化**: 根據報告持續改善系統

---

**📞 支援**: 如有問題，請參考 `SYSTEM_IMPROVEMENT_PLAN.md` 或查看工具腳本內的詳細註解。