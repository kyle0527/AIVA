# 🚀 AIVA 重複定義修復 - 快速開始

## ⚡ 一分鐘快速修復

```powershell
# 1. 試運行預覽 (安全)
.\fix-duplicates.ps1 -Phase 1 -DryRun

# 2. 執行修復 
.\fix-duplicates.ps1 -Phase 1

# 3. 驗證結果
.\fix-duplicates.ps1 -Verify
```

## 📋 詳細步驟

### Step 1: 環境準備
```powershell
# 確認在 AIVA 專案根目錄
ls pyproject.toml

# 創建修復分支
git checkout -b fix/duplicate-definitions-phase-1
```

### Step 2: 試運行預覽
```powershell
# 查看修復計劃（不實際修改檔案）
.\fix-duplicates.ps1 -Phase 1 -DryRun

# 如果需要詳細輸出
.\fix-duplicates.ps1 -Phase 1 -DryRun -Verbose
```

### Step 3: 執行修復
```powershell
# 執行實際修復
.\fix-duplicates.ps1 -Phase 1

# 系統會要求確認，輸入 'y' 繼續
```

### Step 4: 驗證結果
```powershell
# 驗證修復是否成功
.\fix-duplicates.ps1 -Verify

# 運行系統健康檢查
python scripts/utilities/health_check.py
```

### Step 5: 提交變更
```powershell
# 查看修改的檔案
git status

# 提交修復
git add .
git commit -m "🔧 Phase 1: Fix duplicate definitions

✅ Fixed enum duplications: RiskLevel, DataFormat, EncodingType
✅ Unified core models: Target, Finding
✅ All verification tests passed"

# 推送分支 (可選)
git push origin fix/duplicate-definitions-phase-1
```

## 🔍 故障排除

### 問題：找不到 Python
```powershell
# 檢查 Python 安裝
python --version

# 如果沒有安裝，請安裝 Python 3.11+
```

### 問題：缺少依賴
```powershell
# 重新安裝依賴
pip install -e .

# 檢查 aiva_common 模組
ls services/aiva_common/
```

### 問題：權限錯誤
```powershell
# 以管理員權限運行 PowerShell
# 或確保有寫入專案檔案的權限
```

## 📊 預期結果

### 修復項目
- ✅ RiskLevel 枚舉重複 → 統一定義
- ✅ DataFormat vs MimeType → 重命名區分
- ✅ EncodingType 重複 → 合併定義
- ✅ Target 模型重複 → 移除廢棄定義
- ✅ Finding 模型混合 → 統一為 Pydantic 模型

### 驗證測試
- ✅ 導入測試：所有模組可正常導入
- ✅ Schema 一致性：符合 AIVA Common 標準
- ✅ 系統健康：核心功能正常運作

## 🎯 下一步

修復完成後，建議：
1. 運行完整測試套件：`python -m pytest tests/`
2. 檢查文檔是否需要更新
3. 考慮執行階段二修復（跨語言合約統一）

## 📞 需要幫助？

- 查看詳細文檔：[重複定義問題分析報告](reports/analysis/重複定義問題一覽表.md)
- 工具使用說明：`.\fix-duplicates.ps1 -Help`
- 示例用法：`python scripts/analysis/example_usage.py`