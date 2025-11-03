# AIVA 重複定義問題修復工具

[![AIVA Version](https://img.shields.io/badge/AIVA-v5.0-blue.svg)](https://github.com/aiva-platform/aiva)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Ready-success.svg)](https://github.com/aiva-platform/aiva)

## 📝 概述

這是 AIVA v5.0 跨語言統一架構的重複定義問題自動化修復工具。遵循 AIVA Common 開發規範，提供安全、可靠的修復解決方案。

## 🚀 快速開始

### 基本使用

```powershell
# 1. 試運行模式（安全預覽）
.\fix-duplicates.ps1 -Phase 1 -DryRun

# 2. 執行修復
.\fix-duplicates.ps1 -Phase 1

# 3. 驗證結果
.\fix-duplicates.ps1 -Verify
```

### 完整流程

```powershell
# 建立修復分支
git checkout -b fix/duplicate-definitions-phase-1

# 查看工具說明
.\fix-duplicates.ps1 -Help

# 試運行預覽
.\fix-duplicates.ps1 -Phase 1 -DryRun

# 執行修復
.\fix-duplicates.ps1 -Phase 1

# 驗證修復結果
.\fix-duplicates.ps1 -Verify

# 提交變更
git add .
git commit -m "🔧 Phase 1 duplicate definitions fix"
```

## 📂 檔案結構

```
scripts/analysis/
└── duplication_fix_tool.py      # Python 修復工具核心
fix-duplicates.ps1                # PowerShell 執行腳本
reports/analysis/
└── 重複定義問題一覽表.md         # 詳細分析報告
```

## 🔧 工具功能

### 階段一修復 (已實現)
- ✅ **枚舉重複定義修復**
  - RiskLevel 枚舉合併
  - DataFormat 枚舉重命名
  - EncodingType 枚舉統一

- ✅ **核心模型統一**
  - Target 模型統一
  - Finding 模型統一

- ✅ **完整驗證機制**
  - 導入測試
  - Schema 一致性檢查
  - 系統健康檢查

### 後續階段 (規劃中)
- 🔄 **階段二**: 跨語言合約統一
- 🔄 **階段三**: 功能模組整合
- 🔄 **階段四**: 完整驗證與文檔更新

## 📋 使用參數

### PowerShell 腳本參數
| 參數 | 類型 | 說明 |
|------|------|------|
| `-Phase` | int | 指定執行階段 (1-4) |
| `-Verify` | switch | 驗證修復結果 |
| `-DryRun` | switch | 試運行模式（不實際修改檔案） |
| `-Verbose` | switch | 詳細輸出模式 |
| `-Help` | switch | 顯示使用說明 |

### Python 工具參數
| 參數 | 說明 |
|------|------|
| `--phase 1` | 執行階段一修復 |
| `--verify` | 驗證修復結果 |
| `--dry-run` | 試運行模式 |
| `--verbose` | 詳細輸出模式 |

## 🔒 安全特性

- **🔍 試運行模式**: 預覽修復計劃，不實際修改檔案
- **✅ 環境檢查**: 自動檢查 Python 環境和依賴
- **⚠️ 用戶確認**: 重要操作需要用戶確認
- **📝 完整日誌**: 記錄所有操作過程
- **🔄 向後相容**: 保證 100% 向後相容性

## 📊 修復項目

### 當前支援的重複定義問題
1. **枚舉重複定義** (5 項)
   - RiskLevel 重複定義
   - DataFormat vs MimeType 混用
   - EncodingType 重複定義
   - ContentType 衝突
   - ProcessingStatus 不一致

2. **核心模型重複** (2 項)
   - Target 模型重複定義
   - Finding 模型混合定義

## 🧪 驗證機制

### 自動驗證測試
- **導入測試**: 驗證所有模組可正常導入
- **Schema 一致性**: 檢查 Schema 定義符合標準
- **系統健康檢查**: 確保核心功能正常運作

### 手動驗證建議
```bash
# 1. 運行健康檢查
python scripts/utilities/health_check.py

# 2. 執行測試套件
python -m pytest tests/

# 3. Schema 驗證
python scripts/validation/schema_compliance_validator.py
```

## 🐛 故障排除

### 常見問題

**Q: 提示 "Python 未安裝或無法訪問"**
```powershell
# 檢查 Python 安裝
python --version

# 確保在正確環境中
pip install -e .
```

**Q: 提示 "缺少必要檔案"**
```powershell
# 確認在 AIVA 專案根目錄
ls pyproject.toml

# 檢查 aiva_common 模組
ls services/aiva_common/
```

**Q: 修復後出現導入錯誤**
```powershell
# 運行驗證工具
.\fix-duplicates.ps1 -Verify

# 檢查具體錯誤
python scripts/analysis/duplication_fix_tool.py --verify --verbose
```

## 📖 相關文檔

- [AIVA 重複定義問題分析報告](reports/analysis/重複定義問題一覽表.md)
- [AIVA 開發規範指南](guides/AIVA_COMPREHENSIVE_GUIDE.md)
- [Schema 合規驗證工具](scripts/validation/schema_compliance_validator.py)
- [系統健康檢查工具](scripts/utilities/health_check.py)

## 🤝 貢獻指南

1. 確保遵循 AIVA Common 開發規範
2. 所有修復必須通過完整驗證測試
3. 保持 100% 向後相容性
4. 更新相關文檔和測試

## 📧 支援

如果遇到問題或需要協助，請：
1. 查看故障排除章節
2. 運行 `.\fix-duplicates.ps1 -Help` 查看詳細說明
3. 檢查相關文檔
4. 創建 Issue 報告問題

---

**版本**: v1.0.0  
**更新**: 2025-11-03  
**作者**: AIVA 架構團隊