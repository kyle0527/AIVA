# 🚀 System Startup Tools

> **系統啟動工具目錄** - AIVA 系統快速啟動工具  
> **工具類型**: 系統啟動、服務初始化、環境設定  
> **腳本數量**: 1個核心啟動腳本

---

## 📋 目錄概述

此目錄包含 AIVA 系統的啟動腳本，提供快速、安全、可靠的系統初始化功能。確保所有核心服務按正確順序啟動，並進行必要的環境檢查。

---

## 🗂️ 工具列表

### 🎯 啟動工具
- **`start-aiva.sh`** - AIVA 主要啟動腳本
  - 自動環境檢查
  - 服務依賴管理
  - 啟動順序控制
  - 健康狀態監控
  - 錯誤處理和恢復

---

## 🎯 主要功能

### 🔧 系統初始化
- **環境檢查**: 驗證系統環境和依賴
- **配置載入**: 載入系統配置和參數
- **服務啟動**: 按序啟動核心服務
- **健康檢查**: 驗證服務啟動狀態

### 🛡️ 安全啟動
- **權限檢查**: 驗證執行權限
- **安全配置**: 載入安全設定
- **日誌記錄**: 記錄啟動過程
- **錯誤處理**: 處理啟動異常

---

## 🚀 使用說明

### 基本啟動
```bash
# 標準啟動
./start-aiva.sh

# 詳細模式啟動
./start-aiva.sh --verbose

# 開發模式啟動
./start-aiva.sh --dev

# 安全模式啟動
./start-aiva.sh --safe
```

### 進階選項
```bash
# 指定配置文件
./start-aiva.sh --config /path/to/config.yml

# 僅啟動特定服務
./start-aiva.sh --service core,scan

# 跳過健康檢查
./start-aiva.sh --skip-healthcheck

# 後台啟動
./start-aiva.sh --daemon
```

---

## 📋 啟動檢查清單

### ✅ 系統環境
- [ ] Python 環境驗證
- [ ] 必要套件檢查
- [ ] 資料庫連接測試
- [ ] 網路連接檢查
- [ ] 磁碟空間驗證

### 🔒 安全檢查
- [ ] 執行權限確認
- [ ] SSL 憑證驗證
- [ ] API 金鑰檢查
- [ ] 防火牆配置
- [ ] 安全策略載入

### 🚀 服務啟動順序
1. **Core Services** - 核心服務模組
2. **Common Services** - 共用服務模組
3. **Feature Services** - 功能服務模組
4. **Integration Services** - 整合服務模組
5. **Scan Services** - 掃描服務模組
6. **Testing Services** - 測試服務模組

---

## 🔧 環境要求

- **作業系統**: Linux/Unix 或 WSL
- **Shell**: Bash 4.0+
- **Python**: 3.8+
- **權限**: 適當的系統權限
- **網路**: 穩定的網路連接

---

## 📊 啟動監控

### 日誌位置
```
/var/log/aiva/startup.log    # 啟動日誌
/var/log/aiva/error.log      # 錯誤日誌
/var/log/aiva/service.log    # 服務日誌
```

### 健康檢查端點
```
http://localhost:8080/health        # 整體健康狀態
http://localhost:8080/health/core   # 核心服務狀態
http://localhost:8080/health/scan   # 掃描服務狀態
```

---

## ⚠️ 故障排除

### 常見問題
1. **權限不足**: 確保有執行權限 `chmod +x start-aiva.sh`
2. **端口衝突**: 檢查是否有其他服務占用端口
3. **依賴缺失**: 使用 `--check-deps` 檢查依賴
4. **配置錯誤**: 驗證配置文件語法

### 恢復機制
- 自動重試失敗的服務
- 回滾至安全配置
- 緊急停機保護
- 錯誤通知機制

---

**🎯 目標**: 提供可靠、快速、安全的 AIVA 系統啟動體驗，確保所有服務正確初始化並處於健康狀態。

**維護者**: AIVA Operations Team  
**最後更新**: 2025-11-17  
**工具狀態**: ✅ 啟動腳本已測試驗證

---

[← 返回 Scripts 主目錄](../README.md)