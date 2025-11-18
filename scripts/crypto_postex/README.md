# 🔐 Crypto & Post-Exploitation Tools

> **加密與滲透測試工具目錄** - AIVA 安全測試專用工具集  
> **工具類型**: 加密引擎、後滲透工具、Docker 容器化工具  
> **腳本數量**: 7個專業安全測試工具

---

## 📋 目錄概述

此目錄包含 AIVA 系統的加密引擎構建工具和後滲透測試工具。這些工具主要用於安全測試、漏洞評估和加密功能驗證，支援 Docker 容器化部署。

---

## 🗂️ 工具列表

### 🔧 構建工具
- **`build_crypto_engine.sh`** - 加密引擎構建腳本
- **`build_docker_crypto.sh`** - 加密服務 Docker 映像構建
- **`build_docker_postex.sh`** - 後滲透工具 Docker 映像構建

### 🏃 執行工具
- **`run_crypto_worker.sh`** - 加密工作程序啟動器
- **`run_postex_worker.sh`** - 後滲透工作程序啟動器

### 📝 合約與測試
- **`gen_contracts.sh`** - 智能合約生成工具
- **`run_tests.sh`** - 安全測試套件執行器

---

## ⚠️ 使用注意事項

### 🔒 安全警告
- 這些工具僅供**授權的安全測試**使用
- 必須在**隔離環境**中執行
- 使用前請確保擁有**適當的法律授權**

### 🛠️ 環境要求
- Linux/Unix 環境或 WSL
- Docker 引擎已安裝
- 適當的系統權限

---

## 🚀 快速使用

### 構建加密引擎
```bash
./build_crypto_engine.sh
```

### 構建 Docker 映像
```bash
./build_docker_crypto.sh
./build_docker_postex.sh
```

### 執行安全測試
```bash
./run_tests.sh
```

---

**⚠️ 重要提醒**: 本目錄的工具僅供合法的安全測試和研究用途使用。使用者有責任確保符合當地法律法規。

**維護者**: AIVA Security Team  
**最後更新**: 2025-11-17  
**工具狀態**: ✅ 安全測試工具已驗證

---

[← 返回 Scripts 主目錄](../README.md)