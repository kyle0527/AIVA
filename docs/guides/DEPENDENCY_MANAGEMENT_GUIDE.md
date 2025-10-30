# AIVA 依賴管理指引

> **快速參考指南** - 適用於日常開發工作

## 🚀 **快速開始**

### **檢查當前環境**
```bash
# 確認虛擬環境
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" --version

# 檢查核心依賴
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" -c "import fastapi, pydantic, redis; print('✅ 核心依賴正常')"
```

### **執行系統檢查**
```bash
# 完整系統測試
& "C:/D/fold7/AIVA-git/.venv/Scripts/python.exe" testing\common\complete_system_check.py
```

---

## 📦 **依賴安裝指引**

### **1. 基礎開發環境** (已完成 ✅)
```bash
# 已安裝，無需重複執行
pip install -e .
```

### **2. AI 功能依賴** (按需安裝)
```bash
# 機器學習基礎
pip install scikit-learn>=1.3.0 numpy>=1.24.0

# 深度學習 (僅 DQN/PPO 需要)
pip install torch>=2.1.0 torchvision>=0.16.0

# 強化學習環境
pip install gymnasium>=0.29.0
```

### **3. 微服務通訊** (可選)
```bash
# gRPC 支援
pip install grpcio>=1.60.0 grpcio-tools>=1.60.0 protobuf>=4.25.0
```

### **4. 監控和文件** (可選)
```bash
# 監控工具
pip install prometheus-client>=0.20

# PDF 報告
pip install reportlab>=3.6

# 型別提示
pip install types-requests>=2.31.0
```

---

## ⚠️ **常見問題處理**

### **問題 1: FastAPI 循環導入**
```bash
# 解決方案
pip uninstall fastapi -y
pip install fastapi==0.115.0
```

### **問題 2: 模組導入失敗**
```bash
# 重新安裝專案
pip install -e .
```

### **問題 3: 配置屬性缺失**
- 檢查 `services/aiva_common/config/unified_config.py`
- 確認所有必要屬性已定義

### **問題 4: Docker 服務未啟動**
```bash
# 啟動基礎服務 (需要 Docker)
docker-compose up -d redis rabbitmq postgres neo4j
```

---

## 🔍 **依賴健康檢查**

### **定期檢查項目**
```bash
# 1. 檢查過時的套件
pip list --outdated

# 2. 檢查安全漏洞
pip audit

# 3. 檢查依賴樹
pip show --verbose fastapi
```

### **清理未使用依賴**
```bash
# 安裝清理工具
pip install pip-autoremove

# 清理未使用的套件 (謹慎使用)
pip-autoremove -y
```

---

## 📊 **版本管理策略**

### **主要依賴版本鎖定**
| 套件 | 鎖定版本 | 原因 |
|------|----------|------|
| `fastapi` | 0.115.0 | 穩定性 |
| `pydantic` | 2.12.3 | 相容性 |
| `sqlalchemy` | 2.0.44 | 功能完整 |

### **允許彈性更新的套件**
- 開發工具 (black, ruff, mypy)
- 監控工具 (psutil)
- 文件工具 (types-*)

---

## 🎯 **開發階段依賴指引**

### **階段 1: 核心功能開發** (目前階段)
- ✅ 基礎 Web 框架
- ✅ 資料庫連接
- ✅ 訊息佇列
- ⏳ Docker 服務設置

### **階段 2: AI 功能整合**
- 🔄 安裝機器學習依賴
- 🔄 配置深度學習框架
- 🔄 整合 RL 環境

### **階段 3: 生產部署**
- 🔄 監控工具整合
- 🔄 安全性強化
- 🔄 性能優化工具

---

## 📋 **檢查清單**

### **新開發者加入**
- [ ] 檢查 Python 版本 (3.13.9)
- [ ] 創建虛擬環境
- [ ] 安裝基礎依賴 (`pip install -e .`)
- [ ] 執行系統檢查測試
- [ ] 確認配置文件正確

### **功能開發前**
- [ ] 檢查相關依賴是否已安裝
- [ ] 執行依賴健康檢查
- [ ] 更新文件記錄

### **功能完成後**
- [ ] 檢查是否引入新依賴
- [ ] 更新 `pyproject.toml`
- [ ] 更新依賴分析報告
- [ ] 執行完整系統測試

---

## 🔗 **相關資源**

- [依賴分析詳細報告](./DEPENDENCY_ANALYSIS_REPORT.md)
- [系統測試腳本](../testing/common/complete_system_check.py)
- [專案配置](../pyproject.toml)
- [需求清單](../requirements.txt)