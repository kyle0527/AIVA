# AIVA UI 面板啟動指南

> **最後更新**: 2025-11-22

## � 目錄

- [🚀 自動端口版本 (推薦)](#-自動端口版本-推薦)
- [⚙️ 手動端口設定](#-手動端口設定)
- [🔧 配置選項](#-配置選項)
- [🐛 常見問題](#-常見問題)
- [🔗 相關資源](#-相關資源)

## �🚀 自動端口版本 (推薦)

### Windows 用戶

**方法一: 使用 PowerShell 腳本 (最簡單)**
```powershell
# 在專案根目錄執行
.\start_ui_auto.ps1
```

**方法二: 使用 Python 腳本**
```bash
python src/launchers/start_ui_auto.py
# 或者
python examples/start_ui_auto.py
```

**方法三: 手動執行**
```bash
python -m services.core.aiva_core.ui_panel.auto_server
```

### 功能特色

✅ **自動端口選擇** - 不需要擔心端口衝突  
✅ **智慧重試** - 如果選擇的端口被佔用會自動嘗試其他端口  
✅ **偏好端口** - 優先嘗試常用端口 (8080, 8081, 3000, 5000, 9000)  
✅ **詳細日誌** - 清楚顯示啟動狀態和使用的端口  

### 使用範例

#### 1. 預設啟動 (自動端口)
```python
from services.core.aiva_core.ui_panel import start_auto_server

start_auto_server()
# 輸出: 🌐 位址: http://127.0.0.1:8080 (自動選擇)
```

#### 2. 指定偏好端口
```python
start_auto_server(
    preferred_ports=[3000, 8081, 9000]
)
```

#### 3. 指定主機和模式
```python
start_auto_server(
    mode="ai",  # 純 AI 模式
    host="0.0.0.0",  # 允許外部訪問
    preferred_ports=[8080]
)
```

## 🔧 傳統版本 (固定端口)

如果需要使用固定端口，可以使用原始的啟動函數：

```python
from services.core.aiva_core.ui_panel import start_ui_server

start_ui_server(port=8080)  # 固定使用 8080 端口
```

## 📱 訪問 UI

啟動成功後，會顯示訪問地址：
- **主頁**: http://127.0.0.1:端口號
- **API 文檔**: http://127.0.0.1:端口號/docs

## ⚠️ 常見問題

### Q: 啟動失敗怎麼辦？
A: 檢查是否安裝了必要套件：
```bash
pip install fastapi uvicorn pydantic
```

### Q: 無法訪問 UI？
A: 確保：
1. 防火牆沒有阻擋端口
2. 使用正確的 IP 地址 (通常是 127.0.0.1 或 localhost)
3. 伺服器正常啟動且顯示了正確的端口號

### Q: 想要從外部網路訪問？
A: 將 host 參數改為 `"0.0.0.0"`：
```python
start_auto_server(host="0.0.0.0")
```

## 🛠️ 開發模式

對於開發者，可以直接使用 uvicorn：
```bash
uvicorn services.core.aiva_core.ui_panel.improved_ui:app --reload --port 8080
```

---

## 🔗 相關資源

### 開發指南
- 📖 [開發快速指南](./DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [開發任務指南](./DEVELOPMENT_TASKS_GUIDE.md)
- 📖 [開發者指南](./DEVELOPER_GUIDE.md)
- 📖 [依賴管理指南](./DEPENDENCY_MANAGEMENT_GUIDE.md)
- 📖 [API 驗證指南](./API_VERIFICATION_GUIDE.md)
- 📖 [Schema 導入指南](./SCHEMA_IMPORT_GUIDE.md)

### 架構指南
- 📖 [Schema 統一指南](../architecture/SCHEMA_GUIDE.md)
- 📖 [架構兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 故障排除
- 📖 [導入問題解決](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### 使用者手冊
- 📚 [Core 模組手冊](../../docs/user_guides/01_core/AIVA_CORE_使用者手冊.md)

