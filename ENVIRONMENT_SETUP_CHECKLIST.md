# 🔧 AIVA 環境設置快速檢查清單

> **📋 適用情況**: 新設備部署、環境重建、疑難排解  
> **⏱️ 預估時間**: 5-10 分鐘  
> **📅 最後更新**: 2025-10-28

---

## 🚀 快速設置 (5分鐘完成)

### 1️⃣ 環境變數設置 (必須!)

**Windows PowerShell**:
```powershell
$env:AIVA_RABBITMQ_URL = "amqp://localhost:5672"
$env:AIVA_RABBITMQ_USER = "guest"
$env:AIVA_RABBITMQ_PASSWORD = "guest"

# 驗證設置
echo $env:AIVA_RABBITMQ_URL
```

**Windows CMD**:
```cmd
set AIVA_RABBITMQ_URL=amqp://localhost:5672
set AIVA_RABBITMQ_USER=guest
set AIVA_RABBITMQ_PASSWORD=guest

# 驗證設置
echo %AIVA_RABBITMQ_URL%
```

**Linux/macOS**:
```bash
export AIVA_RABBITMQ_URL="amqp://localhost:5672"
export AIVA_RABBITMQ_USER="guest"
export AIVA_RABBITMQ_PASSWORD="guest"

# 驗證設置
echo $AIVA_RABBITMQ_URL
```

### 2️⃣ 快速驗證

```bash
# 檢查系統健康狀態
python health_check.py

# 預期輸出應包含:
# 🧬 Schema 狀態: ✅ Schemas OK (完全可用)
```

---

## 🔄 更換設備完整檢查清單

### ✅ 基礎環境檢查

- [ ] **Python 版本**: `python --version` (需要 3.11+)
- [ ] **Git 可用**: `git --version`
- [ ] **Docker 可用** (如果使用靶場): `docker --version`

### ✅ 專案部署

- [ ] **克隆專案**: `git clone https://github.com/kyle0527/AIVA.git`
- [ ] **進入目錄**: `cd AIVA`
- [ ] **安裝依賴**: `pip install -r requirements.txt`

### ✅ 環境變數配置

- [ ] **設置 RABBITMQ_URL**: `$env:AIVA_RABBITMQ_URL = "amqp://localhost:5672"`
- [ ] **設置 RABBITMQ_USER**: `$env:AIVA_RABBITMQ_USER = "guest"`
- [ ] **設置 RABBITMQ_PASSWORD**: `$env:AIVA_RABBITMQ_PASSWORD = "guest"`
- [ ] **驗證環境變數**: `echo $env:AIVA_RABBITMQ_URL`

### ✅ 系統驗證

- [ ] **健康檢查**: `python health_check.py` ✅ 應顯示「優秀」
- [ ] **Schema 檢查**: `python schema_version_checker.py` ✅ 應無問題
- [ ] **修復 Schema** (如需要): `python schema_version_checker.py --fix`

### ✅ 功能測試

- [ ] **AI 實戰測試**: `python ai_security_test.py`
- [ ] **自主測試**: `python ai_autonomous_testing_loop.py`
- [ ] **系統探索**: `python ai_system_explorer_v3.py`

---

## 🏭 永久環境變數設置 (推薦)

### Windows 系統變數設置

1. **圖形界面方式**:
   - 右鍵「本機」→「內容」→「進階系統設定」
   - 點擊「環境變數」按鈕
   - 在「系統變數」中新增以下變數:

| 變數名稱 | 變數值 |
|---------|--------|
| `AIVA_RABBITMQ_URL` | `amqp://localhost:5672` |
| `AIVA_RABBITMQ_USER` | `guest` |
| `AIVA_RABBITMQ_PASSWORD` | `guest` |

2. **PowerShell 方式** (需管理員權限):
```powershell
[System.Environment]::SetEnvironmentVariable("AIVA_RABBITMQ_URL", "amqp://localhost:5672", "Machine")
[System.Environment]::SetEnvironmentVariable("AIVA_RABBITMQ_USER", "guest", "Machine")
[System.Environment]::SetEnvironmentVariable("AIVA_RABBITMQ_PASSWORD", "guest", "Machine")

# 重啟 PowerShell 後生效
```

### Linux/macOS 永久設置

```bash
# 添加到 shell 配置文件
echo 'export AIVA_RABBITMQ_URL="amqp://localhost:5672"' >> ~/.bashrc
echo 'export AIVA_RABBITMQ_USER="guest"' >> ~/.bashrc
echo 'export AIVA_RABBITMQ_PASSWORD="guest"' >> ~/.bashrc

# 重新載入配置
source ~/.bashrc

# 或者添加到 ~/.zshrc (如果使用 zsh)
echo 'export AIVA_RABBITMQ_URL="amqp://localhost:5672"' >> ~/.zshrc
source ~/.zshrc
```

---

## ❌ 常見問題排除

### 問題 1: 環境變數未生效

**症狀**: 
```
ValueError: AIVA_RABBITMQ_URL or AIVA_RABBITMQ_USER/AIVA_RABBITMQ_PASSWORD must be set
```

**解決方案**:
1. 重新啟動終端/PowerShell
2. 重新設置環境變數
3. 使用 `echo` 命令驗證變數值

### 問題 2: Schema 載入失敗

**症狀**:
```
🧬 Schema 狀態: ⚠️ Schemas: 載入成功但測試失敗
```

**解決方案**:
```bash
# 執行 Schema 修復
python schema_version_checker.py --fix

# 重新執行健康檢查
python health_check.py
```

### 問題 3: 路徑錯誤

**症狀**: 
```
ModuleNotFoundError: No module named 'services'
```

**解決方案**:
```bash
# 確認在正確的專案根目錄
pwd  # 應該顯示 .../AIVA-git
ls   # 應該看到 services/ 目錄

# 如果不在正確目錄，請 cd 到專案根目錄
cd path/to/AIVA-git
```

---

## 🎯 測試建議順序

按照以下順序執行，確保每一步都成功：

```bash
# 1. 基礎驗證
python health_check.py

# 2. AI 實戰測試 (需要靶場運行)
python ai_security_test.py

# 3. AI 自主循環 (完整體驗)
python ai_autonomous_testing_loop.py

# 4. 系統深度分析
python ai_system_explorer_v3.py

# 5. Schema 相容性檢查
python schema_version_checker.py
```

---

## 📞 支援資訊

- **📚 詳細指南**: `AIVA_COMPREHENSIVE_GUIDE.md`
- **🧪 測試報告**: `COMPLETE_TESTING_REPORT.md`
- **📁 日志位置**: `logs/` 目錄
- **⚙️ 配置文令**: `config/` 目錄

---

**📝 檢查清單最後更新**: 2025-10-28  
**✅ 驗證狀態**: 已在 Windows 11 + PowerShell 環境測試通過