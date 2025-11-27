# Integration Module Scripts

## 📑 目錄

- [📂 腳本列表](#腳本列表)
  - [backup.py](#backuppy)
  - [cleanup.py](#cleanuppy)
- [🔄 自動化建議](#自動化建議)
  - [建議排程](#建議排程)
  - [Windows Task Scheduler 完整設定](#windows-task-scheduler-完整設定)
  - [Linux Cron 完整設定](#linux-cron-完整設定)
- [📊 監控建議](#監控建議)
  - [檢查備份狀態](#檢查備份狀態)
  - [檢查磁碟空間](#檢查磁碟空間)
- [🔧 故障排除](#故障排除)
  - [備份失敗](#備份失敗)
  - [清理失敗](#清理失敗)
- [📝 注意事項](#注意事項)
- [🔗 相關文件](#相關文件)
  - [核心文檔](#核心文檔)
  - [配置與建立](#配置與建立)
  - [開發指南](#開發指南)

---


整合模組維護腳本集合

## 📂 腳本列表

### backup.py
備份攻擊路徑圖和經驗資料庫

**用法**:
```bash
# 完整備份 (包含清理舊備份)
python services/integration/scripts/backup.py

# 僅備份攻擊路徑圖
python services/integration/scripts/backup.py --attack-graph-only

# 僅備份經驗資料庫
python services/integration/scripts/backup.py --experience-only

# 備份但不清理舊備份
python services/integration/scripts/backup.py --no-cleanup
```

**排程備份**:
- Windows (Task Scheduler):
  ```powershell
  # 建立每日 2:00 AM 執行的排程任務
  $action = New-ScheduledTaskAction -Execute "python" -Argument "C:\D\fold7\AIVA-git\services\integration\scripts\backup.py"
  $trigger = New-ScheduledTaskTrigger -Daily -At 2am
  Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "AIVA Integration Backup"
  ```

- Linux (crontab):
  ```bash
  # 新增到 crontab
  crontab -e
  
  # 每日 2:00 AM 執行
  0 2 * * * cd /path/to/AIVA && python services/integration/scripts/backup.py
  ```

### cleanup.py
清理舊資料和備份

**用法**:
```bash
# 清理 30 天前的資料 (預設)
python services/integration/scripts/cleanup.py

# 清理 7 天前的資料
python services/integration/scripts/cleanup.py --days 7

# 僅清理備份檔案
python services/integration/scripts/cleanup.py --backup-only

# 僅清理日誌檔案
python services/integration/scripts/cleanup.py --logs-only

# 僅清理匯出檔案
python services/integration/scripts/cleanup.py --exports-only
```

## 🔄 自動化建議

### 建議排程
- **備份**: 每日 2:00 AM
- **清理**: 每週日 3:00 AM

### Windows Task Scheduler 完整設定
```powershell
# 備份任務
$backupAction = New-ScheduledTaskAction -Execute "python" `
    -Argument "C:\D\fold7\AIVA-git\services\integration\scripts\backup.py" `
    -WorkingDirectory "C:\D\fold7\AIVA-git"
$backupTrigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -Action $backupAction -Trigger $backupTrigger `
    -TaskName "AIVA Integration Backup" `
    -Description "每日備份 AIVA 整合模組資料"

# 清理任務
$cleanupAction = New-ScheduledTaskAction -Execute "python" `
    -Argument "C:\D\fold7\AIVA-git\services\integration\scripts\cleanup.py --days 30" `
    -WorkingDirectory "C:\D\fold7\AIVA-git"
$cleanupTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
Register-ScheduledTask -Action $cleanupAction -Trigger $cleanupTrigger `
    -TaskName "AIVA Integration Cleanup" `
    -Description "每週清理 AIVA 整合模組舊資料"
```

### Linux Cron 完整設定
```bash
# 編輯 crontab
crontab -e

# 新增以下行
# 每日 2:00 AM 備份
0 2 * * * cd /path/to/AIVA && python services/integration/scripts/backup.py >> /path/to/AIVA/data/logs/integration/backup.log 2>&1

# 每週日 3:00 AM 清理 30 天前的資料
0 3 * * 0 cd /path/to/AIVA && python services/integration/scripts/cleanup.py --days 30 >> /path/to/AIVA/data/logs/integration/cleanup.log 2>&1
```

## 📊 監控建議

### 檢查備份狀態
```bash
# 列出最近的備份
ls -lh data/integration/backups/attack_paths/ | tail -n 5
ls -lh data/integration/backups/experiences/ | tail -n 5

# 檢查備份大小
du -sh data/integration/backups/*
```

### 檢查磁碟空間
```bash
# Linux/Mac
df -h data/integration/

# Windows PowerShell
Get-PSDrive C | Select-Object Used,Free
```

## 🔧 故障排除

### 備份失敗
1. 檢查磁碟空間
2. 檢查檔案權限
3. 檢查來源檔案是否存在

### 清理失敗
1. 檢查檔案是否被占用
2. 檢查權限
3. 手動刪除後重試

## 📝 注意事項

1. **備份前確認**: 確保攻擊路徑圖和經驗資料庫未被使用
2. **清理謹慎**: 清理前確認不需要這些資料
3. **權限檢查**: 確保腳本有讀寫權限
4. **磁碟監控**: 定期檢查磁碟空間

## 🔗 相關文件

### 核心文檔
- 📖 **[整合模組總覽](../README.md)** - 整合模組主文檔
- 📖 **[資料儲存說明](../../../data/integration/README.md)** - 完整資料儲存結構
- 📖 **[Integration Core](../aiva_integration/README.md)** - 核心模組實現
- 📖 **[Services 總覽](../../README.md)** - 五大核心服務

### 配置與建立
- 📖 **[config.py 文檔](../aiva_integration/config.py)** - 統一配置系統
- 📖 **[建立報告](../../../reports/INTEGRATION_DATA_STORAGE_SETUP_REPORT.md)** - 完整建立過程
- 📖 **[更新計劃](../../../reports/README_UPDATE_PLAN_20251116.md)** - README 更新計劃

### 開發指南
- 📖 **[Data Storage Guide](../../../guides/development/DATA_STORAGE_GUIDE.md)** - 資料儲存總指南
- 📖 **[Attack Path Analyzer](../aiva_integration/attack_path_analyzer/README.md)** - 攻擊路徑分析
