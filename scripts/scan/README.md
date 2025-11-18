# 🔍 Scan Service Scripts

> **掃描服務腳本目錄** - AIVA 智能掃描與監控工具集  
> **服務對應**: AIVA Scan Services  
> **腳本數量**: 2個專業掃描工具

---

## 📋 目錄概述

Scan 服務腳本提供 AIVA 系統的智能掃描與監控功能，包括基礎設施掃描、安全評估、性能監控等核心能力。這些工具確保 AIVA 系統的穩定性、安全性和最佳效能。

---

## 🗂️ 目錄結構

```
scan/
├── 📋 README.md                     # 本文檔
│
├── 🐳 docker/                       # Docker 相關掃描 (1個)
│   └── 🔄 docker_infrastructure_updater.py # Docker 基礎設施更新器
│
└── 📊 reporting/                    # 掃描報告系統 (1個)
    └── 📈 final_report.py           # 最終掃描報告生成器
```

---

## 🐳 Docker 掃描工具

### 🔄 Docker 基礎設施更新器
**檔案**: `docker/docker_infrastructure_updater.py`
```bash
cd docker
python docker_infrastructure_updater.py [operation] [options]
```

**功能**:
- 🐳 自動化 Docker 基礎設施管理
- 🔄 容器鏡像版本更新與維護
- 🔍 Docker 環境安全掃描
- 📊 容器性能監控與分析
- 🛠️ 自動化容器部署最佳化

**主要操作模式**:

#### 🔍 基礎設施掃描
```bash
# 掃描所有 Docker 容器狀態
python docker_infrastructure_updater.py --scan all

# 掃描特定服務容器
python docker_infrastructure_updater.py --scan service --name aiva-core

# 安全漏洞掃描
python docker_infrastructure_updater.py --scan security --deep
```

#### 🔄 自動更新功能
```bash
# 更新所有過期鏡像
python docker_infrastructure_updater.py --update images --auto

# 更新特定服務
python docker_infrastructure_updater.py --update service --name aiva-scan

# 批次更新容器配置
python docker_infrastructure_updater.py --update config --batch
```

#### 📊 監控與分析
```bash
# 性能監控報告
python docker_infrastructure_updater.py --monitor performance --duration 24h

# 資源使用分析
python docker_infrastructure_updater.py --analyze resources --export json

# 網路連接診斷
python docker_infrastructure_updater.py --diagnose network --verbose
```

**使用範例**:
```python
from docker_infrastructure_updater import DockerManager

# 建立 Docker 管理器
docker_mgr = DockerManager()

# 掃描容器狀態
containers = docker_mgr.scan_containers()

# 自動更新過期鏡像
update_result = docker_mgr.auto_update_images()

# 生成基礎設施報告
report = docker_mgr.generate_infrastructure_report()
```

**掃描項目**:
- 🔍 **容器健康狀態**: 運行狀態、資源使用、錯誤日誌
- 🛡️ **安全漏洞檢測**: 鏡像漏洞、配置弱點、權限問題
- 📈 **效能評估**: CPU/記憶體使用、網路延遲、磁碟 I/O
- 🔄 **版本管理**: 鏡像版本追蹤、更新建議、相依性檢查

---

## 📊 報告系統

### 📈 最終掃描報告生成器
**檔案**: `reporting/final_report.py`
```bash
cd reporting
python final_report.py [report_type] [options]
```

**功能**:
- 📈 彙整所有掃描結果的統一報告
- 📊 多維度數據分析與視覺化
- 🔍 異常檢測與趨勢分析
- 📋 自動化報告生成與分發
- 💡 智能建議與行動計劃

**報告類型**:

#### 📊 綜合掃描報告
```bash
# 生成完整系統掃描報告
python final_report.py --type comprehensive --format html

# 安全專項報告
python final_report.py --type security --detailed --output security_report.pdf

# 性能分析報告
python final_report.py --type performance --timeframe 7d --charts
```

#### 📈 趨勢分析報告
```bash
# 系統健康趨勢分析
python final_report.py --type trends --category health --period monthly

# 資源使用趨勢
python final_report.py --type trends --category resources --compare quarterly
```

#### 💡 建議與行動計劃
```bash
# 自動生成優化建議
python final_report.py --type recommendations --priority high

# 生成行動計劃
python final_report.py --type action_plan --timeline 30d
```

**報告功能特色**:
- 📊 **多格式輸出**: HTML、PDF、JSON、Excel
- 📈 **互動式圖表**: 使用 Plotly 生成動態圖表
- 🔍 **深度分析**: 異常檢測、根因分析、影響評估
- 💌 **自動分發**: 郵件、Slack、Teams 整合
- 📱 **移動優化**: 響應式設計支援移動設備

**使用範例**:
```python
from final_report import ReportGenerator

# 建立報告生成器
report_gen = ReportGenerator()

# 收集所有掃描資料
scan_data = report_gen.collect_scan_data()

# 生成綜合分析報告
comprehensive_report = report_gen.generate_comprehensive_report(
    data=scan_data,
    format='html',
    include_charts=True
)

# 生成安全專項報告
security_report = report_gen.generate_security_report(
    severity_filter='high',
    export_format='pdf'
)

# 自動分發報告
report_gen.distribute_reports(
    recipients=['admin@aiva.com', 'devops@aiva.com'],
    channels=['email', 'slack']
)
```

---

## 🎯 使用情境

### 🚀 日常監控作業
```bash
# 1. 執行 Docker 基礎設施掃描
cd docker
python docker_infrastructure_updater.py --scan all --monitor

# 2. 生成日常監控報告
cd ../reporting
python final_report.py --type daily --auto-send
```

### 🔒 安全檢查流程
```bash
# 1. 深度安全掃描
cd docker
python docker_infrastructure_updater.py --scan security --deep --export

# 2. 生成安全分析報告
cd ../reporting
python final_report.py --type security --detailed --priority high
```

### 📈 效能優化分析
```bash
# 1. 效能監控與分析
cd docker
python docker_infrastructure_updater.py --monitor performance --duration 7d

# 2. 生成效能優化建議
cd ../reporting
python final_report.py --type performance --recommendations
```

### 🔄 系統更新維護
```bash
# 1. 檢查並更新 Docker 基礎設施
cd docker
python docker_infrastructure_updater.py --update all --safe-mode

# 2. 生成更新摘要報告
cd ../reporting
python final_report.py --type update_summary --changes-only
```

---

## ⚡ 效能最佳化

### 🐳 Docker 掃描最佳化
- **並行掃描**: 多容器同時掃描提升速度
- **增量掃描**: 只掃描變更的容器和鏡像
- **快取機制**: 掃描結果快取減少重複工作
- **智能排程**: 根據系統負載調整掃描頻率

### 📊 報告生成最佳化
- **模板快取**: 報告模板快取加速生成
- **資料分片**: 大型資料集分片處理
- **非同步處理**: 報告生成不阻塞其他操作
- **壓縮優化**: 報告檔案壓縮減少存儲空間

---

## 🔒 安全性功能

### 🛡️ Docker 安全掃描
- **漏洞資料庫**: 整合 CVE 漏洞資料庫
- **配置檢查**: Docker 安全最佳實踐檢查
- **權限審計**: 容器權限與存取控制檢查
- **網路安全**: 容器網路配置安全分析

### 📊 報告資料保護
- **敏感資料過濾**: 自動過濾敏感資訊
- **存取控制**: 基於角色的報告存取權限
- **資料加密**: 報告檔案加密存儲
- **稽核追蹤**: 報告存取稽核日誌

---

## 🛠️ 配置與自訂

### ⚙️ 掃描配置
```yaml
# docker_scan_config.yaml
scan:
  intervals:
    health_check: 5m
    security_scan: 1h
    performance_monitor: 15m
  
  thresholds:
    cpu_usage_alert: 80%
    memory_usage_alert: 85%
    disk_usage_alert: 90%
```

### 📊 報告配置
```yaml
# report_config.yaml
report:
  formats: [html, pdf, json]
  auto_generate: true
  schedule: "0 8 * * *"  # 每天早上8點
  
  distribution:
    email: [admin@aiva.com]
    slack: "#aiva-monitoring"
```

---

## 🔗 服務整合

### 🤖 與 Core 服務整合
- 為 Core AI 分析提供基礎設施狀態資料
- 支援 AI 系統的容器化部署監控
- 整合 AI 模型的效能分析

### 🔗 與 Common 服務整合
- 使用 Common 啟動器進行掃描服務啟動
- 通過 Common 維護工具進行系統修復
- 利用 Common 驗證器確保掃描環境完整性

### 🎯 與 Features 服務整合
- 為功能模組提供基礎設施監控
- 支援功能的容器化部署掃描
- 整合功能相關的效能指標

### 🔄 與 Integration 服務整合
- 支援多語言容器的統一監控
- 整合跨語言服務的效能分析
- 提供多語言應用的安全掃描

---

## 📋 故障排除

### 常見問題與解決方案

#### 🐳 Docker 連接問題
```bash
# 檢查 Docker 服務狀態
python docker_infrastructure_updater.py --diagnose docker-service

# 修復 Docker 連接
python docker_infrastructure_updater.py --fix connection
```

#### 📊 報告生成失敗
```bash
# 清除報告快取
python final_report.py --clear-cache

# 重新生成報告
python final_report.py --regenerate --force
```

#### 🔍 掃描權限問題
```bash
# 檢查掃描權限
python docker_infrastructure_updater.py --check-permissions

# 修復權限設定
python docker_infrastructure_updater.py --fix-permissions
```

---

## 📅 維護排程

### 🔄 自動化維護
- **每日掃描**: 基礎健康檢查與狀態監控
- **每週報告**: 綜合性能與安全分析報告
- **每月審計**: 深度安全稽核與合規檢查
- **季度優化**: 系統效能優化建議與實施

---

**維護者**: AIVA Scan & Monitoring Team  
**最後更新**: 2025-11-17  
**服務狀態**: ✅ 所有掃描工具已重組並驗證

---

[← 返回 Scripts 主目錄](../README.md)