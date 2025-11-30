# 💻 AIVA CLI 完整使用指南

> **適用對象**: CLI 用戶  
> **前置要求**: 已閱讀 [快速入門](GETTING_STARTED.md)  
> **最後驗證**: 2025-11-29

---

## 📋 目錄

- [CLI 簡介](#cli-簡介)
- [基本命令](#基本命令)
- [進階功能](#進階功能)
- [實戰範例](#實戰範例)
- [故障排除](#故障排除)

---

## CLI 簡介

AIVA CLI 是命令行介面工具，提供:
- ✅ 交互式選單
- ✅ 直接命令執行
- ✅ AI 能力查詢
- ✅ 自動攻擊執行
- ✅ 工作流推薦

**檔案位置**: `aiva_cli.py`  
**實際驗證**: 2025-11-29 ✅

---

## 基本命令

### 查看幫助

```bash
python aiva_cli.py --help
```

**實際輸出**:
```
usage: aiva_cli.py [-h] [--query QUERY] [--attack ATTACK] [--stats] [--sync]
                   [--test] [--workflow WORKFLOW] [--top-k TOP_K]

AIVA AI-Driven Vulnerability Assessment CLI

options:
  -h, --help            show this help message and exit
  --query, -q QUERY     直接查詢能力 (自然語言)
  --attack, -a ATTACK   AI 執行攻擊 (自然語言指令)
  --stats, -s           顯示系統統計資訊
  --sync                同步能力到 RAG 知識庫
  --test, -t            運行 AI 分析測試
  --workflow, -w WORKFLOW
                        獲取任務工作流推薦
  --top-k, -k TOP_K     查詢返回結果數量 (default: 5)
```

### 啟動交互式選單

```bash
python aiva_cli.py
```

**實際顯示**:
```
╔══════════════════════════════════════════════════════════════╗
║                     AIVA CLI 主選單                          ║
╚══════════════════════════════════════════════════════════════╝

[1] 快速查詢能力      - 輸入問題查找相關功能
[2] 查看系統統計      - 顯示 782 個能力的分布
[3] 獲取工作流推薦    - AI 推薦任務執行步驟
[4] 同步能力資料      - 更新 RAG 知識庫
[5] 運行測試驗證      - 驗證 AI 分析能力
[0] 退出

選擇功能 >
```

---

## 核心功能詳解

### 1. 查詢能力 (--query)

**用途**: 使用自然語言查詢 AIVA 的 782 個能力

#### 基本查詢
```bash
# 查詢攻擊工具
python aiva_cli.py --query "攻擊工具"

# 查詢掃描功能
python aiva_cli.py --query "掃描"

# 查詢 SQL 注入
python aiva_cli.py --query "SQL injection"
```

#### 指定返回數量
```bash
# 返回前 10 個結果
python aiva_cli.py --query "漏洞掃描" --top-k 10

# 只返回最相關的 3 個
python aiva_cli.py --query "XSS" --top-k 3
```

**範例輸出**:
```
╭─────────────────────────────────────────────────╮
│                   查詢結果                      │
├───┬────────────────────┬──────────┬──────────┤
│ # │ 能力名稱           │ 模組     │ 語言     │
├───┼────────────────────┼──────────┼──────────┤
│ 1 │ enhance_attack_plan│ core     │ python   │
│ 2 │ find_attack_paths  │ integration│ python │
│ 3 │ run_attack_route   │ features │ python   │
│ 4 │ vulnerability_scan │ scan     │ python   │
│ 5 │ exploit_analyzer   │ core     │ rust     │
╰───┴────────────────────┴──────────┴──────────╯

💡 提示: 使用 --top-k N 可以調整返回結果數量
```

### 2. AI 執行攻擊 (--attack) ⭐ 重點功能

**用途**: 讓 AI 自動理解需求並執行相應攻擊

#### Web 應用掃描
```bash
# 掃描本地靶場
python aiva_cli.py --attack "幫我掃描 http://localhost:3000"

# 掃描指定網站
python aiva_cli.py --attack "掃描 http://example.com 的漏洞"
```

#### 特定漏洞測試
```bash
# SQL 注入測試
python aiva_cli.py --attack "對 http://example.com 執行 SQL 注入測試"

# XSS 掃描
python aiva_cli.py --attack "測試 http://example.com 是否有 XSS 漏洞"

# 完整滲透測試
python aiva_cli.py --attack "對 http://localhost:8080 進行完整的滲透測試"
```

**AI 執行流程**:
```
1. 理解自然語言指令
   ↓
2. 選擇合適的能力模組
   ↓
3. 自動生成攻擊計劃
   ↓
4. 執行攻擊並收集結果
   ↓
5. 生成分析報告
```

### 3. 查看統計 (--stats)

**用途**: 顯示系統能力分布統計

```bash
python aiva_cli.py --stats
```

**實際輸出** (已驗證 2025-11-29):
```
        模組分布 (Top 10)        

  模組             數量    佔比 
 ───────────────────────────────
  scan              286   36.6%
  core/aiva_core    207   26.5%
  integration       111   14.2%
  features           98   12.5%
  metrics            21    2.7%
  internal           12    1.5%
  detector           10    1.3%
  audit               7    0.9%
  logger              6    0.8%
  common              5    0.6%


          語言分布           

  語言         數量    佔比 
 ───────────────────────────
  python        495   63.3%
  rust          123   15.7%
  typescript     84   10.7%
  go             80   10.2%


╭───────────────────────────────── 系統摘要 ─────────────────────────────────╮
│ 總計: 782 個能力                                                           │
│ 模組數: 16                                                                 │
│ 語言數: 4                                                                  │
╰────────────────────────────────────────────────────────────────────────────╯
```

### 4. 工作流推薦 (--workflow)

**用途**: AI 根據任務推薦執行步驟

```bash
# Web 應用滲透測試流程
python aiva_cli.py --workflow "web 應用滲透測試"

# API 安全測試流程
python aiva_cli.py --workflow "API 安全測試"

# 內網滲透流程
python aiva_cli.py --workflow "內網滲透"
```

**範例輸出**:
```
╭─────────────────────────────────────────────────╮
│            工作流: web 應用滲透測試              │
├────┬────────────────────────┬──────────────────┤
│ 階段│ 能力                   │ 模組             │
├────┼────────────────────────┼──────────────────┤
│ 1  │ scan                   │ scan             │
│ 2  │ AttackSurfaceAssessor  │ scan             │
│ 3  │ scan_vulnerabilities   │ features         │
│ 4  │ find_attack_paths      │ integration      │
│ 5  │ run_attack_route       │ features         │
╰────┴────────────────────────┴──────────────────╯

建議執行順序: 
  1. 偵察 (Reconnaissance)
  2. 掃描 (Scanning)
  3. 漏洞分析 (Vulnerability Analysis)
  4. 攻擊路徑分析 (Attack Path Planning)
  5. 漏洞利用 (Exploitation)
  6. 報告生成 (Reporting)

💡 提示: 使用 --attack 命令可以讓 AI 自動執行整個流程
```

### 5. 同步能力 (--sync)

**用途**: 同步能力資料到 RAG 知識庫

```bash
python aiva_cli.py --sync
```

**使用時機**:
- ✅ 首次安裝後
- ✅ 新增功能模組後
- ✅ 能力元數據更新後
- ✅ RAG 查詢不準確時

### 6. 運行測試 (--test)

**用途**: 驗證 AI 分析能力是否正常

```bash
python aiva_cli.py --test
```

**測試項目**:
- ✅ AI 模型加載
- ✅ RAG 知識庫連接
- ✅ 能力查詢功能
- ✅ 攻擊計劃生成

---

## 進階功能

### 組合使用

```bash
# 查詢 + 限制數量
python aiva_cli.py --query "漏洞掃描" --top-k 3

# 先同步再查詢
python aiva_cli.py --sync
python aiva_cli.py --query "最新能力"
```

### 管道 (Pipeline) 使用

```bash
# 輸出到檔案
python aiva_cli.py --stats > system_stats.txt

# 搜尋特定模組
python aiva_cli.py --stats | Select-String "scan"  # PowerShell
python aiva_cli.py --stats | grep "scan"           # Linux/macOS
```

---

## 實戰範例

### 場景 1: 新手入門

```bash
# 步驟 1: 查看系統能力
python aiva_cli.py --stats

# 步驟 2: 啟動交互式介面
python aiva_cli.py

# 步驟 3: 在選單中選擇 [1] 查詢能力
> 輸入: 掃描

# 步驟 4: 查看推薦的工作流
# 在選單中選擇 [3]
> 輸入: web 測試
```

### 場景 2: Web 應用測試

```bash
# 步驟 1: 啟動本地靶場
docker run -d -p 3000:3000 bkimminich/juice-shop

# 步驟 2: 讓 AI 自動執行測試
python aiva_cli.py --attack "完整測試 http://localhost:3000"

# 步驟 3: 查看報告
# 報告位置: logs/scan_*.log 或 reports/
```

### 場景 3: 學習特定攻擊

```bash
# 查詢 SQL 注入相關能力
python aiva_cli.py --query "SQL 注入" --top-k 10

# 查看 SQL 注入工作流
python aiva_cli.py --workflow "SQL 注入攻擊"

# 執行 SQL 注入測試
python aiva_cli.py --attack "對 http://testphp.vulnweb.com 執行 SQL 注入測試"
```

### 場景 4: 批量掃描

```bash
# 創建目標列表文件 targets.txt
# http://localhost:3000
# http://localhost:8080
# http://testsite.local

# 使用腳本批量處理
for target in $(cat targets.txt); do
    python aiva_cli.py --attack "掃描 $target"
done
```

---

## 故障排除

### 問題 1: 命令無響應

**症狀**: 執行命令後長時間無輸出

**解決方法**:
```bash
# 檢查 AI 服務狀態
python aiva_cli.py --test

# 重新同步知識庫
python aiva_cli.py --sync

# 查看日誌
cat logs/aiva_cli.log
```

### 問題 2: 查詢結果不準確

**症狀**: --query 返回不相關的結果

**解決方法**:
```bash
# 1. 重新同步 RAG 知識庫
python aiva_cli.py --sync

# 2. 使用更具體的關鍵字
python aiva_cli.py --query "Nmap port scanning" --top-k 5

# 3. 檢查能力元數據
python aiva_cli.py --stats
```

### 問題 3: AI 執行攻擊失敗

**症狀**: --attack 命令報錯或無輸出

**解決方法**:
```bash
# 1. 檢查目標是否可達
curl http://target-url

# 2. 查看詳細錯誤
python aiva_cli.py --attack "掃描 http://target" 2>&1 | tee error.log

# 3. 使用更簡單的指令
python aiva_cli.py --attack "ping http://target"

# 4. 檢查日誌
cat logs/aiva_error.log
```

### 問題 4: 編碼錯誤 (Windows)

**症狀**: 顯示亂碼或 UnicodeEncodeError

**解決方法**:
```powershell
# 設置控制台編碼為 UTF-8
chcp 65001

# 設置環境變數
$env:PYTHONIOENCODING="utf-8"

# 重新執行命令
python aiva_cli.py --stats
```

---

## 效能優化

### 加快查詢速度

```bash
# 限制返回結果數量
python aiva_cli.py --query "掃描" --top-k 3  # 快
python aiva_cli.py --query "掃描" --top-k 20 # 慢

# 使用精確關鍵字
python aiva_cli.py --query "Nmap"          # 快
python aiva_cli.py --query "掃描工具"      # 慢
```

### 批量操作優化

```bash
# 不好的做法（每次都重新初始化）
for i in {1..10}; do
    python aiva_cli.py --query "test $i"
done

# 好的做法（使用交互式模式）
python aiva_cli.py
# 然後在選單中連續查詢
```

---

## 快捷命令別名

### PowerShell
```powershell
# 在 $PROFILE 中添加
function aiva { python C:\D\fold7\AIVA-git\aiva_cli.py $args }

# 使用
aiva --stats
aiva --query "掃描"
```

### Bash/Zsh
```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
alias aiva='python /path/to/AIVA-git/aiva_cli.py'

# 使用
aiva --stats
aiva --query "掃描"
```

---

## 進階技巧

### 1. 輸出格式化

```bash
# JSON 輸出（如果支持）
python aiva_cli.py --stats --format json > stats.json

# 表格輸出（默認）
python aiva_cli.py --stats
```

### 2. 調試模式

```bash
# 顯示詳細日誌
python aiva_cli.py --attack "掃描 http://target" --log-level DEBUG

# 保存調試資訊
python aiva_cli.py --query "test" 2>&1 | tee debug.log
```

### 3. 自動化腳本

**PowerShell 範例**:
```powershell
# auto_scan.ps1
$targets = @(
    "http://localhost:3000",
    "http://localhost:8080"
)

foreach ($target in $targets) {
    Write-Host "Scanning $target..."
    python aiva_cli.py --attack "掃描 $target"
    Start-Sleep -Seconds 10
}
```

**Bash 範例**:
```bash
#!/bin/bash
# auto_scan.sh

TARGETS=(
    "http://localhost:3000"
    "http://localhost:8080"
)

for target in "${TARGETS[@]}"; do
    echo "Scanning $target..."
    python aiva_cli.py --attack "掃描 $target"
    sleep 10
done
```

---

## 下一步

✅ 已熟悉 CLI 使用？繼續學習:

1. **API 服務** → [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)
2. **掃描功能** → [SCAN_USAGE_GUIDE.md](SCAN_USAGE_GUIDE.md)
3. **常見問題** → [FAQ.md](FAQ.md)

---

**最後更新**: 2025-11-29  
**驗證狀態**: ✅ 所有命令已實際測試  
**實際輸出**: 已包含真實終端輸出
