# AIVA CLI 與 AI 訓練整合指南

## 📖 概述

AIVA 現在提供完整的 CLI 命令行介面和基於真實流程的 AI 訓練系統。

**核心特性:**
- 🎯 真實 CLI 命令操作
- 🧠 500 萬參數 ScalableBioNet
- 🔄 完整訊息流追蹤和學習
- 📊 經驗回放和模型更新

---

## 🚀 快速開始

### 1. CLI 命令示例

#### 掃描網站
```bash
# 基礎掃描
python services/cli/aiva_cli.py scan start https://example.com

# 自定義深度和頁數
python services/cli/aiva_cli.py scan start https://example.com \
  --max-depth 5 \
  --max-pages 200 \
  --wait

# 說明:
#   --max-depth: 最大爬取深度 (預設: 3)
#   --max-pages: 最大頁面數 (預設: 100)
#   --wait: 等待掃描完成並顯示結果
```

#### SQL 注入檢測
```bash
# 檢測單一參數
python services/cli/aiva_cli.py detect sqli \
  https://example.com/login \
  --param username \
  --wait

# 指定檢測引擎
python services/cli/aiva_cli.py detect sqli \
  https://example.com/api/user \
  --param id \
  --method GET \
  --engines error,boolean,time \
  --wait

# 可用引擎:
#   error   - 錯誤基礎檢測
#   boolean - 布爾盲注檢測
#   time    - 時間盲注檢測
#   union   - UNION 查詢檢測
```

#### XSS 檢測
```bash
# 反射型 XSS
python services/cli/aiva_cli.py detect xss \
  https://example.com/search \
  --param q \
  --type reflected \
  --wait

# 存儲型 XSS
python services/cli/aiva_cli.py detect xss \
  https://example.com/comment \
  --param message \
  --type stored \
  --wait

# 可用類型:
#   reflected - 反射型 XSS
#   stored    - 存儲型 XSS
#   dom       - DOM 型 XSS
```

#### 生成報告
```bash
# HTML 報告
python services/cli/aiva_cli.py report generate scan_xxx \
  --format html \
  --output report.html

# PDF 報告
python services/cli/aiva_cli.py report generate scan_xxx \
  --format pdf \
  --output report.pdf

# JSON 報告 (供程序處理)
python services/cli/aiva_cli.py report generate scan_xxx \
  --format json \
  --output report.json \
  --no-findings  # 不包含詳細漏洞資訊

# 支援格式:
#   html - HTML 網頁報告
#   pdf  - PDF 文檔報告
#   json - JSON 數據報告
```

#### AI 訓練
```bash
# 實時訓練模式 (從真實任務學習)
python services/cli/aiva_cli.py ai train \
  --mode realtime \
  --epochs 10

# 回放訓練模式 (從歷史經驗學習)
python services/cli/aiva_cli.py ai train \
  --mode replay \
  --epochs 20 \
  --storage-path ./data/ai

# 模擬訓練模式 (使用模擬場景)
python services/cli/aiva_cli.py ai train \
  --mode simulation \
  --scenarios 100 \
  --epochs 5

# 訓練模式:
#   realtime   - 監聽實際任務執行並學習
#   replay     - 從存儲的歷史經驗回放學習
#   simulation - 使用預定義場景模擬訓練
```

#### 查看 AI 狀態
```bash
python services/cli/aiva_cli.py ai status

# 輸出示例:
# 🤖 AI 系統狀態
#    模型參數量: 5,242,880
#    知識庫條目: 1,234
#    向量維度: 512
#    最後更新: 2025-10-17 10:30:00
```

#### 系統狀態
```bash
python services/cli/aiva_cli.py system status

# 輸出示例:
# ⚙️ AIVA 系統狀態
# 📡 模組狀態:
#    core: 🟢 運行中
#    scan: 🟢 運行中
#    function_sqli: 🟢 運行中
#    function_xss: 🟢 運行中
#    integration: 🟢 運行中
```

---

## 🧠 AI 訓練系統

### 訓練流程

AI 訓練系統會模擬完整的 CLI → Core → Worker → Integration 流程：

```
CLI 命令
   ↓
TaskDispatcher (Core)
   ↓
Worker (Scan/Function)
   ↓
ResultCollector (Core)
   ↓
Integration (Analysis)
   ↓
AI Learning (經驗記錄 + 模型更新)
```

### 使用 integrated_cli_training.py

```bash
# 直接運行訓練
python scripts/ai_training/integrated_cli_training.py

# 輸出示例:
# ============================================================
# AIVA AI 訓練系統
# 基於 500 萬參數 ScalableBioNet
# ============================================================
# 🧠 初始化 ScalableBioNet (500萬參數)...
#    ✅ 神經網路參數量: 5,242,880
# 🚀 初始化 AI 訓練系統...
# ✅ AI 訓練系統初始化完成
# 🎓 開始訓練: 5 個場景, 3 輪
# 
# ============================================================
# 訓練輪次 1/3
# ============================================================
# 
# 場景 1/5
# 🎬 場景 1: 掃描流程模擬
#    場景 ID: scenario_xxx
#    目標 URL: https://example0.com
#    步驟 1/5: CLI 發送掃描請求...
#    步驟 2/5: Scan Worker 處理請求...
#    步驟 3/5: Worker 發送結果到 ResultCollector...
#    步驟 4/5: ResultCollector 轉發到 Integration...
#    步驟 5/5: Integration 分析結果...
#    🧠 AI 學習流程...
# ✅ 場景 1 完成: 發現 5 個資產
```

### 訓練場景

#### 場景 1: 掃描流程
模擬 CLI 命令:
```bash
aiva scan start https://example.com --max-depth 3
```

流程:
1. CLI 發送 TASK_SCAN_START
2. Scan Worker 執行掃描
3. Worker 發送 RESULTS_SCAN_COMPLETED
4. ResultCollector 接收並轉發
5. Integration 分析和存儲
6. AI 學習整個流程

#### 場景 2: SQL 注入檢測
模擬 CLI 命令:
```bash
aiva detect sqli https://example.com/login --param username
```

流程:
1. CLI 發送 TASK_FUNCTION_SQLI
2. SQLi Worker 執行多引擎檢測
3. Worker 發送 FindingPayload
4. Integration 進行風險評估
5. AI 學習檢測策略

#### 場景 3: 完整攻擊鏈
模擬 CLI 命令序列:
```bash
aiva scan start https://example.com
aiva detect sqli <discovered_urls>
aiva detect xss <discovered_urls>
aiva report generate --attack-path
```

流程:
1. 執行掃描發現資產
2. 對資產進行 SQLi 檢測
3. 對資產進行 XSS 檢測
4. 構建攻擊路徑分析
5. AI 學習完整攻擊鏈

---

## 📊 500 萬參數 BioNeuronCore

### 模型架構

```python
ScalableBioNet(
    input_dim=512,              # 輸入層: 512 維
    hidden_dims=[1024, 2048, 1024],  # 隱藏層: 3 層
    output_dim=256,             # 輸出層: 256 維
)

# 參數計算:
# Layer 1: 512 × 1024 = 524,288
# Layer 2: 1024 × 2048 = 2,097,152
# Layer 3: 2048 × 1024 = 2,097,152
# Layer 4: 1024 × 256 = 262,144
# 總計: 5,242,880 參數 (約 500 萬)
```

### 特性

1. **生物啟發式尖峰神經元** (BiologicalSpikingLayer)
   - 模擬真實神經元的尖峰行為
   - 實現不反應期機制
   - 閾值激活函數

2. **抗幻覺模組** (AntiHallucinationModule)
   - 評估決策信心度
   - 低信心時觸發警告
   - 避免過度自信的錯誤決策

3. **RAG 整合** (KnowledgeBase)
   - 檢索增強生成
   - 向量化知識存儲
   - 相似案例查詢

4. **經驗學習** (ExperienceManager)
   - 記錄所有執行經驗
   - 支援經驗回放
   - 持久化存儲

---

## 🔄 完整使用範例

### 範例 1: 掃描並檢測

```bash
# 步驟 1: 掃描目標網站
python services/cli/aiva_cli.py scan start https://testsite.com \
  --max-depth 3 \
  --wait

# 輸出:
# 🚀 啟動掃描任務
#    掃描 ID: scan_xxx
#    任務 ID: task_xxx
#    目標 URL: https://testsite.com
#    最大深度: 3
# ✅ 掃描任務已提交到消息隊列
#    訂閱主題: tasks.scan.start
# ⏳ 等待掃描結果...
# ✅ 掃描完成！
#    資產數量: 25
#    指紋數量: 8
# 
# 📦 發現的資產:
#    - https://testsite.com/login
#    - https://testsite.com/admin
#    - https://testsite.com/api/users
#    - https://testsite.com/search
#    - https://testsite.com/profile

# 步驟 2: 對發現的登錄頁面進行 SQLi 檢測
python services/cli/aiva_cli.py detect sqli \
  https://testsite.com/login \
  --param username \
  --wait

# 輸出:
# 🔍 啟動 SQL 注入檢測
#    任務 ID: task_yyy
#    目標 URL: https://testsite.com/login
#    參數: username
# ✅ SQL 注入檢測任務已提交
# ⏳ 等待 SQLI 檢測結果...
# 🚨 發現 1 個漏洞！
# 
# 漏洞 #1:
#    嚴重程度: HIGH
#    置信度: high
#    描述: SQL injection vulnerability detected in username parameter

# 步驟 3: 生成報告
python services/cli/aiva_cli.py report generate scan_xxx \
  --format html \
  --output vulnerability_report.html

# 輸出:
# 📊 生成報告
#    掃描 ID: scan_xxx
#    格式: html
#    輸出: vulnerability_report.html
# ✅ 報告已生成: vulnerability_report.html
```

### 範例 2: AI 訓練

```bash
# 使用模擬場景訓練 AI
python services/cli/aiva_cli.py ai train \
  --mode simulation \
  --scenarios 50 \
  --epochs 10 \
  --storage-path ./data/ai_training

# 或使用專用訓練腳本
python scripts/ai_training/integrated_cli_training.py

# 訓練完成後查看狀態
python services/cli/aiva_cli.py ai status \
  --storage-path ./data/ai_training

# 輸出:
# 🤖 AI 系統狀態
#    模型參數量: 5,242,880
#    知識庫條目: 450
#    向量維度: 512
#    最後更新: 2025-10-17 11:45:23
```

---

## 📁 文件結構

```
services/
├── cli/
│   └── aiva_cli.py                    # ✨ 主 CLI 入口
│
├── core/aiva_core/
│   ├── ai_engine/
│   │   ├── bio_neuron_core.py         # 🧠 500 萬參數神經網路
│   │   └── knowledge_base.py          # 📚 RAG 知識庫
│   ├── learning/
│   │   └── experience_manager.py      # 💾 經驗管理
│   └── training/
│       └── training_orchestrator.py   # 🎓 訓練編排
│
└── ...

scripts/
└── ai_training/
    └── integrated_cli_training.py     # 🔧 整合訓練腳本
```

---

## 🎯 訓練目標

AI 系統通過訓練學習以下能力：

1. **掃描策略選擇**
   - 根據目標類型選擇最佳掃描深度
   - 學習哪些資產最可能存在漏洞
   - 優化掃描效率

2. **檢測引擎選擇**
   - 根據參數類型選擇最有效的檢測引擎
   - 學習引擎組合策略
   - 減少誤報率

3. **攻擊路徑構建**
   - 識別可行的攻擊路徑
   - 評估攻擊難度和成功率
   - 優先級排序

4. **風險評估**
   - 準確評估漏洞嚴重性
   - 考慮業務影響
   - 提供修復建議

---

## 💡 最佳實踐

### CLI 使用建議

1. **大型掃描時使用 --wait**
   ```bash
   # 建議: 小型網站
   python services/cli/aiva_cli.py scan start https://small-site.com --wait
   
   # 建議: 大型網站 (不使用 --wait，讓掃描在後台運行)
   python services/cli/aiva_cli.py scan start https://large-site.com
   ```

2. **組合使用檢測引擎**
   ```bash
   # 快速檢測 (僅錯誤基礎)
   --engines error
   
   # 全面檢測 (所有引擎)
   --engines error,boolean,time,union
   
   # 平衡檢測 (排除耗時的時間盲注)
   --engines error,boolean,union
   ```

3. **定期訓練 AI**
   ```bash
   # 每週回放訓練
   python services/cli/aiva_cli.py ai train --mode replay --epochs 20
   
   # 測試新策略時使用模擬
   python services/cli/aiva_cli.py ai train --mode simulation --scenarios 100
   ```

### AI 訓練建議

1. **分階段訓練**
   - 第一階段: 模擬訓練 (快速建立基礎)
   - 第二階段: 回放訓練 (從歷史學習)
   - 第三階段: 實時訓練 (持續改進)

2. **定期備份模型**
   ```bash
   # 訓練前備份
   cp -r ./data/ai ./data/ai_backup_$(date +%Y%m%d)
   ```

3. **監控訓練效果**
   - 定期檢查 AI 狀態
   - 比較不同版本模型的表現
   - 調整訓練參數

---

## 🔧 進階配置

### 自定義存儲路徑

```bash
# 使用自定義路徑存儲 AI 數據
export AIVA_AI_STORAGE=/custom/path/to/ai_data

python services/cli/aiva_cli.py ai train \
  --storage-path $AIVA_AI_STORAGE
```

### 調整模型參數

編輯 `scripts/ai_training/integrated_cli_training.py`:

```python
# 修改神經網路架構
net = ScalableBioNet(
    input_dim=512,
    hidden_dims=[2048, 4096, 2048],  # 增加隱藏層大小
    output_dim=512,                  # 增加輸出維度
)
# 新參數量: 約 2000 萬
```

---

## ❓ 常見問題

### Q: CLI 命令找不到？
A: 確保在項目根目錄執行，或設置 PYTHONPATH:
```bash
export PYTHONPATH=/path/to/AIVA:$PYTHONPATH
```

### Q: RabbitMQ 連接失敗？
A: 確保 RabbitMQ 服務運行:
```bash
# 啟動 RabbitMQ
docker-compose up -d rabbitmq

# 或使用本地安裝
sudo systemctl start rabbitmq-server
```

### Q: AI 訓練很慢？
A: 考慮:
1. 減少場景數量和訓練輪數
2. 使用 GPU 加速 (如果可用)
3. 調整批次大小

### Q: 模型參數量如何驗證？
A:
```python
from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet

net = ScalableBioNet(512, [1024, 2048, 1024], 256)
print(f"參數量: {net.count_params():,}")  # 應顯示: 5,242,880
```

---

## 📚 相關文檔

- [SERVICES_ARCHITECTURE_COMPLIANCE_REPORT.md](../SERVICES_ARCHITECTURE_COMPLIANCE_REPORT.md) - 架構合規性報告
- [SERVICES_ORGANIZATION_SUMMARY.md](../SERVICES_ORGANIZATION_SUMMARY.md) - 服務組織總結
- [COMPLETE_BIDIRECTIONAL_FLOW_ANALYSIS.md](../COMPLETE_BIDIRECTIONAL_FLOW_ANALYSIS.md) - 完整訊息流分析

---

## 🎉 總結

AIVA CLI 和 AI 訓練系統提供了：

✅ **完整的命令行介面** - 掃描、檢測、報告、AI 管理  
✅ **500 萬參數 BioNeuron** - 生物啟發式神經網路  
✅ **真實流程訓練** - 基於實際 CLI 命令流程  
✅ **經驗學習系統** - 持續改進和優化  
✅ **RAG 整合** - 知識庫增強決策  

開始使用:
```bash
# 快速測試
python services/cli/aiva_cli.py scan start https://example.com --wait
python scripts/ai_training/integrated_cli_training.py
```
