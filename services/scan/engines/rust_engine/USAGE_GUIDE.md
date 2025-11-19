# Rust Engine 使用指南

**版本**: 1.0.0  
**日期**: 2025-11-19  
**狀態**: ✅ 生產可用

---

## 🚀 快速開始

### 1. 編譯

```bash
cd services/scan/engines/rust_engine
cargo build --release
```

編譯後的可執行文件: `target/release/aiva-info-gatherer.exe` (Windows)

### 2. 基本使用

```bash
# Fast 模式 - 快速偵察 (推薦用於第一階段)
./target/release/aiva-info-gatherer scan \
  --url http://target.com \
  --mode fast \
  --timeout 10

# Deep 模式 - 深度分析
./target/release/aiva-info-gatherer scan \
  --url http://target.com \
  --mode deep \
  --timeout 20
```

---

## 📋 命令行參數

### scan 子命令

| 參數 | 簡寫 | 必填 | 說明 | 示例 |
|------|------|------|------|------|
| `--url` | `-u` | ✅ | 目標URL(可多個) | `--url http://target1.com http://target2.com` |
| `--mode` | `-m` | ✅ | 掃描模式 | `--mode fast` 或 `--mode deep` |
| `--timeout` | `-t` | ❌ | 超時時間(秒) | `--timeout 15` (默認10秒) |

### 掃描模式說明

| 模式 | 用途 | 端點發現 | JS文件 | 執行時間 |
|------|------|---------|--------|---------|
| **fast** | Phase0快速偵察 | 標準 | 3個 | ~200ms/目標 |
| **deep** | Phase1深度分析 | 擴展 | 4-6個 | ~400ms/目標 |

---

## 💡 使用場景

### 場景1: 單目標快速掃描

```bash
# 用於首次接觸目標,快速了解攻擊面
./target/release/aiva-info-gatherer scan \
  --url http://localhost:3000 \
  --mode fast \
  --timeout 10
```

**輸出內容**:
- 端點列表 (40+ 個常見路徑)
- JS Findings (API端點、內部域名、敏感註釋)
- 技術棧識別
- 風險等級分類

### 場景2: 多靶場並行掃描

```bash
# 同時掃描多個目標 (實際場景: Bug Bounty平台多個子域名)
./target/release/aiva-info-gatherer scan \
  --url http://app1.target.com http://app2.target.com http://api.target.com \
  --mode fast \
  --timeout 15
```

**效率提升**:
- 4個目標: ~700ms (vs 順序掃描 ~800ms)
- 並行處理,接近線性加速

### 場景3: 深度分析模式

```bash
# 用於重點目標的詳細偵察
./target/release/aiva-info-gatherer scan \
  --url http://priority-target.com \
  --mode deep \
  --timeout 20
```

**Deep模式額外功能**:
- 更多JS文件 (polyfills.js, scripts.js)
- 攻擊面評估報告
- 推薦測試引擎

### 場景4: 子路徑應用掃描

```bash
# 針對部署在子路徑的應用 (如 WebGoat)
./target/release/aiva-info-gatherer scan \
  --url http://localhost:8080/WebGoat/ \
  --mode fast \
  --timeout 15
```

**注意**: 需提供完整URL,包含子路徑

---

## 📊 輸出格式

### JSON 結構

```json
{
  "mode": "FastDiscovery",
  "targets": [
    {
      "url": "http://target.com",
      "success": true,
      "endpoints": [...],      // 端點列表
      "js_findings": [...],    // JS分析結果
      "sensitive_info": [...], // 敏感資訊
      "technologies": [...],   // 技術棧
      "attack_surface": "..."  // 攻擊面評估
    }
  ],
  "total_execution_time_ms": 713.0,
  "summary": {
    "total_targets": 1,
    "successful_scans": 1,
    "total_endpoints": 40,
    "total_sensitive_info": 0,
    "average_risk_score": 0.0
  }
}
```

### 解析示例 (PowerShell)

```powershell
# 保存輸出到變量
$output = .\target\release\aiva-info-gatherer.exe scan --url http://target.com --mode fast 2>&1 | Out-String

# 解析JSON
$jsonStart = $output.IndexOf('{')
$json = $output.Substring($jsonStart) | ConvertFrom-Json

# 提取關鍵信息
Write-Host "發現 $($json.summary.total_endpoints) 個端點"
Write-Host "JS Findings: $($json.targets[0].js_findings.Count) 個"

# 列出所有Critical端點
$json.targets[0].endpoints | Where-Object { $_.risk_level -eq "critical" } | ForEach-Object {
    Write-Host "⚠️ $($_.path) - $($_.method)"
}
```

---

## 🎯 與 AIVA 架構整合

### Phase0: 快速偵察 (必執行)

```bash
# Rust引擎作為第一階段偵察工具
./target/release/aiva-info-gatherer scan \
  --url $TARGET_URL \
  --mode fast \
  --timeout 10 > phase0_result.json
```

**為Phase1提供的情報**:
1. **端點清單** → 決定測試範圍
2. **技術棧** → 選擇Payload集
3. **風險等級** → 優先級排序
4. **JS Findings** → 發現隱藏攻擊面

### 與Python引擎配合

```python
# Python調用Rust引擎 (未來實現)
import subprocess
import json

result = subprocess.run(
    ['./target/release/aiva-info-gatherer', 'scan', 
     '--url', 'http://target.com', '--mode', 'fast'],
    capture_output=True, text=True
)

# 解析結果
data = json.loads(result.stdout.split('{', 1)[1])
endpoints = data['targets'][0]['endpoints']

# 傳遞給功能引擎
for endpoint in endpoints:
    if endpoint['risk_level'] == 'critical':
        test_sql_injection(endpoint['path'])
```

---

## ⚠️ 注意事項

### 1. 目標URL格式

✅ **正確**:
```
http://example.com
https://api.example.com
http://localhost:3000
http://192.168.1.100:8080/WebGoat/
```

❌ **錯誤**:
```
example.com              # 缺少協議
http://example.com:      # 無效端口
http://example com       # URL中有空格
```

### 2. 超時設置建議

| 目標類型 | 推薦超時 | 原因 |
|---------|---------|------|
| 本地靶場 | 10-15秒 | 網絡快速 |
| 遠程目標 | 20-30秒 | 考慮網絡延遲 |
| 慢速服務器 | 30-60秒 | 避免誤報失敗 |

### 3. 並行數量限制

- **推薦**: 4-8個目標
- **最大**: 取決於系統資源
- **注意**: 過多目標可能觸發WAF

### 4. 子路徑應用處理

對於部署在子路徑的應用:
```bash
# ✅ 正確 - 提供完整路徑
--url http://host:8080/WebGoat/

# ❌ 錯誤 - 僅根路徑
--url http://host:8080
```

---

## 🐛 常見問題

### Q1: 為什麼某些目標掃描不到端點?

**A**: 可能原因:
1. 應用部署在子路徑 → 使用完整URL
2. 所有端點需要認證 → 所有請求返回401/403
3. WAF攔截 → 減少並行數量或添加延遲

### Q2: JS Findings數量為何為0?

**A**: 可能原因:
1. 應用不使用JS框架 (純後端渲染)
2. JS文件路徑不標準 (不是 main.js/vendor.js)
3. JS文件需要認證才能訪問

### Q3: 如何處理HTTPS證書錯誤?

**A**: 當前版本會跳過證書驗證,未來版本可能需要:
```bash
# 未來可能的參數
--insecure  # 忽略證書錯誤
```

### Q4: 掃描是否會觸發WAF?

**A**: 可能會,因為:
- 字典爆破會發送50+個請求
- 快速連續請求可能被識別為攻擊

**緩解方法**:
- 減少並行目標數量
- 增加超時時間 (自然減速)
- 使用代理/VPN分散請求

---

## 📈 性能優化建議

### 1. 本地測試

```bash
# 最快速度
--timeout 5
```

### 2. 遠程掃描

```bash
# 考慮網絡延遲
--timeout 20
```

### 3. 大規模掃描

```bash
# 分批處理
for batch in {1..10}; do
  ./target/release/aiva-info-gatherer scan \
    --url $TARGET1 $TARGET2 $TARGET3 $TARGET4 \
    --mode fast --timeout 15
done
```

---

## 🔄 未來功能 (OPTIMIZATION_ROADMAP.md)

- [ ] A1: 消除重複代碼
- [ ] A2: Regex優化 (性能+15-20%)
- [ ] B1: 修復端點探測 (目前0結果)
- [ ] B2: 增強技術棧檢測 (30+ 種)
- [ ] B3: 自動發現JS文件
- [ ] C1: JS文件快取
- [ ] C2: 並行JS下載

---

## 📞 技術支持

- **文檔**: `README.md`, `WORKING_STATUS_2025-11-19.md`
- **優化計劃**: `OPTIMIZATION_ROADMAP.md`
- **架構**: `PHASE0_IMPLEMENTATION_PLAN.md`
