# 三大決策方法架構對齊問題分析

**發現日期**: 2026年1月7日  
**問題嚴重性**: 🔴 CRITICAL - 架構不一致

---

## 🚨 發現的問題

### 當前實現 vs 13步驟流程對應錯誤

| 當前實現方法 | 預期功能 | 實際13步驟對應 | 問題 |
|-------------|----------|----------------|------|
| `decide_phase1_strategy()` | Phase1 深度掃描決策 | **步驟 6: 策略規劃** | ❌ 名稱不匹配 |
| `decide_phase2_targets()` | Phase2 攻擊目標排序 | **步驟 9: 漏洞確認** | ❌ 功能完全錯誤 |
| `evaluate_phase2_results()` | Phase2 結果評估 | **步驟 11: 攻擊執行** | ❌ 功能完全錯誤 |

### 13步驟中真正的決策點

根據 `13_STEPS_WORKFLOW_VERIFICATION.md`，真正的決策點是：

#### 步驟 6: 策略規劃 (Strategy Planning)
**執行模組**: `services/core/aiva_core/task_planning/`  
**功能**: 
- 基於目標類型選擇掃描策略
- 生成 AICommand 調度序列
- **決策**: 用什麼掃描引擎？什麼參數？

#### 步驟 9: 漏洞確認 (Vulnerability Verification)  
**執行模組**: `services/features/features_ready/`  
**功能**:
- 對高置信度發現進行二次驗證
- 調用 Features 模組進行深度檢測
- **決策**: 哪些發現值得深度驗證？

#### 步驟 11: 攻擊執行 (Attack Execution)
**執行模組**: `services/features/features_ready/`  
**功能**: 
- 執行 SQLi/XSS/SSRF/IDOR 等攻擊
- 記錄攻擊請求和響應
- **決策**: 選擇什麼攻擊向量和 Payload？

---

## 🔧 需要重構的內容

### 1. 方法名稱重新對齊

| 當前錯誤方法名 | 正確方法名 | 對應步驟 |
|---------------|------------|----------|
| ❌ `decide_phase1_strategy()` | ✅ `decide_scan_strategy()` | Step 6 |
| ❌ `decide_phase2_targets()` | ✅ `decide_verification_targets()` | Step 9 |
| ❌ `evaluate_phase2_results()` | ✅ `decide_attack_execution()` | Step 11 |

### 2. 功能重新設計

#### ✅ Step 6: `decide_scan_strategy()` - 掃描策略決策

**輸入**:
- 目標列表 (URLs/IPs)
- 目標類型分析結果
- 可用掃描引擎列表 (from RAG)

**決策邏輯**:
- Web 應用 → TypeScript Engine (SPA 爬蟲)
- API 端點 → Go Engine (高並發模糊測試)  
- 認證系統 → Rust Engine (暴力破解)
- 複雜業務邏輯 → Python Engine (深度分析)

**輸出**:
```python
{
    "selected_engines": [
        {"engine": "typescript", "targets": [...], "params": {...}},
        {"engine": "go", "targets": [...], "params": {...}}
    ],
    "scan_priority": "medium|high|critical",
    "estimated_time": 1800,  # 秒
    "aicommands": [...]  # 生成的調度命令
}
```

#### ✅ Step 9: `decide_verification_targets()` - 漏洞確認決策

**輸入**:
- Phase1 掃描結果 (UnifiedVulnerabilityFinding[])
- 信心度閾值
- 可用 Features 模組

**決策邏輯**:
- 信心度 > 0.8 且影響 High → 立即深度驗證
- 信心度 0.5-0.8 → 批量快速驗證
- 信心度 < 0.5 → 跳過或輕度檢查

**輸出**:
```python
{
    "high_priority_targets": [
        {"finding_id": "...", "verification_method": "deep", "feature_module": "function_sqli"}
    ],
    "batch_targets": [...],
    "skip_targets": [...],
    "verification_plan": {
        "estimated_time": 600,
        "parallel_workers": 3
    }
}
```

#### ✅ Step 11: `decide_attack_execution()` - 攻擊執行決策  

**輸入**:
- 已驗證的漏洞列表
- 攻擊目標偏好 (testing vs exploit)
- 可用攻擊模組

**決策邏輯**:
- SQLi → 選擇 Union-based vs Blind vs Time-based
- XSS → 選擇 DOM vs Stored vs Reflected payload
- SSRF → 選擇內網探測 vs Cloud metadata vs RCE chain

**輸出**:
```python
{
    "attack_plan": [
        {
            "target": {...}, 
            "attack_vector": "sqli_union",
            "payloads": ["' UNION SELECT 1,2,user() --", ...],
            "feature_module": "function_sqli",
            "success_criteria": {...}
        }
    ],
    "execution_order": "sequential|parallel",
    "safety_limits": {...}
}
```

---

## 🏗️ 架構修正方案

### Option A: 重構現有方法 (推薦)

1. 將現有的三個方法重新命名
2. 調整功能邏輯以符合13步驟定義
3. 更新所有調用代碼

### Option B: 保留現有方法，新增正確方法

1. 保持現有 `decide_phase1_strategy()` 等方法不變
2. 新增 `decide_scan_strategy()` 等方法
3. 在 Task Planning 中調用正確的方法

---

## 🎯 立即行動建議

1. **停止使用當前三個決策方法** - 它們與架構不符
2. **重新實現**符合13步驟的決策邏輯
3. **更新文檔**以反映正確的方法映射
4. **重新測試**整個工作流程

---

## 📚 參考文檔

- [13_STEPS_WORKFLOW_VERIFICATION.md](../../新增資料夾 (4)/新增資料夾/services_docs/workflows/13_STEPS_WORKFLOW_VERIFICATION.md)
- [TaskPlanning 模組架構](services/core/aiva_core/task_planning/)
- [Features 模組文檔](services/features/README.md)

---

**結論**: 目前的實現雖然功能完整，但與AIVA架構設計不符。需要重新對齊以確保系統一致性。