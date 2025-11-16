# 📖 AIVA 術語對照表與設計理念統一說明

**創建日期**: 2025年11月15日  
**目的**: 統一 AIVA 系統中容易混淆的術語,確保文檔方向一致

---

## 🎯 核心設計理念

### **AI 自我優化雙重閉環**

AIVA 的核心設計理念是通過**內部閉環**(系統自省)和**外部閉環**(實戰反饋)實現持續自我優化:

```
┌─────────────────────────────────────────────────┐
│         AIVA AI 自我優化雙重閉環                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  內部閉環 (Know Thyself)                        │
│  ├─ 探索 (對內): 掃描 AIVA 自身的五大模組       │
│  ├─ 分析 (靜態): 評估代碼品質和可優化點         │
│  └─ RAG (知識): 檢索相似案例和最佳實踐         │
│       ↓                                         │
│  結果: 知道「我有什麼能力」「哪裡需要改進」      │
│                                                 │
│  ×                                              │
│                                                 │
│  外部閉環 (Learn from Battle)                   │
│  ├─ 掃描 (對外): 探測目標系統的技術棧和漏洞     │
│  ├─ 攻擊 (實戰): 執行安全測試並收集反饋         │
│  └─ 數據收集: 記錄成功/失敗案例和有效技術      │
│       ↓                                         │
│  結果: 知道「外部環境需要什麼」「該朝哪優化」    │
│                                                 │
│       ↓                                         │
│  AI 決策中心: 整合雙重閉環數據                  │
│       ↓                                         │
│  視覺化展示: 用圖表呈現優化方案 (減少NLP負擔)   │
│       ↓                                         │
│  人工審核: 批准/拒絕/修改                       │
│       ↓                                         │
│  自動執行: 代碼生成、CLI優化、策略調整          │
│       ↓                                         │
│  循環回到探索/掃描 (持續進化)                   │
└─────────────────────────────────────────────────┘
```

**詳細設計**: 參見 [AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md](./AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)

---

## 📚 術語對照表

### **容易混淆的術語**

#### **1. "探索" (Exploration)**

| 術語 | 英文 | 含義 | 應用範圍 | 示例模組 | 閉環類型 |
|------|------|------|---------|---------|---------|
| **系統自我探索** | System Self-Exploration<br/>Introspection | AIVA 對**自身系統**的內省和診斷 | AIVA 內部 | `SystemSelfExplorer`<br/>`ai_system_explorer.py` | 內部閉環 |
| **攻擊面探索** | Attack Surface Exploration | 對**目標系統**的攻擊入口發現 | 外部目標 | `attack_surface_mapper` | 外部閉環 |

**關鍵區別**:
- ✅ **對內探索** = AIVA 診斷自己 (我有什麼能力?)
- ✅ **對外探索** = AIVA 掃描目標 (目標有什麼漏洞?)

---

#### **2. "偵察" (Reconnaissance)**

| 術語 | 英文 | 含義 | 應用範圍 | 示例模組 | 閉環類型 |
|------|------|------|---------|---------|---------|
| **目標偵察** | Target Reconnaissance | 收集**目標系統**的信息 | 外部目標 | `real_web_reconnaissance`<br/>`network_scanner` | 外部閉環 |
| **系統診斷** | System Diagnostics | 檢查**AIVA 自身**的健康狀態 | AIVA 內部 | `health_checker`<br/>`module_diagnostics` | 內部閉環 |

**關鍵區別**:
- ✅ **偵察** = 對外收集目標信息
- ✅ **診斷** = 對內檢查系統健康

---

#### **3. "分析" (Analysis)**

| 術語 | 英文 | 含義 | 應用範圍 | 示例模組 | 閉環類型 |
|------|------|------|---------|---------|---------|
| **靜態代碼分析** | Static Code Analysis | 分析**AIVA 自身**的代碼品質 | AIVA 代碼 | `AnalysisEngine`<br/>`ai_analysis/analysis_engine.py` | 內部閉環 |
| **漏洞分析** | Vulnerability Analysis | 分析**目標系統**的安全漏洞 | 外部目標 | `vuln_analyzer`<br/>`exploit_analyzer` | 外部閉環 |

**關鍵區別**:
- ✅ **靜態分析** = 分析 AIVA 自己的代碼
- ✅ **漏洞分析** = 分析目標的安全問題

---

#### **4. "掃描" (Scanning)**

| 術語 | 英文 | 含義 | 應用範圍 | 示例模組 | 閉環類型 |
|------|------|------|---------|---------|---------|
| **模組掃描** | Module Scanning | 掃描**AIVA 自身**的模組狀態 | AIVA 內部 | `SystemSelfExplorer.scan_modules()` | 內部閉環 |
| **目標掃描** | Target Scanning | 掃描**外部目標**的端口和服務 | 外部目標 | `scan_engine`<br/>`nmap_scanner` | 外部閉環 |

**關鍵區別**:
- ✅ **模組掃描** = 掃描 AIVA 的內部模組
- ✅ **目標掃描** = 掃描外部系統

---

### **清晰命名規範**

#### **對內操作 (Internal Operations)**

推薦命名模式:
```
✅ system_self_*       # 系統自我...
✅ internal_*          # 內部...
✅ introspection_*     # 內省...
✅ self_diagnostics_*  # 自我診斷...
✅ health_check_*      # 健康檢查...
```

示例:
- `system_self_explorer.py` (系統自我探索器)
- `internal_diagnostics.py` (內部診斷)
- `self_health_monitor.py` (自我健康監控)

#### **對外操作 (External Operations)**

推薦命名模式:
```
✅ target_*            # 目標...
✅ external_*          # 外部...
✅ reconnaissance_*    # 偵察...
✅ attack_surface_*    # 攻擊面...
✅ vulnerability_*     # 漏洞...
```

示例:
- `target_reconnaissance.py` (目標偵察)
- `external_scanner.py` (外部掃描器)
- `attack_surface_mapper.py` (攻擊面映射)

---

## 🔍 設計理念統一說明

### **1. 三項基礎能力 (數據收集層)**

| 能力 | 範圍 | 目的 | 閉環類型 |
|------|------|------|---------|
| **探索功能** | 對內 | 知道「我有什麼」 | 內部閉環 |
| **靜態分析** | 對內 | 知道「品質如何」 | 內部閉環 |
| **RAG 知識** | 通用 | 知道「如何做」 | 輔助兩者 |

**關鍵點**: 這三項能力都是為了讓 AI 了解自己的現狀

---

### **2. 雙重閉環整合 (優化決策層)**

#### **內部閉環** (Know Thyself)

**問題**: 我有什麼能力? 哪裡需要改進?

**數據來源**:
- 探索: 掃描五大模組,列出能力清單
- 分析: 評估代碼品質,找出可優化點
- RAG: 檢索最佳實踐和改進建議

**輸出**: 能力缺口列表 + 優化目標

#### **外部閉環** (Learn from Battle)

**問題**: 外部環境需要什麼? 實戰中哪些有效?

**數據來源**:
- 掃描: 收集目標系統信息
- 攻擊: 執行安全測試
- 反饋: 記錄成功/失敗案例

**輸出**: 優化方向建議 + 有效技術清單

---

### **3. 視覺化優先策略 (人機協作層)**

**設計理念**: 用圖表而非自然語言展示優化方案

**優勢**:
- ✅ 減少 NLP 資源消耗
- ✅ 提高審核效率 (秒級 vs 分鐘級)
- ✅ 降低理解門檻
- ✅ 支援互動式調整

**實現方式**:
```
AI 生成優化方案
     ↓
轉換為視覺化圖表:
  • 優化方向拓撲圖
  • 能力提升路徑圖
  • 資源分配餅圖
  • 優先級排序矩陣
     ↓
展示給用戶審核
     ↓
用戶一鍵批准/拒絕
     ↓
自動執行優化
```

---

## 📋 文檔更新檢查表

使用此檢查表確保所有文檔方向一致:

### **術語使用檢查**

- [ ] 所有"探索"相關術語明確標註「對內」或「對外」
- [ ] 對內操作使用: `self_*`, `internal_*`, `introspection_*`
- [ ] 對外操作使用: `target_*`, `external_*`, `attack_*`
- [ ] 檔案命名符合命名規範

### **設計理念檢查**

- [ ] 文檔中提到探索系統時,說明其為內部閉環的一部分
- [ ] 文檔中提到掃描/攻擊時,說明其為外部閉環的一部分
- [ ] 說明雙重閉環如何協同工作
- [ ] 強調視覺化優先的優化方案審核機制

### **示例代碼檢查**

- [ ] 示例代碼註釋清楚說明操作範圍 (對內/對外)
- [ ] 變數命名符合術語規範
- [ ] 函數註釋明確說明用途

---

## 🔗 相關文檔

### **核心設計文檔**
- [AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md](./AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md) - 完整雙重閉環設計
- [EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md](./EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md) - 術語混淆根因分析

### **技術實現文檔**
- [services/core/aiva_core/README.md](./services/core/aiva_core/README.md) - 核心模組技術文檔
- [scripts/ai_analysis/ai_system_explorer.py](./scripts/ai_analysis/ai_system_explorer.py) - 系統自我探索實現
- [services/core/aiva_core/ai_analysis/analysis_engine.py](./services/core/aiva_core/ai_analysis/analysis_engine.py) - 靜態分析實現

### **用戶文檔**
- [README.md](./README.md) - 項目主要說明
- [AIVA_USER_MANUAL.md](./AIVA_USER_MANUAL.md) - 用戶手冊

---

## 📞 反饋與改進

如發現文檔中的術語使用不一致或有改進建議,請:

1. **查閱此術語表**確認正確用法
2. **提交 Issue** 說明問題和建議
3. **參考設計文檔**理解完整設計理念

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025年11月15日  
**版本**: v1.0

*此術語表確保 AIVA 所有文檔保持一致的設計方向和術語使用。*
