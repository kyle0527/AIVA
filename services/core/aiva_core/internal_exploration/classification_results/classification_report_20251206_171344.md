# 流程分類分析報告

生成時間: 2025-12-06T17:13:44.790269
分析範圍: c:\D\fold7\AIVA-git\flow_analysis_ai_core

---

## 統計概覽

- 總流程數: 268
- 唯一終點: 21
- 唯一起點: 4
- 唯一模式: 41
- 平均長度: 5.94 步
- 長度範圍: 4 - 8 步

---

## 按終點分類


### models (155次, 57.8%)

範例路徑:
- bio → trainer → neural → network → rl → models
- bio → trainer → neural → network → rl → models
- bio → trainer → neural → network → rl → models

### trainers (30次, 11.2%)

範例路徑:
- bio → trainer → rl → trainers
- bio → trainer → neural → network → rl → trainers
- bio → trainer → neural → network → rl → trainers

### store (25次, 9.3%)

範例路徑:
- bio → trainer → rl → trainers → vector → store
- bio → trainer → rl → trainers → vector → store
- bio → trainer → rl → trainers → vector → store

### orchestrator (13次, 4.9%)

範例路徑:
- bio → trainer → model → trainer → training → orchestrator
- bio → trainer → rl → trainers → capability → orchestrator
- initial → surface → exploit → orchestrator

### manager (6次, 2.2%)

範例路徑:
- strategy → generator → scenario → manager
- bio → trainer → model → trainer → ai → model → manager
- bio → trainer → rl → trainers → scenario → manager

### core (6次, 2.2%)

範例路徑:
- bio → trainer → neural → network → real → neural → core
- bio → trainer → real → neural → core
- bio → trainer → rl → trainers → real → neural → core

### mapper (4次, 1.5%)

範例路徑:
- bio → trainer → permission → matrix → authz → mapper
- bio → trainer → rl → trainers → authz → mapper
- logging → formatter → authz → mapper

### trainer (4次, 1.5%)

範例路徑:
- bio → trainer → model → trainer
- bio → trainer → rl → trainers → model → trainer
- bio → trainer → rl → trainers → model → trainer

### monitor (4次, 1.5%)

範例路徑:
- bio → trainer → rl → trainers → execution → status → monitor
- bio → trainer → rl → trainers → execution → status → monitor
- bio → trainer → rl → trainers → execution → status → monitor

### classifier (4次, 1.5%)

範例路徑:
- bio → trainer → rl → trainers → train → classifier
- bio → trainer → rl → trainers → train → classifier
- bio → trainer → rl → trainers → train → classifier

---

## 關鍵節點分析

- **bio**: 257條路徑 (95.9%) 風險等級: HIGH
- **trainer**: 257條路徑 (95.9%) 風險等級: HIGH
- **rl**: 242條路徑 (90.3%) 風險等級: HIGH
- **network**: 189條路徑 (70.5%) 風險等級: MEDIUM
- **neural**: 191條路徑 (71.3%) 風險等級: MEDIUM
- **models**: 155條路徑 (57.8%) 風險等級: MEDIUM
- **trainers**: 87條路徑 (32.5%) 風險等級: LOW
- **vector**: 25條路徑 (9.3%) 風險等級: LOW
- **store**: 25條路徑 (9.3%) 風險等級: LOW
- **orchestrator**: 13條路徑 (4.9%) 風險等級: LOW