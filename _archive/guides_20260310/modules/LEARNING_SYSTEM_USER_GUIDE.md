# 📚 Learning System 使用者手冊

> **版本**: 2.0  
> **最後更新**: 2026-01-31  
> **位置**: `cognitive_core/learning_system/`

---

## 📑 目錄

1. [概述](#概述)
2. [快速開始](#快速開始)
3. [核心操作](#核心操作)
   - [策略調整 (StrategyAdjuster)](#策略調整-strategyadjuster)
   - [獎勵計算 (RewardConfig)](#獎勵計算-rewardconfig)
   - [參數建議 (suggest_parameters)](#參數建議-suggest_parameters)
4. [操作效果說明](#操作效果說明)
5. [整合流程](#整合流程)
6. [常見問題](#常見問題)

---

## 概述

**Learning System** 是 AIVA 的核心學習引擎，負責：
- 📊 **從執行結果學習** - 分析每次掃描/攻擊的結果
- 🧠 **優化策略** - 根據歷史經驗調整測試策略
- 🎯 **參數建議** - 建議最佳 CLI 參數組合

### 架構位置

```
cognitive_core/
└── learning_system/
    ├── analysis/
    │   └── dynamic_strategy_adjustment.py  ← 核心：StrategyAdjuster
    ├── learning/
    │   └── model_trainer.py
    ├── tracing/
    └── training/
```

---

## 快速開始

### 最小範例

```python
from cognitive_core.learning_system.analysis import StrategyAdjuster

# 1. 創建策略調整器
adjuster = StrategyAdjuster()

# 2. 從結果中學習
adjuster.learn_from_result({
    "scan_id": "scan_001",
    "module": "sqli",
    "success": True,
    "vulnerability": {
        "severity": "HIGH",
        "confidence": "CONFIRMED"
    },
    "waf_status": "bypassed",
    "parameters_used": {"mode": "stealth", "delay": 500}
})

# 3. 獲取統計
stats = adjuster.get_reward_statistics()
print(f"總學習次數: {stats['total_episodes']}")
print(f"平均獎勵: {stats['avg_reward']:.2f}")
```

---

## 核心操作

### 策略調整 (StrategyAdjuster)

#### 初始化

```python
from cognitive_core.learning_system.analysis import StrategyAdjuster, RewardConfig

# 使用預設獎勵配置
adjuster = StrategyAdjuster()

# 或自訂獎勵配置
custom_config = RewardConfig(
    VULN_CONFIRMED=15.0,   # 增加確認漏洞獎勵
    WAF_BYPASS=20.0,       # 增加 WAF 繞過獎勵
    EXECUTION_FAILURE=-10.0  # 增加失敗懲罰
)
adjuster = StrategyAdjuster(reward_config=custom_config)
```

#### 調整策略

```python
# 根據上下文調整測試計畫
adjusted_plan = adjuster.adjust(
    plan={
        "tasks": [
            {"type": "xss", "priority": 5},
            {"type": "sqli", "priority": 5}
        ]
    },
    context={
        "scan_id": "scan_001",
        "fingerprints": {
            "waf_vendor": "Cloudflare",
            "framework": {"name": "Django"}
        },
        "findings_count": 2
    }
)
```

**效果**：
- ✅ 檢測到 WAF → 自動增加延遲、使用繞過技術
- ✅ 根據技術棧 → 選擇適合的 Payload 類型
- ✅ 根據成功率 → 調整任務優先級

---

### 獎勵計算 (RewardConfig)

獎勵值定義了系統如何評估執行結果：

| 結果類型 | 獎勵值 | 說明 |
|---------|-------|------|
| `VULN_CONFIRMED` | +10.0 | 確認漏洞 (confidence=CONFIRMED) |
| `VULN_HIGH` | +8.0 | 高危漏洞 (severity=HIGH/CRITICAL) |
| `VULN_MEDIUM` | +5.0 | 中危漏洞 |
| `VULN_LOW` | +2.0 | 低危/資訊洩露 |
| `WAF_BYPASS` | +15.0 | 成功繞過 WAF |
| `SOFT_BLOCK` | -1.0 | 被 WAF 軟封鎖 |
| `EXECUTION_FAILURE` | -5.0 | 執行失敗 |
| `NO_RESULT` | -2.0 | 無發現 |

#### 獎勵計算流程

```
執行結果 (feedback_data)
    ↓
┌──────────────────────────────────────┐
│ 1. 檢查 success 欄位                  │
│    └─ False → 返回 EXECUTION_FAILURE │
│                                       │
│ 2. 檢查 vulnerability 欄位            │
│    ├─ confidence="CONFIRMED" → +10   │
│    ├─ severity="HIGH" → +8           │
│    ├─ severity="MEDIUM" → +5         │
│    └─ 無漏洞 → -2                     │
│                                       │
│ 3. 檢查 waf_status 欄位               │
│    ├─ "bypassed" → +15               │
│    └─ "blocked" → -1                 │
└──────────────────────────────────────┘
    ↓
總獎勵 = 各項加總
```

---

### 參數建議 (suggest_parameters)

在發送 CLI 指令**前**，詢問學習系統建議的參數：

```python
# 獲取建議參數
suggested = adjuster.suggest_parameters(
    module="xss",
    context={
        "waf_detected": True,
        "fingerprints": {"language": {"php": "7.4"}}
    }
)

print(suggested)
# 輸出: {"mode": "stealth", "delay": 500, "payload_type": "php_specific"}
```

**參數建議邏輯**：
1. 從歷史中找出該模組獎勵最高的參數組合
2. 根據 context 調整：
   - `waf_detected=True` → 切換 stealth 模式、增加延遲
   - PHP 技術棧 → 使用 PHP 專用 Payload

---

## 操作效果說明

### 1. learn_from_result 效果

| 輸入 | 效果 |
|------|------|
| `success=True, vulnerability.confidence=CONFIRMED` | 獎勵 +10，記錄成功模式 |
| `success=True, waf_status=bypassed` | 額外獎勵 +15 |
| `success=False` | 懲罰 -5，記錄失敗模式 |
| `parameters_used={...}` | 記錄參數歷史，供未來建議 |

### 2. adjust 效果

| 上下文 | 調整效果 |
|-------|---------|
| `waf_vendor` 存在 | 增加延遲、啟用繞過模式 |
| `fingerprints.framework=Django` | 檢查 CSRF token |
| `fingerprints.language=PHP` | 啟用 PHP Wrapper |
| `findings_count > 3` | 降低低優先級任務 |

### 3. suggest_parameters 效果

| 條件 | 建議參數 |
|------|---------|
| 無歷史數據 | 返回預設參數 |
| 有歷史，無 context | 返回最高獎勵參數 |
| `waf_detected=True` | mode=stealth, delay≥500 |
| PHP 技術棧 | payload_type=php_specific |

---

## 整合流程

### 與 app.py 的整合

Learning System 已整合到 `app.py`：

```python
# app.py 中的使用方式
strategy_adjuster = StrategyAdjuster()

# 在 process_function_results() 中自動呼叫
feedback_data = {
    "scan_id": scan_id,
    "module": msg.header.source_module,
    "vulnerability": vulnerability_info,
    "success": vulnerability_info.get("confidence") == "CONFIRMED",
}
strategy_adjuster.learn_from_result(feedback_data)
```

### 資料流

```
┌─────────────────────────────────────────────────────────┐
│                    資料流程                              │
└─────────────────────────────────────────────────────────┘

1. 發送前 (決策階段)
   ┌─────────────────────────────────────────────────────┐
   │ Task Planning                                        │
   │     │                                                │
   │     ▼                                                │
   │ suggest_parameters(module, context)                  │
   │     │                                                │
   │     ▼                                                │
   │ 組裝 CLI 指令並發送到 MQ                             │
   └─────────────────────────────────────────────────────┘

2. 執行中 (Worker)
   ┌─────────────────────────────────────────────────────┐
   │ Worker 接收指令 → 執行 CLI 工具 → 發布結果到 MQ      │
   └─────────────────────────────────────────────────────┘

3. 接收後 (學習階段)
   ┌─────────────────────────────────────────────────────┐
   │ app.py 訂閱 MQ                                       │
   │     │                                                │
   │     ▼                                                │
   │ process_function_results()                           │
   │     │                                                │
   │     ▼                                                │
   │ learn_from_result(feedback_data)                     │
   │     │                                                │
   │     ▼                                                │
   │ 更新獎勵歷史、參數歷史                                │
   └─────────────────────────────────────────────────────┘
```

---

## 常見問題

### Q: 如何查看學習統計？

```python
stats = adjuster.get_reward_statistics()
print(f"總學習次數: {stats['total_episodes']}")
print(f"平均獎勵: {stats['avg_reward']:.2f}")
print(f"最高獎勵: {stats['max_reward']}")
print(f"正向比例: {stats['positive_rate']:.1%}")
```

### Q: 如何重置學習歷史？

```python
# 重新創建實例
adjuster = StrategyAdjuster()
```

### Q: 如何自訂獎勵配置？

```python
from cognitive_core.learning_system.analysis import RewardConfig

config = RewardConfig(
    VULN_CONFIRMED=20.0,  # 自訂值
    WAF_BYPASS=25.0
)
adjuster = StrategyAdjuster(reward_config=config)
```

### Q: 參數建議不符合預期？

1. 確認有足夠的歷史數據
2. 檢查 context 是否正確傳入
3. 查看歷史中最高獎勵的參數組合

---

## 相關文檔

- [QUICK_START_GUIDE](../general/QUICK_START_GUIDE.md) - AIVA 快速啟動指南
- [LEARNING_INTEGRATION_WITH_13STEPS.md](../../docs/LEARNING_INTEGRATION_WITH_13STEPS.md) - 13 步驟整合詳細設計
- [dynamic_strategy_adjustment.py](../../services/core/aiva_core/cognitive_core/learning_system/analysis/dynamic_strategy_adjustment.py) - 原始碼

---

**維護者**: AIVA Development Team  
**授權**: MIT License
