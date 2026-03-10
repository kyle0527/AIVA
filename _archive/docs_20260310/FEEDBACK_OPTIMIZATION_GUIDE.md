# 攻擊反饋優化系統使用指南

## 📑 目錄

- [概述](#概述)
- [核心功能](#核心功能)
  - [1. 自動反饋收集](#1-自動反饋收集)
  - [2. 效能評分計算](#2-效能評分計算)
  - [3. 策略優化分析](#3-策略優化分析)
    - [3.1 識別表現不佳的策略](#31-識別表現不佳的策略)
    - [3.2 識別高頻錯誤模式](#32-識別高頻錯誤模式)
    - [3.3 識別 WAF 繞過機會](#33-識別-waf-繞過機會)
- [使用方法](#使用方法)
  - [基本使用](#基本使用)
  - [查看優化報告](#查看優化報告)
  - [查看學習狀態](#查看學習狀態)
- [優化建議類型](#優化建議類型)
  - [1. 表現不佳優化](#1-表現不佳優化)
  - [2. 錯誤修復優化](#2-錯誤修復優化)
  - [3. WAF 繞過優化](#3-waf-繞過優化)
- [優化流程](#優化流程)
- [配置選項](#配置選項)
  - [初始化參數](#初始化參數)
  - [反饋優化參數](#反饋優化參數)
- [實戰案例](#實戰案例)
  - [案例 1：自動修復 WAF 繞過問題](#案例-1自動修復-waf-繞過問題)
  - [案例 2：提升攻擊效率](#案例-2提升攻擊效率)
- [監控和調試](#監控和調試)
  - [啟用詳細日誌](#啟用詳細日誌)
  - [關鍵日誌輸出](#關鍵日誌輸出)
- [最佳實踐](#最佳實踐)
- [架構優勢](#架構優勢)
- [注意事項](#注意事項)
- [後續改進方向](#後續改進方向)
- [參考資料](#參考資料)

---


## 概述

統一攻擊執行器現在包含了基於反饋的智能優化機制，能夠根據每次攻擊的執行結果自動調整和優化攻擊策略。

## 核心功能

### 1. 自動反饋收集

每次攻擊執行後，系統會自動收集以下反饋數據：

- **成功率**：攻擊是否成功達成目標
- **漏洞發現數量**：找到的漏洞總數
- **執行時間**：攻擊完成所需時間
- **錯誤率**：執行過程中出現錯誤的比例
- **WAF 觸發情況**：是否觸發了 Web 應用防火牆

### 2. 效能評分計算

系統會根據以下因素計算攻擊效能分數（0.0 - 1.0）：

```
效能分數 = 成功率 × 40% + 漏洞數量分數 × 40% + 效率分數 × 20%
```

- **成功率分數**（40%）：攻擊是否達成目標
- **漏洞數量分數**（40%）：發現的漏洞數量（歸一化）
- **效率分數**（20%）：執行速度（越快越好）

### 3. 策略優化分析

當累積 10 次反饋後，系統會自動進行優化分析：

#### 3.1 識別表現不佳的策略
- 平均效能分數 < 0.5 的策略將被標記
- 系統建議：增加超時時間、添加重試邏輯、調整 payload 編碼

#### 3.2 識別高頻錯誤模式
- 錯誤率 > 30% 的策略將被分析
- 系統建議：添加錯誤處理、驗證輸入、添加降級策略

#### 3.3 識別 WAF 繞過機會
- WAF 觸發率 > 50% 的策略需要改進
- 系統建議：使用混淆、添加請求延遲、輪換 User-Agent、使用編碼變體

## 使用方法

### 基本使用

```python
from services.core.aiva_core.task_planning.unified_executor import UnifiedAttackExecutor

# 初始化執行器（學習模式開啟）
executor = UnifiedAttackExecutor(
    learning_enabled=True,
    auto_train_threshold=100,
    min_train_interval=3600
)

# 執行攻擊（自動收集反饋）
result = await executor.execute(
    target="https://example.com",
    objective="檢查 XSS 漏洞"
)

# 查看反饋信息
print(f"攻擊成功: {result.success}")
print(f"發現漏洞: {len(result.vulnerabilities)}")
print(f"學習狀態: {result.learning_info}")
```

### 查看優化報告

```python
# 獲取優化報告
report = executor.get_optimization_report()

print(f"總反饋數: {report['total_feedbacks']}")
print(f"追蹤策略數: {report['strategies_tracked']}")

# 查看每個策略的性能
for strategy, perf in report['strategy_performance'].items():
    print(f"\n策略: {strategy}")
    print(f"  平均分數: {perf['avg_score']:.2f}")
    print(f"  最佳分數: {perf['best_score']:.2f}")
    print(f"  使用次數: {perf['total_uses']}")
    print(f"  趨勢: {perf['trend']}")
```

### 查看學習狀態

```python
# 獲取學習狀態
status = executor.get_learning_status()

print(f"學習模式: {'開啟' if status['learning_enabled'] else '關閉'}")
print(f"經驗累積: {status['progress']}")
print(f"距離上次訓練: {status['time_since_last_train']} 秒")
```

## 優化建議類型

### 1. 表現不佳優化

**觸發條件**：策略平均效能 < 0.5

**建議調整**：
- ✅ 增加超時時間
- ✅ 添加重試邏輯
- ✅ 調整 payload 編碼
- ✅ 多樣化攻擊向量

**優先級**：8/10

### 2. 錯誤修復優化

**觸發條件**：策略出現高頻錯誤（≥3 次）

**建議調整**：
- ✅ 添加錯誤處理
- ✅ 驗證輸入參數
- ✅ 添加降級策略
- ✅ 降低並發度

**優先級**：9/10

### 3. WAF 繞過優化

**觸發條件**：WAF 觸發率 > 50%

**建議調整**：
- ✅ 使用混淆技術
- ✅ 添加請求延遲
- ✅ 輪換 User-Agent
- ✅ 使用編碼變體
- ✅ 分割 payload

**優先級**：10/10（最高）

## 優化流程

```
1. 執行攻擊
   ↓
2. 收集反饋數據
   ↓
3. 計算效能分數
   ↓
4. 更新策略性能記錄
   ↓
5. 累積 10 次反饋後觸發優化分析
   ↓
6. 識別問題模式
   ↓
7. 生成優化建議
   ↓
8. 自動應用優化（更新 RAG 知識庫）
   ↓
9. 下次攻擊使用優化後的策略
```

## 配置選項

### 初始化參數

```python
executor = UnifiedAttackExecutor(
    learning_enabled=True,         # 是否啟用學習模式
    auto_train_threshold=100,      # 自動訓練閾值（經驗數量）
    min_train_interval=3600,       # 最小訓練間隔（秒）
    data_directory=Path("./data")  # 數據存儲目錄
)
```

### 反饋優化參數

- **優化觸發閾值**：累積 10 次反饋
- **表現不佳閾值**：平均分數 < 0.5
- **錯誤頻率閾值**：≥3 次錯誤
- **WAF 觸發閾值**：觸發率 > 50%

## 實戰案例

### 案例 1：自動修復 WAF 繞過問題

```python
# 第 1-5 次攻擊：不斷觸發 WAF
# 系統收集反饋：WAF 觸發率 80%

# 第 10 次攻擊後：自動優化觸發
# 系統生成優化建議：
# - 使用 payload 混淆
# - 添加請求延遲
# - 輪換 User-Agent

# 第 11 次攻擊：使用優化後的策略
# 結果：WAF 觸發率降至 20%
```

### 案例 2：提升攻擊效率

```python
# 初始狀態：平均執行時間 45 秒，成功率 60%
# 累積 20 次反饋後

# 優化分析結果：
# - 策略 A：平均分數 0.45（不佳）
# - 策略 B：平均分數 0.75（良好）

# 系統建議：
# - 優化策略 A 的超時配置
# - 增加策略 B 的使用頻率

# 優化後：平均執行時間 30 秒，成功率 80%
```

## 監控和調試

### 啟用詳細日誌

```python
import logging
logging.getLogger("aiva_core.task_planning.unified_executor").setLevel(logging.DEBUG)
```

### 關鍵日誌輸出

```
📊 Collecting feedback for attack: plan_xxx
✅ Feedback collected: effectiveness=0.75, success=1.0, vulns=3
🔧 Optimizing attack strategies based on feedback...
📈 Generated 3 optimization recommendations
  • strategy_xss: WAF 觸發率過高 (80.0%) (priority=10, score=0.20)
  • strategy_sqli: 高頻錯誤模式 (出現 4 次) (priority=9, score=0.00)
✅ Applied optimization for: strategy_xss
```

## 最佳實踐

1. **保持學習模式開啟**：讓系統持續從每次攻擊中學習
2. **定期查看優化報告**：了解策略性能趨勢
3. **分析高優先級建議**：優先處理 WAF 繞過和錯誤修復
4. **調整優化閾值**：根據實際需求調整觸發條件
5. **備份反饋數據**：保存歷史反饋用於長期分析

## 架構優勢

✅ **自動化**：無需手動分析，系統自動優化
✅ **持續改進**：每次攻擊都是學習機會
✅ **數據驅動**：基於實際執行結果做決策
✅ **智能適應**：自動適應不同的目標和防護
✅ **透明可見**：完整的反饋和優化報告

## 注意事項

⚠️ **隱私保護**：反饋數據包含攻擊詳情，需妥善保管
⚠️ **存儲空間**：長期運行會累積大量反饋數據
⚠️ **優化延遲**：需要累積足夠反饋才能觸發優化
⚠️ **策略衝突**：多個優化建議可能存在衝突，需要權衡

## 後續改進方向

- 🔜 支援多目標優化（同時優化成功率和效率）
- 🔜 增加策略 A/B 測試功能
- 🔜 實時優化（不等待累積閾值）
- 🔜 跨目標策略遷移（相似目標共享優化）
- 🔜 可視化反饋分析儀表板

## 參考資料

- [統一執行器架構文檔](./UNIFIED_EXECUTOR_ARCHITECTURE.md)
- [RAG 知識庫集成](./RAG_INTEGRATION.md)
- [持續學習系統](./CONTINUOUS_LEARNING.md)
