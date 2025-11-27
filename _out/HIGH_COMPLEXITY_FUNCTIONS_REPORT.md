# 高複雜度函數記錄報告

## 📑 目錄

- [📊 複雜度過高的函數清單](#-複雜度過高的函數清單)
  - [1. MultiEngineCoordinator - execute_phase0()](#1-multienginecoordinator---execute_phase0)
  - [2. MultiEngineCoordinator - execute_phase1()](#2-multienginecoordinator---execute_phase1)
  - [3. TypeScriptAdapter - scan()](#3-typescriptadapter---scan)
  - [4. RustAdapter - scan()](#4-rustadapter---scan)
- [📋 處理建議](#-處理建議)
  - [高優先級（待 Untitled.md 完成後處理）](#高優先級待-untitledmd-完成後處理)
  - [中優先級](#中優先級)
  - [低優先級](#低優先級)
- [🎯 執行時機](#-執行時機)
- [📝 備註](#-備註)

---

> 生成時間：2025-11-21
> 
> 目的：記錄在重構過程中發現的所有認知複雜度過高的函數，待完成 Untitled.md 所有建議後再處理

## 📊 複雜度過高的函數清單

### 1. MultiEngineCoordinator - execute_phase0()
- **文件**: `services/scan/coordinators/multi_engine_coordinator.py`
- **位置**: Line 374
- **複雜度**: 24 (閾值: 15)
- **超出**: +9
- **問題**: 包含複雜的條件判斷和錯誤處理邏輯
- **優先級**: 中（業務邏輯複雜性）

### 2. MultiEngineCoordinator - execute_phase1()
- **文件**: `services/scan/coordinators/multi_engine_coordinator.py`
- **位置**: Line 489
- **複雜度**: 17 ✅ **已改進** (原 171)
- **改進幅度**: -154 (-90%)
- **狀態**: ✅ 已使用適配器模式重構
- **優先級**: 已完成（從 171 降至 17，接近閾值）

### 3. TypeScriptAdapter - scan()
- **文件**: `services/scan/coordinators/engines/typescript_adapter.py`
- **位置**: Line 75
- **複雜度**: 21 (閾值: 15)
- **超出**: +6
- **問題**: 包含多層錯誤處理和 JSON 解析策略
- **優先級**: 低（已有清晰的註釋和結構）

### 4. RustAdapter - scan()
- **文件**: `services/scan/coordinators/engines/rust_adapter.py`
- **位置**: Line 53
- **複雜度**: 22 (閾值: 15)
- **超出**: +7
- **問題**: 線程池包裝和錯誤處理邏輯
- **優先級**: 低（已有清晰的註釋和結構）

---

## 📋 處理建議

### 高優先級（待 Untitled.md 完成後處理）

#### execute_phase1() - 複雜度 171
建議拆分策略：
1. **提取引擎調用邏輯** → `_execute_python_engine()`, `_execute_typescript_engine()`, 等
2. **提取結果聚合邏輯** → `_aggregate_engine_results()`
3. **提取狀態判斷邏輯** → `_determine_scan_status()`
4. **提取資產去重邏輯** → `_deduplicate_all_assets()`

預期效果：
- 主函數複雜度降至 < 30
- 每個子函數複雜度 < 15
- 提高可讀性和可測試性

### 中優先級

#### execute_phase0() - 複雜度 24
建議：
1. 提取 Rust 引擎調用邏輯
2. 簡化錯誤處理流程

### 低優先級

#### Adapter scan() 方法
- 這些方法的複雜度來自於健壯的錯誤處理
- 已有清晰的註釋和文檔
- 可接受的業務邏輯複雜性

---

## 🎯 執行時機

**當前狀態**: 記錄完成，暫不處理

**處理時機**: 完成以下任務後：
1. ✅ 階段 1：關鍵邏輯修復
2. ✅ 階段 2：死代碼清理
3. ✅ 階段 3：適配器模式實現
4. ⏳ 階段 4：Coordinator 重構（使用適配器）
5. ⏳ 階段 5：複雜度重構（本報告）

---

## 📝 備註

- SonarLint 複雜度閾值設為 15
- 認知複雜度 (Cognitive Complexity) 測量的是代碼的可理解性
- 部分複雜度來自於業務邏輯本身的複雜性，無法完全消除
- 目標：將所有函數複雜度控制在 30 以內
