# 多路徑 Flows 分析報告

## 📑 目錄

- [📊 整體統計](#-整體統計)
- [📈 路徑數量分佈](#-路徑數量分佈)
- [🏆 路徑最多的組合 (Top 10)](#-路徑最多的組合-top-10)
  - [1. session_state_manager → run_analysis](#1-session_state_manager-run_analysis)
  - [2. session_state_manager → core_analyzer](#2-session_state_manager-core_analyzer)
  - [3. session_state_manager → practical_analyzer](#3-session_state_manager-practical_analyzer)
  - [4. session_state_manager → assistant](#4-session_state_manager-assistant)
  - [5. session_state_manager → system_connectivity_checker](#5-session_state_manager-system_connectivity_checker)
  - [6. session_state_manager → app](#6-session_state_manager-app)
  - [7. session_state_manager → bizlogic_attack_executor](#7-session_state_manager-bizlogic_attack_executor)
  - [8. session_state_manager → ai_commander](#8-session_state_manager-ai_commander)
  - [9. session_state_manager → model_trainer](#9-session_state_manager-model_trainer)
  - [10. session_state_manager → aiva_cli_implementation](#10-session_state_manager-aiva_cli_implementation)
- [📏 路徑長度差異最大的組合 (Top 5)](#-路徑長度差異最大的組合-top-5)
  - [1. scalable_bio_trainer → system_connectivity_checker](#1-scalable_bio_trainer-system_connectivity_checker)
  - [2. scalable_bio_trainer → command_repository](#2-scalable_bio_trainer-command_repository)
  - [3. scalable_bio_trainer → model_trainer](#3-scalable_bio_trainer-model_trainer)
  - [4. scalable_bio_trainer → ai_model_manager](#4-scalable_bio_trainer-ai_model_manager)
  - [5. scalable_bio_trainer → real_bio_net_adapter](#5-scalable_bio_trainer-real_bio_net_adapter)
- [💡 設計洞察](#-設計洞察)
  - [多路徑的價值](#多路徑的價值)
  - [系統特徵](#系統特徵)
  - [應用建議](#應用建議)

---


> **生成時間**: 2026-01-01  
> **分析對象**: 起點終點相同、中間路徑不同的 flows

## 📊 整體統計

| 指標 | 數值 |
|------|------|
| 總 flows 數 | 840 |
| 唯一的 (起點, 終點) 組合數 | 158 |
| 有多條路徑的組合數 | 103 |
| 多路徑 flows 總數 | 785 (93.5%) |
| 平均每組路徑數 | 7.6 |

## 📈 路徑數量分佈

| 路徑數量 | 組數 |
|---------|------|
| 44 條路徑 | 1 |
| 33 條路徑 | 1 |
| 30 條路徑 | 1 |
| 20 條路徑 | 1 |
| 19 條路徑 | 1 |
| 18 條路徑 | 1 |
| 17 條路徑 | 2 |
| 14 條路徑 | 2 |
| 13 條路徑 | 4 |
| 12 條路徑 | 3 |
| 11 條路徑 | 5 |
| 10 條路徑 | 6 |
| 9 條路徑 | 2 |
| 8 條路徑 | 3 |
| 7 條路徑 | 9 |
| 6 條路徑 | 20 |
| 5 條路徑 | 9 |
| 4 條路徑 | 6 |
| 3 條路徑 | 10 |
| 2 條路徑 | 16 |

## 🏆 路徑最多的組合 (Top 10)


### 1. session_state_manager → run_analysis

- **路徑數量**: 44 條
- **路徑長度範圍**: 3 - 5 步
- **Flow IDs**: flow142, flow152, flow153, flow186, flow187, flow222, flow226, flow245, flow336, flow340...


### 2. session_state_manager → core_analyzer

- **路徑數量**: 33 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow141, flow148, flow150, flow184, flow220, flow225, flow244, flow284, flow334, flow339...


### 3. session_state_manager → practical_analyzer

- **路徑數量**: 30 條
- **路徑長度範圍**: 3 - 5 步
- **Flow IDs**: flow149, flow151, flow185, flow221, flow224, flow269, flow335, flow338, flow373, flow411...


### 4. session_state_manager → assistant

- **路徑數量**: 20 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow172, flow196, flow199, flow230, flow255, flow272, flow301, flow344, flow397, flow414...


### 5. session_state_manager → system_connectivity_checker

- **路徑數量**: 19 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow133, flow155, flow208, flow237, flow264, flow320, flow322, flow352, flow382, flow406...


### 6. session_state_manager → app

- **路徑數量**: 18 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow291, flow303, flow309, flow447, flow515, flow527, flow536, flow568, flow582, flow586...


### 7. session_state_manager → bizlogic_attack_executor

- **路徑數量**: 17 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow171, flow197, flow205, flow253, flow261, flow300, flow365, flow395, flow403, flow491...


### 8. session_state_manager → ai_commander

- **路徑數量**: 17 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow191, flow248, flow305, flow390, flow418, flow431, flow435, flow444, flow466, flow486...


### 9. session_state_manager → model_trainer

- **路徑數量**: 14 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow169, flow228, flow296, flow315, flow342, flow530, flow566, flow574, flow656, flow685...


### 10. session_state_manager → aiva_cli_implementation

- **路徑數量**: 14 條
- **路徑長度範圍**: 2 - 5 步
- **Flow IDs**: flow182, flow215, flow243, flow279, flow329, flow370, flow378, flow421, flow481, flow544...


## 📏 路徑長度差異最大的組合 (Top 5)


### 1. scalable_bio_trainer → system_connectivity_checker

- **長度差異**: 3 步
- **最短路徑**: 2 步 (Flow 6)
  ```
  scalable_bio_trainer → system_connectivity_checker
  ```
- **最長路徑**: 5 步 (Flow 111)
  ```
  scalable_bio_trainer → neural_network → rl_trainers → ai_model_manager → system_connectivity_checker
  ```


### 2. scalable_bio_trainer → command_repository

- **長度差異**: 3 步
- **最短路徑**: 2 步 (Flow 10)
  ```
  scalable_bio_trainer → command_repository
  ```
- **最長路徑**: 5 步 (Flow 118)
  ```
  scalable_bio_trainer → neural_network → rl_trainers → vector_store → command_repository
  ```


### 3. scalable_bio_trainer → model_trainer

- **長度差異**: 3 步
- **最短路徑**: 2 步 (Flow 11)
  ```
  scalable_bio_trainer → model_trainer
  ```
- **最長路徑**: 5 步 (Flow 104)
  ```
  scalable_bio_trainer → neural_network → rl_trainers → train_classifier → model_trainer
  ```


### 4. scalable_bio_trainer → ai_model_manager

- **長度差異**: 3 步
- **最短路徑**: 2 步 (Flow 78)
  ```
  scalable_bio_trainer → ai_model_manager
  ```
- **最長路徑**: 5 步 (Flow 107)
  ```
  scalable_bio_trainer → neural_network → rl_trainers → model_trainer → ai_model_manager
  ```


### 5. scalable_bio_trainer → real_bio_net_adapter

- **長度差異**: 3 步
- **最短路徑**: 2 步 (Flow 122)
  ```
  scalable_bio_trainer → real_bio_net_adapter
  ```
- **最長路徑**: 5 步 (Flow 119)
  ```
  scalable_bio_trainer → neural_network → rl_trainers → vector_store → real_bio_net_adapter
  ```


## 💡 設計洞察

### 多路徑的價值

1. **靈活性**: 相同目標可通過不同路徑達成，適應不同場景
2. **容錯性**: 某條路徑失敗時可嘗試其他路徑
3. **優化空間**: 可根據性能、成本等因素選擇最優路徑
4. **功能覆蓋**: 不同路徑可能提供不同的中間功能

### 系統特徵

- **93.5%** 的 flows 屬於多路徑組合
- 平均每組有 **7.6** 條路徑
- 最多的組有 **44** 條不同路徑

### 應用建議

1. **路徑選擇策略**: 開發智能路徑選擇算法
2. **性能優化**: 識別並優化常用路徑
3. **冗餘管理**: 評估是否存在過度冗餘
4. **文檔完善**: 說明不同路徑的適用場景

---

**生成時間**: 2026-01-01  
**維護狀態**: 🟢 活躍
