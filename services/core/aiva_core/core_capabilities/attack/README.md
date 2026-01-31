# Attack 攻擊模組

> **路徑**: `services/core/aiva_core/core_capabilities/attack`  
> **狀態**: ✅ 正常 | **文件數**: 4 | **最後更新**: 2026-01-07

## 概述

負責攻擊鏈編排和漏洞利用協調。Core 模組專注於**決策和編排**，不執行實際攻擊測試。

## 核心組件

### attack_chain.py
- `ChainStatus` - 攻擊鏈狀態枚舉（PENDING, RUNNING, COMPLETED, FAILED, PAUSED）
- `AttackChain` - 攻擊鏈編排器
  - 依賴關係管理
  - 執行順序編排
  - 條件分支處理
  - 結果傳遞

### exploit_orchestrator.py
- `register_exploit` - 裝飾器：自動註冊 Exploit 類到全域註冊表
- `ExploitOrchestrator` - 漏洞利用編排器
  - 管理 Exploit 註冊表和元數據
  - 選擇合適的 Exploit 策略
  - 編排 Features 模組執行測試
  - 從 Integration 收集和分析結果

### custom_exploits_example.py ⚠️ 範例代碼
- `TimeBasedSQLiExploit` - 時間盲注 SQL 注入利用示例
- `MongoDBInjectionExploit` - MongoDB 注入利用示例
- `Jinja2SSTIExploit` - Jinja2 SSTI 模板注入示例
- `CustomAuthBypassExploit` - 認證繞過利用示例

**注意**: 此檔案為範例代碼，建議移動至 `examples/` 目錄或在生產環境部署時排除。

## 架構流程

```
Core (決策) → MQ → Features (執行) → MQ → Integration (收集) → MQ → Core (分析)
```

## 依賴關係

- `aiva_common.enums.security` - ExploitType 枚舉
- `yaml` - 漏洞定義配置加載
- 支援熱更新和動態註冊
