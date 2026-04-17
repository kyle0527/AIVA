# Attack 攻擊模組

> **路徑**: `services/core/aiva_core/core_capabilities/attack`  
> **狀態**: ✅ 正常 | **Python 文件數**: 3 | **最後更新**: 2026-04-05

## 概述

負責攻擊鏈編排和漏洞利用協調。Core 模組專注於**決策和編排**，不執行實際攻擊測試。

## 📄 檔案詳細資訊 (Files Details)

### `attack_chain.py`
**說明**: Attack Chain - 攻擊鏈編排器

**類別 (Classes)**:
- `ChainStatus` - 攻擊鏈狀態
- `AttackChain` - 攻擊鏈編排器

### `exploit_orchestrator.py`
**說明**: Exploit Orchestrator - 漏洞利用編排器

**類別 (Classes)**:
- `ExploitOrchestrator` - 漏洞利用編排器
**函式 (Functions)**:
- `register_exploit()` - 裝飾器：自動註冊 Exploit 類到全域註冊表

