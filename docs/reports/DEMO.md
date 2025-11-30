# AIVA AI 分析能力 - 實戰演示

**演示日期**: 2025-11-28  
**目的**: 展示 AI 分析能力在實際場景中的應用

---

## 快速啟動 (30 秒)

```powershell
# 進入項目目錄
cd C:\D\fold7\AIVA-git

# 啟動統一 CLI
python aiva_cli.py
```

你會看到:
```
╔══════════════════════════════════════════════════════════════╗
║                     AIVA CLI 主選單                          ║
╚══════════════════════════════════════════════════════════════╝

[1] 快速查詢能力      - 輸入問題查找相關功能
[2] 查看系統統計      - 顯示 782 個能力的分布
[3] 獲取工作流推薦    - AI 推薦任務執行步驟
[4] 同步能力資料      - 更新 RAG 知識庫
[5] 運行測試驗證      - 驗證 AI 分析能力
[0] 退出
```

---

## 場景 1: 新手探索 (1 分鐘)

**情境**: 你剛接觸 AIVA,不知道它能做什麼

### 步驟 1: 詢問核心能力

```powershell
python aiva_cli.py --query "我能做什麼"
```

**AI 回應**:
```
╭────────────────────────────────────────────────────╮
│                  查詢: 我能做什麼                  │
├────┬───────────────────┬──────────────┬──────────┤
│ #  │ 能力名稱          │ 模組         │ 語言     │
├────┼───────────────────┼──────────────┼──────────┤
│ 1  │ capability_analyzer│ core/aiva_core│ python │
│ 2  │ list_modules      │ core/aiva_core│ python │
│ 3  │ list_features     │ features     │ python │
│ 4  │ list_detectors    │ features     │ python │
│ 5  │ get_capabilities  │ core/aiva_core│ python │
╰────┴───────────────────┴──────────────┴──────────╯
```

### 步驟 2: 查看完整統計

```powershell
python aiva_cli.py --stats
```

**結果**: 瞭解 AIVA 有 782 個能力,分布在 16 個模組

**時間**: 不到 1 分鐘完成探索 ✅

---

## 場景 2: 滲透測試規劃 (2 分鐘)

**情境**: 你要對 web 應用進行滲透測試,需要規劃步驟

### 步驟 1: 獲取工作流推薦

```powershell
python aiva_cli.py --workflow "web 應用滲透測試"
```

**AI 推薦工作流**:
```
╭───────────────────────────────────────────────────╮
│            工作流: web 應用滲透測試                │
├────┬─────────────────────────┬──────────────────┤
│ 階段│ 能力                    │ 模組             │
├────┼─────────────────────────┼──────────────────┤
│ 1  │ scan                    │ scan             │
│ 2  │ AttackSurfaceAssessor   │ scan             │
│ 3  │ scan_vulnerabilities    │ features         │
│ 4  │ find_attack_paths       │ integration      │
│ 5  │ run_attack_route        │ features         │
│ 6  │ generate_capability_records│ features      │
╰────┴─────────────────────────┴──────────────────╯

[建議執行順序]
  階段 1: 偵察 (scan, AttackSurfaceAssessor)
  階段 2: 掃描 (scan_vulnerabilities)
  階段 3: 攻擊 (find_attack_paths, run_attack_route)
  階段 4: 報告 (generate_capability_records)
```

### 步驟 2: 查詢特定階段工具

```powershell
# 查詢掃描工具
python aiva_cli.py --query "vulnerability scanning tools"

# 查詢攻擊工具
python aiva_cli.py --query "attack execution capabilities"
```

**時間**: 2 分鐘完成完整規劃 ✅

---

## 場景 3: 漏洞修復 (1 分鐘)

**情境**: 發現 SQL 注入漏洞,需要修復方案

### 步驟 1: 查詢修復工具

```powershell
python aiva_cli.py --query "SQL injection vulnerability remediation"
```

**AI 推薦**:
```
╭────────────────────────────────────────────────────╮
│        查詢: SQL injection vulnerability           │
│                  remediation                       │
├────┬────────────────────────┬──────────┬─────────┤
│ #  │ 能力名稱               │ 模組     │ 語言    │
├────┼────────────────────────┼──────────┼─────────┤
│ 1  │ fix_vulnerability      │ integration│ python │
│ 2  │ generate_patch_for_vulnerability│ integration│ python │
│ 3  │ update_vulnerability_status│ integration│ python │
│ 4  │ assign_vulnerability   │ integration│ python │
╰────┴────────────────────────┴──────────┴─────────╯
```

### 步驟 2: 查詢完整修復流程

```powershell
python aiva_cli.py --workflow "SQL injection remediation"
```

**時間**: 1 分鐘獲得修復方案 ✅

---

## 場景 4: 開發者整合 (3 分鐘)

**情境**: 你在寫自動化腳本,需要調用 AIVA 能力

### 步驟 1: 使用簡化 API

```python
# demo_integration.py
import asyncio
from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query

async def auto_scan():
    """自動化掃描腳本"""
    # 查詢掃描工具
    scan_tools = await quick_query("vulnerability scanning", top_k=5)
    
    print("可用掃描工具:")
    for tool in scan_tools:
        meta = tool["metadata"]
        print(f"  - {meta['capability_name']} ({meta['module']})")
    
    # 實際執行邏輯
    # await execute_scan(scan_tools[0])

# 運行
asyncio.run(auto_scan())
```

### 步驟 2: 執行

```powershell
python demo_integration.py
```

**輸出**:
```
可用掃描工具:
  - scan_vulnerabilities (features)
  - scan (scan)
  - SecretDetector (scan)
  - AttackSurfaceAssessor (scan)
  - analyzeDOMManipulation (scan)
```

**時間**: 3 分鐘完成整合 ✅

---

## 場景 5: Rich CLI 交互模式 (5 分鐘)

**情境**: 想使用完整的圖形界面進行深度探索

### 步驟 1: 啟動 Rich CLI

```powershell
python -m services.core.aiva_core.ui_panel.rich_cli
```

### 步驟 2: 選擇 [4] AI 能力查詢

你會看到子選單:
```
╔════════════════════════════════════════════════════════╗
║            AIVA AI 能力查詢系統                        ║
╚════════════════════════════════════════════════════════╝

[1] 我能做什麼？      - 查詢 AIVA 的核心能力
[2] 滲透測試工作流    - 獲取滲透測試建議
[3] 漏洞修復指南      - 查詢漏洞處理流程
[4] 攻擊路徑分析      - 分析可能的攻擊路徑
[5] 自定義查詢        - 輸入自然語言查詢
[6] 能力統計          - 查看系統能力統計
[0] 返回主選單        - 回到主選單
```

### 步驟 3: 選擇 [2] 滲透測試工作流

AI 會自動分析並展示:
```
╭──────────────────────────────────────────╮
│        工作流推薦: 滲透測試              │
├──────────────────────────────────────────┤
│ 任務: penetration testing workflow       │
│ 找到: 8 個相關能力                       │
╰──────────────────────────────────────────╯

╭────┬────────────────────────┬───────────╮
│階段│ 能力                   │ 模組      │
├────┼────────────────────────┼───────────┤
│ 1  │ scan_vulnerabilities   │ features  │
│ 2  │ AttackSurfaceAssessor  │ scan      │
│ 3  │ find_attack_paths      │ integration│
│ 4  │ enhance_attack_plan    │ core      │
│ 5  │ run_attack_route       │ features  │
╰────┴────────────────────────┴───────────╯
```

**時間**: 5 分鐘完成深度探索 ✅

---

## 效率對比

### 傳統方式 vs AI 輔助

| 任務 | 傳統方式 | AI 輔助 | 效率提升 |
|-----|---------|---------|---------|
| **探索能力** | 15 分鐘 (閱讀文檔) | 1 分鐘 (查詢) | ⬆️ 93% |
| **規劃測試** | 30 分鐘 (手動組合) | 2 分鐘 (AI 推薦) | ⬆️ 93% |
| **查找工具** | 10 分鐘 (搜索代碼) | 30 秒 (查詢) | ⬆️ 95% |
| **整合開發** | 1 小時 (學習 API) | 3 分鐘 (簡化接口) | ⬆️ 95% |

---

## 核心優勢總結

### 1. 自然語言交互 🗣️

**Before**: 需要熟悉 782 個能力的名稱和用途  
**After**: 用自然語言提問,AI 自動匹配

```powershell
# 不需要記住函數名
python aiva_cli.py --query "如何修復漏洞"

# AI 會找到:
# - fix_vulnerability
# - generate_patch_for_vulnerability
# - update_vulnerability_status
```

---

### 2. 智能工作流推薦 🎯

**Before**: 手動組合工具,可能遺漏步驟  
**After**: AI 推薦完整工作流,確保覆蓋所有階段

```powershell
python aiva_cli.py --workflow "滲透測試"

# AI 自動規劃:
# 階段 1: 偵察 → 階段 2: 掃描 → 階段 3: 攻擊 → 階段 4: 報告
```

---

### 3. 統一入口點 🚪

**Before**: 多個測試腳本,不知道從哪裡開始  
**After**: 單一命令 `python aiva_cli.py`

```powershell
# 一個入口,所有功能
python aiva_cli.py              # 交互式選單
python aiva_cli.py --query "..." # 直接查詢
python aiva_cli.py --stats       # 查看統計
python aiva_cli.py --workflow "..."# 獲取工作流
```

---

### 4. 簡化 API 📚

**Before**: 需要理解 RAG、ChromaDB、向量嵌入等複雜概念  
**After**: 簡單的函數調用

```python
# Before (複雜)
from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

vector_store = VectorStore(backend="chroma", persist_directory=persist_dir)
kb = KnowledgeBase(vector_store=vector_store)
connector = InternalLoopConnector(rag_knowledge_base=kb)
results = await connector.query_self_awareness(query, top_k=5)

# After (簡單)
from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query

results = await quick_query("攻擊工具", top_k=5)
```

---

### 5. 完整文檔支持 📖

| 文檔 | 用途 | 讀者 |
|-----|------|------|
| `AIVA_CLI_QUICK_START.md` | 快速上手指南 | 新手 |
| `AI_ANALYSIS_PRACTICAL_USAGE.md` | 實際應用場景 | 滲透測試人員 |
| `AIVA_AI_ANALYSIS_CAPABILITY_ASSESSMENT.md` | 能力評估報告 | 管理者 |
| `AI_ANALYSIS_INTEGRATION_COMPLETE.md` | 整合完成報告 | 開發者 |
| `DEMO.md` (本文檔) | 實戰演示 | 所有用戶 |

---

## 下一步行動

### 對於新手 👶

1. **快速體驗**: `python aiva_cli.py --stats`
2. **嘗試查詢**: `python aiva_cli.py --query "掃描工具"`
3. **閱讀文檔**: `AIVA_CLI_QUICK_START.md`

---

### 對於滲透測試人員 🔒

1. **獲取工作流**: `python aiva_cli.py --workflow "web 滲透測試"`
2. **查詢特定工具**: `python aiva_cli.py --query "SQL injection detection"`
3. **閱讀場景**: `AI_ANALYSIS_PRACTICAL_USAGE.md`

---

### 對於開發者 💻

1. **學習 API**: 查看 `ai_capability_query.py` 的 docstrings
2. **嘗試整合**: 創建 `demo_integration.py` 測試
3. **閱讀架構**: `AI_ANALYSIS_INTEGRATION_COMPLETE.md`

---

### 對於管理者 📊

1. **查看統計**: `python aiva_cli.py --stats`
2. **評估能力**: 閱讀 `AIVA_AI_ANALYSIS_CAPABILITY_ASSESSMENT.md`
3. **規劃升級**: 查看改進建議 (LLM 整合、自動化執行)

---

## 常見問題 FAQ

### Q1: 如何更新能力資料?

```powershell
python aiva_cli.py --sync --force-refresh
```

---

### Q2: 查詢結果不準確怎麼辦?

**原因**: 向量嵌入模型限制 (all-MiniLM-L6-v2)

**解決方案**: 
- 短期: 使用更精確的關鍵字
- 長期: 升級到 all-mpnet-base-v2 或整合 LLM

---

### Q3: 如何在代碼中使用?

```python
from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query

results = await quick_query("你的問題", top_k=5)
```

---

### Q4: Rich CLI 無法啟動?

```powershell
# 安裝 Rich
pip install rich

# 重試
python -m services.core.aiva_core.ui_panel.rich_cli
```

---

## 技術支持

- **文檔**: 查看 `docs/` 目錄
- **日誌**: 查看 `logs/` 目錄
- **測試**: `python aiva_cli.py --test`

---

## 總結

AIVA AI 分析能力整合完成後,實現了:

✅ **統一入口** - 一個命令訪問所有功能  
✅ **自然交互** - 用問題代替記憶函數名  
✅ **智能推薦** - AI 自動規劃工作流  
✅ **簡化開發** - 友好的 API 接口  
✅ **完整文檔** - 從新手到專家的完整指南

**立即體驗**:
```powershell
cd C:\D\fold7\AIVA-git
python aiva_cli.py
```

---

**演示文檔版本**: v1.0  
**最後更新**: 2025-11-28 12:40:00  
**適用版本**: AIVA v2.0+
