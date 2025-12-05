# AIVA CLI 快速使用指南

## 📑 目錄

1. [快速啟動](#快速啟動)
2. [主選單功能](#主選單功能)
3. [常用操作](#常用操作)
4. [Rich CLI 整合](#rich-cli-整合-高級)
5. [實際使用範例](#實際使用範例)
6. [命令行參數完整列表](#命令行參數完整列表)
7. [故障排除](#故障排除)
8. [進階功能](#進階功能)
9. [核心優勢總結](#核心優勢總結)

---

## 快速啟動

```powershell
# 方法 1: 啟動統一 CLI 選單
python aiva_cli.py

# 方法 2: 直接查詢能力
python aiva_cli.py --query "如何進行滲透測試"

# 方法 3: 查看統計
python aiva_cli.py --stats
```

---

## 主選單功能

啟動 `python aiva_cli.py` 後，你會看到:

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

選擇功能 >
```

---

## 常用操作

### 1. 查詢能力

**場景**: 不知道 AIVA 有哪些攻擊工具

```powershell
# 命令行直接查詢
python aiva_cli.py --query "攻擊工具"

# 或在選單中選擇 [1]，然後輸入:
> 攻擊工具
```

**輸出**:
```
╭─────────────────────────────────────────────────╮
│                   查詢結果                      │
├───┬────────────────────┬──────────┬──────────┤
│ # │ 能力名稱           │ 模組     │ 語言     │
├───┼────────────────────┼──────────┼──────────┤
│ 1 │ enhance_attack_plan│ core     │ python   │
│ 2 │ find_attack_paths  │ integration│ python │
│ 3 │ run_attack_route   │ features │ python   │
╰───┴────────────────────┴──────────┴──────────╯
```

---

### 2. 獲取工作流推薦

**場景**: 想進行 web 應用滲透測試

```powershell
# 命令行方式
python aiva_cli.py --workflow "web 應用滲透測試"

# 或在選單中選擇 [3]
```

**輸出**:
```
╭─────────────────────────────────────────────────╮
│            工作流: web 應用滲透測試              │
├────┬────────────────────────┬──────────────────┤
│ 階段│ 能力                   │ 模組             │
├────┼────────────────────────┼──────────────────┤
│ 1  │ scan                   │ scan             │
│ 2  │ AttackSurfaceAssessor  │ scan             │
│ 3  │ scan_vulnerabilities   │ features         │
│ 4  │ find_attack_paths      │ integration      │
│ 5  │ run_attack_route       │ features         │
╰────┴────────────────────────┴──────────────────╯

建議執行順序: 
  1. 偵察 → 2. 掃描 → 3. 攻擊 → 4. 報告
```

---

### 3. 查看系統統計

```powershell
# 命令行方式
python aiva_cli.py --stats

# 或在選單中選擇 [2]
```

**輸出**:
```
╭─────────────────────────────────────╮
│          模組分布 (Top 10)          │
├──────────────────┬───────┬─────────┤
│ 模組             │ 數量  │ 佔比    │
├──────────────────┼───────┼─────────┤
│ scan             │ 286   │ 36.6%   │
│ core/aiva_core   │ 207   │ 26.5%   │
│ integration      │ 111   │ 14.2%   │
│ features         │ 98    │ 12.5%   │
╰──────────────────┴───────┴─────────╯

╭─────────────────────────────────────╮
│            語言分布                 │
├──────────────┬───────┬─────────────┤
│ 語言         │ 數量  │ 佔比        │
├──────────────┼───────┼─────────────┤
│ python       │ 495   │ 63.3%       │
│ rust         │ 123   │ 15.7%       │
│ typescript   │ 84    │ 10.7%       │
│ go           │ 80    │ 10.2%       │
╰──────────────┴───────┴─────────────╯

系統摘要:
  總計: 782 個能力
  模組數: 16
  語言數: 4
```

---

## Rich CLI 整合 (高級)

如果你想使用完整的 Rich CLI 界面:

```powershell
# 方法 1: 直接運行 Rich CLI
python -m services.core.aiva_core.ui_panel.rich_cli

# 方法 2: 使用 AIVA CLI 的選單模式
python aiva_cli.py
# 然後選擇 [4] AI 能力查詢
```

**Rich CLI 選單**:
```
╔════════════════════════════════════════════════════════╗
║               AIVA Rich CLI - 主選單                   ║
╠════════════════════════════════════════════════════════╣
║ [1] 漏洞掃描      - 啟動 AI 驅動的安全評估            ║
║ [2] 能力管理      - 管理註冊的安全工具和能力          ║
║ [3] AI 對話       - 與 AIVA AI 引擎互動              ║
║ [4] AI 能力查詢   - 查詢 AIVA 的功能與能力  ⭐ 新功能 ║
║ [5] 工具集成      - 整合新的安全工具                  ║
║ [6] 系統監控      - 查看系統狀態和日誌                ║
║ [7] 設定配置      - 調整 AIVA 系統設定                ║
║ [8] 報告生成      - 生成掃描和評估報告                ║
║ [9] 幫助文檔      - 查看使用指南和 API 文檔           ║
║ [A] 關於 AIVA     - 版本資訊和開發團隊                ║
║ [0] 退出          - 安全退出 AIVA CLI                 ║
╚════════════════════════════════════════════════════════╝
```

選擇 `[4] AI 能力查詢` 後:

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

選擇功能 >
```

---

## 實際使用範例

### 範例 1: 新手探索

```powershell
PS> python aiva_cli.py

# 選擇 [1] 快速查詢能力
> 我想學習漏洞掃描

[結果顯示]
  1. scan_vulnerabilities - 漏洞掃描引擎
  2. scan - DOM 掃描與分析
  3. SecretDetector - 敏感資訊偵測

# 繼續查詢
> 如何使用 scan_vulnerabilities

[顯示文檔]
  函數: scan_vulnerabilities(target: str, depth: int = 2)
  參數: 
    - target: 目標 URL
    - depth: 掃描深度
  返回: List[Vulnerability]
```

---

### 範例 2: 滲透測試人員

```powershell
PS> python aiva_cli.py --workflow "web 應用滲透測試"

[AI 推薦工作流]
階段 1: 偵察
  - scan (TypeScript)
  - AttackSurfaceAssessor (Rust)

階段 2: 掃描
  - scan_vulnerabilities (Python)
  - SecretDetector (Rust)

階段 3: 攻擊
  - find_attack_paths (Python)
  - run_attack_route (Python)

階段 4: 後滲透
  - analyze_and_recommend (Python)

階段 5: 報告
  - generate_capability_records (Python)
  - fix_vulnerability (Python)

[用戶下一步]
# 可以根據推薦手動執行，或等待自動化功能完成
```

---

### 範例 3: 開發者整合

```python
# 在你的 Python 代碼中使用
from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query

# 快速查詢
results = await quick_query("攻擊路徑分析", top_k=5)

for result in results:
    meta = result["metadata"]
    print(f"能力: {meta['capability_name']}")
    print(f"模組: {meta['module']}")
    print(f"語言: {meta['language']}")
```

---

## 命令行參數完整列表

```powershell
python aiva_cli.py [OPTIONS]

OPTIONS:
  -q, --query TEXT          直接查詢能力
  -w, --workflow TEXT       獲取工作流推薦
  -s, --stats               顯示統計資訊
  --sync                    同步能力到 RAG
  -t, --test                運行測試
  -k, --top-k INT           查詢結果數量 (default: 5)
  --force-refresh           強制刷新 RAG 資料
  -h, --help                顯示幫助

EXAMPLES:
  python aiva_cli.py
  python aiva_cli.py -q "掃描工具"
  python aiva_cli.py -w "SQL 注入攻擊"
  python aiva_cli.py -s
  python aiva_cli.py --sync --force-refresh
  python aiva_cli.py -q "Rust 掃描器" -k 10
```

---

## 故障排除

### 問題 1: ChromaDB 未找到

```
[Error] Vector database not found!
```

**解決方案**:
```powershell
# 同步能力資料
python aiva_cli.py --sync
```

---

### 問題 2: 模組導入失敗

```
[Warning] Core modules not fully available
```

**解決方案**:
```powershell
# 確認在正確目錄
cd C:\D\fold7\AIVA-git

# 安裝依賴
pip install -r requirements.txt
```

---

### 問題 3: Rich UI 不可用

```
[Warning] Rich UI not available, using plain text
```

**解決方案**:
```powershell
# 安裝 Rich
pip install rich
```

---

## 進階功能

### 1. API 整合

```python
from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery

# 創建查詢系統
query_system = AICapabilityQuery()

# 按模組查詢
scan_capabilities = await query_system.query_by_module("scan", top_k=10)

# 按語言查詢
python_tools = await query_system.query_by_language("python", top_k=10)

# 獲取統計
stats = await query_system.show_statistics()
print(f"Total: {stats['total']} capabilities")
```

---

### 2. 自動化腳本

```python
import asyncio
from services.core.aiva_core.cognitive_core.ai_capability_query import quick_query, quick_stats

async def auto_recon():
    """自動偵察流程"""
    # 獲取偵察工具
    recon_tools = await quick_query("reconnaissance and scanning", top_k=5)
    
    for tool in recon_tools:
        print(f"執行: {tool['metadata']['capability_name']}")
        # 實際執行邏輯
    
    # 顯示統計
    await quick_stats()

# 運行
asyncio.run(auto_recon())
```

---

## 核心優勢總結

### 修改前 vs 修改後

| 項目 | 修改前 | 修改後 |
|-----|--------|--------|
| **啟動方式** | 多個測試腳本分別執行 | 統一入口 `python aiva_cli.py` |
| **查詢能力** | 手動查看文檔或代碼 | AI 自然語言查詢 |
| **工作流規劃** | 手動組合工具 | AI 自動推薦工作流 |
| **統計分析** | 需要編寫查詢代碼 | 一鍵顯示統計 |
| **學習曲線** | 需熟悉所有 782 個能力 | 自然語言交互 |
| **操作步驟** | 15+ 步驟 | 3-5 步驟 |
| **整合難度** | 分散在多個文件 | 統一 API 接口 |

---

## 下一步

1. **測試功能**: `python aiva_cli.py --test`
2. **同步資料**: `python aiva_cli.py --sync`
3. **開始使用**: `python aiva_cli.py`

**提示**: 首次使用建議先執行 `--sync` 確保 RAG 資料最新！

---

**更新日期**: 2025-11-28  
**版本**: v1.0  
**文檔路徑**: `AIVA_CLI_QUICK_START.md`
