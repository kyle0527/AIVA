# AIVA Core v1（真正主核心，純本機、非 LLM）

這是一個可直接放進 AIVA 倉庫的「主核心」落地版本：
- 單一入口：`AivaCore`（Planner / Registry / Executor / Guard / State / Events）
- 內含 5 個內建能力（capabilities）：`echo`, `index_repo`, `parse_ast`, `build_graph`, `render_report`
- 內建 3 條流程（flows）：`scan_minimal`, `fix_minimal`（占位）, `rag_repair`（占位）
- 提供 CLI：`python -m cli_generated.aiva_cli ...`

> 設計原則：只用標準函式庫（可選 `PyYAML` 以支援 .yaml 流程；若未安裝，請把 flow 檔改成 .json）。

## 安裝與放置
1. 把整個資料夾內容放到 AIVA 倉庫根目錄（會合併到 `services/core/`、`cli_generated/`、`config/flows/`）。
2. （可選）安裝 PyYAML 以便解析 YAML 流程檔：`pip install pyyaml`。

## 快速驗收
```bash
# 列出能力
python -m cli_generated.aiva_cli list-caps

# 執行最小掃描（對 repo 根目錄）
python -m cli_generated.aiva_cli scan --target .

# 查看產物與事件
ls -la data/run
tail -n 50 logs/aiva_core/events.log
```

## 導入既有模組
- 若你的 `services/features/*` 裡有模組想掛成能力，只要在該模組提供：
  ```python
  def register_capabilities(registry): 
      registry.register("your_cap", your_callable, desc="...")
  ```
  核心會嘗試自動載入（`autoload.py` 會掃描 `services.features` 子模組）。

## 文件
- `services/core/aiva_core_v1/*`：主核心與能力
- `cli_generated/aiva_cli/*`：命令列介面
- `config/flows/*`：流程範本（可用 JSON 或 YAML）

> 本核心偏重 M1–M3（持續運作、靜態分析/探索、修補管控骨架），M4（RAG）與 M5（攻擊能力）保留掛點。
