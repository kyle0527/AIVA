# Phase 4: Git 狀態清理 + 提交策略

> 優先級: P3  
> 目標: 將 80+ 個未提交變更整理為有意義的 commit  
> 前置條件: Phase 1 修復完成後再提交  
> 注意: 此計畫與其他 Phase 可並行進行

---

## 當前 Git 狀態

```
分支: main (同步 origin/main @ 987d677b)
未提交變更: 80+ 個檔案

分類:
├── 修改 (M): ~45 個檔案
├── 刪除 (D): ~18 個檔案
└── 新增 (?): ~20 個檔案
```

---

## 變更分類

### 類別 A: 空白字符清理（已確認安全）

這些是上一輪刻意做的 trailing whitespace 清理，不影響功能。

```
M services/aiva_common/schemas/api_standards.py
M services/aiva_common/schemas/commands.py
M services/aiva_common/schemas/interfaces/api_standards.py
M services/core/aiva_core/cognitive_core/__init__.py
M services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py
M services/core/aiva_core/core_capabilities/orchestration/two_phase_scan_orchestrator.py
M services/core/aiva_core/internal_exploration/aiva_internal_executor.py
M services/core/aiva_core/service_backbone/api/app.py
M services/core/aiva_core/service_backbone/coordination/ai_controller.py
M services/core/aiva_core/service_backbone/coordination/ai_manager.py
M services/core/aiva_core/task_planning/commander/attack_coordinator.py
M services/core/aiva_core/task_planning/commander/types.py
M services/core/aiva_core/task_planning/planner/tool_selector.py
M services/core/aiva_core/task_planning/unified_executor.py
M services/core/ui/__init__.py
M services/features/function_sqli/config.py
M services/features/function_sqli/engines/boolean_detection_engine.py
M services/features/function_sqli/engines/error_detection_engine.py
M services/features/function_sqli/engines/oob_detection_engine.py
M services/features/function_sqli/engines/time_detection_engine.py
M services/features/function_sqli/engines/union_detection_engine.py
M services/features/function_sqli/payload_wrapper_encoder.py
M services/features/function_sqli/smart_detection_manager.py
M services/features/function_xss/__main__.py
M services/features/function_xss/command_handler.py
M services/features/function_xss/dom_xss_detector.py
M services/features/feature_step_executor.py
M services/features/features_ready/function_crypto/command_handler.py
M services/integration/capability/*.py (多個)
M guides/user_manuals/*.md (多個)
M pyproject.toml
M requirements.txt
```

**建議 commit**:
```bash
git add <上述所有檔案>
git commit -m "chore: remove trailing whitespace across 31 files

Automated cleanup of trailing whitespace and EOF blank lines.
No functional changes."
```

---

### 類別 B: 有意的刪除（舊檔案歸檔）

需要確認這些刪除是有意的重構：

```
D guides/user_manuals/old/*.md              ← 11 個舊手冊（已有新版？）
D services/features/base/__init__.py         ← 舊的 feature base（已歸檔）
D services/features/base/feature_registry.py ← 舊的 registry（已替代）
D services/features/base/integration_helper.py
D services/features/base/result_schema.py
D services/features/function_authn_go/authn_manager.py   ← 需確認
D services/features/function_postex/postex_manager.py    ← 需確認
D services/features/function_web_scanner/scanner_manager.py ← 需確認
D services/integration/capability/minimal_manifest.py    ← 需確認
D services/integration/capability/payload_generator.py   ← 需確認
```

#### 需特別注意的刪除

| 刪除檔案 | 仍有引用 | 風險 |
|----------|----------|------|
| `function_sqli/detector/sqli_detector.py` | 3 個檔案 | **Phase 1 修復項** |
| `integration/alembic/env.py` | alembic.ini 引用 | **Phase 2 修復項** |
| `integration/alembic/versions/001_initial_schema.py` | 無 | 低 |
| `features/base/__init__.py` | 每個 __init__.py 都叫這名字 | 無（名稱巧合） |
| `features/base/feature_registry.py` | 1 個（僅註解引用） | 無 |

**建議 commit**:
```bash
# 確認後
git add <有意刪除的檔案>
git commit -m "refactor: archive legacy feature base and old user manuals

- Remove old user manual versions (replaced by new 6-volume series)
- Remove legacy feature_base framework (replaced by feature_step_executor)
- Remove authn_manager (replaced by Go wrapper)
- Remove scanner_manager (replaced by integration_tools)"
```

---

### 類別 C: 新增檔案

```
?? _archive/06_documentation_archive/user_manuals_old_20260211/
?? _archive/09_integration_archive/
?? _archive/base_feature_infrastructure/
?? _out/AI_CORE_CLI_OPTIMIZATION_PROPOSAL.md
?? _out/CAPABILITY_COMPARISON_ANALYSIS.md
?? _out/CLI_ARCHITECTURE_IMPLEMENTATION_COMPLETE.md
?? _out/IMPLEMENTATION_RISK_ASSESSMENT.md
?? config/dashboard_config.yaml
?? docs/CODE_QUALITY_FIX_REPORT.md
?? docs/UI_DASHBOARD_IMPLEMENTATION_PLAN.md
?? docs/UI_DASHBOARD_USER_GUIDE.md
?? guides/user_manuals/檢查報告_20260211.md
?? requirements-dashboard.txt
?? scripts/start_dashboard.py
?? services/core/aiva_core/cognitive_core/decision/bounty_strategy_agent.py
?? services/core/aiva_core/cognitive_core/decision/knowledge_decision_mixin.py
?? services/core/aiva_core/service_backbone/REPAIR_REPORT.md
?? services/core/aiva_core/service_backbone/api/sse.py
?? services/core/aiva_core/service_backbone/coordination/REPAIR_*.md
?? services/dashboard/
?? services/features/function_sqli/README.md
?? services/features/function_sqli/data/
?? services/features/function_sqli/engines/base_detector.py
?? services/features/function_sqli/verify_refactor_v2.py
?? services/features/function_xss/payloads.json
?? services/features/function_xss/scanner.py
?? tests/test_cli_architecture.py
```

**分批提交建議**:

```bash
# Commit 1: 歸檔目錄
git add _archive/
git commit -m "archive: move legacy docs and feature infrastructure"

# Commit 2: SQLi 重構
git add services/features/function_sqli/
git commit -m "refactor(sqli): engine-based detection architecture

- Add base_detector.py for engine abstraction
- Add detection data and README
- Replace monolithic detector with 5 specialized engines"

# Commit 3: 新增功能
git add services/core/aiva_core/cognitive_core/decision/bounty_strategy_agent.py
git add services/core/aiva_core/cognitive_core/decision/knowledge_decision_mixin.py
git add services/core/aiva_core/service_backbone/api/sse.py
git add services/dashboard/
git commit -m "feat: add bounty strategy agent, SSE support, dashboard service"

# Commit 4: 文件
git add docs/ guides/ _out/
git commit -m "docs: add repair reports, dashboard docs, quality reports"

# Commit 5: 配置
git add config/dashboard_config.yaml requirements-dashboard.txt scripts/start_dashboard.py
git commit -m "config: add dashboard configuration and startup script"

# Commit 6: 測試
git add tests/test_cli_architecture.py
git commit -m "test: add CLI architecture tests"
```

---

### 類別 D: 功能變更（需 review）

以下修改檔案可能包含功能性改動（不只是空白清理）：

```
M .env.example
M AIVA.code-workspace
M services/README.md
M services/features/README.md
M services/features/__init__.py
```

**建議**: 先 `git diff <file>` 檢查實際改動內容，再決定歸入哪個 commit。

---

## 建議的提交順序

```
提交 1: chore: remove trailing whitespace (純清理)
提交 2: archive: move legacy files (歸檔)
提交 3: refactor: replace sqli detector with engine architecture
提交 4: feat: add new cognitive components (bounty_strategy, SSE)
提交 5: feat: add dashboard service
提交 6: docs: add repair reports and documentation
提交 7: config: add dashboard and environment configs
提交 8: test: add CLI architecture tests
提交 9: fix: Phase 1 critical fixes (修復完成後)
```

---

## 注意事項

1. **先修復再提交**: Phase 1 的修復應該和當前變更一起整理，而不是分開
2. **不要 `git add .`**: 逐類分批提交才能保持 git history 清晰
3. **確認刪除**: 標記為 `D` 的檔案，逐一確認是否真的不需要了
4. **_out/ 目錄**: 考慮加入 `.gitignore`（分析輸出不該追蹤）
5. **repair_plan/ 目錄**: 此修復計畫本身也在 `_out/` 下，提交時可考慮是否需要追蹤

---

## .gitignore 建議新增

```gitignore
# 分析/報告輸出
_out/

# 清理腳本
clean_whitespace.py

# 本地資料庫
data/database/*.db

# 日誌
logs/*.log

# Python 快取
__pycache__/
*.pyc
```
