# Phase 3: 功能模組補齊 + 品質提升

> 優先級: P2  
> 目標: 多漏洞類型支援 + 測試覆蓋 + 清除技術債  
> 前置條件: Phase 1、Phase 2 完成  
> 驗證方式: 各模組獨立測試 + 整合掃描靶場

---

## 3.1 功能模組現狀與補齊計畫

### 模組完整度一覽

| 模組 | `__main__.py` | `command_handler` | import 成功 | CLI 可執行 | 狀態 |
|------|---------------|-------------------|-------------|-----------|------|
| **function_xss** | ✅ | ✅ | ✅ | ✅ | 完整 |
| **function_sqli** | ❌ | ⚠️ (有但 import 失敗) | ⚠️ | ❌ | Phase 1 修復後半完成 |
| **function_ssrf** | ❌ | ❌ | ✅ | ❌ | 骨架 |
| **function_idor** | ❌ | ❌ | ✅ | ❌ | 骨架 |
| **function_bizlogic** | ✅ | ❌ | ✅ | ❌ | 半完成 |
| **function_web_scanner** | ❌ | ❌ | ⚠️ | ❌ | 骨架 |
| **function_authn_go** | ❌ | ✅ (Go wrapper) | ✅ | ⚠️ (需 Go binary) | Go 模組 |
| **function_postex** | ❌ | ❌ | ⚠️ | ❌ | 骨架 |
| **function_exploit** | ❌ | ❌ | ⚠️ | ❌ | 骨架 |
| **function_crypto** | ❌ | ❌ | ❌ | ❌ | Rust only |
| function_csrf | — | — | — | — | **目錄不存在** |
| function_cors | — | — | — | — | **目錄不存在** |

### 優先順序建議

**第一批（核心漏洞類型，必須補齊）:**
1. `function_sqli` — 補 `__main__.py`，確認 SmartDetectionManager 能跑
2. `function_ssrf` — 補 `__main__.py` + command_handler
3. `function_idor` — 補 `__main__.py` + command_handler

**第二批（加值功能）:**
4. `function_bizlogic` — 補 command_handler
5. `function_web_scanner` — 修復 import + 補入口點

**第三批（進階/可延後）:**
6. `function_csrf` — 從零建立
7. `function_cors` — 從零建立
8. `function_crypto` — 需 Rust 工具鏈
9. `function_authn_go` — 需 Go 工具鏈
10. `function_postex` — 後滲透功能
11. `function_exploit` — 漏洞利用

---

### 3.1.1 SQLi 模組補齊

**現狀**: 6 個 detection engine 已實作、SmartDetectionManager 存在、但缺少 `__main__.py`

**所需檔案**: `services/features/function_sqli/__main__.py`

```python
"""AIVA SQL Injection Detection Module — CLI 入口點

使用方式:
    python -m services.features.function_sqli --url <target> [--method GET|POST] [--params key=value]
"""
import asyncio
import argparse
import json
import sys

from .smart_detection_manager import SmartDetectionManager
from .config import SqliConfig


async def main():
    parser = argparse.ArgumentParser(description="AIVA SQLi Detector")
    parser.add_argument("--url", required=True, help="Target URL")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--params", nargs="*", help="Parameters (key=value)")
    parser.add_argument("--engines", nargs="*", help="Specific engines to use")
    parser.add_argument("--output", default="-", help="Output file (- for stdout)")
    args = parser.parse_args()

    # 解析參數
    params = {}
    if args.params:
        for p in args.params:
            k, v = p.split("=", 1)
            params[k] = v

    config = SqliConfig()
    manager = SmartDetectionManager(config=config)
    
    print(f"🔍 Scanning {args.url} for SQL injection...")
    results = await manager.run_detection(
        target_url=args.url,
        params=params,
        method=args.method,
    )
    
    output = json.dumps([r.__dict__ for r in results], indent=2, default=str)
    if args.output == "-":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)


if __name__ == "__main__":
    asyncio.run(main())
```

**同時需要**: 確認 `SmartDetectionManager.run_detection()` 方法存在。
若不存在，需檢查實際的公開方法名稱並適配。

---

### 3.1.2 SSRF 模組補齊

**所需檔案**: 
- `services/features/function_ssrf/__main__.py`
- `services/features/function_ssrf/command_handler.py`

**需調查**: 模組內部結構（detector 類名稱、公開 API）

---

### 3.1.3 IDOR 模組補齊

**所需檔案**: 
- `services/features/function_idor/__main__.py`  
- `services/features/function_idor/command_handler.py`

**需調查**: 模組內部結構

---

### 模組入口點模板

所有功能模組的 `__main__.py` 應遵循統一模板：

```python
"""AIVA {ModuleName} Module — CLI 入口點"""
import asyncio
import argparse
import json

from .{detector_module} import {DetectorClass}


async def main():
    parser = argparse.ArgumentParser(description="AIVA {ModuleName}")
    parser.add_argument("--url", required=True, help="Target URL")
    parser.add_argument("--output", default="-", help="Output file")
    # 模組特定參數...
    args = parser.parse_args()
    
    detector = {DetectorClass}()
    results = await detector.detect(args.url)
    
    output = json.dumps(results, indent=2, default=str)
    print(output) if args.output == "-" else open(args.output, "w").write(output)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3.2 NotImplementedError 清理

### 統計

| 檔案 | 數量 | 類型 | 是否影響核心 |
|------|------|------|-------------|
| `aiva_services_pb2_grpc.py` | 12 | gRPC 服務 stub | ❌ 不影響 |
| `messaging/mq.py` | 8 | 抽象基類方法 | ❌ 設計如此 |
| `function_web_scanner/.../web_tools.py` | 4 | 降級 fallback | ❌ 不影響 |
| `cross_language/core.py` | 3 | 抽象基類方法 | ❌ 設計如此 |
| `cli/__init__.py` | 1 | CLI 入口 | ⚠️ 需檢查 |
| `capability/models.py` | 1 | 能力模型 | ⚠️ 需檢查 |
| **合計** | **29** | | |

### 處理策略

1. **gRPC stubs (12 個)**: 保留不動，這是 protobuf 自動生成的服務端 stub，正常做法
2. **抽象基類 (11 個)**: 保留不動，這是設計模式，子類負責實作
3. **Fallback stubs (4 個)**: 保留不動，import 失敗時的安全降級
4. **需檢查 (2 個)**:
   - `cli/__init__.py` — 確認 CLI 入口是否有實際實作
   - `capability/models.py` — 確認是空殼還是有部分實作

**結論**: 29 個 NotImplementedError 中，**0 個需要緊急修復**。它們要麼是設計如此（抽象方法），要麼是選用功能（gRPC）。

---

## 3.3 測試補寫計畫

### 現狀

```
tests/
├── test_attack_coordinator_simple.py  (1,755 bytes)
├── test_cli_architecture.py           (8,529 bytes)
├── test_direct_import.py              (2,101 bytes)
├── verify_attack_coordinator.py       (2,180 bytes)
├── verify_dispatcher.py               (592 bytes)
└── verify_internal_loop.py            (590 bytes)
```

**6 個測試 vs 505 個源碼檔案 = 1.2% 覆蓋**

### 補寫優先順序

#### 第一層: 冒煙測試（確保不會 crash）

```
tests/
├── test_smoke_imports.py          ← 所有核心模組都能 import
├── test_smoke_app_startup.py      ← FastAPI app 能建立 + 路由正確
└── test_smoke_neural_core.py      ← 神經網路能初始化 + 推理
```

**test_smoke_imports.py 範例:**

```python
"""冒煙測試 — 確保所有核心模組能匯入"""
import pytest
import sys
sys.path.insert(0, '.')


class TestCoreImports:
    def test_app(self):
        from services.core.aiva_core.service_backbone.api.app import app
        assert app is not None
    
    def test_decision_agent(self):
        from services.core.aiva_core.cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
        agent = EnhancedDecisionAgent()
        assert agent is not None
    
    def test_neural_core(self):
        from services.core.aiva_core.cognitive_core.neural.real_neural_core import RealDecisionEngine
        engine = RealDecisionEngine(use_5m_model=True)
        assert engine is not None
    
    def test_commander(self):
        from services.core.aiva_core.task_planning.commander import CommanderCoordinator
        c = CommanderCoordinator()
        assert c is not None
    
    def test_attack_coordinator(self):
        from services.core.aiva_core.task_planning.commander.attack_coordinator import AttackCoordinator
        ac = AttackCoordinator()
        assert ac is not None
    
    def test_unified_executor(self):
        from services.core.aiva_core.task_planning.unified_executor import UnifiedAttackExecutor
        ue = UnifiedAttackExecutor()
        assert ue is not None


class TestFeatureImports:
    def test_xss(self):
        from services.features.function_xss import TraditionalXssDetector
        assert TraditionalXssDetector is not None
    
    def test_sqli(self):
        from services.features.function_sqli import SmartDetectionManager  # 或 SqliDetector
        assert SmartDetectionManager is not None
    
    def test_ssrf(self):
        import services.features.function_ssrf
        assert True
    
    def test_idor(self):
        import services.features.function_idor
        assert True
```

#### 第二層: 單元測試（核心邏輯）

```
tests/
├── unit/
│   ├── test_real_decision_engine.py    ← encode_input, decide, generate_decision
│   ├── test_smart_detection_manager.py ← SQLi 各引擎
│   ├── test_xss_detector.py           ← XSS 檢測邏輯
│   └── test_scan_request_validation.py ← API 請求驗證
```

#### 第三層: 整合測試

```
tests/
├── integration/
│   ├── test_scan_flow.py              ← POST /scan → 13 步驟
│   ├── test_commander_pipeline.py     ← Commander → AttackCoordinator → Executor
│   └── test_internal_loop.py          ← RAG 知識庫同步
```

### 測試框架配置

確認 `pyproject.toml` 中有 pytest 配置：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
```

---

## 3.4 TODO/FIXME 清理

### 統計

- `TODO`: 散佈在 services/ 下的 42 個標記
- 不需要全部立即處理，但需要分類

### 處理策略

1. **查詢所有 TODO**: `grep -rn "TODO\|FIXME\|HACK" services/ --include="*.py"`
2. **分類**:
   - 「已完成但忘了刪 TODO」→ 直接刪除標記
   - 「確實需要做」→ 轉為 GitHub Issue
   - 「不再相關」→ 刪除
3. **建立 Issue 追蹤**: 每個有效 TODO 轉成 GitHub Issue 並加標籤

---

## 3.5 靶場整合測試

### 目的

確認 AIVA 對已知漏洞靶場能正確检测。

### 建議靶場

| 靶場 | 類型 | 用途 |
|------|------|------|
| [DVWA](https://github.com/digininja/DVWA) | PHP | SQLi, XSS, CSRF, Command Injection |
| [Juice Shop](https://github.com/juice-shop/juice-shop) | Node.js | OWASP Top 10 全覆蓋 |
| [WebGoat](https://github.com/WebGoat/WebGoat) | Java | 學習型靶場 |

### Docker 化靶場環境

```yaml
# docker-compose.test-targets.yml
version: '3.8'
services:
  dvwa:
    image: vulnerables/web-dvwa
    ports:
      - "8080:80"
  
  juice-shop:
    image: bkimminich/juice-shop
    ports:
      - "3000:3000"
```

### 驗收標準

```
✅ AIVA 能偵測到 DVWA 的 SQL 注入漏洞
✅ AIVA 能偵測到 DVWA 的 XSS 反射型漏洞
✅ AIVA 能完成 13 步驟掃描流程不 crash
✅ 掃描結果包含至少 1 個有效的漏洞報告
```

---

## 完成清單

```
3.1 功能模組
    [ ] SQLi: 建立 __main__.py
    [ ] SQLi: 確認 SmartDetectionManager.run_detection() 可用
    [ ] SSRF: 建立 __main__.py + command_handler
    [ ] IDOR: 建立 __main__.py + command_handler
    [ ] BizLogic: 補 command_handler
    [ ] WebScanner: 修復 import + 補入口點

3.2 NotImplementedError
    [ ] 確認 cli/__init__.py 的 stub 狀態
    [ ] 確認 capability/models.py 的 stub 狀態
    [ ] (其餘 27 個保留不動)

3.3 測試
    [ ] 建立 test_smoke_imports.py
    [ ] 建立 test_smoke_app_startup.py
    [ ] 建立 test_smoke_neural_core.py
    [ ] 建立 test_real_decision_engine.py
    [ ] 建立 test_scan_flow.py (整合測試)

3.4 TODO 清理
    [ ] 掃描並分類 42 個 TODO/FIXME
    [ ] 有效的轉為 GitHub Issue
    [ ] 無效的刪除標記

3.5 靶場測試
    [ ] 啟動 DVWA
    [ ] 執行 POST /scan 對 DVWA
    [ ] 驗證結果包含已知漏洞
```
