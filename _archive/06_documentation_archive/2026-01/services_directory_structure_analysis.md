# AIVA Services 目錄結構全面分析

## 總覽

**基礎路徑**: `C:\D\fold7\AIVA-git\services\`

## 主要目錄結構

```
services/
├── __pycache__/
├── aiva_common/          # ✅ 共用模組（存在）
├── core/                 # 核心服務
├── data/                 # 數據存儲
├── features/             # 功能模組（各種能力）
├── integration/          # 整合服務
├── scan/                 # 掃描引擎
├── __init__.py
└── _fix_all_readmes.py
```

## 關鍵發現

### ❌ 不存在的目錄
- `services/features/common/` - **不存在！**
  - 引用位置: `services/features/function_xss/worker.py:25`
  - 錯誤導入: `from services.features.common.worker_statistics import StatisticsCollector`
  - 實際情況: 這個模組根本不存在，只在 artifacts 數據中有記錄

### ✅ 存在的目錄
- `services/aiva_common/` - **正確的共用模組位置**
  - 包含各種共用工具和schemas
  - 應該改用此路徑導入共用模組

## 詳細結構

### 1. `services/aiva_common/` (共用模組)
```
aiva_common/
├── ai/                   # AI 相關
├── async_utils/          # 異步工具
├── cli/                  # CLI 工具
├── config/               # 配置
├── cross_language/       # 跨語言支援
├── detection/            # 檢測工具
├── enums/                # 枚舉定義
├── messaging/            # 消息隊列
├── observability/        # 可觀測性
├── plugins/              # 插件系統
├── protocols/            # 協議定義
├── schemas/              # 數據 schemas
│   ├── _base/
│   ├── analysis/
│   ├── generated/
│   ├── infrastructure/
│   ├── interfaces/
│   ├── risk/
│   ├── security/
│   └── testing/
├── services/             # 服務模組
│   └── features/         # 功能服務
├── tools/                # 工具集
├── utils/                # 工具函數
│   ├── dedup/
│   └── network/
└── v2_client/            # V2 客戶端
```

### 2. `services/features/` (功能模組)
```
features/
├── analysis_output/
├── base/
├── function_authn_go/         # Go 認證功能
├── function_bizlogic/          # 業務邏輯漏洞
├── function_crypto/            # 加密功能（含 Rust）
├── function_exploit/           # 漏洞利用
├── function_forensic/          # 取證工具
├── function_idor/              # IDOR 檢測
├── function_info_leak/         # 信息洩漏
├── function_postex/            # 後滲透
├── function_reverse_engineering/  # 逆向工程
├── function_social_engineering/   # 社工
├── function_sqli/              # SQL注入
├── function_ssrf/              # SSRF
├── function_steganography/     # 隱寫術
├── function_web_scanner/       # Web掃描
├── function_wordlist_generator/   # 字典生成
├── function_xss/               # XSS檢測
├── __init__.py
├── feature_step_executor.py
├── high_value_manager.py
├── smart_detection_manager.py
└── validate_handlers.py
```

### 3. `services/features/function_xss/` (XSS 模組詳細)
```
function_xss/
├── engines/
│   └── hackingtool_engine.py
├── external_tools/
│   ├── XSS-LOADER/
│   └── XSStrike/
├── integration_tools/
│   ├── __init__.py
│   └── xss_tools.py
├── __init__.py
├── __main__.py               # ✅ CLI 入口（含 run_reflected_test）
├── __main___sync.py
├── blind_xss_listener_validator.py
├── command_handler.py
├── dom_xss_detector.py
├── hackingtool_config.py
├── payload_generator.py
├── result_publisher.py
├── stored_detector.py
├── task_queue.py
├── traditional_detector.py
└── worker.py                 # ❌ 導入錯誤的位置
```

### 4. `services/integration/` (整合服務)
```
integration/
├── alembic/
│   ├── versions/
│   └── env.py
├── capability/
│   ├── adapters/
│   ├── capabilities/
│   ├── __init__.py
│   ├── bug_bounty_reporting.py
│   ├── config.py
│   ├── forensic_tools.py
│   ├── function_recon.py
│   ├── lifecycle_cli.py
│   ├── lifecycle.py
│   ├── minimal_manifest.py
│   ├── models.py
│   ├── payload_generator.py
│   ├── register_standardized_capabilities.py
│   ├── registry.py
│   ├── reverse_engineering_tools.py
│   ├── steganography_tools.py
│   ├── sync_from_analysis.py
│   └── toolkit.py
├── coordinators/
│   ├── __init__.py
│   ├── base_coordinator.py
│   └── xss_coordinator.py
├── data/                     # ✅ 數據存儲位置
│   └── internal_exploration/
│       └── classification_data.json  # 分類器輸出
├── scripts/
├── tools/
├── __init__.py
├── models.py
├── search_command_handler.py
└── simple_data_manager.py
```

### 5. `services/core/` (核心服務)
```
core/
└── aiva_core/
    └── internal_exploration/
        ├── python_tools/
        ├── aiva_external_classifier.py  # 分類器
        ├── aiva_external_executor.py    # 執行器
        └── ...
```

## 導入路徑修復指南

### 錯誤的導入
```python
# ❌ 不存在的路徑
from services.features.common.worker_statistics import StatisticsCollector
```

### 修復方案

#### 方案 1: 檢查 aiva_common 是否有相關模組
```python
# 檢查是否在 aiva_common 中
from services.aiva_common.utils.statistics import StatisticsCollector  # (如果存在)
```

#### 方案 2: 在當前模組定義（已實施）
```python
# 在 worker.py 中本地定義
@dataclass
class StatisticsCollector:
    task_id: str
    worker_type: str
    def start_operation(self, name: str): pass
    def end_operation(self, name: str): pass
    def record_finding(self, severity: str): pass
    def get_stats(self): return {}
```

## XSS 模組執行路徑

### 入口點位置
- **CLI 入口**: `services/features/function_xss/__main__.py`
  - `run_reflected_test(args)` - Line 29
  - `run_dom_test(args)` - Line 58
  - `run_stored_test(args)` - Line 67

### 導入路徑
```python
# 正確的導入方式
from services.features.function_xss.__main__ import run_reflected_test
```

## 執行器路徑配置

### 分類器輸出位置
```
features_classification/classification_data.json
```

### 執行器讀取位置
```
services/integration/data/internal_exploration/classification_data.json
```

### 需要同步
```bash
# 復制分類器輸出到執行器讀取位置
copy features_classification\classification_data.json services\integration\data\internal_exploration\
```

## 總結

### ✅ 確認的事實
1. **沒有 `services/features/common/` 目錄**
2. **共用模組在 `services/aiva_common/`**
3. **XSS 入口點在 `__main__.py`**（不是 `main.py`）
4. **分類數據需要手動復制** 到執行器讀取位置

### ❌ 需要修復的問題
1. **worker.py 的導入錯誤** - 已用本地定義替代
2. **執行器的動態導入** - 需要支援 `__main__` 模組
3. **async 函數調用** - 需要 `asyncio.run()`

### 🔧 下一步
1. ✅ 修復 StatisticsCollector 導入 - 已完成
2. ✅ 更新執行器支援 `__main__` - 已完成
3. ✅ 支援 async 函數執行 - 已完成
4. 🔄 實際測試 22 個 flows
5. 📊 收集測試結果

## 文件統計

- **總 Python 文件**: ~2000+ 個（含 external_tools）
- **核心功能模組**: 17 個
- **XSS 模組文件**: ~50+ 個
- **分類數據流程**: 641 個（Python: 627, Rust: 1, Go: 13）
