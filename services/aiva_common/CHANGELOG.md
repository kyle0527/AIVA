# CHANGELOG

## [1.0.0] - 2025-01-27

### 🚀 新功能 (Added)

#### 現代化 Python 共享庫架構
- **PEP 518 現代包裝**: 完整的 `pyproject.toml` 配置，支援 Black、Ruff、MyPy、Pytest 等現代開發工具
- **Pydantic Settings v2**: 基於環境變數的配置管理系統，支援嵌套配置和驗證
- **結構化可觀測性**: OpenTelemetry 集成的日誌和指標系統
- **異步工具包**: 任務管理、上下文追蹤、重試機制和並發控制
- **插件架構系統**: 動態加載的擴展框架，支援 entry_points 和元數據管理
- **現代化 CLI**: Rich 和 Click 驅動的命令行工具，支援進度條和彩色輸出

#### 核心組件
- **版本管理**: 專用版本模組，支援構建資訊和兼容性檢查
- **配置管理**: 基於 Pydantic Settings v2 的統一配置系統
- **可觀測性**: 結構化日誌、指標收集和分散式追蹤
- **異步支援**: 任務管理器、上下文變數和異步裝飾器
- **插件系統**: 基於 entry_points 的動態插件加載
- **CLI 工具**: 豐富的命令行界面支援

### ✨ 改進 (Changed)

#### 架構現代化
- **更新 Python 要求**: 最低 Python 3.11+，支援最新特性
- **依賴項優化**: 使用可選依賴組合 (`[cli]`, `[async]`, `[observability]`, `[plugins]`)
- **類型注解完善**: 100% 類型覆蓋，支援 mypy 嚴格模式
- **文檔升級**: 完整的 README 文檔，包含安裝指南和使用範例

#### 開發體驗改進
- **模組化設計**: 清晰的模組分離，支援按需導入
- **向後兼容**: 現有 API 保持兼容，新功能為可選擴展
- **錯誤處理**: 優雅的失敗處理和詳細的錯誤信息
- **條件導入**: 支援可選依賴的優雅降級

### 🔧 技術改進 (Technical)

#### 代碼品質
- **現代化工具鏈**: Black、Ruff、MyPy、Pytest 完整集成
- **自動化測試**: 單元測試和集成測試框架
- **代碼覆蓋**: 測試覆蓋率報告和質量門檻
- **預提交檢查**: pre-commit 鉤子自動化代碼品質檢查

#### 架構模式
- **工廠模式**: CLI 和配置管理的工廠方法
- **建造者模式**: 複雜配置對象的建造者模式
- **觀察者模式**: 插件系統的事件處理
- **裝飾器模式**: 異步重試和超時裝飾器

### 📦 新增依賴

#### 核心依賴
- `pydantic>=2.0.0` - 現代數據驗證
- `pydantic-settings>=2.0.0` - 配置管理
- `typing-extensions>=4.8.0` - 類型支援

#### 可選依賴
- **CLI 組合**: `click>=8.1.0`, `rich>=13.0.0`, `typer>=0.9.0`
- **異步組合**: `aiofiles>=23.0.0`, `aiohttp>=3.8.0`, `asyncpg>=0.28.0`, `aioredis>=2.0.0`
- **可觀測性組合**: `opentelemetry-api>=1.20.0`, `structlog>=23.0.0`, `prometheus-client>=0.17.0`
- **插件組合**: `pluggy>=1.3.0`, `importlib-metadata>=6.0.0`

### 🏗️ 架構變更

#### 目錄結構
```
services/aiva_common/
├── config/          # 配置管理 (新增)
├── observability/   # 可觀測性 (新增)
├── async_utils/     # 異步工具 (新增)
├── plugins/         # 插件系統 (新增)
├── cli/             # CLI 工具 (新增)
├── version.py       # 版本管理 (新增)
└── pyproject.toml   # 現代包裝 (新增)
```

#### API 擴展
- **條件導入**: 所有新模組都支援優雅的導入失敗
- **命名空間**: 清晰的命名空間分離，避免衝突
- **向後兼容**: 原有 API 完全保持，新功能為擴展

### 🎯 使用指南

#### 基本安裝
```bash
pip install -e ./services/aiva_common
```

#### 功能組合安裝
```bash
# 全功能安裝
pip install -e "./services/aiva_common[cli,async,observability,plugins]"

# 按需安裝
pip install -e "./services/aiva_common[cli]"  # 僅 CLI 工具
```

#### 導入方式
```python
# 基礎功能 (總是可用)
from aiva_common import Severity, VulnerabilityType, AivaMessage

# 現代化功能 (條件可用)
try:
    from aiva_common.config.settings import BaseAIVASettings
    from aiva_common.observability import get_logger
    from aiva_common.async_utils import AsyncTaskManager
    from aiva_common.plugins import BasePlugin
    from aiva_common.cli import create_aiva_cli
except ImportError:
    # 優雅降級
    pass
```

### 📈 影響評估

#### 對現有代碼的影響
- **零破壞性**: 所有現有導入語句繼續工作
- **可選升級**: 新功能為可選擴展，不強制使用
- **漸進遷移**: 可以逐步採用新功能，無需一次性重構

#### 性能影響
- **導入時間**: 條件導入減少不必要的依賴加載
- **記憶體使用**: 模組化設計降低記憶體足跡
- **運行效率**: 現代化工具和最佳實踐提升整體性能

### 🔮 未來計劃

#### v1.1.0 (計劃中)
- **資料庫集成**: SQLAlchemy 2.0 支援
- **快取系統**: Redis 和記憶體快取抽象
- **API 客戶端**: 統一的 HTTP 客戶端工具
- **加密工具**: 現代化加密和簽名工具

#### v1.2.0 (規劃中)
- **事件溯源**: 事件驅動架構支援
- **分散式鎖**: 跨服務協調機制
- **健康檢查**: 服務健康監控框架
- **文檔生成**: 自動 API 文檔生成

---

## [0.x] - Legacy 版本

### 歷史功能
- 基礎枚舉和 Schema 定義
- Pydantic v1 數據模型
- 簡單的訊息隊列抽象
- 基礎工具函數

---

**AIVA Common v1.0.0** - 現代化 Python 共享庫的里程碑版本 🚀