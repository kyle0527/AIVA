# AIVA 開發者指南 v5.0

> **📋 使用者指南**: 如何使用已完成的AIVA功能和標準化系統  
> **🎯 適用對象**: 開發者、貢獻者、新手入門  
> **📅 版本**: v5.0 - Schema標準化完成版本  
> **✅ 系統狀態**: 8/8 模組 100% Schema 合規

## 🚀 快速開始

### ⚡ 5分鐘上手AIVA
```bash
# 1. 環境檢查
python tools/schema_compliance_validator.py  # 應顯示 8/8 模組 100% 合規

# 2. 生成標準Schema (如需要)
python services/aiva_common/tools/schema_codegen_tool.py --lang all

# 3. 運行測試確保一切正常
python -m pytest tests/ -v

# 4. 開始開發！
```

## 🛠️ 開發環境設置

### 1. 環境要求
- Python 3.8+
- Git
- VS Code (推薦)

### 2. 專案設置
```bash
# 克隆專案
git clone <repository-url>
cd AIVA-git

# 設置虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt

# 複製環境配置
cp .env.example .env
```

## 📝 開發規範

### 程式碼風格
- 使用 `ruff` 進行格式化
- 使用 `mypy` 進行型別檢查
- 遵循 PEP 8 規範

### 🎯 跨語言 Schema 標準使用 (已完成標準化)
**狀態**: ✅ 已實現 100% Schema 標準化 (8/8 模組完全合規)

#### 1. 標準Schema使用方法

**Go 模組**:
```go
// ✅ 使用標準生成的 schema
import schemas "github.com/kyle0527/aiva/services/features/common/go/aiva_common_go/schemas/generated"

// 創建標準Finding
finding := &schemas.FindingPayload{
    FindingId: "example-001",
    TaskId:    taskId,
    ScanId:    scanId, 
    Status:    "confirmed",
    Vulnerability: &schemas.Vulnerability{
        Name:        "SQL_INJECTION",
        Severity:    "high",
        Confidence:  "high",
        Description: "SQL injection vulnerability detected",
    },
    // ... 其他字段按標準填寫
}
```

**Rust 模組**:
```rust
// ✅ 使用生成的標準 schema
use aiva_common_rust::schemas::generated::{FindingPayload, Vulnerability, VulnerabilityType};

let finding = FindingPayload {
    finding_id: "example-001".to_string(),
    task_id: task_id.clone(),
    scan_id: scan_id.clone(),
    status: "confirmed".to_string(),
    vulnerability: Some(Vulnerability {
        name: VulnerabilityType::SqlInjection,
        severity: "high".to_string(),
        confidence: "high".to_string(),
        description: "SQL injection vulnerability detected".to_string(),
        cwe: Some("CWE-89".to_string()),
    }),
    // ... 其他字段
};
```

**TypeScript 模組**:
```typescript
// ✅ 使用標準生成的類型定義
import { FindingPayload, VulnerabilityType } from '../../features/common/typescript/aiva_common_ts/schemas/generated/schemas';

const finding: FindingPayload = {
    finding_id: "example-001",
    task_id: taskId,
    scan_id: scanId,
    status: "confirmed",
    vulnerability: {
        name: VulnerabilityType.SQL_INJECTION,
        severity: "high",
        confidence: "high", 
        description: "SQL injection vulnerability detected",
        cwe: "CWE-89"
    },
    // ... 其他字段
};
```

#### 2. Schema合規檢查工具
```bash
# 檢查所有模組的合規性
python tools/schema_compliance_validator.py

# 預期輸出：8/8 模組 100% 合規
```

#### 3. 禁止的操作
- ❌ **不要手動創建** Finding 相關的結構定義
- ❌ **不要修改** generated/ 目錄中的檔案
- ❌ **不要重新創建** 已清理的過時schema工具
- ❌ **不要引用** schemas/aiva_schemas (已移除)

### 提交規範
```bash
# 執行預提交檢查
pre-commit run --all-files

# 提交格式
git commit -m "feat: 添加新功能"
git commit -m "fix: 修復bug"
git commit -m "docs: 更新文件"
```

## 🏗️ 模組開發 (使用標準化Schema)

### 新增功能檢測模組
```python
# services/features/function_newattack/
# ├── __init__.py
# ├── detector.py
# ├── payload_generator.py
# └── validator.py

from services.aiva_common.schemas.generated.schemas import FindingPayload, Vulnerability, VulnerabilityType

class NewAttackDetector:
    def detect(self, target, task_id: str, scan_id: str) -> FindingPayload:
        """使用標準Schema創建Finding"""
        
        # 檢測邏輯...
        
        # ✅ 使用標準化的FindingPayload
        return FindingPayload(
            finding_id=f"newattack-{int(time.time())}",
            task_id=task_id,
            scan_id=scan_id,
            status="confirmed",
            vulnerability=Vulnerability(
                name=VulnerabilityType.CUSTOM_VULNERABILITY,  # 或適當的類型
                severity="medium",
                confidence="high",
                description="New attack pattern detected",
                cwe="CWE-xxx"
            ),
            target=target,
            strategy="new_attack_detection",
            # ... 其他必要字段
        )
```

### Go模組開發範例
```go
// services/features/function_newattack_go/detector.go
package main

import (
    schemas "github.com/kyle0527/aiva/services/features/common/go/aiva_common_go/schemas/generated"
)

func DetectNewAttack(target *schemas.Target, taskID, scanID string) *schemas.FindingPayload {
    // 檢測邏輯...
    
    // ✅ 使用標準化的schemas
    return &schemas.FindingPayload{
        FindingId: fmt.Sprintf("newattack-%d", time.Now().Unix()),
        TaskId:    taskID,
        ScanId:    scanID,
        Status:    "confirmed",
        Vulnerability: &schemas.Vulnerability{
            Name:        "CUSTOM_VULNERABILITY",
            Severity:    "medium", 
            Confidence:  "high",
            Description: "New attack pattern detected",
            Cwe:         "CWE-xxx",
        },
        Target:   target,
        Strategy: "new_attack_detection",
        // ... 其他字段
    }
}
```

### Rust模組開發範例
```rust
// services/features/function_newattack_rust/src/detector.rs
use aiva_common_rust::schemas::generated::{FindingPayload, Vulnerability, VulnerabilityType};

pub fn detect_new_attack(target: &Target, task_id: &str, scan_id: &str) -> FindingPayload {
    // 檢測邏輯...
    
    // ✅ 使用標準化的schemas
    FindingPayload {
        finding_id: format!("newattack-{}", chrono::Utc::now().timestamp()),
        task_id: task_id.to_string(),
        scan_id: scan_id.to_string(),
        status: "confirmed".to_string(),
        vulnerability: Some(Vulnerability {
            name: VulnerabilityType::CustomVulnerability,
            severity: "medium".to_string(),
            confidence: "high".to_string(),
            description: "New attack pattern detected".to_string(),
            cwe: Some("CWE-xxx".to_string()),
        }),
        target: target.clone(),
        strategy: "new_attack_detection".to_string(),
        // ... 其他字段
    }
}
```

## 🧪 測試指南

### 單元測試
```bash
# 執行所有測試
pytest tests/

# 執行特定模組測試
pytest tests/test_core/

# 測試覆蓋率
pytest --cov=services
```

### 整合測試
```bash
# API 測試
python api/test_api.py

# 系統整合測試
python services/core/aiva_core/ai_integration_test.py
```

## 📊 監控與除錯

### 日誌系統
```python
import logging
from services.aiva_common.logging import get_logger

logger = get_logger(__name__)
logger.info("處理開始")
logger.error("發生錯誤: %s", error_msg)
```

### 效能監控
```python
from services.integration.aiva_integration.system_performance_monitor import monitor

@monitor
def your_function():
    # 自動監控函數效能
    pass
```

## 🔧 常見開發任務

### 1. 新增 API 端點
```python
# api/routers/new_router.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint(data: dict):
    return {"result": "success"}
```

### 2. 新增資料庫模型
```python
# services/integration/models.py
from sqlalchemy import Column, String, Integer

class NewModel(Base):
    __tablename__ = "new_table"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
```

### 3. 新增配置選項
```python
# config/settings.py
NEW_FEATURE_ENABLED = True
NEW_FEATURE_CONFIG = {
    "timeout": 30,
    "retries": 3
}
```

## 📦 部署指南

### Docker 部署
```bash
# 構建映像
docker-compose build

# 啟動服務
docker-compose up -d

# 生產環境
docker-compose -f docker-compose.production.yml up -d
```

### 本地部署
```bash
# 啟動所有服務
python scripts/launcher/aiva_launcher.py

# 或分別啟動
python api/start_api.py &
python services/integration/aiva_integration/trigger_ai_continuous_learning.py &
```

## 🐛 故障排除

### 常見問題
1. **導入錯誤**: 檢查 `sys.path` 設置
2. **資料庫連接**: 檢查 `.env` 配置
3. **端口衝突**: 修改配置檔案中的端口設置

### 除錯工具
```bash
# 檢查套件狀態
python aiva_package_validator.py

# 檢查系統狀態
python -c "from services.integration.aiva_integration.system_performance_monitor import check_system; check_system()"
```

## 🎯 Schema標準化使用總結

### ✅ 當前系統狀態 (2025-10-28)
- **合規狀態**: 8/8 模組 100% 合規 
- **編譯狀態**: 所有語言編譯成功
- **維護狀態**: 零手動維護需求
- **標準遵循**: 100% 符合國際標準

### 🛡️ 開發者須知重點
1. **只使用生成的Schema**: 絕對不要手動創建Finding相關結構
2. **統一的引用路徑**: 使用標準的import路徑
3. **合規檢查**: 開發完成後務必運行合規檢查
4. **避免過時工具**: 不要使用已清理的過時schema工具

### � 開發效率提升
使用標準化Schema後，您將享受到：
- **開發時間減少**: 不需要重複定義數據結構
- **錯誤率降低**: 自動生成避免人為錯誤
- **維護成本減少**: 統一維護入口點
- **國際標準合規**: 自動符合CVSS、SARIF等標準

## �📚 參考資源

- [SCHEMA_PROJECT_FINAL_REPORT.md](SCHEMA_PROJECT_FINAL_REPORT.md) - Schema標準化完整報告
- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) - 完整專案結構
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速參考
- [API 文件](api/README.md) - API 使用說明
- [services/aiva_common/README.md](services/aiva_common/README.md) - Schema使用規範

## 🤝 貢獻指南

### 開發流程
1. **Fork 專案** 並 Clone 到本地
2. **檢查合規性**: `python tools/schema_compliance_validator.py`
3. **創建功能分支**: `git checkout -b feature/new-feature`
4. **開發功能** (使用標準Schema)
5. **再次檢查合規**: 確保仍為 8/8 模組 100% 合規
6. **提交變更**: `git commit -am 'feat: Add new feature'`
7. **推送分支**: `git push origin feature/new-feature`
8. **提交 Pull Request**

### 代碼審查重點
- ✅ 使用標準生成的Schema
- ✅ 通過合規性檢查
- ✅ 所有語言編譯成功
- ❌ 沒有手動創建的Schema定義

---

*🎉 恭喜！您現在可以使用業界領先的跨語言統一Schema系統進行開發！*