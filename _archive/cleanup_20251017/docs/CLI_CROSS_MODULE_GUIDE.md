# AIVA 跨模組命令詳解

## 🌐 什麼是跨模組命令？

`aiva tools` 是 AIVA 統一 CLI 中專門用於**跨模組整合**的命令組，它整合了 `aiva-contracts` 工具，提供：

1. **跨語言協定支援** - JSON Schema 可用於任何語言
2. **型別安全** - TypeScript 定義確保前端型別安全
3. **API 文件基礎** - 自動生成的 Schema 可用於文件
4. **多語言開發** - 為 Go/Rust/TypeScript 等語言提供協定基礎

---

## 📋 四個核心命令

### 1. `aiva tools schemas` - 導出 JSON Schema

**用途：** 生成標準的 JSON Schema，可供任何語言使用

```bash
aiva tools schemas [--out <檔案>] [--format <human|json>]
```

**範例：**
```bash
# 基本使用
aiva tools schemas

# 指定輸出檔案
aiva tools schemas --out contracts/schemas.json

# JSON 格式輸出（用於自動化）
aiva tools schemas --out schemas.json --format json
```

**輸出內容：**
- 所有 Pydantic v2 模型的 JSON Schema
- 包含欄位定義、驗證規則、預設值
- 可用於生成其他語言的型別定義

**適用場景：**
- ✅ Go 專案：使用 `go-jsonschema` 生成 Go struct
- ✅ Rust 專案：使用 `schemars` 生成 Rust struct
- ✅ API 文件：用於生成 OpenAPI/Swagger 文件
- ✅ 資料驗證：驗證 JSON 資料是否符合 Schema

---

### 2. `aiva tools typescript` - 導出 TypeScript 型別

**用途：** 生成 TypeScript 型別定義檔（.d.ts）

```bash
aiva tools typescript [--out <檔案>] [--format <human|json>]
```

**範例：**
```bash
# 基本使用
aiva tools typescript

# 指定輸出檔案
aiva tools typescript --out types/aiva.d.ts

# JSON 格式輸出
aiva tools typescript --out aiva.d.ts --format json
```

**輸出內容：**
- TypeScript 介面定義
- 完整的型別註解
- 可直接在 TypeScript/JavaScript 專案中使用

**適用場景：**
- ✅ React/Vue/Angular 前端專案
- ✅ Node.js 後端專案
- ✅ TypeScript 工具開發
- ✅ IDE 自動補全和型別檢查

---

### 3. `aiva tools models` - 列出所有模型

**用途：** 列出專案中所有可用的 Pydantic 模型

```bash
aiva tools models [--format <human|json>]
```

**範例：**
```bash
# 人類可讀格式
aiva tools models

# JSON 格式（用於程式處理）
aiva tools models --format json
```

**輸出內容：**
- 模型名稱列表
- 模型所在模組
- 模型的基本資訊

**適用場景：**
- ✅ 查看可用的資料模型
- ✅ 文件生成前的模型清單
- ✅ 了解專案結構

---

### 4. `aiva tools export-all` - 一鍵導出全部 ⭐

**用途：** 最常用！一次導出 JSON Schema + TypeScript 定義

```bash
aiva tools export-all [--out-dir <目錄>] [--format <human|json>]
```

**範例：**
```bash
# 基本使用（導出到 _out 目錄）
aiva tools export-all

# 指定輸出目錄
aiva tools export-all --out-dir contracts

# JSON 格式輸出
aiva tools export-all --out-dir exports --format json
```

**輸出檔案：**
```
<out-dir>/
├── aiva.schemas.json    # JSON Schema
└── aiva.d.ts            # TypeScript 定義
```

**JSON 輸出範例：**
```json
{
  "ok": true,
  "command": "export-all",
  "exports": [
    {
      "type": "json-schema",
      "path": "C:\\path\\to\\aiva.schemas.json"
    },
    {
      "type": "typescript",
      "path": "C:\\path\\to\\aiva.d.ts"
    }
  ],
  "message": "已導出 2 個檔案到 C:\\path\\to"
}
```

**適用場景：**
- ✅ CI/CD 自動化流程
- ✅ 前後端協作開發
- ✅ 多語言專案整合
- ✅ API 協定更新

---

## 🔗 與其他模組的整合

### 整合 1：與 Python 核心模組

```python
# services/aiva_common/schemas.py 中的模型
from pydantic import BaseModel

class ScanRequest(BaseModel):
    target_url: str
    max_depth: int = 3
    
# ↓ 透過 aiva tools 導出

# aiva.schemas.json 中自動包含：
{
  "$defs": {
    "ScanRequest": {
      "properties": {
        "target_url": {"type": "string"},
        "max_depth": {"type": "integer", "default": 3}
      }
    }
  }
}
```

### 整合 2：與 TypeScript 前端

```typescript
// 1. 導出型別定義
// $ aiva tools typescript --out src/types/aiva.d.ts

// 2. 在前端使用
import type { ScanRequest, ScanResponse } from './types/aiva';

const request: ScanRequest = {
  target_url: 'https://example.com',
  max_depth: 3
};

// TypeScript 會提供自動補全和型別檢查！
```

### 整合 3：與 Go 後端

```bash
# 1. 導出 JSON Schema
aiva tools schemas --out contracts/aiva.schemas.json

# 2. 使用 Go 工具生成 struct
go-jsonschema -p models contracts/aiva.schemas.json > models/aiva.go

# 3. 在 Go 中使用
package main

import "myproject/models"

func HandleScan(req models.ScanRequest) {
    // 與 Python 後端保持型別一致！
}
```

### 整合 4：與 Rust SAST 引擎

```bash
# 1. 導出 JSON Schema
aiva tools schemas --out contracts/aiva.schemas.json

# 2. 使用 Rust 工具生成型別
# 使用 schemars 或 typify

# 3. 在 Rust 中使用
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct ScanRequest {
    target_url: String,
    max_depth: i32,
}
```

---

## 🎯 實際使用場景

### 場景 1：前後端分離開發

```bash
# 後端更新 Pydantic 模型後
cd backend
aiva tools export-all --out-dir ../frontend/src/types

# 前端自動獲得最新型別定義
cd ../frontend
npm run type-check  # 型別檢查通過！
```

### 場景 2：CI/CD 自動化

```yaml
# .github/workflows/update-contracts.yml
name: Update Contracts

on:
  push:
    paths:
      - 'services/**/schemas.py'
      
jobs:
  export-types:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Export contracts
        run: |
          pip install -e .
          aiva tools export-all --out-dir contracts --format json
      - name: Commit changes
        run: |
          git add contracts/
          git commit -m "chore: update contracts"
          git push
```

### 場景 3：多語言微服務架構

```
┌─────────────────────────────────────────────────┐
│          Python Core (Pydantic Models)         │
└────────────────┬────────────────────────────────┘
                 │
         aiva tools export-all
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼──────┐          ┌──────▼────┐
│ JSON     │          │ TypeScript │
│ Schema   │          │ Types      │
└───┬──────┘          └──────┬─────┘
    │                        │
┌───▼────┐  ┌────────┐  ┌───▼────┐
│   Go   │  │  Rust  │  │   TS   │
│ Service│  │  SAST  │  │Frontend│
└────────┘  └────────┘  └────────┘
```

### 場景 4：API 文件生成

```bash
# 1. 導出 JSON Schema
aiva tools schemas --out api-docs/schemas.json

# 2. 使用工具生成文件
redoc-cli bundle api-docs/schemas.json -o api-docs/index.html

# 3. 部署文件站
# 文件會自動包含所有型別定義和驗證規則
```

---

## 🔄 工作流程建議

### 日常開發

```bash
# 1. 修改 Python 模型
vim services/aiva_common/schemas.py

# 2. 立即導出新協定
aiva tools export-all --out-dir contracts

# 3. 提交變更
git add services/ contracts/
git commit -m "feat: update data models"
```

### 版本發布

```bash
# 發布前確保協定同步
aiva tools export-all --out-dir release/v1.0.0/contracts --format json

# 檢查輸出
cat release/v1.0.0/contracts/aiva.schemas.json | jq .

# 打包發布
tar -czf aiva-contracts-v1.0.0.tar.gz release/v1.0.0/
```

---

## 📊 輸出格式對比

### Human 格式（預設）

```bash
$ aiva tools export-all
✅ 已導出 JSON Schema: /path/to/aiva.schemas.json
✅ 已導出 TypeScript: /path/to/aiva.d.ts
✨ 完成！共導出 2 個檔案
```

### JSON 格式（自動化）

```bash
$ aiva tools export-all --format json
{
  "ok": true,
  "command": "export-all",
  "exports": [
    {"type": "json-schema", "path": "/path/to/aiva.schemas.json"},
    {"type": "typescript", "path": "/path/to/aiva.d.ts"}
  ],
  "message": "已導出 2 個檔案到 /path/to"
}
```

---

## 🎓 進階技巧

### 技巧 1：搭配 jq 處理 JSON

```bash
# 導出並提取檔案路徑
SCHEMA_PATH=$(aiva tools schemas --format json | jq -r '.output')
echo "Schema 已儲存至: $SCHEMA_PATH"

# 驗證 Schema 是否有效
cat "$SCHEMA_PATH" | jq . > /dev/null && echo "✅ 有效的 JSON"
```

### 技巧 2：整合到 Makefile

```makefile
.PHONY: contracts
contracts:
	@echo "📦 導出協定定義..."
	@aiva tools export-all --out-dir contracts --format json
	@echo "✅ 完成"

.PHONY: check-contracts
check-contracts: contracts
	@echo "🔍 驗證協定..."
	@cat contracts/aiva.schemas.json | jq . > /dev/null
	@echo "✅ Schema 有效"
```

### 技巧 3：版本控制

```bash
# 為每個版本儲存協定快照
VERSION="v1.2.3"
aiva tools export-all --out-dir "contracts/versions/$VERSION"

# 比較版本差異
diff contracts/versions/v1.2.2/aiva.schemas.json \
     contracts/versions/v1.2.3/aiva.schemas.json
```

---

## 🆚 與原工具的關係

### 原工具（aiva-contracts）

```bash
# 仍然可以直接使用
aiva-contracts list-models
aiva-contracts export-jsonschema --out schemas.json
aiva-contracts gen-ts --out types.d.ts
```

### 統一 CLI（aiva tools）

```bash
# 透過統一入口使用（推薦）
aiva tools models
aiva tools schemas --out schemas.json
aiva tools typescript --out types.d.ts
aiva tools export-all  # 🌟 一鍵導出
```

**關係：**
- `aiva tools` 是 `aiva-contracts` 的包裝器
- 提供統一的介面和輸出格式
- 添加 `export-all` 便利命令
- 保持向後相容

---

## 📚 相關資源

### 文件
- [完整命令參考](./CLI_COMMAND_REFERENCE.md) - 所有命令詳解
- [快速參考](./CLI_QUICK_REFERENCE.md) - 速查表
- [安裝指南](./CLI_UNIFIED_SETUP_GUIDE.md) - 設定說明

### 工具
- **JSON Schema**: https://json-schema.org/
- **TypeScript**: https://www.typescriptlang.org/
- **Pydantic**: https://docs.pydantic.dev/

### 程式碼
- `services/cli/tools.py` - 實作程式碼
- `tools/aiva-contracts-tooling/` - 底層工具

---

## ❓ 常見問題

### Q: 為什麼需要跨模組命令？

**A:** 現代應用通常使用多種語言：
- 前端：TypeScript
- 後端：Python
- 效能關鍵模組：Go/Rust

跨模組命令確保**型別定義在所有語言間保持一致**。

### Q: JSON Schema 有什麼用？

**A:** JSON Schema 是語言中立的型別定義格式：
- ✅ 可用於任何程式語言
- ✅ 可自動生成對應語言的型別
- ✅ 可用於資料驗證
- ✅ 可生成 API 文件

### Q: 我只用 Python，需要這些命令嗎？

**A:** 即使只用 Python，這些命令也很有用：
- ✅ 生成 API 文件
- ✅ 驗證資料格式
- ✅ 為未來擴展準備
- ✅ 與第三方服務整合

### Q: 多久需要重新導出？

**A:** 建議在以下情況導出：
- ✅ 修改 Pydantic 模型後
- ✅ 版本發布前
- ✅ CI/CD 自動化中
- ✅ 前後端開始協作時

---

## 🎉 總結

`aiva tools` 跨模組命令提供：

1. ✅ **統一的協定導出** - 一個命令，多種格式
2. ✅ **跨語言支援** - JSON Schema + TypeScript
3. ✅ **自動化友善** - JSON 輸出格式
4. ✅ **簡單易用** - 一鍵導出全部

**最常用命令：**
```bash
aiva tools export-all --out-dir contracts
```

這一個命令就能滿足大多數跨模組整合需求！🚀

---

**版本**: 1.0.0  
**更新日期**: 2025-10-17  
**維護者**: AIVA Team
