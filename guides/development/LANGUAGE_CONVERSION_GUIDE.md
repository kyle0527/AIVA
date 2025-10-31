# AIVA 程式語言轉換指南

## 📋 文件資訊
- **創建日期**: 2025-10-31
- **最後更新**: 2025-10-31
- **適用版本**: AIVA v2.0
- **狀態**: ✅ 已驗證 (10/31實測驗證)

## 🎯 指南目標

提供 AIVA 專案中不同程式語言間的轉換標準，包含：
- 🔄 **代碼轉換**: Python ↔ TypeScript ↔ Go ↔ Rust
- 📝 **Schema 轉換**: 統一資料結構跨語言實現
- 🤖 **AI 組件轉換**: AI 功能跨語言移植
- 🔧 **工具鏈整合**: 自動化轉換工具使用

## 🌍 支援的語言轉換

### 核心語言矩陣

| 來源語言 | 目標語言 | 轉換難度 | 工具支援 | 狀態 |
|---------|---------|----------|----------|------|
| Python | TypeScript | 🟡 中等 | ✅ 手動+工具 | 可用 |
| Python | Go | 🟠 困難 | ⚠️ 半自動 | 開發中 |
| Python | Rust | 🔴 最困難 | ⚠️ 手動為主 | 實驗性 |
| TypeScript | Python | 🟢 簡單 | ✅ 自動化 | 完整 |
| Go | Python | 🟡 中等 | ⚠️ 半自動 | 可用 |
| Rust | Python | 🟠 困難 | ❌ 手動 | 基礎 |

## 🔄 Schema 跨語言轉換

### 1. 統一 Schema 系統

AIVA 使用 YAML 定義的統一 Schema，自動生成多語言實現：

```yaml
# core_schema_sot.yaml
schemas:
  Message:
    description: "AI 組件間通訊訊息"
    fields:
      id:
        type: "string"
        description: "唯一識別碼"
        required: true
      content:
        type: "string"
        description: "訊息內容"
        required: true
      timestamp:
        type: "datetime"  
        description: "建立時間"
        required: true
```

### 2. 自動代碼生成

**使用 Schema 代碼生成工具**:
```powershell
# 生成所有語言的 Schema
python services/aiva_common/tools/schema_codegen_tool.py --all

# 生成特定語言
python services/aiva_common/tools/schema_codegen_tool.py --language python
python services/aiva_common/tools/schema_codegen_tool.py --language typescript
python services/aiva_common/tools/schema_codegen_tool.py --language go
python services/aiva_common/tools/schema_codegen_tool.py --language rust
```

### 3. 生成結果範例

#### Python (Pydantic v2)
```python
from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    """AI 組件間通訊訊息"""
    id: str
    content: str
    timestamp: datetime
```

#### TypeScript
```typescript
export interface Message {
  /** AI 組件間通訊訊息 */
  id: string;
  content: string;
  timestamp: Date;
}
```

#### Go
```go
package schemas

import "time"

// Message AI 組件間通訊訊息
type Message struct {
    ID        string    `json:"id" yaml:"id"`
    Content   string    `json:"content" yaml:"content"`
    Timestamp time.Time `json:"timestamp" yaml:"timestamp"`
}
```

#### Rust
```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// AI 組件間通訊訊息
#[derive(Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}
```

## 🐍 Python 轉換指南

### Python → TypeScript

#### 基本類型轉換
```python
# Python
def process_data(items: List[str], count: int) -> Dict[str, Any]:
    return {"items": items, "count": count}
```

```typescript
// TypeScript
function processData(items: string[], count: number): Record<string, any> {
    return { items, count };
}
```

#### 類別轉換
```python
# Python
class DataProcessor:
    def __init__(self, name: str):
        self.name = name
    
    def process(self, data: str) -> str:
        return f"Processed: {data}"
```

```typescript
// TypeScript
class DataProcessor {
    constructor(private name: string) {}
    
    process(data: string): string {
        return `Processed: ${data}`;
    }
}
```

### Python → Go

#### 基本結構轉換
```python
# Python
@dataclass
class Config:
    host: str
    port: int
    enabled: bool = True
```

```go
// Go
type Config struct {
    Host    string `json:"host"`
    Port    int    `json:"port"`
    Enabled bool   `json:"enabled"`
}

func NewConfig(host string, port int) *Config {
    return &Config{
        Host:    host,
        Port:    port,
        Enabled: true,
    }
}
```

#### 錯誤處理轉換
```python
# Python
def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}")
```

```go
// Go
func ReadFile(path string) (string, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return "", fmt.Errorf("file not found: %s", path)
    }
    return string(data), nil
}
```

### Python → Rust

#### 所有權和借用
```python
# Python
class Buffer:
    def __init__(self, data: str):
        self.data = data
    
    def get_data(self) -> str:
        return self.data
```

```rust
// Rust
pub struct Buffer {
    data: String,
}

impl Buffer {
    pub fn new(data: String) -> Self {
        Self { data }
    }
    
    pub fn get_data(&self) -> &str {
        &self.data
    }
}
```

#### Result 類型處理
```python
# Python
def parse_number(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"Invalid number: {s}")
```

```rust
// Rust
fn parse_number(s: &str) -> Result<i32, String> {
    s.parse::<i32>()
        .map_err(|_| format!("Invalid number: {}", s))
}
```

## 🔧 TypeScript 轉換指南

### TypeScript → Python

#### 介面轉類別
```typescript
// TypeScript
interface User {
    id: number;
    name: string;
    email?: string;
}

function createUser(data: Partial<User>): User {
    return { id: Date.now(), name: "Unknown", ...data };
}
```

```python
# Python
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: Optional[str] = None

def create_user(**data) -> User:
    defaults = {"id": int(time.time() * 1000), "name": "Unknown"}
    return User(**{**defaults, **data})
```

#### Promise → Async/Await
```typescript
// TypeScript
async function fetchData(url: string): Promise<any> {
    const response = await fetch(url);
    return response.json();
}
```

```python
# Python
import aiohttp

async def fetch_data(url: str) -> any:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## 🐹 Go 轉換指南

### Go → Python

#### 結構體轉類別
```go
// Go
type Service struct {
    Name    string
    Port    int
    running bool
}

func (s *Service) Start() error {
    s.running = true
    return nil
}
```

```python
# Python
class Service:
    def __init__(self, name: str, port: int):
        self.name = name
        self.port = port
        self._running = False
    
    def start(self) -> None:
        self._running = True
```

#### Channel → Queue
```go
// Go
func processMessages(ch <-chan string) {
    for msg := range ch {
        fmt.Println("Processing:", msg)
    }
}
```

```python
# Python
import asyncio

async def process_messages(queue: asyncio.Queue):
    while True:
        msg = await queue.get()
        print(f"Processing: {msg}")
        queue.task_done()
```

## 🦀 Rust 轉換指南

### Rust → Python

#### 枚舉轉換
```rust
// Rust
#[derive(Debug)]
pub enum Status {
    Pending,
    Running(String),
    Completed(i32),
    Failed(String, i32),
}
```

```python
# Python
from enum import Enum
from typing import Union
from dataclasses import dataclass

class StatusType(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Status:
    type: StatusType
    message: str = ""
    code: int = 0
```

#### Option → Optional
```rust
// Rust
fn find_user(id: u32) -> Option<String> {
    if id > 0 {
        Some(format!("User-{}", id))
    } else {
        None
    }
}
```

```python
# Python
from typing import Optional

def find_user(id: int) -> Optional[str]:
    if id > 0:
        return f"User-{id}"
    return None
```

## 🤖 AI 組件轉換

### AI 功能跨語言實現

AIVA 的 AI 組件支援跨語言調用：

#### 1. 統一接口定義
```python
# Python AI 組件接口
from abc import ABC, abstractmethod

class AIComponent(ABC):
    @abstractmethod
    async def process(self, input_data: dict) -> dict:
        pass
```

#### 2. 跨語言橋接器
```python
from services.aiva_common.ai.cross_language_bridge import CrossLanguageBridge

# 調用 Go 實現的 AI 組件
bridge = CrossLanguageBridge()
result = await bridge.execute_go_component(
    component="authentication",
    input_data={"token": "abc123"}
)
```

#### 3. Rust 安全組件整合
```python
# 調用 Rust 安全掃描組件
security_result = await bridge.execute_rust_component(
    component="security_scanner",
    input_data={"code": source_code}
)
```

## 🔧 轉換工具和輔助

### 1. AIVA 內建工具

#### Schema 代碼生成器
```powershell
# 生成跨語言 Schema
python services/aiva_common/tools/schema_codegen_tool.py --all
```

#### 跨語言接口工具
```powershell
# AI 組件跨語言轉換
python services/aiva_common/tools/cross_language_interface.py
```

#### 跨語言驗證工具
```powershell
# 驗證跨語言一致性
python services/aiva_common/tools/cross_language_validator.py
```

### 2. 外部工具建議

#### Python → TypeScript
- **py2ts**: 基本類型轉換
- **mypy**: 類型檢查確保轉換準確性
- **手動調整**: 複雜邏輯需人工優化

#### Python → Go
- **gopy**: Python-Go 綁定
- **手動重寫**: 建議重新設計適應 Go 慣例
- **protobuf**: 數據結構統一

#### Python → Rust
- **PyO3**: Python-Rust 整合
- **手動重寫**: 完全重新設計推薦
- **Serde**: 序列化統一

### 3. 轉換輔助腳本

建立自動化轉換輔助：

```powershell
# 創建轉換輔助腳本
@"
# 語言轉換輔助工具

param(
    [string]`$SourceLang,
    [string]`$TargetLang,
    [string]`$SourceFile,
    [string]`$OutputDir = "converted"
)

Write-Host "轉換 `$SourceLang -> `$TargetLang" -ForegroundColor Cyan

switch ("`$SourceLang-`$TargetLang") {
    "python-typescript" {
        Write-Host "使用 Schema 生成器轉換..."
        python services/aiva_common/tools/schema_codegen_tool.py --language typescript
    }
    "python-go" {
        Write-Host "使用跨語言接口轉換..."
        python services/aiva_common/tools/cross_language_interface.py --target go
    }
    "python-rust" {
        Write-Host "建議手動轉換，參考 Rust 轉換指南"
    }
    default {
        Write-Host "不支援的轉換: `$SourceLang -> `$TargetLang" -ForegroundColor Red
    }
}
"@ | Out-File -FilePath "scripts/language_converter.ps1" -Encoding UTF8
```

## 📋 轉換最佳實践

### 1. 轉換前準備
- ✅ 確保原始程式碼類型完整
- ✅ 理解目標語言慣例
- ✅ 準備測試用例
- ✅ 評估性能需求

### 2. 轉換過程
- 🔄 先轉換資料結構 (Schema)
- 🔄 再轉換業務邏輯
- 🔄 最後整合測試
- 🔄 性能調優

### 3. 轉換後驗證
- ✅ 功能等價性測試
- ✅ 性能基準對比  
- ✅ 錯誤處理驗證
- ✅ 整合測試通過

### 4. 語言特定考量

#### Python → TypeScript
- 注意動態類型轉靜態類型
- Promise/async 模式調整
- 錯誤處理機制差異

#### Python → Go
- 錯誤處理範式完全不同
- 記憶體管理手動化
- 併發模型差異巨大

#### Python → Rust
- 所有權系統學習曲線陡峭
- 生命週期管理複雜
- 類型系統更嚴格

## 🚨 常見轉換陷阱

### 1. 類型系統差異
```python
# Python - 動態類型可能的問題
def process(data):  # 缺少類型註解
    return data.get("key", None)  # 假設 data 是 dict
```

```typescript
// TypeScript - 需要明確類型
function process(data: Record<string, any>): any | null {
    return data.key ?? null;
}
```

### 2. 錯誤處理差異
```python
# Python - 異常模式
try:
    result = risky_operation()
except ValueError as e:
    handle_error(e)
```

```go
// Go - 返回值模式
result, err := riskyOperation()
if err != nil {
    handleError(err)
}
```

### 3. 記憶體管理
```python
# Python - 自動垃圾回收
data = load_large_data()  # 自動管理記憶體
```

```rust
// Rust - 手動所有權管理
let data = load_large_data();  // 編譯時確定生命週期
drop(data);  // 明確釋放
```

## 📊 轉換品質檢查

### 自動化檢查清單
```powershell
# 轉換品質檢查腳本
@"
Write-Host "=== 語言轉換品質檢查 ===" -ForegroundColor Cyan

# 1. 語法檢查
Write-Host "1. 語法檢查..." -ForegroundColor Yellow
# Python: python -m py_compile
# TypeScript: tsc --noEmit
# Go: go vet
# Rust: cargo check

# 2. 類型檢查
Write-Host "2. 類型檢查..." -ForegroundColor Yellow
# Python: mypy
# TypeScript: 內建
# Go: 內建  
# Rust: 內建

# 3. 功能測試
Write-Host "3. 功能測試..." -ForegroundColor Yellow
# 執行對應的測試套件

# 4. 性能基準
Write-Host "4. 性能基準..." -ForegroundColor Yellow
# 執行性能測試

Write-Host "=== 檢查完成 ===" -ForegroundColor Cyan
"@ | Out-File -FilePath "scripts/conversion_quality_check.ps1" -Encoding UTF8
```

## 🔗 相關資源

### AIVA 內建資源
- [`guides/architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md`](../architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md) - 跨語言 Schema 詳細指南
- [`guides/architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md`](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md) - 兼容性分析
- [`guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md`](../development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) - 環境配置標準

### 工具和服務
- `services/aiva_common/tools/schema_codegen_tool.py` - Schema 代碼生成
- `services/aiva_common/ai/cross_language_bridge.py` - 跨語言橋接
- `services/aiva_common/tools/cross_language_interface.py` - AI 組件接口

### 外部參考
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Go Language Specification](https://golang.org/ref/spec)
- [Rust Book](https://doc.rust-lang.org/book/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**✅ 驗證狀態**: 此轉換指南已整合 AIVA 現有跨語言工具，並於 2025-10-31 驗證可用性