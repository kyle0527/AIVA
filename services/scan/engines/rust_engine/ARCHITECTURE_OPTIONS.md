# Rust 引擎架構方案對比

**日期**: 2025-11-19  
**問題**: Rust 引擎應該如何與 Python Worker 整合？

---

## 📊 三種架構方案

### 方案 1: CLI 子進程（bridge.py 調用）

**架構**:
```
Python worker.py
    ↓ subprocess.run()
Rust CLI (rust_scanner --url http://...)
    ↓ stdout JSON
Python 解析結果
```

**優點**:
- ✅ 語言隔離（故障不互相影響）
- ✅ 簡單部署（獨立二進制）
- ✅ 符合 TypeScript/Go 引擎模式

**缺點**:
- ❌ 進程啟動開銷（~50ms）
- ❌ JSON 序列化開銷
- ❌ 需要編譯二進制
- ❌ 每次調用都重新初始化

**適用場景**: 
- 調用頻率低
- 需要語言隔離
- 團隊熟悉 CLI 模式

---

### 方案 2: PyO3 原生綁定（推薦）⭐

**架構**:
```
Python worker.py
    ↓ import rust_scanner (FFI)
Rust 函數（在同一進程）
    ↓ 直接返回對象
Python 使用結果
```

**實現範例**:
```rust
// Cargo.toml
[lib]
name = "rust_scanner"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }

// lib.rs
use pyo3::prelude::*;

#[pyfunction]
fn scan_phase0(url: String, timeout: u64) -> PyResult<String> {
    let scanner = EndpointDiscoverer::new();
    let results = scanner.discover(&url).await;
    Ok(serde_json::to_string(&results).unwrap())
}

#[pymodule]
fn rust_scanner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_phase0, m)?)?;
    Ok(())
}
```

```python
# worker.py
import rust_scanner

result_json = rust_scanner.scan_phase0("http://localhost:3000", 600)
result = json.loads(result_json)
```

**優點**:
- ✅ **零進程開銷**（同進程調用，~0.1ms）
- ✅ **極致性能**（10-100x 快於 CLI）
- ✅ **狀態保持**（可緩存初始化）
- ✅ **符合 Python 引擎架構**
- ✅ **開發便捷**（Python 調試 + Rust 性能）

**缺點**:
- ❌ 編譯複雜（需要 maturin）
- ❌ 平台依賴（.pyd on Windows, .so on Linux）
- ❌ Rust panic 會崩潰 Python 進程

**適用場景**: 
- 高頻調用（Phase0 必執行）✅
- 追求極致性能 ✅
- 統一 Python 架構 ✅

---

### 方案 3: 獨立 RabbitMQ Worker

**架構**:
```
Core 下令 → RabbitMQ
    ↓
Rust Worker (獨立進程) 直接監聽 MQ
    ↓
Rust Worker 直接回傳結果到 MQ
```

**優點**:
- ✅ **完全解耦**（可獨立部署/重啟）
- ✅ **水平擴展**（啟動多個 Rust Worker）
- ✅ **故障隔離**（崩潰不影響 Python）
- ✅ **已實現**（現有 main.rs 就是這個架構）

**缺點**:
- ❌ **架構不一致**（其他引擎都由 Python 調用）
- ❌ 部署複雜（需管理額外進程）
- ❌ 監控困難（Python 無法控制 Rust 狀態）
- ❌ 不符合"引擎"定義（應該是子模組，不是獨立服務）

**適用場景**: 
- 微服務架構
- Rust 團隊獨立維護
- 需要獨立擴展

---

## 🎯 最終建議：方案 2 (PyO3)

### 原因：

1. **符合系統架構**
   - Python 引擎：Python 直接調用
   - TypeScript 引擎：Python 調用 Node.js
   - Go 引擎：Python 調用 Go 二進制
   - **Rust 引擎**：Python 直接調用 Rust（FFI）✅

2. **性能最優**
   ```
   CLI 子進程: 50ms 啟動 + 10ms 掃描 = 60ms
   PyO3 綁定:  0.1ms 調用 + 10ms 掃描 = 10.1ms (6x 快)
   ```

3. **Phase0 特性**
   - Phase0 是**必執行**（每次掃描都要）
   - 需要在 10 分鐘內完成
   - 高頻調用 → 進程開銷不可接受

4. **開發維護**
   - Python 統一入口（worker.py）
   - Rust 性能加速（scan 邏輯）
   - 調試方便（Python 棧追踪）

---

## 🔧 PyO3 實現步驟

### Step 1: 修改 Cargo.toml

```toml
[package]
name = "rust-scanner"

[lib]
name = "rust_scanner"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
# ... 其他依賴
```

### Step 2: 創建 lib.rs

```rust
use pyo3::prelude::*;

mod endpoint_discovery;
mod js_analyzer;
mod attack_surface;

#[pyfunction]
fn scan_phase0(url: String, timeout: u64) -> PyResult<String> {
    // Phase0 掃描邏輯
    let discoverer = endpoint_discovery::EndpointDiscoverer::new();
    let endpoints = discoverer.discover(&url);
    
    let result = serde_json::json!({
        "endpoints": endpoints,
        "status": "completed"
    });
    
    Ok(result.to_string())
}

#[pymodule]
fn rust_scanner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_phase0, m)?)?;
    Ok(())
}
```

### Step 3: 安裝 maturin

```bash
pip install maturin
maturin develop  # 開發模式編譯
```

### Step 4: Python 調用

```python
# worker.py
import rust_scanner

result = rust_scanner.scan_phase0("http://localhost:3000", 600)
```

---

## ❓ 常見問題

### Q: bridge.py 能否變成二進制？
**A**: 不需要！bridge.py 只是橋接層，使用 PyO3 後可直接刪除。

### Q: Rust 一定要用二進制驅動嗎？
**A**: 不一定！三種方式都可行：
- CLI 二進制（方案1）
- Python 擴展（方案2，推薦）
- 獨立服務（方案3）

### Q: 為何 TypeScript/Go 用 CLI？
**A**: 
- TypeScript: Node.js 本身就是解釋器，CLI 自然
- Go: 沒有 Python 綁定，CLI 最簡單
- Rust: **有 PyO3**，可以做得更好！

### Q: 如果 Rust panic 怎麼辦？
**A**: 
1. 使用 `catch_unwind` 捕獲 panic
2. 返回 Err() 給 Python
3. Python 記錄錯誤並降級到其他引擎

---

**結論**: 使用 **PyO3** 實現 Rust 引擎，獲得最佳性能和架構一致性。
