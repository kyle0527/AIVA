# Performance - 性能監控與優化

**導航**: [← 返回 Service Backbone](../README.md) | [← 返回 AIVA Core](../../README.md) | [← 返回項目根目錄](../../../../../README.md)

## 📑 目錄

- [📋 概述](#-概述)
- [📂 文件結構](#-文件結構)
- [🎯 核心功能](#-核心功能)
  - [unified_memory_manager.py](#unified_memory_managerpy-439-行-)
  - [monitoring.py](#monitoringpy-140-行)
  - [parallel_processor.py](#parallel_processorpy-54-行)
- [💾 內存管理策略](#-內存管理策略)
- [📊 性能監控](#-性能監控)
- [⚡ 並行處理](#-並行處理)
- [📚 相關模組](#-相關模組)
- [🔧 配置最佳實踐](#-配置最佳實踐)

---

## 📋 概述

**定位**: 性能監控、內存管理和並行處理  
**狀態**: ✅ 已實現  
**文件數**: 3 個 Python 文件 (633 行)

## 📂 文件結構

```
performance/
├── unified_memory_manager.py (439 行) ⭐ - 統一內存管理器
├── monitoring.py (140 行) - 性能監控
├── parallel_processor.py (54 行) - 並行處理器
├── __init__.py
└── README.md (本文檔)
```

## 🎯 核心功能

### unified_memory_manager.py (439 行) ⭐

**職責**: 統一的內存管理和優化

**主要類/函數**:
- `UnifiedMemoryManager` - 內存管理器主類
- `allocate(size)` - 內存分配
- `deallocate(ptr)` - 內存釋放
- `get_memory_stats()` - 獲取內存統計
- `optimize()` - 內存優化和垃圾回收

**管理的內存類型**:
- 向量存儲內存 (RAG)
- 模型參數內存 (Neural)
- 任務數據緩存
- 臨時計算結果

**使用範例**:
```python
from aiva_core.service_backbone.performance import UnifiedMemoryManager

memory_mgr = UnifiedMemoryManager(
    max_memory_mb=2048,
    enable_auto_cleanup=True
)

# 分配內存
buffer = memory_mgr.allocate(size_mb=100, label="scan_cache")

# 獲取內存統計
stats = memory_mgr.get_memory_stats()
print(f"已使用: {stats['used_mb']} MB")
print(f"可用: {stats['available_mb']} MB")

# 觸發優化
memory_mgr.optimize()
```

**內存策略**:
- ✅ 內存池管理
- ✅ 自動垃圾回收
- ✅ 內存洩漏檢測
- ✅ OOM 預防機制

---

### monitoring.py (140 行)

**職責**: 實時性能監控和指標收集

**監控指標**:
| 類別 | 指標 | 描述 |
|------|------|------|
| **CPU** | usage, load_avg | CPU 使用率和負載 |
| **Memory** | used, available, swap | 內存使用情況 |
| **Disk** | read_bytes, write_bytes | 磁盤 I/O |
| **Network** | sent_bytes, recv_bytes | 網絡流量 |
| **Task** | queue_size, processing_time | 任務執行情況 |

**使用範例**:
```python
from aiva_core.service_backbone.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# 開始監控
monitor.start(interval=5)  # 每 5 秒收集一次

# 獲取當前指標
metrics = monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_percent']}%")
print(f"Memory: {metrics['memory_used_mb']} MB")

# 設置告警
monitor.set_alert(
    metric="cpu_percent",
    threshold=80,
    callback=lambda: send_alert("High CPU usage")
)
```

**告警機制**:
```python
# 自動告警配置
monitor.configure_alerts({
    "cpu_percent": {"threshold": 80, "action": "warn"},
    "memory_percent": {"threshold": 90, "action": "critical"},
    "queue_size": {"threshold": 1000, "action": "scale"}
})
```

---

### parallel_processor.py (54 行)

**職責**: 並行任務處理和加速

**主要功能**:
- 多進程並行處理
- 多線程並行處理
- 異步任務處理
- 自動負載均衡

**使用範例**:
```python
from aiva_core.service_backbone.performance import ParallelProcessor

processor = ParallelProcessor(max_workers=8)

# 並行處理任務列表
results = processor.map(
    func=scan_target,
    items=["site1.com", "site2.com", "site3.com"],
    mode="process"  # 'process' or 'thread'
)

# 異步並行處理
async def scan_all_targets(targets):
    processor = ParallelProcessor()
    return await processor.async_map(scan_target, targets)
```

## 📊 性能優化策略

### 1. 內存優化

```python
# 自動內存優化
memory_mgr.configure({
    "gc_threshold": 0.8,  # 80% 時觸發 GC
    "cache_size_mb": 512,
    "enable_compression": True
})
```

### 2. 並行優化

```python
# 根據 CPU 核心數自動配置
import os
cpu_count = os.cpu_count()

processor = ParallelProcessor(
    max_workers=cpu_count * 2,
    chunk_size="auto"
)
```

### 3. 監控優化

```python
# 分級監控頻率
monitor = PerformanceMonitor()
monitor.set_intervals({
    "critical": 1,   # 1 秒
    "warning": 5,    # 5 秒
    "normal": 30     # 30 秒
})
```

## 📈 性能指標報表

### 實時儀表板

```python
from aiva_core.service_backbone.performance import PerformanceDashboard

dashboard = PerformanceDashboard(monitor)
dashboard.serve(port=8080)  # http://localhost:8080
```

**顯示內容**:
- CPU/內存使用趨勢圖
- 任務處理吞吐量
- 平均響應時間
- 錯誤率統計

### 性能報告導出

```python
# 生成性能報告
report = monitor.generate_report(
    period="last_24h",
    format="json"
)

# 導出為 JSON/CSV/PDF
monitor.export_report("performance_report.pdf")
```

## 🔧 配置最佳實踐

### 生產環境配置

```python
# 生產環境推薦配置
config = {
    "memory": {
        "max_mb": 4096,
        "gc_threshold": 0.75,
        "enable_swap": False
    },
    "monitoring": {
        "enabled": True,
        "interval": 10,
        "retention_days": 30
    },
    "parallel": {
        "max_workers": cpu_count * 2,
        "enable_profiling": True
    }
}
```

### 開發環境配置

```python
# 開發環境配置
config = {
    "memory": {
        "max_mb": 1024,
        "gc_threshold": 0.8
    },
    "monitoring": {
        "enabled": True,
        "interval": 5,
        "verbose": True
    },
    "parallel": {
        "max_workers": 4
    }
}
```

## 📚 相關模組

- [coordination](../coordination/README.md) - 資源協調
- [messaging](../messaging/README.md) - 異步處理
- [storage](../storage/README.md) - 數據持久化

## 🚨 告警閾值建議

| 指標 | 警告 | 嚴重 | 行動 |
|------|------|------|------|
| **CPU** | 70% | 90% | 限流/擴容 |
| **Memory** | 80% | 95% | 清理/擴容 |
| **Queue Size** | 500 | 1000 | 增加工作者 |
| **Response Time** | 2s | 5s | 優化/快取 |

---

## 🔨 aiva_common 修復規範

> **核心原則**: 本模組必須嚴格遵循 [`services/aiva_common`](../../../../aiva_common/README.md#-開發指南) 的修復規範。

**完整規範**: [aiva_common 開發指南](../../../../aiva_common/README.md#-開發指南)

### 性能監控特別注意

```python
# ✅ 正確：使用標準配置
from aiva_common.config import UnifiedConfig

class PerformanceConfig(UnifiedConfig):
    monitoring_interval: int = 60
    alert_threshold: float = 0.8

# ✅ 合理的性能專屬枚舉
class MetricType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
```

📖 **完整規範**: [aiva_common 修復指南](../../../../aiva_common/README.md#-開發規範與最佳實踐)

---

**文檔版本**: v1.0  
**最後更新**: 2025-11-16  
**維護者**: Service Backbone 團隊
