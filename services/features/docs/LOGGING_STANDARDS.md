# Features 模組 - 日誌記錄標準

**版本**: 1.0  
**最後更新**: 2025-10-25  
**適用範圍**: services/features/ 所有模組

---

## 📋 目錄

- [總覽](#總覽)
- [日誌級別使用規範](#日誌級別使用規範)
- [日誌格式標準](#日誌格式標準)
- [結構化日誌](#結構化日誌)
- [最佳實踐](#最佳實踐)
- [實例參考](#實例參考)

---

## 🎯 總覽

### **核心原則**

1. **一致性**: 所有 Features 模組使用統一的日誌標準
2. **可觀測性**: 日誌應提供足夠信息用於問題追蹤和性能分析
3. **結構化**: 使用結構化日誌便於自動化分析和告警
4. **性能**: 避免過度日誌影響系統性能

### **使用 aiva_common 日誌工具**

```python
# ✅ 推薦：使用 aiva_common 提供的統一日誌工具
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# ❌ 避免：直接使用 logging 模組
import logging
logger = logging.getLogger(__name__)  # 不推薦，除非有特殊需求
```

---

## 📊 日誌級別使用規範

### **DEBUG** - 調試信息

**使用場景**:
- 詳細的函數執行流程
- 變數值追蹤
- 性能計時詳情
- 開發期間的臨時輸出

**示例**:
```python
logger.debug("Starting XSS detection", extra={
    "task_id": task.task_id,
    "target_url": target_url,
    "payload_count": len(payloads)
})

logger.debug(f"Fetched external script: {script_url}")
logger.debug(f"DOM analysis completed in {elapsed:.2f}s")
```

**何時使用**:
- 生產環境默認**不啟用**
- 用於開發和測試階段
- 幫助開發者理解代碼執行路徑

---

### **INFO** - 一般信息

**使用場景**:
- 系統啟動/關閉
- 重要功能執行開始/完成
- 配置加載
- 正常業務流程的關鍵步驟

**示例**:
```python
logger.info("SmartDetectionManager initialized")
logger.info(f"Registered detector: {name}")
logger.info(f"Starting Client-Side Auth Bypass check for task {task_id} on {url}")
logger.info(f"Task {task_id} completed successfully: {findings_count} findings")
```

**何時使用**:
- 生產環境**默認啟用**
- 記錄正常運行的重要事件
- 便於追蹤系統運行狀態

---

### **WARNING** - 警告信息

**使用場景**:
- 可恢復的錯誤
- 配置問題
- 性能降級
- 預期外但不影響核心功能的情況

**示例**:
```python
logger.warning(f"Detector '{name}' already registered, overwriting")
logger.warning(f"No JavaScript found on {target_url}")
logger.warning(f"Failed to fetch external script {src}: {e}")

# 結構化警告
logger.warning(
    "Payload attempt failed",
    extra={
        "task_id": task.task_id,
        "payload": error.payload,
        "vector": error.vector,
        "attempts": error.attempts
    }
)
```

**何時使用**:
- 非致命性問題
- 需要關注但不需要立即處理
- 可能影響結果質量但不影響系統運行

---

### **ERROR** - 錯誤信息

**使用場景**:
- 功能執行失敗
- 數據驗證失敗
- 外部服務調用失敗
- 需要立即關注的問題

**示例**:
```python
logger.error(f"Failed to fetch or parse scripts from {url}: {e}")
logger.error(f"Regex error in pattern {pattern_name}: {e}")

# 包含異常堆疊
logger.exception(
    "Failed to provision blind XSS payload",
    extra={"task_id": task.task_id}
)

# 結構化錯誤
logger.error(
    "Detector execution failed",
    extra={
        "detector_name": name,
        "error": str(e),
        "task_id": task_id,
        "execution_time": elapsed
    }
)
```

**何時使用**:
- 任務執行失敗
- 數據處理錯誤
- 需要告警和後續處理的問題

---

### **CRITICAL** - 嚴重錯誤

**使用場景**:
- 系統級故障
- 數據損壞
- 安全問題
- 需要立即干預的緊急情況

**示例**:
```python
logger.critical(
    "Database connection lost",
    extra={
        "service": "features_module",
        "impact": "all_detections_stopped"
    }
)

logger.critical(
    "Security violation detected",
    extra={
        "violation_type": "unauthorized_access",
        "source_ip": source_ip
    }
)
```

**何時使用**:
- 影響整個模組或系統的問題
- 需要立即告警和人工介入
- 生產環境慎用（頻繁 CRITICAL 會導致告警疲勞）

---

## 📝 日誌格式標準

### **基本格式**

```python
# 簡單訊息
logger.info("Operation completed successfully")

# 帶變數的訊息（使用 f-string）
logger.info(f"Processed {count} items in {elapsed:.2f}s")

# 結構化訊息（使用 extra 參數）
logger.info(
    "Detection completed",
    extra={
        "task_id": task.task_id,
        "findings": len(findings),
        "execution_time": elapsed
    }
)
```

### **日誌訊息撰寫原則**

1. **清晰簡潔**: 訊息應該一眼看出發生了什麼
2. **包含上下文**: 提供足夠的資訊定位問題
3. **避免敏感信息**: 不要記錄密碼、token、個人信息
4. **使用英文**: 核心訊息使用英文，描述可用中文

```python
# ✅ 好的日誌
logger.info("XSS detection started", extra={"task_id": task_id, "url": url})

# ❌ 不好的日誌
logger.info("開始了")  # 訊息不明確
logger.info(f"Token: {user_token}")  # 洩露敏感信息
logger.info("Something happened")  # 缺少上下文
```

---

## 🏗️ 結構化日誌

### **使用 `extra` 參數**

結構化日誌便於自動化分析、搜索和告警。

```python
logger.info(
    "HTTP request completed",
    extra={
        # 請求標識
        "task_id": task.task_id,
        "request_id": request.id,
        
        # 請求詳情
        "method": "POST",
        "url": target_url,
        "status_code": response.status_code,
        
        # 性能指標
        "response_time": elapsed,
        "payload_size": len(payload),
        
        # 業務指標
        "vulnerability_found": bool(findings),
        "confidence": finding.confidence if findings else None
    }
)
```

### **常用結構化字段**

| 字段類別 | 字段名稱 | 說明 | 示例 |
|---------|---------|------|------|
| **標識符** | `task_id` | 任務ID | `"task_123"` |
|  | `session_id` | 會話ID | `"sess_abc"` |
|  | `finding_id` | 發現ID | `"finding_456"` |
| **目標信息** | `target_url` | 目標URL | `"https://example.com"` |
|  | `target_param` | 目標參數 | `"id"` |
| **性能指標** | `execution_time` | 執行時間(秒) | `1.23` |
|  | `payload_count` | Payload數量 | `50` |
|  | `request_count` | 請求數量 | `100` |
| **結果信息** | `findings` | 發現數量 | `3` |
|  | `severity` | 嚴重程度 | `"high"` |
|  | `confidence` | 信心度 | `"high"` |
| **錯誤信息** | `error` | 錯誤訊息 | `"Connection timeout"` |
|  | `error_type` | 錯誤類型 | `"NetworkError"` |
|  | `attempts` | 重試次數 | `3` |

---

## ✅ 最佳實踐

### **1. 日誌級別選擇決策樹**

```
是否影響核心功能？
├─ 是 → 是否可恢復？
│        ├─ 否 → CRITICAL
│        └─ 是 → ERROR
└─ 否 → 是否預期行為？
         ├─ 否 → WARNING
         └─ 是 → 是否重要？
                  ├─ 是 → INFO
                  └─ 否 → DEBUG
```

### **2. 性能考量**

```python
# ✅ 好：避免在熱路徑中使用字符串格式化
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Complex calculation: {expensive_operation()}")

# ✅ 更好：使用惰性求值
logger.debug("Complex calculation: %s", lambda: expensive_operation())

# ❌ 避免：即使 DEBUG 未啟用也會執行
logger.debug(f"Complex calculation: {expensive_operation()}")
```

### **3. 異常記錄**

```python
# ✅ 使用 exception() 自動包含堆疊追蹤
try:
    risky_operation()
except Exception as e:
    logger.exception("Operation failed", extra={"task_id": task_id})

# ✅ 選擇性記錄特定異常
try:
    risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}", extra={"value": bad_value})
except Exception as e:
    logger.exception("Unexpected error")
    raise  # 重新拋出
```

### **4. 避免日誌洪水**

```python
# ❌ 避免：在循環中記錄每個項目
for item in items:
    logger.info(f"Processing {item}")  # 可能產生數千條日誌

# ✅ 推薦：聚合後記錄
logger.info(f"Processing {len(items)} items")
# 或者使用 DEBUG 級別記錄詳情
for item in items:
    logger.debug(f"Processing {item}")
```

---

## 📚 實例參考

### **SmartDetectionManager 範例**

```python
class SmartDetectionManager:
    def __init__(self) -> None:
        self._detectors: Dict[str, DetectorFunc] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        logger.info("SmartDetectionManager initialized")  # ✅ INFO: 初始化
    
    def register(self, name: str, fn: DetectorFunc) -> None:
        if name in self._detectors:
            logger.warning(  # ✅ WARNING: 覆蓋現有註冊
                f"Detector '{name}' already registered, overwriting"
            )
        
        self._detectors[name] = fn
        logger.info(f"Registered detector: {name}")  # ✅ INFO: 成功註冊
    
    def run_detector(
        self, 
        name: str, 
        input_data: Dict[str, Any]
    ) -> DetectionResult:
        start_time = time.time()
        
        logger.debug(  # ✅ DEBUG: 執行詳情
            f"Running detector: {name}",
            extra={"input_keys": list(input_data.keys())}
        )
        
        try:
            result = self._detectors[name](input_data)
            execution_time = time.time() - start_time
            
            logger.info(  # ✅ INFO: 成功完成
                f"Detector '{name}' completed",
                extra={
                    "execution_time": execution_time,
                    "success": True
                }
            )
            
            return DetectionResult(
                detector_name=name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(  # ✅ ERROR: 執行失敗
                f"Detector '{name}' failed",
                extra={
                    "error": str(e),
                    "execution_time": execution_time,
                    "input_data": input_data
                }
            )
            
            return DetectionResult(
                detector_name=name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
```

### **Worker 範例**

```python
class ClientSideAuthBypassWorker(FeatureBaseWorker):
    def __init__(self, ...):
        super().__init__(...)
        logger.info("ClientSideAuthBypassWorker initialized.")  # ✅ INFO
    
    async def execute_task(self, payload: FunctionTaskPayload):
        task_id = payload.task_id
        target_url = payload.target.url
        
        logger.info(  # ✅ INFO: 任務開始
            f"Starting Client-Side Auth Bypass check for task {task_id} on {target_url}"
        )
        
        try:
            # 步驟 1
            logger.debug("Fetching page content and scripts...")  # ✅ DEBUG
            html_content, scripts = await self._fetch_page_and_scripts(target_url)
            
            if not scripts:
                logger.warning(f"No JavaScript found on {target_url}")  # ✅ WARNING
                return FunctionTaskResult(...)
            
            # 步驟 2
            logger.debug("Analyzing JavaScript for auth bypass patterns...")  # ✅ DEBUG
            issues = await self.js_analyzer.analyze(target_url, scripts)
            
            # 記錄結果
            for issue in issues:
                logger.warning(  # ✅ WARNING: 發現潛在問題
                    f"Potential client-side auth issue found: {issue['description']}"
                )
            
            logger.info(  # ✅ INFO: 任務完成
                f"Task {task_id} completed: {len(issues)} issues found"
            )
            
        except Exception as e:
            logger.error(  # ✅ ERROR: 任務失敗
                f"Task {task_id} failed: {str(e)}",
                extra={
                    "target_url": target_url,
                    "error_type": type(e).__name__
                }
            )
            raise
```

---

## 🔍 檢查清單

在提交代碼前，確保：

- [ ] 使用 `get_logger(__name__)` 獲取 logger
- [ ] 日誌級別使用正確（INFO/WARNING/ERROR/DEBUG）
- [ ] 重要操作使用結構化日誌（`extra` 參數）
- [ ] 錯誤包含足夠的上下文信息
- [ ] 沒有記錄敏感信息（密碼、token、個人數據）
- [ ] 避免在熱路徑中過度日誌
- [ ] 使用異常處理時正確使用 `logger.exception()`

---

## 📖 相關文件

- [DEVELOPMENT_STANDARDS.md](../DEVELOPMENT_STANDARDS.md) - 開發標準
- [README.md](../README.md) - 模組總覽
- [aiva_common 日誌工具文檔](../../aiva_common/utils/logger.py)

---

**版本歷史**:
- v1.0 (2025-10-25): 初始版本，定義日誌標準規範

**維護團隊**: AIVA Features Architecture Team
