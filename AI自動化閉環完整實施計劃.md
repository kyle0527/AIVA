# 🚀 AIVA AI 自動化閉環完整實施計劃

> **目標**: 實現 AI 通過一個指令自動完成內外閉環，無需人工逐個啟動腳本  
> **預計工作量**: 11-17 天（2-3.5 週）  
> **最後更新**: 2025年11月29日

---

## 📋 目錄

- [核心目標](#核心目標)
- [當前系統狀態](#當前系統狀態)
- [架構設計方案](#架構設計方案)
- [實施計劃](#實施計劃)
  - [Phase 1: Core → Features 調用打通](#phase-1-core--features-調用打通)
  - [Phase 2: Features → Integration 自動觸發](#phase-2-features--integration-自動觸發)
  - [Phase 3: 反饋循環與學習優化](#phase-3-反饋循環與學習優化)
- [詳細實作規範](#詳細實作規範)
- [測試驗證計劃](#測試驗證計劃)
- [風險與應對](#風險與應對)

---

## 🎯 核心目標

### 期望的用戶體驗

```bash
# 用戶只需執行一個命令
python aiva_cli.py --attack "掃描 http://target.com 的 XSS"

# AI 自動完成：
# 1. AICommanderV2 接收任務並規劃
# 2. AttackCoordinator 選擇 ScannerPlugin
# 3. ScannerPlugin 調用 func_xss Feature
# 4. func_xss 執行掃描並返回結果
# 5. XSSCoordinator 自動處理結果，生成內外循環數據
# 6. Core 接收反饋，更新優化策略
# 7. 下次執行時自動使用優化參數

# 輸出：完整的掃描報告 + 優化建議
```

### 核心需求

1. ✅ **統一入口**: AICommanderV2 接收所有任務
2. ✅ **自動調用**: Core 自動調用 Features（不需要手動啟動）
3. ✅ **自動觸發**: Features 完成後自動觸發雙閉環處理
4. ✅ **自動學習**: Core 接收反饋並應用優化建議
5. ✅ **閉環循環**: 持續優化，每次執行都比上次更好

---

## 📊 當前系統狀態

### ✅ 已完成的組件（87%）

| 組件 | 完成度 | 狀態 | 位置 |
|------|--------|------|------|
| **AICommanderV2** | 100% | ✅ 完整 | `services/core/aiva_core/task_planning/ai_commander_v2.py` |
| **AttackCoordinator** | 100% | ✅ 完整 | `services/core/aiva_core/task_planning/coordinators/attack_coordinator.py` |
| **BaseCoordinator** | 100% | ✅ 完整 | `services/integration/coordinators/base_coordinator.py` |
| **XSSCoordinator** | 87% | ✅ 可用 | `services/integration/coordinators/xss_coordinator.py` |
| **MessageBroker** | 100% | ✅ 完整 | `services/core/aiva_core/service_backbone/messaging/message_broker.py` |
| **CoreFeedback** | 100% | ✅ 完整 | `services/integration/coordinators/base_coordinator.py` |

### ❌ 缺失的關鍵環節

| 缺失組件 | 優先級 | 影響 |
|---------|--------|------|
| **FeaturesInvoker** | 🔴 P0 | Core 無法調用 Features |
| **自動觸發機制** | 🔴 P0 | Features 完成後無法自動處理 |
| **FeedbackProcessor** | 🟡 P1 | Core 無法接收和應用優化 |
| **MessageBroker 初始化** | 🟡 P1 | 組件間無法通信 |

---

## 🏗️ 架構設計方案

### 核心設計原則

1. **數據合約優先**: 使用 Pydantic Schema 統一數據格式
2. **異步優先**: 所有組件使用 `async/await`
3. **錯誤隔離**: 單個組件失敗不影響整體
4. **優雅降級**: 組件不可用時返回空結果
5. **參考成功案例**: 複用掃描模組的適配器模式

### 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│  用戶指令: python aiva_cli.py --attack "掃描 XSS"             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  AICommanderV2 (統一入口)                                     │
│  ✅ 已存在                                                    │
│  1. 接收任務描述                                              │
│  2. 識別領域 (ATTACK/DEFENSE/ANALYSIS)                       │
│  3. 檢查優化建議緩存 (新增)                                   │
│  4. 分發給對應 Coordinator                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  AttackCoordinator (任務規劃)                                │
│  ✅ 已存在                                                    │
│  1. 分解任務為子任務                                          │
│  2. 選擇合適的 Plugin (ScannerPlugin)                        │
│  3. 傳遞優化參數 (新增)                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ScannerPlugin (執行入口)                                     │
│  ⚠️ 需修改                                                    │
│  1. 接收任務                                                  │
│  2. 調用 FeaturesInvoker (新增)                              │
│  3. 傳遞參數給 Features                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  FeaturesInvoker (統一調用接口)                               │
│  ❌ 需新建 - P0 優先級                                        │
│                                                              │
│  功能:                                                        │
│  1. 統一調用各語言 Features                                   │
│  2. Python Features: 直接 import 調用                        │
│  3. Rust Features: subprocess 調用                           │
│  4. Go Features: subprocess 調用                             │
│  5. TypeScript Features: subprocess 調用                     │
│                                                              │
│  設計原則:                                                    │
│  • 使用數據合約 (FeatureRequest/FeatureResult)               │
│  • 簡單的 dict mapping (不需要複雜的 Adapter Pattern)        │
│  • 異步調用 (async/await)                                    │
│  • 錯誤處理 (try-except)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Features (實際執行)                                          │
│  ✅ 已存在                                                    │
│  • func_xss: XSS 掃描                                        │
│  • func_sqli: SQL 注入                                       │
│  • func_ssrf: SSRF 檢測                                      │
│  • ... 等                                                    │
│                                                              │
│  執行掃描並返回 FeatureResult                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  自動觸發 Integration Coordinator                             │
│  ❌ 需實現 - P0 優先級                                        │
│                                                              │
│  方案 A (推薦): 直接調用                                      │
│    ScannerPlugin 收到 Feature 結果後                         │
│    直接調用 XSSCoordinator.collect_result()                  │
│                                                              │
│  方案 B: MessageBroker                                       │
│    Feature 完成後發送消息到 MQ                                │
│    Coordinator 監聽隊列並處理                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  XSSCoordinator (雙閉環處理)                                  │
│  ✅ 87% 完成                                                  │
│                                                              │
│  1. collect_result(feature_result)                          │
│     • 驗證結果                                                │
│     • 提取內循環數據 (payload 效率、成功模式)                  │
│     • 提取外循環數據 (漏洞統計、Bug Bounty 評估)               │
│     • 生成 CoreFeedback                                      │
│                                                              │
│  2. _send_feedback_to_core(feedback)                        │
│     • 通過 MessageBroker 發送反饋 (需修改)                    │
│     • 發送到 "coordinator.feedback" 隊列                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  FeedbackProcessor (反饋處理)                                │
│  ❌ 需新建 - P1 優先級                                        │
│                                                              │
│  功能:                                                        │
│  1. 監聽 "coordinator.feedback" 隊列                         │
│  2. 接收 CoreFeedback                                        │
│  3. 提取優化建議:                                             │
│     • recommended_concurrency                               │
│     • recommended_timeout_ms                                │
│     • successful_patterns (有效 payload)                    │
│     • strategy_adjustments                                  │
│  4. 緩存優化建議 (按 feature_module 分類)                     │
│  5. 更新執行策略                                              │
│                                                              │
│  集成到 AICommanderV2:                                       │
│  • initialize() 時啟動監聽                                   │
│  • execute_task() 前查詢優化建議                             │
│  • 自動應用優化參數                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  閉環完成 - 持續優化                                          │
│                                                              │
│  下次用戶執行相同任務時:                                       │
│  1. AICommanderV2 檢查優化建議緩存                           │
│  2. 自動使用優化參數 (並發數、timeout、payload)               │
│  3. 執行效率更高、成功率更高                                   │
│  4. 生成新的優化建議                                          │
│  5. 持續進化                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📅 實施計劃

### Phase 1: Core → Features 調用打通 (4-5 天)

**目標**: 讓 AI 能夠通過一個指令執行到底

**調整說明**: 
- ❌ 取消新增數據合約（原計劃 1 天）
- ✅ 直接使用現有 `FeatureResult`（節省維護成本）
- ✅ 總工時從 5-7 天減少到 4-5 天

#### 任務 1.1: 使用現有數據合約 (0.5 天)

**決策**: 直接使用 `services/features/base/result_schema.py` 的現有合約

**理由**:
1. ✅ **數據量小** - 幾 KB 完全可接受，不需要優化
2. ✅ **功能完整** - 已有完整的 Finding、severity、confidence
3. ✅ **無需維護兩套** - 避免格式轉換和維護成本
4. ✅ **全系統統一** - Core、Features、Coordinator 都用同一套

**使用的合約**:

```python
# 直接從現有模組導入
from services.features.base import (
    FeatureResult,           # 功能執行結果
    Finding,                 # 漏洞發現
    FeatureExecutionStatus,  # 執行狀態
    FindingSeverity,         # 嚴重程度
    FindingConfidence        # 置信度
)
```

**FeatureResult 已有的完整功能**:
```python
class FeatureResult(BaseModel):
    feature_name: str               # 功能名稱
    task_id: str                    # 任務 ID
    status: FeatureExecutionStatus  # 執行狀態
    execution_time: float           # 執行時間
    findings: List[Finding]         # 漏洞列表
    statistics: Dict[str, Any]      # 統計信息
    error_message: Optional[str]    # 錯誤信息
    metadata: Optional[Dict[str, Any]]  # 元數據
    
    # 已有的便捷方法
    @property
    def has_findings(self) -> bool
    
    @property
    def critical_findings_count(self) -> int
    
    @property
    def high_findings_count(self) -> int
    
    def get_findings_by_severity(self, severity: FindingSeverity)
    def get_findings_by_confidence(self, confidence: FindingConfidence)
```

**需要新增的補充**:

```python
# services/features/base/result_schema.py 新增
class FeatureType(str, Enum):
    """Feature 類型枚舉"""
    XSS = "xss"
    SQLI = "sqli"
    SSRF = "ssrf"
    CSRF = "csrf"
    IDOR = "idor"
    LFI = "lfi"
    RCE = "rce"
    XXE = "xxe"
    # ... 更多類型
```

**驗收標準**:
- ✅ 確認現有 FeatureResult 滿足所有需求
- ✅ 新增 FeatureType 枚舉到 result_schema.py
- ✅ 更新 `__init__.py` 導出新枚舉

---

#### 任務 1.2: 實作 FeaturesInvoker (2-3 天)

**文件**: `services/core/aiva_core/plugins/features_invoker.py`

```python
"""
Features 統一調用接口

設計原則：
1. 使用現有 FeatureResult 合約（統一全系統）
2. 簡單的 dict mapping（不需要複雜的 Adapter Pattern）
3. 異步調用（async/await）
4. 支持多語言 Features（Python/Rust/Go/TypeScript）
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from services.features.base import (
    FeatureResult,
    FeatureType,
    FeatureExecutionStatus
)
from services.aiva_common.utils import get_logger

logger = get_logger("FeaturesInvoker")


class FeaturesInvoker:
    """Features 統一調用接口"""
    
    def __init__(self):
        # Feature 配置映射
        self.features_config = {
            FeatureType.XSS: {
                "type": "python",
                "module": "services.features.func_xss.xss_detector",
                "class": "XSSDetector"
            },
            FeatureType.SQLI: {
                "type": "python",
                "module": "services.features.func_sqli.sqli_detector",
                "class": "SQLiDetector"
            },
            FeatureType.SSRF: {
                "type": "rust",
                "binary": "services/features/function_ssrf_rust/target/release/ssrf_scanner"
            },
            FeatureType.CSRF: {
                "type": "go",
                "binary": "services/features/function_csrf_go/worker"
            },
            # 更多 Features...
        }
        
        # 可用的 Features
        self.available_features = set()
    
    async def initialize(self) -> None:
        """初始化並檢查所有 Features 的可用性"""
        logger.info("🔍 檢查 Features 可用性...")
        
        for feature_type, config in self.features_config.items():
            try:
                if config["type"] == "python":
                    is_available = await self._check_python_feature(config)
                elif config["type"] in ["rust", "go", "typescript"]:
                    is_available = await self._check_binary_feature(config)
                else:
                    is_available = False
                
                if is_available:
                    self.available_features.add(feature_type)
                    logger.info(f"  ✅ {feature_type.value.upper()} Feature 可用")
                else:
                    logger.warning(f"  ❌ {feature_type.value.upper()} Feature 不可用")
                    
            except Exception as e:
                logger.error(f"  ❌ {feature_type.value.upper()} 檢查失敗: {e}")
        
        if not self.available_features:
            logger.warning("⚠️ 警告：沒有任何 Feature 可用！")
        else:
            available_list = [f.value for f in self.available_features]
            logger.info(f"✅ Features 初始化完成，可用: {', '.join(available_list)}")
    
    async def invoke_feature(
        self,
        feature_type: FeatureType,
        target: str,
        task_id: str,
        options: Dict[str, Any] = None
    ) -> FeatureResult:
        """統一調用 Features
        
        Args:
            feature_type: Feature 類型
            target: 目標 URL
            task_id: 任務 ID
            options: 執行選項（可包含優化參數）
        
        Returns:
            FeatureResult: 執行結果（使用統一格式）
        """
        options = options or {}
        
        # 檢查 Feature 是否可用
        if feature_type not in self.available_features:
            return FeatureResult(
                feature_name=feature_type.value,
                task_id=task_id,
                status=FeatureExecutionStatus.ERROR,
                execution_time=0.0,
                findings=[],
                error_message=f"Feature {feature_type.value} not available"
            )
        
        # 獲取 Feature 配置
        config = self.features_config[feature_type]
        
        # 根據類型調用
        try:
            if config["type"] == "python":
                return await self._invoke_python_feature(
                    config, feature_type, target, task_id, options
                )
            elif config["type"] in ["rust", "go", "typescript"]:
                return await self._invoke_binary_feature(
                    config, feature_type, target, task_id, options
                )
            else:
                raise ValueError(f"Unknown feature type: {config['type']}")
        
        except Exception as e:
            logger.error(f"Feature {feature_type.value} 執行失敗: {e}")
            return FeatureResult(
                feature_name=feature_type.value,
                task_id=task_id,
                status=FeatureExecutionStatus.ERROR,
                execution_time=0.0,
                findings=[],
                error_message=str(e)
            )
    
    async def _invoke_python_feature(
        self,
        config: Dict[str, Any],
        feature_type: FeatureType,
        target: str,
        task_id: str,
        options: Dict[str, Any]
    ) -> FeatureResult:
        """調用 Python Feature"""
        import importlib
        
        # 動態導入
        module = importlib.import_module(config["module"])
        feature_class = getattr(module, config["class"])
        
        # 創建實例
        feature_instance = feature_class()
        
        # 執行掃描（直接返回 FeatureResult）
        result = await feature_instance.scan(
            target=target,
            task_id=task_id,
            options=options
        )
        
        return result
    
    async def _invoke_binary_feature(
        self,
        config: Dict[str, Any],
        feature_type: FeatureType,
        target: str,
        task_id: str,
        options: Dict[str, Any]
    ) -> FeatureResult:
        """調用編譯型 Feature（Rust/Go/TypeScript）"""
        import time
        start_time = time.time()
        
        binary_path = config["binary"]
        
        # 構建命令行參數
        cmd = [
            str(binary_path),
            "--target", target,
            "--task-id", task_id,
            "--output", "json"
        ]
        
        # 添加選項
        if "timeout" in options:
            cmd.extend(["--timeout", str(options["timeout"])])
        if "concurrency" in options:
            cmd.extend(["--concurrency", str(options["concurrency"])])
        
        # 執行子進程
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        execution_time = time.time() - start_time
        
        if proc.returncode != 0:
            return FeatureResult(
                feature_name=feature_type.value,
                task_id=task_id,
                status=FeatureExecutionStatus.ERROR,
                execution_time=execution_time,
                findings=[],
                error_message=stderr.decode()
            )
        
        # 解析 JSON 輸出為 FeatureResult
        result_data = json.loads(stdout.decode())
        return FeatureResult(**result_data)
```

**關鍵設計**:
1. ✅ **統一返回格式** - 所有 Features 都返回 `FeatureResult`
2. ✅ **無需格式轉換** - 直接使用全系統統一的合約
3. ✅ **自動觸發 Coordinator** - 通過事件系統（後續 Phase 2）
    
    async def initialize(self) -> None:
        """初始化並檢查所有 Features 的可用性"""
        logger.info("🔍 檢查 Features 可用性...")
        
        for feature_type, config in self.features_config.items():
            try:
                if config["type"] == "python":
                    # Python Features: 嘗試導入
                    is_available = await self._check_python_feature(config)
                elif config["type"] in ["rust", "go", "typescript"]:
                    # 編譯型 Features: 檢查二進制文件
                    is_available = await self._check_binary_feature(config)
                else:
                    is_available = False
                
                if is_available:
                    self.available_features.add(feature_type)
                    logger.info(f"  ✅ {feature_type.value.upper()} Feature 可用")
                else:
                    logger.warning(f"  ❌ {feature_type.value.upper()} Feature 不可用")
                    
            except Exception as e:
                logger.error(f"  ❌ {feature_type.value.upper()} 檢查失敗: {e}")
        
        if not self.available_features:
            logger.warning("⚠️ 警告：沒有任何 Feature 可用！")
        else:
            available_list = [f.value for f in self.available_features]
            logger.info(f"✅ Features 初始化完成，可用: {', '.join(available_list)}")
    
    async def _check_python_feature(self, config: Dict[str, Any]) -> bool:
        """檢查 Python Feature 是否可用"""
        try:
            # 動態導入模組
            module_path = config["module"]
            class_name = config["class"]
            
            # 使用 importlib 動態導入
            import importlib
            module = importlib.import_module(module_path)
            
            # 檢查類是否存在
            if hasattr(module, class_name):
                return True
            return False
        except ImportError:
            return False
    
    async def _check_binary_feature(self, config: Dict[str, Any]) -> bool:
        """檢查編譯型 Feature 是否可用"""
        binary_path = Path(config["binary"])
        return binary_path.exists() and binary_path.is_file()
    
    async def invoke_feature(
        self,
        request: FeatureRequest
    ) -> FeatureResult:
        """統一調用 Features
        
        Args:
            request: Feature 執行請求
        
        Returns:
            FeatureResult: 執行結果
        """
        feature_type = request.feature_type
        
        # 檢查 Feature 是否可用
        if feature_type not in self.available_features:
            return FeatureResult(
                success=False,
                feature_type=feature_type,
                target=str(request.target),
                vulnerabilities=[],
                error=f"Feature {feature_type.value} not available"
            )
        
        # 獲取 Feature 配置
        config = self.features_config[feature_type]
        
        # 根據類型調用
        try:
            if config["type"] == "python":
                return await self._invoke_python_feature(config, request)
            elif config["type"] in ["rust", "go", "typescript"]:
                return await self._invoke_binary_feature(config, request)
            else:
                raise ValueError(f"Unknown feature type: {config['type']}")
        
        except Exception as e:
            logger.error(f"Feature {feature_type.value} 執行失敗: {e}")
            return FeatureResult(
                success=False,
                feature_type=feature_type,
                target=str(request.target),
                vulnerabilities=[],
                error=str(e)
            )
    
    async def _invoke_python_feature(
        self,
        config: Dict[str, Any],
        request: FeatureRequest
    ) -> FeatureResult:
        """調用 Python Feature"""
        import importlib
        
        # 動態導入
        module = importlib.import_module(config["module"])
        feature_class = getattr(module, config["class"])
        
        # 創建實例
        feature_instance = feature_class()
        
        # 執行掃描
        result = await feature_instance.scan(
            target=str(request.target),
            options=request.options
        )
        
        # 確保返回符合 FeatureResult 格式
        if isinstance(result, FeatureResult):
            return result
        elif isinstance(result, dict):
            return FeatureResult(**result)
        else:
            raise ValueError(f"Invalid result type: {type(result)}")
    
    async def _invoke_binary_feature(
        self,
        config: Dict[str, Any],
        request: FeatureRequest
    ) -> FeatureResult:
        """調用編譯型 Feature（Rust/Go/TypeScript）"""
        binary_path = config["binary"]
        
        # 構建命令行參數
        cmd = [
            str(binary_path),
            "--target", str(request.target),
            "--timeout", str(request.options.get("timeout", 10))
        ]
        
        # 添加額外選項
        if request.concurrency:
            cmd.extend(["--concurrency", str(request.concurrency)])
        
        # 執行子進程
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # 等待完成（帶超時）
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=request.options.get("timeout", 10) + 10
            )
        except asyncio.TimeoutError:
            proc.kill()
            return FeatureResult(
                success=False,
                feature_type=request.feature_type,
                target=str(request.target),
                vulnerabilities=[],
                error="Execution timeout"
            )
        
        # 解析 JSON 結果
        if proc.returncode != 0:
            return FeatureResult(
                success=False,
                feature_type=request.feature_type,
                target=str(request.target),
                vulnerabilities=[],
                error=stderr.decode('utf-8', errors='ignore')
            )
        
        # 解析 stdout
        result_dict = json.loads(stdout.decode('utf-8'))
        
        # 轉換為 FeatureResult
        return FeatureResult(**result_dict)
```

**驗收標準**:
- ✅ 支持 Python Features（原生調用）
- ✅ 支持 Rust/Go/TypeScript Features（subprocess 調用）
- ✅ 有完整的錯誤處理
- ✅ 有初始化檢查機制
- ✅ 返回統一的 FeatureResult

---

#### 任務 1.3: 修改 ScannerPlugin (1 天)

**文件**: `services/core/aiva_core/plugins/scanner_plugin.py`

修改重點：

```python
from .features_invoker import FeaturesInvoker
from services.aiva_common.schemas.feature_schemas import (
    FeatureType, FeatureRequest, FeatureResult
)

class ScannerPlugin(AIModulePlugin):
    
    def __init__(self):
        super().__init__()
        # ✅ 添加 FeaturesInvoker
        self.features_invoker = FeaturesInvoker()
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        # 現有初始化邏輯...
        
        # ✅ 初始化 FeaturesInvoker
        await self.features_invoker.initialize()
        
        self.initialized = True
        return True
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行掃描任務"""
        if not self.initialized:
            return AIResult(
                success=False,
                error="Scanner not initialized"
            )
        
        # 提取參數
        target = task.parameters.get("target")
        scan_type = task.parameters.get("scan_type", "xss")
        
        # 映射 scan_type 到 FeatureType
        feature_type_map = {
            "xss": FeatureType.XSS,
            "sqli": FeatureType.SQLI,
            "ssrf": FeatureType.SSRF,
            # ...
        }
        
        feature_type = feature_type_map.get(scan_type.lower())
        if feature_type is None:
            return AIResult(
                success=False,
                error=f"Unknown scan type: {scan_type}"
            )
        
        # ✅ 構建 FeatureRequest
        request = FeatureRequest(
            feature_type=feature_type,
            target=target,
            options=task.parameters.get("options", {}),
            concurrency=task.parameters.get("concurrency"),
            timeout_ms=task.parameters.get("timeout_ms")
        )
        
        # ✅ 調用 FeaturesInvoker
        logger.info(f"🔍 調用 {feature_type.value.upper()} Feature...")
        feature_result = await self.features_invoker.invoke_feature(request)
        
        # ✅ 轉換為 AIResult
        return AIResult(
            success=feature_result.success,
            data=feature_result.dict(),
            task_id=task.task_id,
            execution_time=feature_result.metadata.get("scan_duration", 0.0),
            metrics={
                "vulnerabilities_found": len(feature_result.vulnerabilities),
                "requests_sent": feature_result.metadata.get("requests_sent", 0)
            },
            error=feature_result.error
        )
```

**驗收標準**:
- ✅ ScannerPlugin 集成 FeaturesInvoker
- ✅ 能夠成功調用 Python Features
- ✅ 能夠成功調用編譯型 Features
- ✅ 返回標準的 AIResult

---

#### 任務 1.4: 端到端測試 (1-2 天)

**測試腳本**: `tests/integration/test_core_features_integration.py`

```python
import asyncio
from services.core.aiva_core.task_planning.ai_commander_v2 import AICommanderV2

async def test_xss_scan():
    """測試 XSS 掃描完整流程"""
    commander = AICommanderV2()
    await commander.initialize()
    
    # 執行任務
    result = await commander.execute_task(
        task_description="掃描 XSS 漏洞",
        parameters={
            "target": "http://testphp.vulnweb.com",
            "scan_type": "xss"
        }
    )
    
    print(f"執行結果: {result}")
    assert result["success"] == True
    assert "result" in result

if __name__ == "__main__":
    asyncio.run(test_xss_scan())
```

**驗收標準**:
- ✅ 端到端測試通過
- ✅ AI 能夠通過一個指令執行到底
- ✅ 返回正確的掃描結果

---

### Phase 2: Features → Integration 自動觸發 (3-4 天)

**目標**: Features 完成後自動處理雙閉環數據

#### 任務 2.1: 選擇自動觸發方案 (0.5 天)

**方案 A（推薦）: 直接調用**
- ✅ 簡單直接
- ✅ 無需額外組件
- ✅ 容易調試
- ❌ 耦合度較高

**方案 B: MessageBroker**
- ✅ 解耦合
- ✅ 支持異步處理
- ❌ 需要 RabbitMQ
- ❌ 增加複雜度

**建議**: 先用方案 A，未來可重構為方案 B

---

#### 任務 2.2: 實現直接調用方案 (1.5 天)

**文件**: `services/core/aiva_core/plugins/scanner_plugin.py`

修改重點：

```python
async def execute_task(self, task: AITask) -> AIResult:
    """執行掃描任務"""
    # ... 前面的代碼不變
    
    # 調用 FeaturesInvoker
    feature_result = await self.features_invoker.invoke_feature(request)
    
    # ✅ 新增：自動觸發 Integration Coordinator
    if feature_result.success:
        await self._trigger_integration_coordinator(
            feature_type=request.feature_type,
            feature_result=feature_result,
            task_id=task.task_id
        )
    
    # 返回結果
    return AIResult(...)

async def _trigger_integration_coordinator(
    self,
    feature_type: FeatureType,
    feature_result: FeatureResult,
    task_id: str
) -> None:
    """觸發 Integration Coordinator 處理雙閉環數據"""
    try:
        # 根據 feature_type 選擇對應的 Coordinator
        coordinator_map = {
            FeatureType.XSS: "services.integration.coordinators.xss_coordinator",
            FeatureType.SQLI: "services.integration.coordinators.sqli_coordinator",
            # ...
        }
        
        coordinator_module = coordinator_map.get(feature_type)
        if coordinator_module is None:
            logger.warning(f"No coordinator for {feature_type.value}")
            return
        
        # 動態導入 Coordinator
        import importlib
        module = importlib.import_module(coordinator_module)
        
        # 獲取 Coordinator 類（假設類名是 XSSCoordinator）
        coordinator_class_name = f"{feature_type.value.upper()}Coordinator"
        coordinator_class = getattr(module, coordinator_class_name)
        
        # 創建實例
        coordinator = coordinator_class()
        
        # 轉換 FeatureResult 為 Coordinator 需要的格式
        result_dict = {
            "task_id": task_id,
            "feature_module": feature_type.value,
            "results": feature_result.dict()
        }
        
        # ✅ 調用 collect_result 處理雙閉環數據
        logger.info(f"🔄 觸發 {coordinator_class_name}.collect_result()...")
        processed = await coordinator.collect_result(result_dict)
        
        logger.info(
            f"✅ 雙閉環處理完成: "
            f"內循環={processed.get('internal_loop', {}).get('optimization_score', 0)}, "
            f"外循環={processed.get('external_loop', {}).get('total_findings', 0)}"
        )
        
    except Exception as e:
        logger.error(f"Integration Coordinator 觸發失敗: {e}")
```

**驗收標準**:
- ✅ Feature 完成後自動調用 Coordinator
- ✅ Coordinator 成功生成內外循環數據
- ✅ 錯誤不影響主流程

---

#### 任務 2.3: 修改 BaseCoordinator 發送反饋 (1 天)

**文件**: `services/integration/coordinators/base_coordinator.py`

修改重點：

```python
async def _send_feedback_to_core(self, feedback: CoreFeedback):
    """發送反饋給 Core"""
    try:
        # ✅ 確保 mq_client 已初始化
        if self.mq_client is None:
            from services.core.aiva_core.service_backbone.messaging.message_broker import MessageBroker
            self.mq_client = MessageBroker()
            await self.mq_client.connect()
        
        # ✅ 發送到 "coordinator.feedback" 隊列
        await self.mq_client.publish_message(
            exchange_name="aiva.feedback",
            routing_key="coordinator.feedback",
            message=feedback.dict()
        )
        
        logger.info(f"✅ Feedback 已發送: {feedback.task_id}")
        
    except Exception as e:
        logger.error(f"發送 Feedback 失敗: {e}")
        # 不拋出異常，避免影響主流程
```

**驗收標準**:
- ✅ Coordinator 能夠成功發送 Feedback
- ✅ MessageBroker 正確初始化
- ✅ 消息發送到正確的隊列

---

#### 任務 2.4: 端到端測試雙閉環 (1 天)

**測試腳本**: `tests/integration/test_dual_loop.py`

```python
async def test_dual_loop_generation():
    """測試雙閉環數據生成"""
    commander = AICommanderV2()
    await commander.initialize()
    
    result = await commander.execute_task(
        task_description="掃描 XSS",
        parameters={
            "target": "http://testphp.vulnweb.com",
            "scan_type": "xss"
        }
    )
    
    # 驗證內循環數據
    assert "internal_loop" in result
    assert "optimization_score" in result["internal_loop"]
    
    # 驗證外循環數據
    assert "external_loop" in result
    assert "total_findings" in result["external_loop"]
    
    print("✅ 雙閉環測試通過")
```

**驗收標準**:
- ✅ 自動生成內循環數據
- ✅ 自動生成外循環數據
- ✅ Feedback 成功發送到 MQ

---

### Phase 3: 反饋循環與學習優化 (3-6 天)

**目標**: Core 能夠接收並應用優化建議

#### 任務 3.1: 實作 FeedbackProcessor (2-3 天)

**文件**: `services/core/aiva_core/task_planning/feedback_processor.py`

```python
"""
反饋處理器 - 接收並應用優化建議

職責:
1. 監聽 coordinator.feedback 隊列
2. 接收 CoreFeedback
3. 提取優化建議
4. 緩存優化建議
5. 更新執行策略
"""

import asyncio
from typing import Dict, Any, Optional
from services.aiva_common.utils import get_logger
from services.integration.coordinators.base_coordinator import CoreFeedback

logger = get_logger("FeedbackProcessor")


class FeedbackProcessor:
    """反饋處理器"""
    
    def __init__(self, ai_commander):
        """初始化
        
        Args:
            ai_commander: AICommanderV2 實例
        """
        self.ai_commander = ai_commander
        self.mq_client = None
        
        # 優化建議緩存（按 feature_module 分類）
        self.optimization_cache: Dict[str, Dict[str, Any]] = {}
        
        # 學習歷史
        self.learning_history: list = []
    
    async def start_listening(self) -> None:
        """啟動反饋監聽"""
        try:
            # 初始化 MessageBroker
            from services.core.aiva_core.service_backbone.messaging.message_broker import MessageBroker
            self.mq_client = MessageBroker()
            await self.mq_client.connect()
            
            # 訂閱 coordinator.feedback 隊列
            await self.mq_client.subscribe(
                queue_name="core_feedback_queue",
                routing_keys=["coordinator.feedback"],
                exchange_name="aiva.feedback",
                callback=self._on_feedback_received
            )
            
            logger.info("✅ FeedbackProcessor 開始監聽...")
            
        except Exception as e:
            logger.error(f"啟動監聽失敗: {e}")
    
    async def _on_feedback_received(self, message) -> None:
        """處理接收到的反饋"""
        try:
            # 解析消息
            import json
            body = message.body.decode('utf-8')
            feedback_dict = json.loads(body)
            
            # 轉換為 CoreFeedback
            feedback = CoreFeedback(**feedback_dict)
            
            logger.info(
                f"📨 收到反饋: {feedback.task_id}, "
                f"Feature={feedback.feature_module.value}, "
                f"Findings={feedback.findings_count}"
            )
            
            # 提取並緩存優化建議
            await self._process_optimization(feedback)
            
            # 確認消息
            await message.ack()
            
        except Exception as e:
            logger.error(f"處理反饋失敗: {e}")
            await message.nack(requeue=False)
    
    async def _process_optimization(self, feedback: CoreFeedback) -> None:
        """處理優化建議"""
        feature_name = feedback.feature_module.value
        optimization = feedback.optimization_suggestions
        
        # 緩存優化建議
        self.optimization_cache[feature_name] = {
            "concurrency": optimization.recommended_concurrency,
            "timeout_ms": optimization.recommended_timeout_ms,
            "successful_patterns": optimization.successful_patterns,
            "failed_patterns": optimization.failed_patterns,
            "strategy_adjustments": optimization.strategy_adjustments,
            "updated_at": feedback.timestamp
        }
        
        # 記錄學習歷史
        self.learning_history.append({
            "timestamp": feedback.timestamp,
            "feature": feature_name,
            "optimization_score": optimization.optimization_score
        })
        
        # 應用到執行策略
        await self._apply_optimization(feature_name, optimization)
        
        logger.info(
            f"✅ 優化建議已應用: {feature_name}, "
            f"併發數={optimization.recommended_concurrency}, "
            f"超時={optimization.recommended_timeout_ms}ms"
        )
    
    async def _apply_optimization(
        self,
        feature_name: str,
        optimization: Any
    ) -> None:
        """應用優化建議到執行策略"""
        # 更新 Plugin 配置
        # 這裡可以通知 AICommanderV2 更新配置
        pass
    
    def get_optimization(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """獲取緩存的優化建議
        
        Args:
            feature_name: Feature 名稱
        
        Returns:
            優化建議字典，如果沒有則返回 None
        """
        return self.optimization_cache.get(feature_name)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """獲取學習統計"""
        return {
            "total_optimizations": len(self.learning_history),
            "features_optimized": len(self.optimization_cache),
            "latest_updates": self.learning_history[-5:] if self.learning_history else []
        }
```

**驗收標準**:
- ✅ 能夠監聽 MQ 隊列
- ✅ 成功解析 CoreFeedback
- ✅ 正確提取優化建議
- ✅ 緩存管理正常

---

#### 任務 3.2: 集成到 AICommanderV2 (1-2 天)

**文件**: `services/core/aiva_core/task_planning/ai_commander_v2.py`

修改重點：

```python
from .feedback_processor import FeedbackProcessor

class AICommanderV2:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # ... 現有初始化
        
        # ✅ 添加 FeedbackProcessor
        self.feedback_processor: Optional[FeedbackProcessor] = None
    
    async def initialize(self) -> bool:
        """初始化 AI Commander"""
        # ... 現有初始化邏輯
        
        # ✅ 初始化並啟動 FeedbackProcessor
        self.feedback_processor = FeedbackProcessor(self)
        await self.feedback_processor.start_listening()
        
        logger.info("✅ FeedbackProcessor 已啟動")
        
        return True
    
    async def execute_task(
        self,
        task_description: str,
        parameters: Dict[str, Any],
        domain: Optional[TaskDomain] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行任務"""
        
        # ✅ 檢查是否有優化建議
        scan_type = parameters.get("scan_type")
        if scan_type and self.feedback_processor:
            optimization = self.feedback_processor.get_optimization(scan_type)
            
            if optimization:
                logger.info(f"🎯 應用優化建議: {scan_type}")
                
                # 應用優化參數
                parameters["concurrency"] = optimization.get("concurrency")
                parameters["timeout_ms"] = optimization.get("timeout_ms")
                parameters["preferred_payloads"] = optimization.get("successful_patterns")
                
                logger.debug(f"優化參數: {parameters}")
        
        # ... 繼續執行任務
        result = await self._execute_task_internal(...)
        
        return result
```

**驗收標準**:
- ✅ FeedbackProcessor 在初始化時啟動
- ✅ execute_task 前檢查優化建議
- ✅ 自動應用優化參數

---

#### 任務 3.3: 完整閉環測試 (1 天)

**測試腳本**: `tests/integration/test_complete_loop.py`

```python
async def test_complete_optimization_loop():
    """測試完整的優化閉環"""
    commander = AICommanderV2()
    await commander.initialize()
    
    # 第一次執行（無優化）
    result1 = await commander.execute_task(
        task_description="掃描 XSS",
        parameters={
            "target": "http://testphp.vulnweb.com",
            "scan_type": "xss"
        }
    )
    
    print(f"第一次執行: {result1}")
    
    # 等待 Feedback 處理
    await asyncio.sleep(5)
    
    # 第二次執行（應該應用優化）
    result2 = await commander.execute_task(
        task_description="掃描 XSS",
        parameters={
            "target": "http://testphp.vulnweb.com",
            "scan_type": "xss"
        }
    )
    
    print(f"第二次執行: {result2}")
    
    # 驗證優化已應用
    stats = commander.feedback_processor.get_learning_stats()
    print(f"學習統計: {stats}")
    
    assert stats["total_optimizations"] > 0
    print("✅ 完整閉環測試通過")
```

**驗收標準**:
- ✅ 第一次執行生成優化建議
- ✅ 第二次執行自動應用優化
- ✅ 執行效率提升
- ✅ 學習統計正確

---

## 📝 詳細實作規範

### 代碼規範

1. **類型註解**: 所有函數都要有類型註解
2. **文檔字符串**: 所有公開方法都要有 docstring
3. **錯誤處理**: 使用 try-except，不要讓異常中斷整個流程
4. **日誌記錄**: 關鍵步驟都要記錄日誌
5. **異步優先**: 所有 I/O 操作使用 async/await

### 測試規範

1. **單元測試**: 每個類都要有單元測試
2. **集成測試**: 每個 Phase 完成後要有集成測試
3. **端到端測試**: 最終要有完整的端到端測試
4. **測試覆蓋率**: 目標 > 80%

### 文檔規範

1. **代碼註釋**: 複雜邏輯要有註釋
2. **API 文檔**: 所有公開接口要有 API 文檔
3. **架構文檔**: 重要設計決策要記錄
4. **變更日誌**: 每次修改要更新 CHANGELOG

---

## 🧪 測試驗證計劃

### Phase 1 驗收測試

```bash
# 測試 FeaturesInvoker
python -m pytest tests/unit/test_features_invoker.py -v

# 測試 ScannerPlugin
python -m pytest tests/unit/test_scanner_plugin.py -v

# 端到端測試
python tests/integration/test_core_features_integration.py
```

**預期結果**:
- ✅ 所有單元測試通過
- ✅ 端到端測試成功
- ✅ 能夠成功調用 Features

### Phase 2 驗收測試

```bash
# 測試雙閉環生成
python tests/integration/test_dual_loop.py

# 測試 Feedback 發送
python tests/integration/test_feedback_sending.py
```

**預期結果**:
- ✅ 自動生成內外循環數據
- ✅ Feedback 成功發送到 MQ
- ✅ Coordinator 正常工作

### Phase 3 驗收測試

```bash
# 測試 FeedbackProcessor
python -m pytest tests/unit/test_feedback_processor.py -v

# 測試完整閉環
python tests/integration/test_complete_loop.py
```

**預期結果**:
- ✅ FeedbackProcessor 正常監聽
- ✅ 優化建議正確緩存
- ✅ 第二次執行自動應用優化
- ✅ 學習統計正確

### 最終驗收

```bash
# 完整系統測試
python tests/e2e/test_full_automation.py
```

**驗收標準**:
1. ✅ 用戶執行一個命令，AI 自動完成所有步驟
2. ✅ 自動調用 Features
3. ✅ 自動生成雙閉環數據
4. ✅ 自動應用優化建議
5. ✅ 持續學習和優化

---

## ⚠️ 風險與應對

### 風險 1: MessageBroker 連接失敗

**影響**: FeedbackProcessor 無法接收反饋

**應對**:
- ✅ 檢查 RabbitMQ 是否運行
- ✅ 實現重連機制
- ✅ 降級方案：暫時使用內存緩存

### 風險 2: Features 不可用

**影響**: ScannerPlugin 無法執行掃描

**應對**:
- ✅ 初始化時檢查可用性
- ✅ 提供友好的錯誤訊息
- ✅ 降級方案：使用備用 Feature

### 風險 3: 性能問題

**影響**: 系統響應變慢

**應對**:
- ✅ 使用異步調用
- ✅ 實現超時機制
- ✅ 監控性能指標

### 風險 4: 數據格式不一致

**影響**: 組件間無法通信

**應對**:
- ✅ 使用 Pydantic 嚴格驗證
- ✅ 提供數據轉換工具
- ✅ 詳細的錯誤日誌

---

## 📊 進度追蹤

### Phase 1: Core → Features 調用打通

| 任務 | 預計工時 | 實際工時 | 狀態 | 完成日期 |
|------|---------|---------|------|---------|
| 1.1 使用現有數據合約 | 0.5 天 | - | ⏳ 待開始 | 補充 FeatureType 枚舉 |
| 1.2 實作 FeaturesInvoker | 2-3 天 | - | ⏳ 待開始 | 直接使用 FeatureResult |
| 1.3 修改 ScannerPlugin | 1 天 | - | ⏳ 待開始 | - |
| 1.4 端到端測試 | 1-2 天 | - | ⏳ 待開始 | - |

### Phase 2: Features → Integration 自動觸發

| 任務 | 預計工時 | 實際工時 | 狀態 | 完成日期 |
|------|---------|---------|------|---------|
| 2.1 選擇自動觸發方案 | 0.5 天 | - | ⏳ 待開始 | - |
| 2.2 實現直接調用方案 | 1.5 天 | - | ⏳ 待開始 | - |
| 2.3 修改 BaseCoordinator | 1 天 | - | ⏳ 待開始 | - |
| 2.4 端到端測試 | 1 天 | - | ⏳ 待開始 | - |

### Phase 3: 反饋循環與學習優化

| 任務 | 預計工時 | 實際工時 | 狀態 | 完成日期 |
|------|---------|---------|------|---------|
| 3.1 實作 FeedbackProcessor | 2-3 天 | - | ⏳ 待開始 | - |
| 3.2 集成到 AICommanderV2 | 1-2 天 | - | ⏳ 待開始 | - |
| 3.3 完整閉環測試 | 1 天 | - | ⏳ 待開始 | - |

---

## 🎯 成功標準

### 功能標準

- ✅ 用戶執行一個命令，AI 自動完成所有步驟
- ✅ Core 能夠成功調用所有 Python Features
- ✅ Core 能夠成功調用編譯型 Features（Rust/Go）
- ✅ Features 完成後自動觸發雙閉環處理
- ✅ 自動生成內循環數據（優化建議）
- ✅ 自動生成外循環數據（漏洞報告）
- ✅ Core 能夠接收並應用優化建議
- ✅ 第二次執行時自動使用優化參數
- ✅ 持續學習和優化

### 性能標準

- ✅ 端到端執行時間 < 60 秒
- ✅ Feature 調用延遲 < 5 秒
- ✅ Feedback 處理延遲 < 2 秒
- ✅ 內存使用 < 2GB
- ✅ CPU 使用 < 80%

### 質量標準

- ✅ 單元測試覆蓋率 > 80%
- ✅ 集成測試全部通過
- ✅ 端到端測試全部通過
- ✅ 無嚴重 Bug
- ✅ 代碼符合規範

---

## 📚 參考文檔

- [AI自動化閉環執行流程分析.md](./AI自動化閉環執行流程分析.md) - 詳細架構分析
- [掃描模組多語言協調機制分析.md](./掃描模組多語言協調機制分析.md) - 參考實現
- [AIVA系統深度分析報告.md](./AIVA系統深度分析報告.md) - 系統現狀
- `services/scan/coordinators/` - 掃描模組適配器實現
- `services/integration/coordinators/` - Integration 協調器實現

---

## 🔄 更新記錄

| 日期 | 版本 | 更新內容 | 負責人 |
|------|------|---------|--------|
| 2025-11-29 | v1.0 | 初始版本 | AI Assistant |

---

## 📞 聯絡方式

如有問題或建議，請聯絡開發團隊。

---

**最後更新**: 2025年11月29日  
**狀態**: 📋 計劃階段  
**預計完成**: 2-3.5 週後
