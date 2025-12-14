# 🎯 能力範圍管理與架構優化方案

> **創建日期**: 2025-12-14  
> **問題**: 能力都在 services/core，需區分可操作整個 services 的能力  
> **目標**: 建立清晰的能力範圍分層架構

---

## 📋 目錄

1. [問題分析](#-問題分析)
2. [業界最佳實踐](#-業界最佳實踐)
3. [架構設計方案](#-架構設計方案)
4. [實施方案](#-實施方案)
5. [能力範圍分類標準](#-能力範圍分類標準)
6. [遷移計劃](#-遷移計劃)

---

## 🔍 問題分析

### 當前狀況

**問題描述**:
- ✅ 670 條能力全部在 `services/core/aiva_core` 中探索和記錄
- ❌ 實際上有些能力是**跨 services 目錄**的（features, scan, integration）
- ❌ 其他模組（features/*）的 CLI 指令尚未完善
- ❌ 無法區分哪些是 core 內部能力，哪些是全局可用能力

**目錄結構現狀**:
```
services/
├── core/                    # ✅ 已探索 670 條能力
│   └── aiva_core/          # AI 核心、任務規劃、能力分析
├── features/               # ⚠️ 功能模組，CLI 未完善
│   ├── function_sqli/      # SQL 注入檢測
│   ├── function_xss/       # XSS 檢測
│   ├── function_ssrf/      # SSRF 檢測
│   ├── function_crypto/    # 加密檢測 (Rust)
│   ├── function_authn_go/  # 認證檢測 (Go)
│   └── ... (15+ 功能模組)
├── scan/                   # ⚠️ 掃描引擎，部分完成
│   ├── engines/            # 多引擎掃描
│   └── coordinators/       # 掃描協調器
└── integration/            # ⚠️ 整合層，基礎設施
    └── capability/         # 能力註冊中心
```

### 根本問題

**能力範圍混淆**:
1. **Core 能力** - 只能在 aiva_core 內部使用（如內部探索、RAG 管理）
2. **Service 能力** - 可以跨 services 目錄使用（如 SQL 注入檢測）
3. **External 能力** - 調用外部工具（如 sqlmap, XSStrike）

**當前所有能力都標記為 "core" 範圍，無法區分**

---

## 🌐 業界最佳實踐

### 1. 微服務能力範圍管理 (Service Mesh)

**參考**: Kubernetes Service Mesh (Istio, Linkerd)

```yaml
# 能力範圍分層
Namespace Scope:
  - cluster-wide (全局)
  - namespace-local (服務組內)
  - pod-local (單服務內)

Service Discovery:
  - global registry (全局註冊)
  - local cache (本地緩存)
  - capability routing (能力路由)
```

**應用到 AIVA**:
```
Global Scope (services/*)    - 所有服務可用
Service Scope (features/*)   - 功能服務組內
Core Scope (core/aiva_core)  - 核心內部
```

---

### 2. API Gateway 能力分類 (Kong, AWS API Gateway)

**參考**: Kong 的插件範圍管理

```yaml
Plugin Scopes:
  - global: 所有服務
  - service: 特定服務
  - route: 特定路由
  - consumer: 特定消費者

Capability Attributes:
  - visibility: public/internal/private
  - access_level: system/service/module
  - dependencies: []
```

**應用到 AIVA**:
```python
CapabilityScope:
  - PUBLIC: 可被 AI Commander 直接調用
  - INTERNAL: 只能服務內部使用
  - SYSTEM: 系統級能力
```

---

### 3. 單體應用的邊界上下文 (DDD - Domain-Driven Design)

**參考**: Eric Evans 的 Bounded Context

```
Bounded Context (邊界上下文):
  - Core Domain (核心域) - 核心業務邏輯
  - Supporting Subdomain (支撐子域) - 輔助功能
  - Generic Subdomain (通用子域) - 通用能力
  
Context Mapping (上下文映射):
  - Shared Kernel (共享內核)
  - Customer/Supplier (客戶/供應商)
  - Conformist (遵循者)
```

**應用到 AIVA**:
```
Core Domain (aiva_core):
  - AI 決策
  - 任務規劃
  - 內部探索
  
Supporting Domain (features):
  - SQL 注入檢測
  - XSS 檢測
  - 加密檢測
  
Generic Domain (aiva_common):
  - 日誌
  - 錯誤處理
  - 數據模型
```

---

### 4. 雲原生應用的 RBAC (Role-Based Access Control)

**參考**: Kubernetes RBAC

```yaml
Capabilities by Scope:
  - ClusterRole: 集群級別
  - Role: 命名空間級別
  - ServiceAccount: 服務賬戶綁定

Rules:
  - resources: [pods, services]
  - verbs: [get, list, create]
  - apiGroups: [v1, apps/v1]
```

**應用到 AIVA**:
```python
CapabilityAccessLevel:
  - SYSTEM: 系統管理能力
  - SERVICE: 服務協調能力
  - MODULE: 模組功能能力
  - INTERNAL: 內部實現能力
```

---

## 🏗️ 架構設計方案

### 方案 A: 能力範圍標註 (推薦) ⭐

**核心思想**: 在現有 670 條能力上添加 `scope` 和 `visibility` 屬性

#### 新增能力屬性

```python
# aiva_common/schemas/dual_loop.py

from enum import Enum

class CapabilityScope(str, Enum):
    """能力範圍"""
    CORE = "core"              # 核心內部能力（只能 aiva_core 使用）
    SERVICE = "service"        # 服務級能力（services/* 可用）
    GLOBAL = "global"          # 全局能力（整個項目可用）
    EXTERNAL = "external"      # 外部工具能力

class CapabilityVisibility(str, Enum):
    """能力可見性"""
    PUBLIC = "public"          # AI Commander 可直接調用
    INTERNAL = "internal"      # 只能內部調用
    SYSTEM = "system"          # 系統級（需特殊權限）
    DEPRECATED = "deprecated"  # 已棄用

class CapabilityAccessLevel(str, Enum):
    """能力訪問級別"""
    L0_SYSTEM = "system"       # 系統管理（如服務重啟）
    L1_SERVICE = "service"     # 服務協調（如多引擎協調）
    L2_MODULE = "module"       # 模組功能（如 SQL 注入檢測）
    L3_INTERNAL = "internal"   # 內部實現（如 AST 分析）

# 更新 ModuleCapability
class ModuleCapability(BaseModel):
    """模組能力定義 (v3.0 - 範圍管理增強)"""
    
    # ... 原有欄位 ...
    
    # ✅ 新增: 範圍管理
    scope: CapabilityScope = CapabilityScope.CORE
    visibility: CapabilityVisibility = CapabilityVisibility.INTERNAL
    access_level: CapabilityAccessLevel = CapabilityAccessLevel.L3_INTERNAL
    
    # ✅ 新增: 可用性條件
    available_in: List[str] = []  # 可用的服務路徑 ["core", "features/sqli"]
    depends_on_services: List[str] = []  # 依賴的服務
    
    # ✅ 新增: CLI 相關
    has_cli: bool = False  # 是否有 CLI 接口
    cli_command: Optional[str] = None  # CLI 命令（如 "aiva sqli scan"）
    cli_maturity: str = "none"  # CLI 成熟度: none/alpha/beta/stable
```

---

#### 能力分類規則

**規則 1: 基於文件路徑自動判斷**

```python
def classify_capability_scope(file_path: str) -> tuple[CapabilityScope, CapabilityVisibility]:
    """
    根據文件路徑自動分類能力範圍
    
    規則:
    - services/core/aiva_core/internal_exploration/* → CORE, INTERNAL
    - services/core/aiva_core/task_planning/* → SERVICE, PUBLIC
    - services/core/aiva_core/core_capabilities/* → SERVICE, PUBLIC
    - services/features/* → GLOBAL, PUBLIC
    - services/scan/* → GLOBAL, PUBLIC
    - services/integration/* → GLOBAL, SYSTEM
    """
    
    if "internal_exploration" in file_path:
        return CapabilityScope.CORE, CapabilityVisibility.INTERNAL
    
    elif "task_planning" in file_path or "core_capabilities" in file_path:
        return CapabilityScope.SERVICE, CapabilityVisibility.PUBLIC
    
    elif "services/features" in file_path:
        return CapabilityScope.GLOBAL, CapabilityVisibility.PUBLIC
    
    elif "services/scan" in file_path:
        return CapabilityScope.GLOBAL, CapabilityVisibility.PUBLIC
    
    elif "services/integration" in file_path:
        return CapabilityScope.GLOBAL, CapabilityVisibility.SYSTEM
    
    else:
        return CapabilityScope.CORE, CapabilityVisibility.INTERNAL
```

**規則 2: 基於能力類別判斷訪問級別**

```python
def classify_access_level(category: str, sub_category: str) -> CapabilityAccessLevel:
    """
    根據能力類別判斷訪問級別
    
    規則:
    - Scanning, Attacking → MODULE (功能模組)
    - Analysis, Reporting → SERVICE (服務協調)
    - Integration, Utility → SYSTEM (系統級)
    - Internal → INTERNAL (內部實現)
    """
    
    if category in ["Scanning", "Attacking"]:
        return CapabilityAccessLevel.L2_MODULE
    
    elif category in ["Analysis", "Reporting"]:
        return CapabilityAccessLevel.L1_SERVICE
    
    elif category in ["Integration", "Utility"]:
        return CapabilityAccessLevel.L0_SYSTEM
    
    else:
        return CapabilityAccessLevel.L3_INTERNAL
```

---

#### InternalLoopConnector 增強

```python
# services/core/aiva_core/cognitive_core/internal_loop_connector.py

class InternalLoopConnector:
    """內部閉環連接器 (v11.0 - 範圍管理增強)"""
    
    def __init__(self, rag_knowledge_base=None, pg_session=None):
        # ... 原有初始化 ...
        self.scope_classifier = CapabilityScopeClassifier()
    
    def _enhance_capability_with_scope(self, cap: dict) -> dict:
        """為能力添加範圍信息
        
        ✅ 新增邏輯:
        1. 根據文件路徑自動分類 scope 和 visibility
        2. 根據能力類別自動分類 access_level
        3. 檢測是否有 CLI 接口
        4. 標記依賴的服務
        """
        
        # 1. 獲取文件路徑
        file_path = cap.get("file_path", "")
        
        # 2. 自動分類範圍
        scope, visibility = self.scope_classifier.classify_scope(file_path)
        cap["scope"] = scope.value
        cap["visibility"] = visibility.value
        
        # 3. 自動分類訪問級別
        category = cap.get("category", "Utility")
        sub_category = cap.get("sub_category")
        access_level = self.scope_classifier.classify_access_level(category, sub_category)
        cap["access_level"] = access_level.value
        
        # 4. 檢測可用範圍
        cap["available_in"] = self._detect_available_in(file_path)
        
        # 5. 檢測 CLI 成熟度
        cap["has_cli"], cap["cli_command"], cap["cli_maturity"] = self._detect_cli_info(cap)
        
        # 6. 檢測服務依賴
        cap["depends_on_services"] = self._detect_service_dependencies(cap)
        
        return cap
    
    def _detect_available_in(self, file_path: str) -> List[str]:
        """檢測能力可用的服務路徑"""
        available = []
        
        if "services/core" in file_path:
            available.append("core")
        
        if "services/features" in file_path:
            # 提取具體功能模組
            import re
            match = re.search(r"features/(function_\w+)", file_path)
            if match:
                available.append(f"features/{match.group(1)}")
        
        if "services/scan" in file_path:
            available.append("scan")
        
        if "services/integration" in file_path:
            available.append("integration")
        
        return available if available else ["core"]
    
    def _detect_cli_info(self, cap: dict) -> tuple[bool, Optional[str], str]:
        """檢測 CLI 信息
        
        Returns:
            (has_cli, cli_command, cli_maturity)
        """
        name = cap.get("name", "")
        file_path = cap.get("file_path", "")
        
        # 檢查是否在 CLI 相關文件中
        if "cli" in file_path.lower() or "command" in file_path.lower():
            # 嘗試推斷 CLI 命令
            cli_command = self._infer_cli_command(name, file_path)
            
            # 判斷成熟度
            if "services/features" in file_path:
                maturity = "alpha"  # features 模組 CLI 尚未完善
            elif "services/core/aiva_core" in file_path:
                maturity = "beta"  # core 模組較成熟
            else:
                maturity = "alpha"
            
            return True, cli_command, maturity
        
        return False, None, "none"
    
    def _infer_cli_command(self, func_name: str, file_path: str) -> str:
        """推斷 CLI 命令
        
        例如:
        - function_sqli/scanner.py → aiva sqli scan
        - function_xss/detector.py → aiva xss detect
        """
        import re
        
        # 提取功能模組名
        match = re.search(r"function_(\w+)", file_path)
        if match:
            module = match.group(1)
            # 簡化函數名為動詞
            action = func_name.split("_")[0] if "_" in func_name else func_name
            return f"aiva {module} {action}"
        
        return f"aiva {func_name}"
    
    def _detect_service_dependencies(self, cap: dict) -> List[str]:
        """檢測服務依賴
        
        通過分析導入語句和函數調用檢測依賴
        """
        dependencies = []
        
        # 從參數中提取依賴線索
        params = cap.get("parameters", [])
        for param in params:
            param_type = param.get("type", "")
            
            if "RAGEngine" in param_type:
                dependencies.append("core/rag")
            elif "ScanEngine" in param_type:
                dependencies.append("scan")
            elif "FeatureExecutor" in param_type:
                dependencies.append("features")
        
        return list(set(dependencies))  # 去重
```

---

#### AICommander 查詢增強

```python
# services/core/aiva_core/task_planning/ai_commander.py

class AICommander:
    """AI 指揮官 (範圍感知增強)"""
    
    async def _query_relevant_capabilities(
        self,
        task_type: AITaskType,
        context: dict,
        required_scope: CapabilityScope = CapabilityScope.GLOBAL  # ✅ 新增參數
    ) -> list[dict]:
        """查詢相關能力（範圍感知）
        
        ✅ 增強:
        1. 根據 required_scope 過濾能力
        2. 優先返回 PUBLIC visibility 的能力
        3. 檢查 CLI 成熟度
        4. 驗證服務依賴是否滿足
        """
        
        if not self.internal_loop:
            return []
        
        try:
            # 構建查詢（同之前）
            query = self._build_capability_query(context.get("objective"), context.get("target"))
            
            from aiva_common.schemas.dual_loop import RAGQueryRequest
            query_req = RAGQueryRequest(
                query=query,
                query_type="capability_search",
                top_k=20,  # 多取一些，後續過濾
                filters={
                    "category": self._map_task_to_category(task_type),
                    # ✅ 新增: 範圍過濾
                    "scope": [required_scope.value, CapabilityScope.GLOBAL.value],
                    "visibility": CapabilityVisibility.PUBLIC.value
                }
            )
            
            result = self.internal_loop.query_self_awareness(query_req)
            raw_capabilities = result.results
            
            # ✅ 新增: 後處理過濾
            filtered_capabilities = []
            for cap in raw_capabilities:
                # 檢查範圍匹配
                cap_scope = cap.get("scope", "core")
                if cap_scope not in [required_scope.value, "global"]:
                    continue
                
                # 檢查 CLI 成熟度（如果需要）
                if context.get("require_cli"):
                    cli_maturity = cap.get("cli_maturity", "none")
                    if cli_maturity == "none":
                        logger.debug(f"Skipping {cap['name']}: no CLI")
                        continue
                
                # 檢查服務依賴
                depends_on = cap.get("depends_on_services", [])
                if not self._check_dependencies_available(depends_on):
                    logger.warning(f"Skipping {cap['name']}: dependencies not met")
                    continue
                
                filtered_capabilities.append(cap)
            
            logger.info(f"✅ Found {len(filtered_capabilities)} capabilities in scope '{required_scope.value}'")
            logger.debug(f"   (Filtered from {len(raw_capabilities)} raw results)")
            
            return filtered_capabilities
            
        except Exception as e:
            logger.error(f"Capability query failed: {e}")
            return []
    
    def _check_dependencies_available(self, dependencies: List[str]) -> bool:
        """檢查依賴的服務是否可用"""
        # TODO: 實現服務健康檢查
        # 目前簡單返回 True
        return True
```

---

### 方案 B: 多 InternalLoopConnector 實例

**核心思想**: 為每個服務創建獨立的 InternalLoopConnector

```python
# 不推薦，因為：
# 1. 管理複雜度高
# 2. RAG 知識庫需要分離（浪費資源）
# 3. 查詢時需要跨多個實例
```

**決定**: 不採用此方案

---

## 📊 能力範圍分類標準

### 分類矩陣

| 能力類型 | Scope | Visibility | Access Level | 範例 |
|---------|-------|------------|--------------|------|
| **AI 決策** | SERVICE | PUBLIC | L1_SERVICE | _plan_attack() |
| **SQL 注入檢測** | GLOBAL | PUBLIC | L2_MODULE | sqli_scan() |
| **XSS 檢測** | GLOBAL | PUBLIC | L2_MODULE | xss_detect() |
| **加密掃描** (Rust) | GLOBAL | PUBLIC | L2_MODULE | crypto_scan() |
| **認證檢測** (Go) | GLOBAL | PUBLIC | L2_MODULE | authn_check() |
| **多引擎協調** | SERVICE | PUBLIC | L1_SERVICE | multi_engine_scan() |
| **AST 分析** | CORE | INTERNAL | L3_INTERNAL | ast_parse() |
| **RAG 管理** | CORE | INTERNAL | L3_INTERNAL | rag_add_knowledge() |
| **內部探索** | CORE | INTERNAL | L3_INTERNAL | explore_module() |
| **服務註冊** | GLOBAL | SYSTEM | L0_SYSTEM | register_service() |
| **健康檢查** | GLOBAL | SYSTEM | L0_SYSTEM | health_check() |

### CLI 成熟度標準

| 成熟度 | 定義 | 條件 | 是否可用 |
|-------|-----|------|---------|
| **none** | 無 CLI | 只有 Python API | ❌ |
| **alpha** | 早期 CLI | 有基本命令，未測試 | ⚠️ 謹慎使用 |
| **beta** | 測試中 | 有完整命令，部分測試 | ✅ 可用 |
| **stable** | 穩定版 | 完整測試，文檔齊全 | ✅ 推薦 |

### 當前模組狀態評估

| 模組路徑 | CLI 成熟度 | Scope | 說明 |
|---------|-----------|-------|------|
| `services/core/aiva_core` | beta | SERVICE | 部分有 CLI |
| `services/features/function_sqli` | alpha | GLOBAL | CLI 未完善 |
| `services/features/function_xss` | alpha | GLOBAL | CLI 未完善 |
| `services/features/function_crypto` | alpha | GLOBAL | Rust CLI 存在 |
| `services/features/function_authn_go` | alpha | GLOBAL | Go CLI 存在 |
| `services/scan/engines` | beta | GLOBAL | 有基本 CLI |
| `services/integration` | none | GLOBAL | 無 CLI |

---

## 🚀 實施方案

### Phase 1: 數據模型更新 (1 小時)

#### Task 1.1: 更新 Schema 定義

**文件**: `services/aiva_common/schemas/dual_loop.py`

```python
# 添加新的枚舉類型
class CapabilityScope(str, Enum): ...
class CapabilityVisibility(str, Enum): ...
class CapabilityAccessLevel(str, Enum): ...

# 更新 ModuleCapability
class ModuleCapability(BaseModel):
    # ... 原有欄位 ...
    scope: CapabilityScope = CapabilityScope.CORE
    visibility: CapabilityVisibility = CapabilityVisibility.INTERNAL
    access_level: CapabilityAccessLevel = CapabilityAccessLevel.L3_INTERNAL
    available_in: List[str] = []
    depends_on_services: List[str] = []
    has_cli: bool = False
    cli_command: Optional[str] = None
    cli_maturity: str = "none"
```

---

### Phase 2: InternalLoopConnector 增強 (2 小時)

#### Task 2.1: 添加範圍分類器

**文件**: `services/core/aiva_core/cognitive_core/internal_loop_connector.py`

```python
class CapabilityScopeClassifier:
    """能力範圍分類器"""
    
    def classify_scope(self, file_path: str) -> tuple: ...
    def classify_access_level(self, category: str, sub_category: str) -> CapabilityAccessLevel: ...
```

#### Task 2.2: 增強能力處理

```python
def _enhance_capability_with_scope(self, cap: dict) -> dict:
    """為能力添加範圍信息"""
    # 實現如上
```

---

### Phase 3: AICommander 查詢增強 (1.5 小時)

#### Task 3.1: 添加範圍感知查詢

**文件**: `services/core/aiva_core/task_planning/ai_commander.py`

```python
async def _query_relevant_capabilities(
    self,
    task_type: AITaskType,
    context: dict,
    required_scope: CapabilityScope = CapabilityScope.GLOBAL
) -> list[dict]:
    """範圍感知的能力查詢"""
    # 實現如上
```

---

### Phase 4: 驗證和測試 (1 小時)

#### Task 4.1: 重新同步能力

```python
# 觸發完整同步，添加範圍信息
await internal_loop.sync_capabilities_to_rag(force_refresh=True)
```

#### Task 4.2: 測試範圍過濾

```python
# 測試查詢不同範圍的能力
global_caps = await commander._query_relevant_capabilities(
    task_type=AITaskType.VULNERABILITY_DETECTION,
    context={},
    required_scope=CapabilityScope.GLOBAL
)

core_caps = await commander._query_relevant_capabilities(
    task_type=AITaskType.KNOWLEDGE_RETRIEVAL,
    context={},
    required_scope=CapabilityScope.CORE
)
```

---

## 📝 遷移計劃

### 短期 (本週)

1. ✅ 更新數據模型（Phase 1）
2. ✅ 增強 InternalLoopConnector（Phase 2）
3. ✅ 重新同步能力，添加範圍標註
4. ✅ 驗證範圍分類正確性

### 中期 (下週)

5. ✅ AICommander 查詢增強（Phase 3）
6. ✅ 測試範圍感知查詢
7. ⚠️ 逐步完善 features 模組的 CLI（alpha → beta）
8. ⚠️ 為 scan 模組添加 CLI 接口

### 長期 (下個月)

9. ✅ 創建服務健康檢查機制
10. ✅ 實現依賴驗證邏輯
11. ✅ CLI 成熟度自動檢測
12. ✅ 建立能力註冊中心（integration/capability）

---

## ✅ 驗證檢查清單

### 數據模型

- [ ] CapabilityScope 枚舉定義正確
- [ ] CapabilityVisibility 枚舉定義正確
- [ ] CapabilityAccessLevel 枚舉定義正確
- [ ] ModuleCapability 新增欄位正確

### 分類邏輯

- [ ] 文件路徑 → Scope 映射正確
- [ ] 能力類別 → Access Level 映射正確
- [ ] CLI 成熟度檢測正確
- [ ] 服務依賴檢測正確

### 查詢功能

- [ ] 範圍過濾生效
- [ ] Visibility 過濾生效
- [ ] CLI 成熟度過濾生效
- [ ] 依賴檢查生效

### 效果驗證

- [ ] 查詢 GLOBAL 能力返回 features/* 的能力
- [ ] 查詢 CORE 能力只返回 aiva_core 內部能力
- [ ] CLI 未完善的模組被正確標記為 alpha
- [ ] PUBLIC 能力可被 AI Commander 使用

---

## 🎯 總結

### 核心方案

**推薦**: 方案 A - 能力範圍標註

**優點**:
- ✅ 無需重構現有架構
- ✅ 向後兼容
- ✅ 實施成本低（4-5 小時）
- ✅ 可漸進式完善

**關鍵改進**:
1. **清晰的範圍分層**: CORE → SERVICE → GLOBAL
2. **可見性控制**: PUBLIC vs INTERNAL
3. **CLI 成熟度追蹤**: alpha → beta → stable
4. **依賴管理**: 自動檢測服務依賴

### 預期效果

**修復前**:
```
所有 670 條能力混在一起
❌ 無法區分哪些是 core 內部能力
❌ 無法知道哪些有 CLI
❌ AI Commander 可能調用不可用的能力
```

**修復後**:
```
能力按範圍分類
✅ CORE: 123 條（內部使用）
✅ SERVICE: 287 條（服務級）
✅ GLOBAL: 260 條（全局可用）
✅ CLI 成熟度明確標註
✅ AI Commander 智能選擇可用能力
```

---

**創建時間**: 2025-12-14  
**預計實施時間**: 4-5 小時  
**優先級**: 🟡 MEDIUM (優化項，不阻塞)  
**建議**: 在完成雙閉環修復後實施
