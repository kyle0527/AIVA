# 🔹 組件補強報告：IDOR + SSRF 模組

**報告編號**: FEAT-003  
**日期**: 2025年11月6日  
**狀態**: 🔹 中優先級 - 組件補全  
**負責模組**: IDOR (部分實現) + SSRF (部分實現)

---

## 📊 模組現況分析

### 🔐 IDOR 模組 - 垂直權限繞過檢測

#### **現況評估**
- **完善度**: 🔹 部分實現 (2/4 組件)
- **程式規模**: 6 檔案, 2,090 行程式碼
- **開發語言**: Python (純架構)
- **組件狀態**: Worker✅ | Detector✅ | Engine❌ | Config❌

#### **現有優勢**
```
function_idor/
├── ✅ worker.py                         # 標準 Worker (519行)
├── ✅ enhanced_worker.py                # 增強 Worker (420行)
├── ✅ smart_idor_detector.py            # 智能檢測器 (547行)
├── ✅ cross_user_tester.py              # 橫向權限測試 (159行)
├── ✅ vertical_escalation_tester.py     # 垂直權限測試 (191行)
├── ✅ resource_id_extractor.py          # 資源 ID 提取器 (254行)
└── ❌ 缺少統一的 Engine 和 Config 組件
```

#### **架構特色**
- **智能檢測管理**: 整合 UnifiedSmartDetectionManager
- **多維度測試**: 橫向 + 垂直權限檢測
- **資源識別**: 自動提取和分析資源 ID
- **自適應策略**: 動態超時和速率限制

---

### 🌐 SSRF 模組 - 服務端請求偽造檢測

#### **現況評估**
- **完善度**: 🔹 部分實現 (2/4 組件)
- **程式規模**: 6 檔案, 1,924 行程式碼
- **開發語言**: Python (純架構)
- **組件狀態**: Worker✅ | Detector✅ | Engine❌ | Config❌

#### **現有結構**
```
function_ssrf/
├── ✅ worker.py                         # 標準 Worker (587行)
├── ✅ smart_ssrf_detector.py            # 智能檢測器 (532行)
├── ✅ internal_address_detector.py      # 內網地址檢測 (339行)
├── ✅ param_semantics_analyzer.py       # 參數語義分析 (225行)
├── ✅ oast_dispatcher.py                # OAST 調度器 (143行)
├── ✅ result_publisher.py               # 結果發布器 (98行)
└── ❌ 缺少統一的 Engine 和 Config 組件
```

#### **Python 深度檢測優勢**
- **語義分析**: 智能參數語義理解和測試向量生成
- **OAST 整合**: Out-of-Band Application Security Testing
- **內網探測**: 專業的內部地址和服務發現
- **智能調度**: 動態負載和回調管理

---

## 🎯 組件補全策略

### 🔐 IDOR 模組補強方案

#### **缺失組件**: Engine + Config
**目標**: 實現標準化的 IDOR 檢測引擎和配置管理

```python
# 新增 engine.py - IDOR 檢測引擎
class IDOREngine:
    """統一 IDOR 檢測引擎"""
    
    def __init__(self, config: IDORConfig):
        self.config = config
        self.detectors = {
            'horizontal': CrossUserTester(config.cross_user_config),
            'vertical': VerticalEscalationTester(config.vertical_config),
            'resource_extraction': ResourceIdExtractor(config.extraction_config)
        }
        self.smart_manager = UnifiedSmartDetectionManager()
    
    async def detect_idor(self, target: str, context: dict) -> List[IDORFinding]:
        """智能 IDOR 檢測
        
        1. 資源 ID 自動提取和分析
        2. 橫向權限測試 (不同用戶存取)
        3. 垂直權限測試 (權限提升)
        4. 智能早期停止和自適應調優
        """
        # 階段 1: 資源發現
        resources = await self.detectors['resource_extraction'].extract_resources(target)
        
        # 階段 2: 權限矩陣構建
        permission_matrix = await self._build_permission_matrix(resources, context)
        
        # 階段 3: 並行權限測試
        horizontal_results = await self.detectors['horizontal'].test_cross_user_access(
            permission_matrix, context.get('users', [])
        )
        
        vertical_results = await self.detectors['vertical'].test_privilege_escalation(
            permission_matrix, context.get('roles', [])
        )
        
        # 階段 4: 結果融合和風險評估
        return self._merge_and_evaluate(horizontal_results, vertical_results)
```

#### **新增 Config 管理**
```python
# 新增 config.py - IDOR 配置管理
@dataclass
class IDORConfig(BaseConfig):
    """IDOR 檢測配置"""
    
    # 檢測策略
    enable_horizontal_testing: bool = True
    enable_vertical_testing: bool = True
    enable_resource_enumeration: bool = True
    
    # 用戶和角色配置
    test_users: List[UserCredential] = field(default_factory=list)
    privilege_levels: List[PrivilegeLevel] = field(default_factory=lambda: [
        PrivilegeLevel.GUEST,
        PrivilegeLevel.USER, 
        PrivilegeLevel.ADMIN
    ])
    
    # 資源 ID 配置
    resource_patterns: List[str] = field(default_factory=lambda: [
        r'/api/users/(\d+)',
        r'/documents/([a-f0-9-]+)',
        r'id=(\d+)',
        r'userId=([^&]+)'
    ])
    
    # 智能檢測配置
    max_concurrent_tests: int = 5
    adaptive_timeout: bool = True
    early_stop_threshold: float = 0.8
    
    # CVSS 評分配置
    horizontal_base_score: float = 6.5  # 橫向存取 (Medium-High)
    vertical_base_score: float = 8.5    # 垂直提權 (High)
```

### 🌐 SSRF 模組補強方案

#### **目標**: 發揮 Python 深度檢測和語義分析優勢

```python
# 新增 engine.py - SSRF 檢測引擎
class SSRFEngine:
    """統一 SSRF 檢測引擎"""
    
    def __init__(self, config: SSRFConfig):
        self.config = config
        self.analyzers = {
            'semantic': ParamSemanticsAnalyzer(config.semantic_config),
            'internal': InternalAddressDetector(config.internal_config),
            'oast': OastDispatcher(config.oast_config)
        }
        self.payload_generator = SSRFPayloadGenerator()
    
    async def detect_ssrf(self, target: str, params: dict) -> List[SSRFFinding]:
        """智能 SSRF 檢測
        
        1. 參數語義分析和測試向量生成
        2. 內網服務發現和探測
        3. OAST 回調驗證
        4. 時間盲注和錯誤分析
        """
        # 階段 1: 語義分析
        analysis_plan = await self.analyzers['semantic'].analyze_parameters(target, params)
        
        # 階段 2: 測試向量生成
        test_vectors = await self.payload_generator.generate_vectors(analysis_plan)
        
        # 階段 3: 並行檢測
        tasks = []
        
        # 內網探測任務
        if self.config.enable_internal_detection:
            tasks.append(self.analyzers['internal'].test_internal_access(test_vectors))
        
        # OAST 回調任務
        if self.config.enable_oast_detection:
            tasks.append(self.analyzers['oast'].test_oast_callbacks(test_vectors))
        
        # 執行並行檢測
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self._merge_and_score(results)
```

#### **智能配置管理**
```python
# 新增 config.py - SSRF 配置管理
@dataclass  
class SSRFConfig(BaseConfig):
    """SSRF 檢測配置"""
    
    # 檢測策略
    enable_internal_detection: bool = True
    enable_oast_detection: bool = True
    enable_blind_detection: bool = True
    enable_time_based_detection: bool = True
    
    # 內網目標配置
    internal_targets: List[str] = field(default_factory=lambda: [
        'http://127.0.0.1:22',
        'http://localhost:3306',
        'http://169.254.169.254/latest/meta-data/',
        'file:///etc/passwd',
        'gopher://127.0.0.1:25/'
    ])
    
    # OAST 配置
    oast_domain: str = 'ssrf-callback.aiva.local'
    oast_timeout: int = 30
    callback_validation: bool = True
    
    # 語義分析配置
    url_parameter_patterns: List[str] = field(default_factory=lambda: [
        'url', 'link', 'src', 'href', 'callback', 'webhook', 'proxy'
    ])
    
    semantic_confidence_threshold: float = 0.7
    
    # 檢測時序配置  
    time_delay_detection: bool = True
    baseline_timeout: float = 5.0
    delay_multiplier: float = 2.0
    
    # CVSS 評分配置
    internal_network_score: float = 8.5    # 內網存取 (High)
    metadata_access_score: float = 9.0     # 雲元數據 (Critical)
    file_access_score: float = 7.5         # 檔案讀取 (High)
```

---

## 💪 技術實現細節

### 🔐 IDOR Engine 核心演算法

#### **權限矩陣構建**
```python
async def _build_permission_matrix(self, resources: List[ResourceId], context: dict) -> PermissionMatrix:
    """構建權限測試矩陣"""
    matrix = PermissionMatrix()
    
    # 用戶角色組合
    users = context.get('users', [])
    roles = context.get('privilege_levels', [])
    
    for resource in resources:
        for user in users:
            for role in roles:
                permission = Permission(
                    resource=resource,
                    user=user,
                    role=role,
                    expected_access=self._predict_access(resource, user, role)
                )
                matrix.add_permission(permission)
    
    return matrix

async def _predict_access(self, resource: ResourceId, user: User, role: PrivilegeLevel) -> bool:
    """預測合法存取權限 (用於檢測 IDOR)"""
    # 基於資源類型、用戶關係、角色等級預測正常存取權限
    if resource.owner_id == user.id:
        return True  # 自己的資源
    
    if role == PrivilegeLevel.ADMIN:
        return True  # 管理員全存取
    
    if resource.is_public:
        return True  # 公開資源
    
    return False  # 預期無存取權限 - IDOR 檢測目標
```

### 🌐 SSRF Engine 語義分析

#### **參數語義理解**
```python
class ParamSemanticsAnalyzer:
    """參數語義分析器 - SSRF 專精"""
    
    def __init__(self, config: SSRFSemanticConfig):
        self.url_indicators = ['url', 'link', 'src', 'href', 'callback', 'webhook']
        self.file_indicators = ['file', 'path', 'document', 'image', 'download']
        self.api_indicators = ['api', 'endpoint', 'service', 'proxy', 'forward']
    
    async def analyze_parameters(self, target: str, params: dict) -> AnalysisPlan:
        """深度參數語義分析"""
        plan = AnalysisPlan()
        
        for param_name, param_value in params.items():
            # 語義分類
            semantic_type = self._classify_parameter(param_name, param_value)
            confidence = self._calculate_confidence(param_name, param_value, semantic_type)
            
            if confidence >= self.config.confidence_threshold:
                test_vector = SsrfTestVector(
                    parameter=param_name,
                    original_value=param_value,
                    semantic_type=semantic_type,
                    confidence=confidence,
                    test_payloads=self._generate_targeted_payloads(semantic_type)
                )
                plan.add_vector(test_vector)
        
        return plan
    
    def _classify_parameter(self, name: str, value: str) -> SemanticType:
        """參數語義分類"""
        name_lower = name.lower()
        
        if any(indicator in name_lower for indicator in self.url_indicators):
            return SemanticType.URL_PARAMETER
        
        if any(indicator in name_lower for indicator in self.file_indicators):
            return SemanticType.FILE_PARAMETER
            
        if any(indicator in name_lower for indicator in self.api_indicators):
            return SemanticType.API_PARAMETER
        
        # 值語義分析
        if self._looks_like_url(value):
            return SemanticType.URL_PARAMETER
        
        if self._looks_like_file_path(value):
            return SemanticType.FILE_PARAMETER
            
        return SemanticType.UNKNOWN
```

---

## 📋 實現里程碑

### 🎯 第一階段 (1週) - IDOR Engine 實現
- [ ] 設計 IDOREngine 統一接口
- [ ] 實現權限矩陣構建演算法
- [ ] 整合現有的橫向和垂直測試器
- [ ] 實現 IDORConfig 配置管理
- [ ] 單元測試和基準測試

### 🎯 第二階段 (1週) - SSRF Engine 實現  
- [ ] 設計 SSRFEngine 統一接口
- [ ] 增強參數語義分析能力
- [ ] 整合 OAST 和內網檢測器
- [ ] 實現 SSRFConfig 智能配置
- [ ] 時間盲注和錯誤分析優化

### 🎯 第三階段 (1週) - 整合與優化
- [ ] 跨模組功能測試
- [ ] 與 XSS 模組架構對齊驗證
- [ ] 效能基準對比 (vs 現有實現)
- [ ] SARIF 輸出格式標準化
- [ ] 文檔更新和使用範例

---

## 🚀 團隊分工建議

### **Team A - IDOR 補全** (1人，1週)
- **Python 權限專家**
  - Engine 組件設計實現
  - 權限矩陣演算法優化
  - Config 管理系統建立

### **Team B - SSRF 補全** (1人，1週)  
- **Python 網路安全專家**
  - Engine 組件設計實現
  - 語義分析增強
  - OAST 整合優化

---

## ⚡ 性能提升預期

### **IDOR 模組**
- **檢測覆蓋率**: +25% (權限矩陣全面覆蓋)
- **誤報率**: -30% (智能預測合法存取)
- **檢測速度**: 3-5倍提升 (並行權限測試)
- **支援場景**: 橫向 + 垂直權限檢測全覆蓋

### **SSRF 模組**
- **參數識別**: +40% (語義分析增強)
- **檢測準確率**: +35% (OAST 回調驗證)
- **內網發現**: 支援 20+ 內網服務類型
- **時間檢測**: 盲注檢測能力大幅提升

---

## 🏆 與 XSS 最佳實踐對齊

### **架構一致性**
```
標準 4 組件架構:
XSS  模組: Worker✅ | Detector✅ | Engine✅ | Config✅  (100%)
IDOR 模組: Worker✅ | Detector✅ | Engine❌ | Config❌  (50% → 100%)
SSRF 模組: Worker✅ | Detector✅ | Engine❌ | Config❌  (50% → 100%)
```

### **設計模式統一**
- **引擎抽象**: 統一 Engine 接口，支援多檢測器整合
- **配置管理**: 結構化配置，支援動態調優
- **智能檢測**: 整合 UnifiedSmartDetectionManager
- **結果標準**: SARIF 2.1.0 格式，CVSS v3 評分

---

## 🔧 技術決策說明

### **為什麼 IDOR 需要權限矩陣？**
- **現況**: 現有檢測器分散，缺乏統一權限模型
- **問題**: 無法系統化測試複雜的權限關係
- **解決**: 權限矩陣提供全面的權限測試覆蓋

### **為什麼 SSRF 強調語義分析？**
- **SSRF 特性**: 高度依賴參數語義理解
- **Python 優勢**: 適合複雜語義分析和機器學習
- **效果需求**: 減少誤報，提升檢測精準度

---

## 📈 成功指標

### **IDOR 模組**
- [ ] 4/4 標準組件完整實現
- [ ] 權限檢測覆蓋率 > 90% (橫向 + 垂直)
- [ ] 並行權限測試效能提升 3x+
- [ ] 支援 10+ 種資源類型和權限模型

### **SSRF 模組**
- [ ] 4/4 標準組件完整實現  
- [ ] 參數語義識別準確率 > 85%
- [ ] OAST 回調驗證成功率 > 95%
- [ ] 支援 15+ 內網服務和協議檢測

---

**報告結論**: IDOR 和 SSRF 模組具備優秀的基礎架構，通過補全 Engine 和 Config 組件，可快速達到與 XSS 模組同等的完整實現狀態。建議優先完成這兩個模組的補強，為其他模組提供標準化架構參考。

**架構價值**: 完成後將建立統一的 Features 模組標準架構，所有深度檢測模組都遵循相同的設計模式，大幅提升開發效率和代碼質量。