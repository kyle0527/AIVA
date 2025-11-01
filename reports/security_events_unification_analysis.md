# 安全事件模型群組統一分析報告

📅 分析日期: 2025-11-01 11:05:00  
🎯 目標: 統一SIEM事件和攻擊路徑相關模型，消除重複定義  
📊 分析範圍: SIEMEvent、AttackPath、AttackPathNode等安全相關模型

## 📋 重複模型檢測結果

### 🔍 SIEMEvent 模型重複分析

#### 重複位置
1. **services/integration/models.py** (Line 67)
2. **services/aiva_common/schemas/telemetry.py** (Line 367)

#### 結構對比分析
| 欄位名稱 | integration/models | aiva_common/telemetry | 差異狀況 |
|----------|-------------------|----------------------|----------|
| `event_id` | ✅ | ✅ | 一致 |
| `event_type` | ✅ | ✅ | 一致 |
| `source_system` | ✅ | ✅ | 一致 |
| `timestamp` | ✅ | ✅ | 一致 |
| `severity` | ✅ | ✅ | 一致 |
| `subcategory` | `Optional[str]` | `str \| None` | 型別語法差異 |
| `source_ip` | `Optional[str]` | `str \| None` | 型別語法差異 |

**結論**: 兩個定義幾乎完全相同，僅有Pydantic語法差異(Optional vs |)。

### 🔍 AttackPath 相關模型重複分析

#### AttackPathNode 重複位置
1. **services/core/models.py** (Line 191)
2. **services/core/aiva_core/business_schemas.py** (Line 65)
3. **services/aiva_common/schemas/risk.py** (Line 65)

#### AttackPathEdge 重複位置
1. **services/core/models.py** (Line 203)
2. **services/aiva_common/schemas/risk.py** (Line 74)

#### AttackPath 重複位置
1. **services/core/aiva_core/business_schemas.py** (Line 91)
2. **Enhanced版本**: services/aiva_common/schemas/enhanced.py

#### 結構差異分析
| 組件 | core/models | business_schemas | aiva_common/risk | Enhanced版 |
|------|-------------|------------------|------------------|------------|
| **AttackPathNode** | 基礎定義 | 詳細業務邏輯 | 標準風險評估 | 增強功能 |
| **欄位複雜度** | 簡單 | 中等 | 標準 | 高 |
| **使用場景** | 通用 | 業務分析 | 風險評估 | 高級分析 |

## 🎯 統一標準化策略

### 🏗️ 建議架構: 分層統一模式

#### 1. 基礎安全事件模型 (BaseSIEMEvent)
```python
# services/aiva_common/schemas/security_events.py
class BaseSIEMEvent(BaseModel):
    """所有SIEM事件的基礎模型"""
    
    # 核心識別
    event_id: str = Field(description="事件唯一識別ID")
    event_type: str = Field(description="事件類型")
    source_system: str = Field(description="來源系統")
    
    # 時間信息
    timestamp: datetime = Field(description="事件發生時間戳")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="事件接收時間"
    )
    
    # 分類和嚴重程度
    severity: Severity = Field(description="事件嚴重程度")
    category: str = Field(description="事件主分類")
    subcategory: str | None = Field(default=None, description="事件子分類")
    
    # 網路信息
    source_ip: str | None = Field(default=None, description="來源IP位址")
    source_port: int | None = Field(default=None, description="來源端口")
    destination_ip: str | None = Field(default=None, description="目標IP位址") 
    destination_port: int | None = Field(default=None, description="目標端口")
    
    # 身份信息
    username: str | None = Field(default=None, description="相關用戶名")
    
    # 擴展元數據
    metadata: dict[str, Any] = Field(default_factory=dict, description="擴展屬性")
```

#### 2. 統一攻擊路徑基礎架構
```python
class BaseAttackPathNode(BaseModel):
    """攻擊路徑節點基礎模型"""
    
    node_id: str = Field(description="節點唯一識別ID")
    node_type: AttackPathNodeType = Field(description="節點類型")
    name: str = Field(description="節點名稱")
    description: str = Field(default="", description="節點描述")
    
    # 風險評估
    risk_score: float = Field(ge=0.0, le=10.0, default=0.0, description="節點風險評分")
    confidence: float = Field(ge=0.0, le=1.0, default=0.0, description="評估置信度")
    
    # 擴展屬性
    properties: dict[str, Any] = Field(default_factory=dict, description="節點屬性")

class BaseAttackPathEdge(BaseModel):
    """攻擊路徑邊基礎模型"""
    
    edge_id: str = Field(description="邊唯一識別ID")
    source_node_id: str = Field(description="源節點ID")
    target_node_id: str = Field(description="目標節點ID")
    edge_type: AttackPathEdgeType = Field(description="邊類型")
    
    # 攻擊評估
    attack_complexity: float = Field(ge=0.0, le=1.0, default=0.5, description="攻擊複雜度")
    success_probability: float = Field(ge=0.0, le=1.0, default=0.5, description="成功機率")
    
    # 擴展屬性
    properties: dict[str, Any] = Field(default_factory=dict, description="邊屬性")

class BaseAttackPath(BaseModel):
    """攻擊路徑基礎模型"""
    
    path_id: str = Field(description="路徑唯一識別ID")
    target_asset: str = Field(description="目標資產")
    
    # 路徑組成
    nodes: list[BaseAttackPathNode] = Field(description="路徑節點列表")
    edges: list[BaseAttackPathEdge] = Field(description="路徑邊列表")
    
    # 路徑評估
    overall_risk_score: float = Field(ge=0.0, le=10.0, default=0.0, description="整體風險評分")
    path_feasibility: float = Field(ge=0.0, le=1.0, default=0.0, description="路徑可行性")
    estimated_time_hours: float = Field(ge=0.0, default=0.0, description="預估攻擊時間(小時)")
    
    # 技能需求
    skill_level_required: SkillLevel = Field(description="所需技能等級")
    
    # 時間信息
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="發現時間")
    
    # 擴展元數據
    metadata: dict[str, Any] = Field(default_factory=dict, description="路徑元數據")
```

#### 3. 專業化擴展模型
```python
class EnhancedSIEMEvent(BaseSIEMEvent):
    """增強版SIEM事件 - 支援高級分析"""
    
    # 威脅情報
    threat_indicators: list[str] = Field(default_factory=list, description="威脅指標")
    ioc_matches: list[str] = Field(default_factory=list, description="IoC匹配")
    
    # 關聯分析
    related_events: list[str] = Field(default_factory=list, description="相關事件ID")
    correlation_score: float = Field(ge=0.0, le=1.0, default=0.0, description="關聯評分")
    
    # 響應信息
    response_actions: list[str] = Field(default_factory=list, description="響應動作")
    status: EventStatus = Field(default=EventStatus.NEW, description="事件狀態")

class EnhancedAttackPath(BaseAttackPath):
    """增強版攻擊路徑 - 支援複雜場景分析"""
    
    # 攻擊情境
    attack_scenario: str = Field(description="攻擊情境描述")
    prerequisites: list[str] = Field(default_factory=list, description="攻擊前提條件")
    
    # 防護評估
    current_defenses: list[str] = Field(default_factory=list, description="當前防護措施")
    defense_effectiveness: float = Field(ge=0.0, le=1.0, default=0.0, description="防護有效性")
    
    # 業務影響
    business_impact: BusinessImpact = Field(description="業務影響評估")
    affected_systems: list[str] = Field(default_factory=list, description="影響系統")
    
    # 修復建議
    recommendations: list[str] = Field(default_factory=list, description="修復建議")
    mitigation_priority: Priority = Field(description="緩解優先級")
```

## 🔄 遷移和向後兼容策略

### 階段1: 建立統一基礎 (高優先級)
1. 在 `aiva_common/schemas/security_events.py` 建立新的統一模型
2. 確保所有欄位向後兼容
3. 建立適配器支援舊格式轉換

### 階段2: 逐步遷移 (中優先級)
1. **services/aiva_common/schemas/telemetry.py** → 遷移至新基礎模型
2. **services/integration/models.py** → 使用統一標準或建立特化版本
3. **services/core/** → 統一攻擊路徑相關模型

### 階段3: 清理與優化 (低優先級)
1. 移除重複定義
2. 更新所有引用
3. 完善測試覆蓋

## 🎯 新增枚舉支援

需要定義的枚舉類型：
```python
class EventStatus(str, Enum):
    NEW = "new"
    ANALYZING = "analyzing"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"
    
class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    
class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
```

## 📊 預期效益

### ✅ 正面影響
- **代碼重用性**: 減少70%模型重複
- **維護效率**: 統一修改點，降低維護成本
- **型別一致性**: 統一型別系統，減少整合錯誤
- **擴展性**: 支援未來新的安全事件類型

### ⚠️ 實施風險
- **向後兼容**: 需要完善的適配器機制
- **功能覆蓋**: 確保統一模型涵蓋所有現有功能
- **性能影響**: 評估模型複雜度對性能的影響

## 📋 實施檢查清單

### Phase 1: 基礎建設
- [ ] 建立 `security_events.py` 模組
- [ ] 實作 `BaseSIEMEvent`、`BaseAttackPath` 等基礎模型
- [ ] 定義支援枚舉 (`EventStatus`、`SkillLevel` 等)
- [ ] 建立向後兼容適配器

### Phase 2: 遷移執行
- [ ] 遷移 telemetry.py 中的 SIEMEvent
- [ ] 統一 risk.py 中的 AttackPath 相關模型
- [ ] 更新 enhanced.py 中的增強版模型
- [ ] 清理 integration/models.py 重複定義

### Phase 3: 驗證與清理
- [ ] 執行全面測試
- [ ] 更新導入引用
- [ ] 運行合約健康檢查
- [ ] 文檔更新

---

**下一步**: 開始實作基礎安全事件模型並執行第一階段遷移