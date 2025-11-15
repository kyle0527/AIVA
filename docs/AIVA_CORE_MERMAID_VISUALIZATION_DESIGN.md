# AIVA Core Mermaid 組圖視覺化設計文件

**文件版本**: v1.0  
**創建日期**: 2025年11月14日  
**設計目標**: 為 AIVA Core 現有分析功能提供 Mermaid 組圖視覺化支援

## 📑 目錄

- [📊 組圖邏輯架構設計](#組圖邏輯架構設計)
  - [1. 整合位置建議](#1-整合位置建議)
- [🎯 核心組圖邏輯設計](#核心組圖邏輯設計)
  - [A. AI 分析結果組圖](#a-ai-分析結果組圖)
  - [B. 攻擊路徑圖](#b-攻擊路徑圖)
  - [C. 系統依賴圖](#c-系統依賴圖)
  - [D. 執行流程圖](#d-執行流程圖)
- [🔧 技術實現架構](#技術實現架構)
  - [1. 組圖生成器設計](#1-組圖生成器設計)
  - [2. 圖表配置管理](#2-圖表配置管理)
  - [3. 導出功能](#3-導出功能)
- [📊 具體組圖類型與實現](#具體組圖類型與實現)
- [🎨 圖表樣式與主題](#圖表樣式與主題)
- [⚙️ 配置與客製化](#配置與客製化)
- [🚀 部署與整合建議](#部署與整合建議)
- [📈 效能考量](#效能考量)
- [🔮 未來擴展計畫](#未來擴展計畫)

---

## 📊 組圖邏輯架構設計

### 1. **整合位置建議**

**推薦方案**: 將 Mermaid 組圖功能整合至 `services/core/aiva_core/analysis/` 模組

**理由**:
- Analysis 模組已具備完整的分析能力
- 組圖是分析結果的視覺化展現
- 避免創建獨立模組，保持架構簡潔
- 與現有分析工作流自然整合

---

## 🎯 核心組圖邏輯設計

### **A. AI 分析結果組圖**

基於 `ai_analysis/analysis_engine.py` 的分析結果：

```
AIAnalysisResult → Mermaid 組圖轉換器 → 視覺化圖表
```

**支援的圖表類型**:
1. **安全風險分布圖**: 危險函數、SQL注入風險、硬編碼密碼分佈
2. **複雜度熱力圖**: 循環複雜度、嵌套深度視覺化
3. **架構關係圖**: 組件依賴關係和交互流程
4. **漏洞檢測結果圖**: 漏洞類型分佈和嚴重程度

### **B. 技能圖網絡組圖**

基於 `decision/skill_graph.py` 的技能關係：

```
NetworkX Graph → Mermaid 轉換 → 技能關係圖
```

**支援的圖表類型**:
1. **技能關係網絡圖**: 能力節點和依賴關係
2. **執行路徑流程圖**: 從起點到目標的技能路徑
3. **中心性分析圖**: 關鍵技能節點識別
4. **技能推薦圖**: 相關技能建議鏈

### **C. 攻擊面分析組圖**

基於 `analysis/initial_surface.py` 的攻擊面分析：

```
AttackSurfaceAnalysis → Mermaid 生成 → 攻擊面視覺圖
```

**支援的圖表類型**:
1. **攻擊向量分布圖**: XSS、SQLi、SSRF、IDOR 候選點分佈
2. **風險評估矩陣圖**: 資產重要性 vs 威脅程度
3. **攻擊路徑流程圖**: 從入口點到目標的攻擊鏈
4. **測試策略規劃圖**: 測試任務的優先級和順序

### **D. 計畫執行對比組圖**

基於 `analysis/plan_comparator.py` 的對比分析：

```
PlanExecutionMetrics → 視覺化對比 → 執行差異圖
```

**支援的圖表類型**:
1. **計畫 vs 執行對比圖**: 預期步驟與實際軌跡對比
2. **步驟匹配分析圖**: 精確匹配、部分匹配、遺漏步驟
3. **執行效能圖表**: 時間軸、成功率、錯誤分佈
4. **獎勵分數趨勢圖**: AI 學習進度和改進趨勢

---

## 🔧 技術實現方案

### **1. 組圖生成器設計**

```python
class AnalysisVisualizer:
    """分析結果視覺化生成器"""
    
    def generate_ai_analysis_diagram(self, results: Dict[AnalysisType, AIAnalysisResult]) -> str
    def generate_skill_graph_diagram(self, skill_graph: nx.DiGraph) -> str
    def generate_attack_surface_diagram(self, attack_surface: AttackSurfaceAnalysis) -> str
    def generate_plan_comparison_diagram(self, metrics: PlanExecutionMetrics) -> str
```

### **2. Mermaid 語法模板系統**

**模板分類**:
- **流程圖 (Flowchart)**: 攻擊路徑、執行流程
- **網絡圖 (Graph)**: 技能關係、組件依賴  
- **序列圖 (Sequence)**: 時間軸執行過程
- **甘特圖 (Gantt)**: 測試任務排程
- **餅圖 (Pie)**: 風險分佈、漏洞類型統計
- **桑基圖 (Sankey)**: 數據流向和轉換

### **3. 動態配置系統**

```python
@dataclass
class DiagramConfig:
    """圖表配置"""
    diagram_type: str  # flowchart, graph, sequence, pie, etc.
    theme: str        # default, dark, base, forest, etc.
    direction: str    # TD, LR, BT, RL
    max_nodes: int    # 節點數量限制
    show_labels: bool # 是否顯示標籤
    color_scheme: Dict[str, str] # 顏色配置
```

---

## 🚀 整合工作流建議

### **Phase 1: 基礎整合**
1. 在 `analysis/` 目錄下創建 `diagram_generator.py`
2. 實現基礎的 AI 分析結果轉 Mermaid 功能
3. 整合到現有的 `AIAnalysisEngine` 工作流

### **Phase 2: 功能擴展**
1. 添加技能圖視覺化支援
2. 實現攻擊面分析圖表生成
3. 添加計畫對比分析圖表

### **Phase 3: 高級特性**
1. 實現交互式圖表配置
2. 添加圖表匯出功能 (PNG, SVG, PDF)
3. 整合到 Dashboard UI 中

---

## 📈 具體使用場景

### **場景 1: 代碼安全分析報告**
```python
# AI 分析 + 組圖生成
analysis_engine = AIAnalysisEngine()
results = analysis_engine.analyze_code(source_code, file_path)

# 生成視覺化圖表
visualizer = AnalysisVisualizer()
security_diagram = visualizer.generate_ai_analysis_diagram(results)
```

### **場景 2: 攻擊路徑規劃**
```python
# 技能圖分析 + 路徑視覺化
skill_graph = AIVASkillGraph()
paths = skill_graph.find_execution_path(start, goal)

# 生成路徑圖
path_diagram = visualizer.generate_skill_path_diagram(paths)
```

### **場景 3: 測試策略視覺化**
```python
# 攻擊面分析 + 策略圖表
surface_analyzer = InitialAttackSurface()
attack_surface = surface_analyzer.compute(scan_results)

# 生成測試策略圖
strategy_diagram = visualizer.generate_attack_surface_diagram(attack_surface)
```

---

## 🎨 視覺設計原則

### **色彩編碼標準**
- 🔴 **高風險/嚴重**: `#FF6B6B` (紅色)
- 🟡 **中風險/警告**: `#FFD93D` (黃色)  
- 🟢 **低風險/安全**: `#6BCF7F` (綠色)
- 🔵 **信息/正常**: `#4ECDC4` (藍色)
- ⚪ **未知/中性**: `#95A5A6` (灰色)

### **節點形狀約定**
- `[]` **方形**: 功能模組、組件
- `()` **圓形**: 開始/結束節點
- `{}` **菱形**: 決策點、判斷
- `((雙圓))` **雙圓**: 重要節點、關鍵點
- `[/平行四邊形/]` **平行四邊形**: 輸入/輸出

### **連接線樣式**
- `-->` **實線箭頭**: 正常流程、依賴關係
- `-.->` **虛線箭頭**: 可選路徑、建議關係  
- `==>` **粗箭頭**: 重要流程、主要路徑
- `--x` **錯誤連接**: 失敗路徑、錯誤流程

---

## 🔄 與現有系統整合點

### **1. Analysis Module 整合**
- 擴展 `__init__.py` 添加 `AnalysisVisualizer` 導出
- 在 `AIAnalysisEngine` 中添加可選的圖表生成
- 修改 `PlanComparator` 支援圖表輸出

### **2. UI Panel 整合**
- 在 `ui_panel/dashboard.py` 中添加圖表顯示
- 修改 `ui_panel/server.py` 添加圖表端點
- 整合到 Web UI 的分析結果頁面

### **3. AI Controller 整合**
- 在 `ai_controller.py` 的報告生成中包含圖表
- 添加圖表格式的摘要報告選項
- 支援批量圖表生成和匯出

---

## 📝 開發優先級建議

### **高優先級** (立即實現)
1. ✅ AI 安全分析結果圖表
2. ✅ 攻擊面分析視覺化
3. ✅ 基礎 Mermaid 語法生成

### **中優先級** (後續迭代)
1. 🔶 技能圖網絡視覺化
2. 🔶 計畫執行對比圖表
3. 🔶 交互式圖表配置

### **低優先級** (長期規劃)
1. 🔸 高級圖表類型支援
2. 🔸 圖表動畫和交互
3. 🔸 自定義主題系統

---

**總結**: 此設計將 Mermaid 組圖功能無縫整合到現有的分析工作流中，為各種 AI 分析結果提供直觀的視覺化支援，提升 AIVA Core 的可用性和分析洞察能力。