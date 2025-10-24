# 架構圖表自動化組合優化框架

## 🎯 目標
建立一套通用的方法論，優化從腳本產出的大量單一圖表到有意義組合圖表的過程，適用於 AIVA 系統的所有模組。

## 📊 當前問題分析

### 問題模式識別
基於掃描模組的分析，發現以下普遍問題：

1. **數量爆炸**：單一模組產生 289+ 個細粒度圖表
2. **重複冗餘**：相似功能的圖表重複產生  
3. **缺乏層次**：無法區分核心架構與實作細節
4. **組合困難**：缺乏自動化的圖表分類和組合機制
5. **維護負擔**：手動管理數百個檔案的成本過高

### 成功經驗總結
通過掃描模組的組圖過程，識別出以下成功模式：
- 策略驅動的架構發現
- 分層抽象的視覺呈現
- 關鍵路徑的突出顯示
- 語法相容性的確保

---

## 🏗️ 優化框架設計

### 階段一：智能分類系統

```python
@dataclass
class DiagramClassification:
    """圖表分類元資料"""
    category: Literal["core", "detail", "integration", "example"]
    priority: int  # 1-10，數字越小優先級越高
    complexity: Literal["low", "medium", "high"]
    dependencies: List[str]
    abstraction_level: Literal["system", "module", "component", "function"]

class DiagramAnalyzer:
    """自動圖表分析器"""
    
    def classify_diagram(self, diagram_path: str) -> DiagramClassification:
        """基於檔案名稱、內容和依賴關係自動分類"""
        
        # 檔名模式匹配
        filename_patterns = {
            r".*_Module\.mmd$": ("core", 1, "medium", "module"),
            r".*_Function_.*__init__\.mmd$": ("detail", 8, "low", "function"),
            r".*_integration_.*\.mmd$": ("integration", 3, "medium", "component"),
            r".*_examples?_.*\.mmd$": ("example", 9, "low", "function"),
            r"\d{2}_.*\.mmd$": ("core", 1, "high", "system")  # 手動核心圖
        }
        
        # 內容複雜度分析
        content = self.read_diagram_content(diagram_path)
        complexity = self.analyze_complexity(content)
        
        # 依賴關係分析
        dependencies = self.extract_dependencies(content)
        
        return DiagramClassification(...)
    
    def analyze_complexity(self, content: str) -> str:
        """分析圖表複雜度"""
        node_count = content.count("-->") + content.count("-.->")
        subgraph_count = content.count("subgraph")
        
        if node_count > 20 or subgraph_count > 3:
            return "high"
        elif node_count > 8 or subgraph_count > 1:
            return "medium"
        else:
            return "low"
```

### 階段二：智能組合引擎

```python
class DiagramComposer:
    """圖表組合引擎"""
    
    def create_module_overview(self, module_diagrams: List[DiagramInfo]) -> str:
        """創建模組概覽圖"""
        
        # 1. 識別核心組件
        core_components = self.identify_core_components(module_diagrams)
        
        # 2. 分析組件間關係
        relationships = self.analyze_relationships(core_components)
        
        # 3. 生成分層架構
        architecture_layers = self.generate_layers(core_components, relationships)
        
        # 4. 創建 Mermaid 語法
        return self.generate_mermaid_syntax(architecture_layers)
    
    def identify_core_components(self, diagrams: List[DiagramInfo]) -> List[Component]:
        """識別核心組件"""
        
        # 基於以下規則識別核心組件：
        rules = [
            # 1. 高優先級且被多個圖表引用
            lambda d: d.classification.priority <= 3 and d.reference_count > 2,
            
            # 2. 模組級別的圖表
            lambda d: d.classification.abstraction_level == "module",
            
            # 3. 包含關鍵字的組件
            lambda d: any(keyword in d.name.lower() 
                         for keyword in ["controller", "manager", "orchestrator", "engine"])
        ]
        
        core_components = []
        for diagram in diagrams:
            if any(rule(diagram) for rule in rules):
                core_components.append(self.extract_component(diagram))
                
        return core_components
    
    def generate_layers(self, components: List[Component], 
                       relationships: List[Relationship]) -> Dict[str, List[Component]]:
        """生成分層架構"""
        
        layers = {
            "interface": [],      # 介面層
            "control": [],        # 控制層
            "service": [],        # 服務層  
            "data": [],          # 資料層
            "integration": []     # 整合層
        }
        
        # 基於組件類型和依賴關係分配到不同層級
        for component in components:
            layer = self.determine_layer(component, relationships)
            layers[layer].append(component)
            
        return layers
```

### 階段三：品質保證系統

```python
class DiagramQualityAssurance:
    """圖表品質保證"""
    
    def validate_syntax(self, mermaid_code: str) -> ValidationResult:
        """驗證 Mermaid 語法"""
        # 使用 Mermaid 驗證器
        pass
    
    def check_completeness(self, generated_diagram: str, 
                          source_diagrams: List[str]) -> CompletenessReport:
        """檢查組合圖表的完整性"""
        
        # 1. 確保核心組件都包含在內
        # 2. 檢查關鍵關係是否遺漏  
        # 3. 驗證抽象層次的一致性
        pass
    
    def optimize_layout(self, diagram: str) -> str:
        """優化圖表佈局"""
        
        # 1. 減少交叉線
        # 2. 平衡子圖大小
        # 3. 優化標籤可讀性
        pass
```

---

## 🔄 自動化工作流程

### 工作流程設計

```yaml
# diagram_optimization_workflow.yml
name: 架構圖表自動化組合

on:
  - script_generation_complete
  - manual_trigger

jobs:
  analyze_and_classify:
    steps:
      - name: "掃描圖表檔案"
        uses: ./scripts/scan_diagrams.py
        
      - name: "自動分類"
        uses: ./scripts/classify_diagrams.py
        with:
          input_dir: "_out/architecture_diagrams"
          output: "diagram_classification.json"
          
  generate_compositions:
    needs: analyze_and_classify
    steps:
      - name: "生成模組概覽圖"
        uses: ./scripts/compose_module_overview.py
        
      - name: "生成整合架構圖"  
        uses: ./scripts/compose_integration_diagram.py
        
      - name: "生成問題分析報告"
        uses: ./scripts/analyze_architecture_issues.py
        
  quality_assurance:
    needs: generate_compositions  
    steps:
      - name: "驗證語法"
        uses: ./scripts/validate_mermaid_syntax.py
        
      - name: "檢查完整性"
        uses: ./scripts/check_completeness.py
        
      - name: "優化佈局"
        uses: ./scripts/optimize_layout.py
        
  cleanup_and_archive:
    needs: quality_assurance
    steps:
      - name: "清理冗餘檔案"
        uses: ./scripts/cleanup_redundant_diagrams.py
        with:
          preserve_patterns:
            - "**/SCAN_MODULE_*.mmd"
            - "**/\d{2}_*.mmd"  # 手動核心圖
            - "**/*_INTEGRATED_*.mmd"  # 組合圖
            
      - name: "歸檔原始檔案"
        uses: ./scripts/archive_source_diagrams.py
```

### 配置驅動的模組適配

```yaml
# module_optimization_config.yml
modules:
  scan:
    core_patterns:
      - "*_strategy_controller_*"
      - "*_config_control_center_*"
      - "*_scan_orchestrator_*"
    integration_patterns:
      - "*_integration_*scan*"
    priority_keywords:
      - "orchestrator"
      - "controller" 
      - "manager"
      - "engine"
      
  analysis:
    core_patterns:
      - "*_risk_assessment_*"
      - "*_correlation_analyzer_*"
    integration_patterns:
      - "*_integration_analysis_*"
    priority_keywords:
      - "analyzer"
      - "engine"
      - "processor"
      
  reception:
    core_patterns:
      - "*_lifecycle_manager_*"
      - "*_data_reception_*"
    integration_patterns:
      - "*_integration_reception_*"
    priority_keywords:
      - "manager"
      - "repository"
      - "handler"
```

---

## 📈 預期效益

### 量化指標
- **檔案數量減少 80%**：從 300+ 個細節圖減少到 20-30 個有意義的組合圖
- **維護時間節省 70%**：自動化分類和組合減少手動工作
- **架構洞察提升 50%**：通過智能組合發現隱藏的架構模式
- **品質一致性提升 90%**：標準化的驗證和優化流程

### 定性改進
- **可讀性**：分層抽象使架構更易理解
- **可維護性**：自動化流程減少人工錯誤
- **可擴展性**：配置驅動的方法適用於所有模組
- **可重用性**：建立的框架可應用於其他專案

---

## 🚀 實施計劃

### 第一階段（2週）：核心框架開發
1. 實現 DiagramAnalyzer 類別
2. 建立基礎的分類規則
3. 創建簡單的組合演算法
4. 驗證掃描模組的效果

### 第二階段（3週）：自動化工作流程
1. 實現完整的 DiagramComposer
2. 建立品質保證系統
3. 創建配置驅動的模組適配
4. 測試其他模組（analysis, reception）

### 第三階段（2週）：優化和推廣
1. 效能優化和錯誤處理
2. 建立詳細的使用文件
3. 訓練團隊使用新流程
4. 建立持續改進機制

---

## ⚠️ 重要使用須知：檔案管理策略

### **問題背景與核心理念**

**現象**: `diagram_auto_composer.py` 會產生大量個別組件圖檔（301個），造成目錄混亂。

**核心理念**: **完整產出優於預先過濾**
- 🎯 **無法預知價值**: 在分析前無法確定哪些組件包含關鍵架構洞察
- 🔍 **發現驚喜**: 最重要的架構模式往往隱藏在看似次要的組件中
- 🛡️ **零風險策略**: 寧可產出 300 個圖後刪除 295 個，也不要遺漏 1 個關鍵發現
- 💡 **笨方法智慧**: 先全面掃描，再智能篩選，確保完整性

### **必要的檔案管理流程**

#### **🔄 標準作業流程**
```bash
# 1. 執行圖表分析和組合
python scripts/diagram_auto_composer.py --module [MODULE_NAME]

# 2. 立即備份重要檔案
cp _out/[MODULE]_INTEGRATED_ARCHITECTURE.mmd backup/

# 3. 清理自動產生的個別圖檔  
Remove-Item "_out/architecture_diagrams/aiva_[MODULE]*" -Confirm:$false

# 4. 保留關鍵檔案
# ✅ [MODULE]_INTEGRATED_ARCHITECTURE.mmd (手工整合)
# ✅ [MODULE]_AUTO_INTEGRATED.mmd (自動產出參考) 
# ✅ [module]_diagram_classification.json (分類數據)
# ✅ [MODULE]_ARCHITECTURE_ANALYSIS.md (分析報告)
```

#### **📋 檔案保留決策矩陣**

| 檔案類型 | 保留 | 刪除 | 原因 |
|---------|------|------|------|
| 手工整合架構圖 | ✅ | | 經過人工優化，語法正確 |
| 自動產出整合圖 | ✅ | | 作為參考和改進基準 |
| 分類數據 JSON | ✅ | | 重要的分析資產 |
| 分析報告 MD | ✅ | | 架構洞察和問題識別 |
| 個別函數圖 | | ❌ | 過於細碎，重複率高 |
| 個別模組圖 | | ❌ | 缺乏整合，語法錯誤多 |

#### **⚡ 自動化改進建議**

**在腳本中加入清理選項**：
```python
def main():
    # 執行分析和組合
    analyzer = DiagramAnalyzer()
    composer = DiagramComposer()
    
    # 產生結果
    results = analyzer.analyze_module(module_name)
    integrated_diagram = composer.compose_integrated_diagram(results)
    
    # 可選：自動清理
    if args.auto_cleanup:
        cleanup_individual_diagrams(output_dir)
        print(f"🧹 已清理 {cleanup_count} 個個別組件圖")
        print("📋 保留重要整合圖和分析數據")
```

**在 CLI 中加入選項**：
```bash
python scripts/diagram_auto_composer.py \
  --module scan \
  --auto-cleanup \
  --keep-integrated-only
```

### **💡 最佳實踐建議**

#### **核心原則：完整性第一，效率第二**

1. **完整產出策略** - 永遠先產生所有可能的組件圖
   ```
   ✅ 產出 301 個圖 → 發現 6 個關鍵模式 → 刪除 295 個
   ❌ 預篩選產出 20 個圖 → 可能遺漏最重要的 1 個洞察
   ```

2. **人工價值判斷** - 機器分類 + 人工最終決策
   ```python
   # 機器：提供分類和統計
   classification = analyzer.classify_components()
   
   # 人工：基於完整理解做最終決策
   valuable_patterns = human_review(classification)
   ```

3. **延遲清理策略** - 理解後再清理，不要邊產出邊刪除
   - 📊 先完整分析和分類
   - 🧠 人工識別真正的價值
   - 🗑️ 最後階段批量清理

4. **保護意外發現** - 為突然的靈感留空間
   ```bash
   # 好的做法：分階段進行
   python diagram_auto_composer.py        # 1. 完整產出
   # 花時間分析和理解...                    # 2. 深度分析  
   python cleanup_diagram_output.py       # 3. 智能清理
   ```

**笨方法的智慧：寧可事後清理 1000 個檔案，也不要事前遺漏 1 個關鍵洞察！**

---

*這個框架將成為 AIVA 專案架構視覺化的標準方法，確保所有模組都能產生高品質、有意義的架構圖表。*