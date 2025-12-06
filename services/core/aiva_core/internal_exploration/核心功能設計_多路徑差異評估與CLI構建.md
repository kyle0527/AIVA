# Internal Exploration 核心功能設計 - 基於數據流分析

**日期**: 2025-12-06  
**分析基礎**: 268 個流程圖 + 10 個多路徑終點

---

## 一、「註冊」概念澄清

### 🔍 什麼是「能力註冊」？

**註冊 = 將發現的能力存入數據庫**

從代碼看 `capability_registry.py`:
```python
def register_capabilities(self, capabilities: list[ModuleCapability]) -> dict:
    """註冊/更新能力列表
    
    接收內循環掃描到的能力列表，進行批量註冊
    """
```

**完整流程**:
```
掃描代碼 → 識別能力 → 註冊到數據庫 → 供 RAG 查詢使用
```

**為什麼需要註冊？**
- 將 268 個流程圖中的能力**結構化存儲**
- 支持 AI 查詢「我有什麼攻擊 SQL 的能力？」
- 避免重複掃描，支持增量更新

---

## 二、多路徑終點差異評估系統

### 📊 發現的多路徑終點

**統計結果**: 10 個終點有多條路徑，需要評估差異

#### 最重要的多路徑案例

**1. `orchestrator` (能力編排器) - 3 種路徑**:
```
路徑 1: bio → trainer → rl → trainers → capability → orchestrator (11次)
路徑 2: initial → surface → exploit → orchestrator (1次)  
路徑 3: bio → trainer → model → trainer → training → orchestrator (1次)
```

**差異分析**:
- 路徑 1: **訓練路線** - 通過 RL 訓練生成能力
- 路徑 2: **攻擊路線** - 直接從攻擊入口進入編排
- 路徑 3: **模型路線** - 通過模型訓練進入編排

**2. `core` (核心) - 4 種路徑**:
```
路徑 1: bio → trainer → neural → network → optimized → core (3次)
路徑 2: bio → trainer → real → neural → core (1次)
路徑 3: bio → trainer → rl → trainers → real → neural → core (1次)
```

**差異分析**:
- 路徑 1: **優化路線** - 經過優化的標準路徑
- 路徑 2: **直達路線** - 跳過部分中間步驟
- 路徑 3: **RL路線** - 經過強化學習訓練

### 🎯 設計路徑差異評估功能

```python
class PathDifferenceAnalyzer:
    """路徑差異分析器 - 評估到達同一終點的不同路徑"""
    
    def analyze_path_differences(self, endpoint: str) -> PathAnalysisReport:
        """分析到達指定終點的路徑差異
        
        Args:
            endpoint: 終點名稱 (如 'orchestrator')
            
        Returns:
            PathAnalysisReport: 包含路徑效率、成本、風險評估
        """
        paths = self._get_all_paths_to_endpoint(endpoint)
        
        analysis = {
            "endpoint": endpoint,
            "total_paths": len(paths),
            "path_efficiency": {},
            "path_risks": {},
            "recommended_path": None
        }
        
        for path in paths:
            # 1. 效率評估 (路徑長度)
            efficiency = self._calculate_path_efficiency(path)
            
            # 2. 風險評估 (依賴節點的風險)
            risk = self._calculate_path_risk(path)
            
            # 3. 成本評估 (計算複雜度)
            cost = self._calculate_path_cost(path)
            
            analysis["path_efficiency"][path.id] = {
                "length": len(path.nodes),
                "efficiency_score": efficiency,
                "risk_score": risk,
                "cost_score": cost,
                "frequency": path.frequency  # 使用次數
            }
        
        # 4. 推薦最佳路徑
        analysis["recommended_path"] = self._select_best_path(paths)
        
        return PathAnalysisReport(**analysis)
    
    def _calculate_path_efficiency(self, path: Path) -> float:
        """計算路徑效率 (越短越好)"""
        base_score = 1.0 / len(path.nodes)  # 基礎效率
        frequency_bonus = path.frequency / 100  # 使用頻率加成
        return base_score + frequency_bonus
    
    def _calculate_path_risk(self, path: Path) -> float:
        """計算路徑風險 (依賴高風險節點)"""
        high_risk_nodes = ["rl", "trainer"]  # 94% 依賴的節點
        risk_score = 0
        for node in path.nodes:
            if node in high_risk_nodes:
                risk_score += 0.3  # 每個高風險節點 +0.3
        return min(risk_score, 1.0)
    
    def _select_best_path(self, paths: List[Path]) -> str:
        """選擇最佳路徑 (綜合效率、風險、頻率)"""
        best_score = -1
        best_path = None
        
        for path in paths:
            # 綜合評分: 效率 - 風險 + 頻率
            score = (
                self._calculate_path_efficiency(path) * 0.4 +  # 40% 權重
                (1 - self._calculate_path_risk(path)) * 0.3 +  # 30% 權重
                (path.frequency / max(p.frequency for p in paths)) * 0.3  # 30% 權重
            )
            
            if score > best_score:
                best_score = score
                best_path = path.id
        
        return best_path
```

---

## 三、CLI 指令建立功能

### 🎯 基於數據流鏈建立指令

你提到「建立 CLI 的功能，依照數據流練立」- 理解了！

**目標**: 將流程圖轉化為可執行的命令

#### 當前 CLI 狀況

**已存在的 CLI**:
```python
# aiva_flow_analyzer.py 第 1173 行
parser = argparse.ArgumentParser(
    description="AIVA Flow Analyzer - 產圖+組圖工具"
)
```

**功能**: 分析和生成流程圖

#### 設計新的 CLI: 能力執行器

```python
class CapabilityCommandBuilder:
    """能力指令構建器 - 將數據流轉化為可執行命令"""
    
    def build_command_from_flow(self, flow_path: str) -> ExecutableCommand:
        """從數據流路徑構建可執行命令
        
        Args:
            flow_path: 如 "bio → trainer → neural → network → rl → models"
            
        Returns:
            ExecutableCommand: 可執行的命令對象
        """
        nodes = flow_path.split(" → ")
        
        # 1. 驗證路徑可行性
        validation_result = self._validate_flow_path(nodes)
        if not validation_result.is_valid:
            raise FlowValidationError(validation_result.error_message)
        
        # 2. 構建執行步驟
        execution_steps = []
        for i, node in enumerate(nodes):
            if i == 0:
                # 起始節點：初始化
                step = self._create_init_step(node)
            elif i == len(nodes) - 1:
                # 終止節點：輸出結果
                step = self._create_output_step(node, nodes[i-1])
            else:
                # 中間節點：數據處理
                step = self._create_process_step(node, nodes[i-1])
            
            execution_steps.append(step)
        
        # 3. 生成命令
        command = ExecutableCommand(
            flow_id=self._generate_flow_id(nodes),
            steps=execution_steps,
            estimated_duration=self._estimate_duration(nodes),
            required_capabilities=self._extract_required_capabilities(nodes)
        )
        
        return command
    
    def _validate_flow_path(self, nodes: List[str]) -> ValidationResult:
        """驗證流程路徑是否可執行"""
        # 檢查每個節點是否存在對應的實現
        for node in nodes:
            if not self._node_exists(node):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"節點 '{node}' 沒有對應的實現"
                )
        
        # 檢查節點間連接是否有效
        for i in range(len(nodes) - 1):
            from_node, to_node = nodes[i], nodes[i+1]
            if not self._connection_exists(from_node, to_node):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"節點 '{from_node}' 到 '{to_node}' 的連接無效"
                )
        
        return ValidationResult(is_valid=True)
    
    def _create_process_step(self, node: str, prev_node: str) -> ExecutionStep:
        """創建處理步驟"""
        # 根據節點類型生成對應的執行步驟
        node_mapping = {
            "trainer": "python -m external_learning.learning.scalable_bio_trainer",
            "neural": "python -m cognitive_core.neural.ai_model_manager --mode=neural",
            "network": "python -m cognitive_core.neural.ai_model_manager --mode=network",
            "rl": "python -m external_learning.learning.rl_trainers --mode=dqn",
            "models": "python -m external_learning.learning.rl_models --save-model"
        }
        
        base_command = node_mapping.get(node, f"echo 'Processing {node}'")
        
        return ExecutionStep(
            name=f"process_{node}",
            command=base_command,
            input_from=prev_node,
            output_to=f"{node}_output",
            timeout=300  # 5 分鐘超時
        )
```

### 📋 CLI 使用範例

```bash
# 1. 列出所有可用的數據流路徑
python capability_cli.py list-flows --endpoint models

# 輸出:
# 可用路徑:
# 1. bio → trainer → neural → network → rl → models (155次使用)
# 2. bio → trainer → rl → models (29次使用)

# 2. 執行特定路徑
python capability_cli.py execute-flow "bio → trainer → neural → network → rl → models"

# 輸出:
# ✅ 驗證路徑... 成功
# 🔄 執行步驟 1/5: 初始化 bio_trainer
# 🔄 執行步驟 2/5: 處理 trainer
# 🔄 執行步驟 3/5: 處理 neural
# 🔄 執行步驟 4/5: 處理 network  
# 🔄 執行步驟 5/5: 處理 rl
# ✅ 生成模型: ./output/rl_models_20251206.pkl

# 3. 比較路徑差異
python capability_cli.py compare-paths orchestrator

# 輸出:
# 📊 orchestrator 的 3 種路徑分析:
# 路徑 1: bio → ... → orchestrator (推薦, 效率: 0.8, 風險: 0.2)
# 路徑 2: initial → ... → orchestrator (不推薦, 效率: 0.6, 風險: 0.7)

# 4. 驗證能力可用性
python capability_cli.py validate-capability "SQL注入攻擊"

# 輸出:
# 🔍 查找能力: SQL注入攻擊
# ✅ 找到 3 個相關流程:
# 1. exploit → sql_injector (可用)
# 2. scanner → vulnerability_detector → sql_injector (可用)
# 3. trainer → rl → attack_planner → sql_injector (不可用，缺少 attack_planner 實現)
```

---

## 四、整合模組資料存儲

### 📁 數據存儲架構

從整合模組文檔可見，AIVA 有統一的資料儲存標準：

```python
# 整合模組資料儲存路徑
data_storage_paths = {
    "flow_analysis": "./flow_analysis_results/",      # 流程分析結果
    "capability_metadata": "./capability_metadata/", # 能力元數據  
    "path_analysis": "./path_analysis_results/",     # 路徑分析結果
    "execution_logs": "./execution_logs/",           # 執行日誌
    "model_weights": "./model_weights/",             # 模型權重
    "training_data": "./training_data/"              # 訓練數據
}
```

### 🗄️ 資料庫整合設計

```python
class IntegratedDataManager:
    """整合數據管理器 - 統一管理所有內部探索數據"""
    
    def __init__(self):
        self.flow_db = FlowAnalysisDB()      # 流程圖數據庫
        self.capability_db = CapabilityDB()  # 能力數據庫
        self.path_db = PathAnalysisDB()      # 路徑分析數據庫
        self.execution_db = ExecutionDB()    # 執行記錄數據庫
    
    def store_flow_analysis_results(self, flow_dir: Path):
        """存儲流程分析結果到整合模組"""
        # 1. 解析所有流程圖
        flows = self._parse_all_flows(flow_dir)
        
        # 2. 存入流程數據庫
        self.flow_db.batch_insert(flows)
        
        # 3. 生成路徑分析
        path_analysis = self._analyze_paths(flows)
        self.path_db.store_analysis(path_analysis)
        
        # 4. 更新能力註冊
        capabilities = self._extract_capabilities(flows)
        self.capability_db.register_capabilities(capabilities)
        
        # 5. 同步到整合模組
        self._sync_to_integration_module(flows, capabilities)
    
    def query_executable_capabilities(self, query: str) -> List[ExecutableCapability]:
        """查詢可執行的能力"""
        # 1. RAG 查詢匹配能力
        raw_capabilities = self.capability_db.vector_search(query)
        
        # 2. 檢查可執行性
        executable_caps = []
        for cap in raw_capabilities:
            # 驗證對應的流程是否完整
            flow_paths = self.flow_db.get_paths_to_capability(cap.name)
            
            for path in flow_paths:
                validation = self._validate_path_executability(path)
                if validation.is_executable:
                    executable_caps.append(ExecutableCapability(
                        capability=cap,
                        execution_path=path,
                        estimated_success_rate=validation.success_rate
                    ))
        
        return executable_caps
```

---

## 五、完整功能實現計劃

### 🎯 階段 1: 路徑差異評估 (立即實施)

**實現文件**: `path_difference_analyzer.py`

```python
# 主要功能
class PathDifferenceAnalyzer:
    def analyze_all_endpoints(self) -> Dict[str, PathAnalysisReport]
    def compare_path_efficiency(self, endpoint: str) -> EfficiencyReport
    def recommend_optimal_path(self, endpoint: str, criteria: str) -> str
```

### 🎯 階段 2: 能力分類與比較 (優先實施)

**基於最後一個腳本分析的能力分類**:

```python
# capability_classifier.py
class CapabilityClassifier:
    def classify_by_endpoint(self) -> Dict[str, List[Capability]]
    def compare_capabilities(self, cap1: str, cap2: str) -> ComparisonReport
    def find_capability_alternatives(self, capability: str) -> List[Alternative]
```

### 🎯 階段 3: CLI 指令建立 (重要實施)

**實現文件**: `capability_cli.py`

```python
# 主要命令
commands = [
    "list-flows",           # 列出所有流程
    "execute-flow",         # 執行指定流程  
    "validate-capability",  # 驗證能力可用性
    "compare-paths",        # 比較路徑差異
    "build-command",        # 構建執行命令
    "test-flow"            # 測試流程可行性
]
```

### 🎯 階段 4: 整合模組同步 (最終實施)

**與整合模組的資料同步**:

```python
# integration_sync.py
class IntegrationModuleSync:
    def sync_flow_analysis_to_integration(self)
    def sync_capability_registry_to_integration(self)
    def export_executable_commands_to_integration(self)
```

---

## 六、驗證測試

### 測試用例 1: 路徑差異評估
```python
analyzer = PathDifferenceAnalyzer()
report = analyzer.analyze_path_differences("orchestrator")

assert len(report.paths) == 3
assert report.recommended_path == "bio → trainer → rl → trainers → capability → orchestrator"
```

### 測試用例 2: CLI 指令建立
```bash
python capability_cli.py execute-flow "bio → trainer → rl → models" --dry-run
# 應該輸出執行步驟但不實際執行
```

### 測試用例 3: 能力可用性驗證  
```python
capabilities = data_manager.query_executable_capabilities("DQN 訓練")
assert len(capabilities) > 0
assert all(cap.execution_path for cap in capabilities)
```

---

## 總結

**你的需求分析**:
1. ✅ **註冊概念**: 將 268 個流程圖中的能力存入數據庫
2. ✅ **路徑差異評估**: 10 個多路徑終點需要效率/風險/成本比較  
3. ✅ **能力分類比較**: 基於終點類型分類，比較不同實現方式
4. ✅ **CLI 指令建立**: 將數據流轉化為可執行命令
5. ✅ **整合模組**: 統一資料存儲和管理

**實現優先級**:
1. 路徑差異分析器 (解決多路徑問題)
2. 能力分類器 (解決分類比較問題)  
3. CLI 指令構建器 (解決可執行性問題)
4. 整合模組同步 (解決資料統一問題)