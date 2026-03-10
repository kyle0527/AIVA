# AIVA 經驗學習規劃設計

**生成時間**: 2025-12-13  
**基於**: 雙閉環架構 + 實際可獲得數據

---

## 📑 目錄

- [🎯 核心原則](#-核心原則)
- [📊 可獲得的數據類型](#-可獲得的數據類型data-sources)
  - [內閉環數據](#1-內閉環數據-internal-loop)
  - [外閉環數據](#2-外閉環數據-external-loop)
- [🧠 學習系統設計](#-學習系統設計)
- [📈 實施路線圖](#-實施路線圖)

---

## 🎯 核心原則

**基於實際可獲得的數據進行學習，不做白工**

根據 AIVA 雙閉環架構分析，AI 可以獲得以下類型的數據：

---

## 📊 可獲得的數據類型（Data Sources）

### 1. **內閉環數據** (Internal Loop)
來源: `internal_loop_connector.py` + `internal_exploration`

#### 1.1 能力數據 ([ModuleCapability](../aiva_common/schemas/dual_loop.py#L240-L314))
```python
{
    "capability_id": "flow_123",
    "name": "exec_flow_123",
    "module": "aiva_flows",
    "function": "execute_flow",
    "language": "python",  # CLI 格式類型
    "file_path": "latest_classification.json",
    "description": "[FLOW] ID:123 | LEN:5 | MOD:scanner | PATH:...",
    
    "category": "SCANNING",  # 能力類別
    "sub_category": "PORT_SCAN",
    "complexity": 3,  # 1-5
    "tags": ["flow", "scanner", "network"],
    
    "parameters": [...],  # 參數定義
    "invocation": {  # 調用元數據
        "protocol": "cli_python_flow",
        "cli_command": "python -m module",
        "cli_args": ["--flow", "{flow_id}"],
        "timeout_seconds": 300
    },
    
    "health_score": 1.0,  # 健康分數
    "availability": 1.0,
    "error_rate": 0.0,
    "avg_latency_ms": null,
    "last_used": null
}
```

**可學習內容**：
- ✅ 哪些能力可用
- ✅ 每種能力的成功率（health_score）
- ✅ 每種能力的錯誤率（error_rate）
- ✅ 每種能力的執行時間（avg_latency_ms）
- ✅ 能力之間的依賴關係
- ❌ **無法學習**：能力的實際效果（需要外閉環數據）

---

#### 1.2 數據流分析 ([CompleteDataFlow](../aiva_common/schemas/dual_loop.py#L141-L169))
```python
{
    "flow_id": "flow_001",
    "entry_point": "main",
    "exit_point": "return_result",
    "path": [
        {
            "node_id": "node_1",
            "function_signature": "ClassName.method_name",
            "file_path": "/path/to/file.py",
            "module_name": "scanner",
            "is_async": true,
            "parameters": [...]
        }
    ],
    "path_length": 5,
    "crosses_files": true,
    "file_count": 3
}
```

**可學習內容**：
- ✅ 功能調用路徑
- ✅ 跨文件調用模式
- ✅ 異步/同步執行方式
- ❌ **無法學習**：路徑的實際執行效果

---

#### 1.3 問題診斷 ([BrokenChainDiagnosis](../aiva_common/schemas/dual_loop.py#L172-L187))
```python
{
    "caller_signature": "Caller.method",
    "caller_file": "/path/caller.py",
    "missing_function": "MissingFunc",
    "line_number": 42,
    "possible_causes": [
        "Function not implemented",
        "Import missing"
    ],
    "severity": "error",
    "auto_fix_suggestion": "Implement function or add import"
}
```

**可學習內容**：
- ✅ 系統中的斷鏈問題
- ✅ 潛在的修復方案
- ❌ **無法學習**：修復後的實際效果

---

### 2. **外閉環數據** (External Loop)
來源: `integration/coordinators` + Features 執行結果

#### 2.1 任務執行記錄 ([ExecutionTrace](tracing/trace_recorder.py#L60-L104))
```python
{
    "trace_session_id": "trace_abc123",
    "plan_id": "plan_001",
    "start_time": "2025-12-13T10:00:00Z",
    "end_time": "2025-12-13T10:05:00Z",
    "entries": [
        {
            "trace_id": "entry_001",
            "timestamp": "2025-12-13T10:00:01Z",
            "trace_type": "TASK_START",
            "task_id": "task_001",
            "content": {...},
            "metadata": {...}
        },
        {
            "trace_type": "HTTP_REQUEST",
            "content": {
                "method": "POST",
                "url": "https://target.com/api/login",
                "payload": "..."
            }
        },
        {
            "trace_type": "TOOL_OUTPUT",
            "content": {
                "tool": "sqlmap",
                "result": "SQL injection found"
            }
        },
        {
            "trace_type": "DECISION",
            "content": {
                "decision": "EXPLOIT_SQL_INJECTION",
                "confidence": 0.85,
                "reasoning": "..."
            }
        }
    ]
}
```

**可學習內容**：
- ✅ **執行軌跡**：每一步的操作序列
- ✅ **決策點**：AI 在何時做了什麼決定
- ✅ **工具輸出**：每個工具的實際返回結果
- ✅ **HTTP 交互**：請求和響應的完整記錄
- ✅ **執行時間**：每一步花費的時間

---

#### 2.2 漏洞發現結果 (Features Output)
```python
{
    "task_id": "uuid",
    "feature_module": "function_xss",
    "status": "completed",
    "success": true,
    "duration_ms": 1234,
    
    "findings": [
        {
            "id": "vuln_001",
            "vulnerability_type": "XSS",
            "severity": "high",
            "cvss_score": 7.5,
            "cwe_id": "CWE-79",
            "owasp_category": "A03:2021",
            
            "evidence": {
                "payload": "<script>alert(1)</script>",
                "request": "POST /search?q=...",
                "response": "...<script>alert(1)</script>...",
                "confidence": 0.95
            },
            
            "poc": {
                "steps": ["1. Navigate to...", "2. Input..."],
                "curl_command": "curl -X POST...",
                "exploit_code": "..."
            },
            
            "impact": {
                "confidentiality": "high",
                "integrity": "high",
                "availability": "none"
            }
        }
    ],
    
    "statistics": {
        "payloads_tested": 150,
        "requests_sent": 150,
        "false_positives_filtered": 10,
        "success_rate": 0.05  # 5% payload 成功
    },
    
    "performance": {
        "avg_response_time_ms": 250,
        "rate_limit_hits": 5,
        "retries": 3,
        "network_errors": 1
    }
}
```

**可學習內容**：
- ✅ **Payload 效果**：哪些 payload 成功，哪些失敗
- ✅ **成功率模式**：不同 payload 類型的成功率
- ✅ **目標特徵**：目標網站的防禦機制（WAF、Rate Limit）
- ✅ **性能數據**：響應時間、錯誤率
- ✅ **漏洞嚴重度分佈**：發現的漏洞類型和嚴重程度

---

#### 2.3 優化反饋 (Coordinator Output - Internal Loop)
```python
{
    "internal_loop": {
        "payload_efficiency": {
            "script_tag": 0.15,      # <script> 成功率 15%
            "img_onerror": 0.08,     # <img onerror> 成功率 8%
            "svg_onload": 0.03       # <svg onload> 成功率 3%
        },
        "successful_patterns": [
            "<script>alert(1)</script>",
            "<img src=x onerror=alert(1)>"
        ],
        "recommended_concurrency": 5,
        "recommended_timeout_ms": 3000,
        "strategy_adjustments": {
            "increase_script_payload": true,
            "decrease_svg_payload": true
        }
    }
}
```

**可學習內容**：
- ✅ **最佳策略**：哪種策略最有效
- ✅ **參數優化**：最佳並發數、超時時間
- ✅ **模式識別**：成功的 payload 模式

---

#### 2.4 經驗轉換 ([ExperienceTransition](experience_manager.py#L21-L62))
```python
{
    "experience_id": "exp_abc123",
    "state": {
        "ast": {...},  # 代碼結構
        "target": "https://target.com",
        "target_info": {
            "technology": "PHP",
            "framework": "Laravel",
            "waf": "Cloudflare"
        },
        "context": {
            "previous_findings": [...],
            "available_tools": ["sqlmap", "xsstrike"],
            "current_phase": "exploitation"
        }
    },
    "action": {
        "type": "sql_injection",
        "tool": "sqlmap",
        "params": {
            "url": "https://target.com/api/user",
            "data": "id=1",
            "technique": "BEUSTQ"
        }
    },
    "next_state": {
        "success": true,
        "findings": [
            {
                "vulnerability": "SQL Injection",
                "database": "MySQL 8.0",
                "injectable_param": "id"
            }
        ]
    },
    "reward": 0.85,  # 獎勵值 (0-1)
    "metadata": {
        "execution_time": 45.2,
        "requests_sent": 150,
        "success": true
    }
}
```

**可學習內容**：
- ✅ **狀態-動作-回報**：完整的 RL 三元組
- ✅ **上下文關聯**：在什麼情況下採取什麼行動
- ✅ **成功模式**：哪些組合導致成功
- ✅ **失敗教訓**：哪些組合導致失敗

---

## 🧠 經驗學習設計（基於可獲得數據）

### 階段 1：能力效能學習 (Capability Performance Learning)
**數據來源**: 內閉環 + 外閉環執行記錄

#### 學習目標
1. **能力可靠性評估**
   - 輸入：能力執行記錄（成功/失敗）
   - 輸出：更新 `health_score`, `error_rate`, `avg_latency_ms`
   - 方法：滑動窗口統計

2. **能力選擇優化**
   - 輸入：歷史執行結果 + 目標特徵
   - 輸出：能力推薦排序
   - 方法：協同過濾（類似目標 → 推薦類似能力）

```python
# 偽代碼示例
def update_capability_metrics(capability_id: str, execution_result: dict):
    """更新能力指標"""
    success = execution_result["success"]
    duration_ms = execution_result["duration_ms"]
    
    # 更新成功率（滑動窗口，最近100次）
    recent_executions = get_recent_executions(capability_id, limit=100)
    success_count = sum(1 for e in recent_executions if e["success"])
    health_score = success_count / len(recent_executions)
    
    # 更新延遲
    avg_latency = np.mean([e["duration_ms"] for e in recent_executions])
    
    # 更新錯誤率
    error_count = sum(1 for e in recent_executions if not e["success"])
    error_rate = error_count / len(recent_executions)
    
    update_capability(capability_id, {
        "health_score": health_score,
        "avg_latency_ms": avg_latency,
        "error_rate": error_rate,
        "last_used": datetime.now()
    })
```

---

### 階段 2：攻擊策略學習 (Attack Strategy Learning)
**數據來源**: Features Output + Coordinator Feedback

#### 學習目標
1. **Payload 效果預測**
   - 輸入：目標特徵 + Payload 類型
   - 輸出：成功概率
   - 方法：分類模型（Logistic Regression / Random Forest）

2. **參數優化**
   - 輸入：歷史執行數據
   - 輸出：最佳並發數、超時時間、重試次數
   - 方法：貝葉斯優化

3. **WAF 繞過學習**
   - 輸入：WAF 類型 + 成功的 payload
   - 輸出：繞過策略庫
   - 方法：模式匹配 + 規則提取

```python
# 偽代碼示例
class PayloadSuccessPredictor:
    """Payload 成功概率預測器"""
    
    def train(self, training_data: List[ExperienceSample]):
        """訓練模型"""
        X = []  # 特徵
        y = []  # 標籤（成功/失敗）
        
        for sample in training_data:
            features = self.extract_features(sample)
            X.append(features)
            y.append(sample.success)
        
        self.model = RandomForestClassifier()
        self.model.fit(X, y)
    
    def extract_features(self, sample: ExperienceSample) -> np.ndarray:
        """提取特徵"""
        return np.array([
            # 目標特徵
            hash(sample.target_info.technology),
            hash(sample.target_info.waf),
            
            # Payload 特徵
            len(sample.action.payload),
            sample.action.payload.count('<'),
            sample.action.payload.count('script'),
            
            # 上下文特徵
            len(sample.state.previous_findings),
            sample.performance.rate_limit_hits
        ])
    
    def predict(self, target_info: dict, payload: str) -> float:
        """預測成功概率"""
        features = self.extract_features_from_context(target_info, payload)
        probability = self.model.predict_proba([features])[0][1]
        return probability
```

---

### 階段 3：決策序列學習 (Decision Sequence Learning)
**數據來源**: ExecutionTrace (完整執行軌跡)

#### 學習目標
1. **最佳執行序列**
   - 輸入：初始狀態 + 目標
   - 輸出：動作序列
   - 方法：序列模型（LSTM / Transformer）

2. **決策點優化**
   - 輸入：當前狀態 + 可用動作
   - 輸出：最佳動作
   - 方法：強化學習（DQN / PPO）

```python
# 偽代碼示例
class AttackSequenceLearner:
    """攻擊序列學習器"""
    
    def learn_from_trace(self, trace: ExecutionTrace):
        """從執行軌跡學習"""
        # 提取決策序列
        decisions = trace.get_entries_by_type(TraceType.DECISION)
        
        # 構建狀態-動作對
        for i, decision in enumerate(decisions):
            state = self.build_state(trace, i)
            action = decision.content["decision"]
            next_state = self.build_state(trace, i + 1)
            
            # 計算回報（基於最終結果）
            reward = self.calculate_reward(trace, i)
            
            # 保存經驗
            self.experience_manager.push(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward
            )
    
    def calculate_reward(self, trace: ExecutionTrace, step: int) -> float:
        """計算回報"""
        # 基於後續發現的漏洞數量和嚴重度
        findings = trace.get_entries_by_type(TraceType.TOOL_OUTPUT)
        findings_after_step = [f for f in findings if f.timestamp > trace.entries[step].timestamp]
        
        reward = 0.0
        for finding in findings_after_step:
            severity = finding.content.get("severity", "low")
            reward += {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.1}[severity]
        
        # 懲罰花費時間
        time_penalty = -0.01 * (trace.end_time - trace.entries[step].timestamp).seconds
        
        return reward + time_penalty
```

---

### 階段 4：目標特徵學習 (Target Profiling)
**數據來源**: Features Output + 成功/失敗模式

#### 學習目標
1. **目標分類**
   - 輸入：目標 URL + 初步掃描結果
   - 輸出：目標類型（技術棧、防禦機制）
   - 方法：多標籤分類

2. **防禦機制識別**
   - 輸入：HTTP 響應 + 錯誤模式
   - 輸出：WAF 類型、Rate Limit 規則
   - 方法：特徵匹配 + 規則學習

3. **成功模式遷移**
   - 輸入：相似目標的成功經驗
   - 輸出：推薦策略
   - 方法：案例基推理（CBR）

```python
# 偽代碼示例
class TargetProfiler:
    """目標特徵分析器"""
    
    def profile_target(self, url: str, initial_scan: dict) -> dict:
        """分析目標特徵"""
        features = {
            "technology": self.detect_technology(initial_scan),
            "framework": self.detect_framework(initial_scan),
            "waf": self.detect_waf(initial_scan),
            "rate_limit": self.detect_rate_limit(initial_scan)
        }
        
        # 查找相似目標的成功策略
        similar_targets = self.find_similar_targets(features)
        recommended_strategies = self.extract_strategies(similar_targets)
        
        return {
            "features": features,
            "similar_targets": similar_targets,
            "recommended_strategies": recommended_strategies
        }
    
    def find_similar_targets(self, features: dict) -> List[dict]:
        """查找相似目標"""
        # 基於特徵相似度
        all_targets = self.load_historical_targets()
        similarities = []
        
        for target in all_targets:
            sim = self.calculate_similarity(features, target["features"])
            similarities.append((target, sim))
        
        # 返回最相似的前10個
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in similarities[:10]]
```

---

## 📈 學習評估指標

### 1. **能力效能指標**
- Success Rate: 成功執行的能力調用比例
- Average Latency: 平均執行時間
- Error Rate: 錯誤率

### 2. **攻擊效果指標**
- Findings per Hour: 每小時發現的漏洞數
- High-Value Findings: 高價值漏洞數（Critical/High）
- False Positive Rate: 誤報率
- Time to First Finding: 首次發現漏洞的時間

### 3. **決策質量指標**
- Decision Accuracy: 決策準確率（與專家比較）
- Reward per Episode: 每次攻擊的總回報
- Exploration Efficiency: 探索效率（發現新漏洞/總請求數）

### 4. **學習效率指標**
- Sample Efficiency: 樣本效率（學習速度）
- Generalization: 泛化能力（在新目標上的表現）
- Improvement Rate: 改進速度（隨時間提升的比例）

---

## 🔄 持續學習流程

```
1. 執行攻擊任務
   ↓
2. 記錄完整軌跡 (ExecutionTrace)
   ↓
3. 提取經驗轉換 (ExperienceTransition)
   ↓
4. 存入經驗池 (ExperienceManager)
   ↓
5. 定期訓練模型
   │  - 每 N 個經驗觸發
   │  - 或每 X 小時觸發
   ↓
6. 評估模型表現
   ↓
7. 更新策略
   ↓
8. 返回步驟 1（使用新策略）
```

---

## 🎓 關鍵洞察

### ✅ **可行的學習方向**

1. **Payload 效果預測**：基於歷史數據，預測哪些 payload 在哪種目標上會成功
2. **參數自動調優**：學習最佳的並發數、超時時間等參數
3. **執行序列優化**：學習最有效的攻擊步驟順序
4. **目標特徵識別**：學習快速識別目標的技術棧和防禦機制
5. **策略遷移**：將成功經驗應用到類似目標

### ❌ **需要避免的陷阱**

1. **不要學習無法驗證的東西**：例如"AI認為這個漏洞很重要"（主觀判斷）
2. **不要依賴不存在的數據**：例如"用戶反饋"（目前沒有這個數據流）
3. **不要過度擬合**：避免只在特定目標上表現好，需要泛化能力
4. **不要忽略探索**：不能只利用已知策略，需要探索新方法

---

## 🚀 實施優先級

### P0（立即實施）
1. ✅ 能力效能統計：更新 health_score, error_rate, avg_latency_ms
2. ✅ 經驗記錄：保存所有 ExecutionTrace 到數據庫

### P1（短期實施）
3. Payload 效果分析：統計不同 payload 的成功率
4. 參數優化：學習最佳並發數、超時時間
5. 目標特徵提取：記錄每個目標的技術棧和防禦機制

### P2（中期實施）
6. Payload 成功預測模型：基於目標特徵預測 payload 成功率
7. 策略推薦系統：基於相似目標推薦策略
8. WAF 繞過學習：學習不同 WAF 的繞過方法

### P3（長期實施）
9. 決策序列學習：強化學習模型學習最佳執行序列
10. 端到端攻擊規劃：給定目標，自動生成完整攻擊計劃

---

## 📚 參考資料

- [雙閉環架構 Schema](../aiva_common/schemas/dual_loop.py)
- [Integration Coordinators](../integration/coordinators/README.md)
- [Experience Manager](experience_manager.py)
- [Trace Recorder](tracing/trace_recorder.py)
- [Model Trainer](learning/model_trainer.py)

---

**結論**：基於實際可獲得的數據，我們可以建立完整的經驗學習系統，從能力效能學習到決策序列優化，逐步提升 AI 的攻擊能力。關鍵是從簡單的統計學習開始（P0），逐步過渡到複雜的強化學習（P3）。
