## 數據差異原因分析報告

### 發現的真相

**classification_data.json 中 function_xss 有 107 個 flows**
- 這些是**經過分類器處理後的 flows**
- 每個 flow 都有完整的 metadata（module_type, use_case, parameters 等）
- 這些是**有效的、可執行的流程**

**analysis_results.json 中有 274 個 graphs**
- 這些是**原始的 AST 分析結果**
- 使用舊格式 `graphs`（不是 `flows`）
- 包含所有的代碼結構（包括類定義、內部方法等）

### 差異原因

#### 274 graphs → 107 flows 的轉換過程：

1. **過濾無效項目**：
   - Class definitions（只是結構定義）
   - Internal helper methods（內部輔助方法）
   - 沒有實際調用關係的節點

2. **提取有效流程**：
   - 只保留有實際調用鏈的 graphs
   - 轉換為 flows 格式
   - 添加分類 metadata

3. **智能分類**：
   - 加上 module_type: "injection"
   - 加上 use_case 描述
   - 提取函數參數信息

### 數據流向

```
analysis_results.json (原始)
   274 graphs (所有 AST 結構)
         ↓
   [分類器過濾 + 轉換]
         ↓
classification_data.json (分類後)
   107 flows (有效流程 + metadata)
```

### 結論

**這不是數據不一致，而是正常的處理流程！**

- analysis_results.json = 完整的原始 AST 分析（274 個結構）
- classification_data.json = 過濾後的有效流程（107 個 flows）

**兩者的關係是：分析 → 過濾 → 分類**

差異是合理的，因為不是所有的 AST 結構都能轉換為可執行的 flows。
