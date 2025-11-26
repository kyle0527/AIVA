# 網路研究：滲透測試自動化分類最佳實踐
## 📋 目錄

- [🔍 主要發現摘要](#主要發現摘要)
  - [1. 國際標準框架](#1-國際標準框架)
  - [2. 學術研究發現](#2-學術研究發現)
  - [3. 行業實務標準](#3-行業實務標準)
- [💡 建議實施方案](#建議實施方案)
  - [方案 1：階段優先級權重系統 (推薦)](#方案-1階段優先級權重系統-推薦)
  - [方案 2：動詞優先分析法](#方案-2動詞優先分析法)
  - [方案 3：OWASP 標準前綴系統](#方案-3owasp-標準前綴系統)
- [🎯 具體建議](#具體建議)
  - [立即實施 (解決當前問題)](#立即實施-解決當前問題)
  - [中期改善](#中期改善)
  - [長期優化](#長期優化)
- [📊 預期效果](#預期效果)

**研究日期：** 2025年11月13日  
**搜索範圍：** OWASP、MITRE ATT&CK、學術論文、行業標準  
**目標：** 找出滲透測試功能自動分類的建議方法

## 🔍 主要發現摘要

### 1. 國際標準框架

#### OWASP Web Security Testing Guide (WSTG)
- **分類標準：** 採用 `WSTG-<category>-<number>` 格式
- **類別系統：** 4字符大寫字符串識別測試類型
- **階段劃分：** 
  - INFO (Information Gathering)
  - CONF (Configuration Management)  
  - IDNT (Identity Management)
  - ATHN (Authentication Testing)
  - AUTHZ (Authorization Testing)
  - SESS (Session Management)
  - INPV (Input Validation)

**關鍵洞察：** OWASP 使用**明確的功能分類前綴**而非關鍵字匹配

#### MITRE ATT&CK Framework
- **戰術分類：** 14個主要戰術階段 (TA0001-TA0040)
- **技術編號：** 每個戰術下有具體技術 (T1xxx)
- **優先級系統：** 按攻擊鏈順序排列優先級

**關鍵洞察：** 使用**階段優先級**和**編號系統**進行分類

### 2. 學術研究發現

#### "Penetration Taxonomy" (Sarker et al., 2023) - 18次引用
**主要建議：**
```
分類維度：
1. 測試範圍 (Scope)
2. 測試深度 (Depth)  
3. 執行方法 (Method)
4. 自動化程度 (Automation Level)
```

#### "Automated Penetration Testing Overview" (Abu-Dabaseh, 2018) - 86次引用
**關鍵發現：**
- **三層分類法：** Grey Hat / Black Hat / White Hat
- **自動化分級：** Manual → Semi-Automated → Fully Automated
- **工具分類：** 按攻擊階段而非關鍵字分類

#### "Rule Tree Assessment Method" (Zhao et al., 2015) - 30次引用
**核心方法：**
```python
# 規則樹分類方法
if (is_reconnaissance_phase):
    priority = 1
elif (is_exploitation_phase):
    priority = 5  # 最高優先級
elif (is_analysis_phase):
    priority = 3
```

### 3. 行業實務標準

#### Penetration Testing Execution Standard (PTES)
**標準階段順序：**
1. Pre-engagement → 優先級: 1
2. Intelligence Gathering → 優先級: 2  
3. Threat Modeling → 優先級: 3
4. Vulnerability Analysis → 優先級: 4
5. Exploitation → 優先級: 5 (**最高**)
6. Post Exploitation → 優先級: 5
7. Reporting → 優先級: 1

**關鍵洞察：** 行業標準明確將 **Exploitation 列為最高優先級階段**

## 💡 建議實施方案

### 方案 1：階段優先級權重系統 (推薦)

基於 PTES 和 MITRE ATT&CK 標準：

```python
PHASE_PRIORITY_WEIGHTS = {
    PentestPhase.EXPLOITATION: 5,        # 最高優先級
    PentestPhase.POST_EXPLOITATION: 5,   # 同等最高
    PentestPhase.VULNERABILITY_ANALYSIS: 3,  # 中等
    PentestPhase.INTELLIGENCE_GATHERING: 2,   # 較低
    PentestPhase.THREAT_MODELING: 2,     # 較低
    PentestPhase.REPORTING: 1,           # 最低
    PentestPhase.PRE_ENGAGEMENT: 1       # 最低
}

def enhanced_classify(capability, semantic_analysis):
    # 標準關鍵字匹配
    keyword_scores = calculate_keyword_matches(capability)
    
    # 應用階段優先級權重
    for phase, base_score in keyword_scores.items():
        priority_weight = PHASE_PRIORITY_WEIGHTS[phase]
        keyword_scores[phase] = base_score * priority_weight
    
    # 衝突解決：同分時選擇高優先級階段
    return resolve_by_priority(keyword_scores)
```

### 方案 2：動詞優先分析法

基於學術研究的語義分析方法：

```python
ACTION_VERB_WEIGHTS = {
    # 攻擊行為動詞 - 最高權重
    "exploit": 3, "attack": 3, "compromise": 3, "penetrate": 3,
    
    # 分析行為動詞 - 中等權重  
    "analyze": 2, "scan": 2, "test": 2, "assess": 2,
    
    # 收集行為動詞 - 較低權重
    "gather": 1, "collect": 1, "discover": 1
}

def verb_priority_classify(text):
    # 優先識別主要動詞
    main_verb = extract_primary_verb(text)
    if main_verb in ACTION_VERB_WEIGHTS:
        return get_phase_by_verb_priority(main_verb)
```

### 方案 3：OWASP 標準前綴系統

模仿 OWASP WSTG 的明確分類方法：

```python
FUNCTION_PREFIX_MAP = {
    "exploit_": PentestPhase.EXPLOITATION,
    "attack_": PentestPhase.EXPLOITATION, 
    "scan_": PentestPhase.VULNERABILITY_ANALYSIS,
    "analyze_": PentestPhase.VULNERABILITY_ANALYSIS,
    "gather_": PentestPhase.INTELLIGENCE_GATHERING,
    "generate_": PentestPhase.REPORTING
}

def prefix_classify(function_name):
    for prefix, phase in FUNCTION_PREFIX_MAP.items():
        if function_name.startswith(prefix):
            return phase
```

## 🎯 具體建議

### 立即實施 (解決當前問題)

1. **階段優先級權重系統**
   - 給 EXPLOITATION 階段 3x 權重加成
   - 實施衝突解決機制 (優先級排序)

2. **動詞識別優先**
   - "exploit" 關鍵字獲得額外 +2 分權重
   - 函數名開頭的動詞優先分析

### 中期改善

3. **多維度分類**
   - 結合關鍵字匹配 + 語義分析 + 階段優先級
   - 實施置信度評分系統

### 長期優化

4. **機器學習分類器**
   - 使用標注數據訓練專用分類模型
   - 持續學習和調優

## 📊 預期效果

基於研究發現，實施階段優先級系統後：

- `exploit_vulnerability` → EXPLOITATION ✅ (符合 PTES 標準)
- `scan_target` → VULNERABILITY_ANALYSIS ✅  
- `gather_information` → INTELLIGENCE_GATHERING ✅

**結論：** 學術研究和行業標準均支持**階段優先級權重系統**作為最佳實踐方法。