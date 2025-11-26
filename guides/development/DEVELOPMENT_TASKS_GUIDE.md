# AIVA 開發任務流程指南

## 📋 目錄

- [📑 目錄](#目錄)
- [🎯 Week 1: AI攻擊計畫映射器開發](#week-1-ai攻擊計畫映射器開發)
  - [Day 1-2: 核心邏輯實現 🔴](#day-1-2-核心邏輯實現)
  - [Day 3-4: AI決策整合 🟠](#day-3-4-ai決策整合)
  - [Day 5-7: 測試與優化 🟡](#day-5-7-測試與優化)
- [🎯 Week 2: 進階SSRF微服務探測開發](#week-2-進階ssrf微服務探測開發)
  - [Day 1-2: 探測邏輯完善 🔴](#day-1-2-探測邏輯完善)
  - [Day 3-4: 雲端元數據強化 🟠](#day-3-4-雲端元數據強化)
  - [Day 5-7: 整合與接口 🟡](#day-5-7-整合與接口)
- [🎯 Week 3: 客戶端授權繞過 - 靜態分析](#week-3-客戶端授權繞過-靜態分析)
  - [Day 1-3: 靜態分析強化 🔴](#day-1-3-靜態分析強化)
  - [Day 4-7: 動態測試準備 🟠](#day-4-7-動態測試準備)
- [🎯 Week 4: 客戶端授權繞過 - 完整整合](#week-4-客戶端授權繞過-完整整合)
  - [Day 1-4: 高級檢測功能 🔴](#day-1-4-高級檢測功能)
  - [Day 5-7: 最終整合 🟠](#day-5-7-最終整合)
- [🧪 每日檢查腳本](#每日檢查腳本)
  - [開發環境驗證](#開發環境驗證)
  - [測試執行腳本](#測試執行腳本)
- [📊 進度追蹤表格](#進度追蹤表格)
- [🚨 風險點檢查清單](#風險點檢查清單)
  - [技術風險監控](#技術風險監控)
  - [時程風險應對](#時程風險應對)
- [🔗 相關資源](#相關資源)
  - [開發指南](#開發指南)
  - [架構指南](#架構指南)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

> **📋 適用對象**: 開發團隊、項目經理、技術負責人  
> **🎯 使用場景**: 開發計劃執行、任務追蹤、進度管理  
> **⏱️ 時間範圍**: 4週開發週期  
> **🔧 工具需求**: Python、AI模組、測試環境

---

## 📑 目錄

1. [🎯 Week 1: AI攻擊計畫映射器開發](#-week-1-ai攻擊計畫映射器開發)
2. [🧠 Week 2: AI決策優化與整合](#-week-2-ai決策優化與整合)
3. [🔍 Week 3: 客戶端檢測功能](#-week-3-客戶端檢測功能)
4. [⚡ Week 4: 性能優化與測試](#-week-4-性能優化與測試)
5. [📊 每日任務追蹤](#-每日任務追蹤)
6. [🔧 技術債務管理](#-技術債務管理)
7. [🧪 測試策略](#-測試策略)
8. [📈 進度評估指標](#-進度評估指標)

---

## 🎯 Week 1: AI攻擊計畫映射器開發

### Day 1-2: 核心邏輯實現 🔴
```python
# 任務清單 - AI攻擊計畫映射器
□ 完善 map_decision_to_tasks() 具體映射規則
  └── 實現 SQLi, XSS, SSRF, IDOR, RCE 映射邏輯
  └── 添加參數驗證和型別檢查
  
□ 擴展 _map_vulnerability_to_module() 方法
  └── 支援 15+ 漏洞類型映射
  └── 添加自定義模組映射規則
  
□ 實現上下文傳遞邏輯
  └── session_id, scan_config 狀態管理
  └── 父子任務關聯性建立
  
□ 添加決策鏈依賴處理
  └── 任務執行順序優化
  └── 條件分支邏輯處理
```

### Day 3-4: AI決策整合 🟠
```python
# 任務清單 - BioNeuronRAGAgent 整合
□ 深度整合 BioNeuronRAGAgent
  └── decision 結果格式標準化
  └── 錯誤決策重試機制
  
□ 實現決策品質評估
  └── 置信度評分系統
  └── 決策有效性驗證
  
□ 添加決策回饋學習
  └── 成功/失敗案例記錄
  └── 模型權重動態調整
  
□ 實現動態策略調整
  └── 基於目標環境自適應
  └── 攻擊強度動態調節
```

### Day 5-7: 測試與優化 🟡
```python
# 任務清單 - 測試與優化
□ 建立完整測試套件
  └── 單元測試覆蓋率 > 90%
  └── 模擬決策數據測試
  
□ 整合測試實現
  └── 與掃描引擎通信測試
  └── 端到端工作流驗證
  
□ 效能優化
  └── 映射邏輯效能基準
  └── 記憶體使用優化
  
□ 錯誤處理完善
  └── 異常情況恢復機制
  └── 詳細錯誤日誌記錄
```

**Week 1 驗收檢查項目**:
- [ ] `python -c "from services.core.aiva_core.execution.attack_plan_mapper import AttackPlanMapper; print('✅ 導入成功')"`
- [ ] 執行映射器單元測試: `pytest tests/test_attack_plan_mapper.py -v`
- [ ] 整合測試通過率 > 95%
- [ ] 記憶體使用 < 100MB，響應時間 < 2s

---

## 🎯 Week 2: 進階SSRF微服務探測開發

### Day 1-2: 探測邏輯完善 🔴
```go
// 任務清單 - Go SSRF 模組
□ 擴展 ProbeCommonPorts() 功能
  └── 常見端口擴展到 50+ 個
  └── 服務指紋識別實現
  
□ 實現智能超時機制
  └── 動態超時調整
  └── 重試邏輯優化
  
□ 添加結果聚合分析
  └── 服務類型自動分類
  └── 漏洞風險評估
  
□ Kubernetes 服務發現
  └── kube-dns 解析
  └── service mesh 探測
```

### Day 3-4: 雲端元數據強化 🟠
```go
// 任務清單 - 雲端元數據掃描
□ 完善 AWS IMDSv2 流程
  └── Token 獲取到數據請求完整鏈路
  └── 錯誤處理和重試邏輯
  
□ 添加更多雲服務商
  └── Oracle Cloud, Tencent Cloud
  └── OpenStack, VMware vSphere
  
□ 元數據內容深度解析
  └── IAM 角色信息提取
  └── 網路配置分析
  
□ 憑證洩露檢測
  └── AWS Access Keys 檢測
  └── GCP Service Account 檢測
```

### Day 5-7: 整合與接口 🟡
```go
// 任務清單 - 整合開發
□ 實現 HTTP API 接口
  └── RESTful API 設計
  └── JSON 數據格式標準化
  
□ Python 掃描引擎通信
  └── 異步請求處理
  └── 結果格式轉換
  
□ 添加詳細報告功能
  └── 漏洞報告生成
  └── 修復建議提供
  
□ 效能基準測試
  └── 並發能力測試
  └── 記憶體使用監控
```

**Week 2 驗收檢查項目**:
- [ ] `go build ./cmd/worker/... && echo "✅ Go 編譯成功"`
- [ ] API 響應時間 < 5s，併發支援 > 100 requests
- [ ] 元數據檢測成功率 > 90%
- [ ] 與 Python 掃描引擎通信穩定

---

## 🎯 Week 3: 客戶端授權繞過 - 靜態分析

### Day 1-3: 靜態分析強化 🔴
```python
# 任務清單 - JavaScript 分析強化
□ 擴展檢測模式到 15+ 種
  └── React Router 權限檢測
  └── Vue.js 路由守衛分析
  └── Angular CanActivate 檢查
  
□ 實現 AST 語法樹分析
  └── esprima 整合
  └── 複雜表達式解析
  
□ 代碼混淆反向解析
  └── UglifyJS 混淆識別
  └── Webpack bundle 分析
  
□ TypeScript 支援
  └── .d.ts 類型定義檢查
  └── 介面權限檢測
```

### Day 4-7: 動態測試準備 🟠
```python
# 任務清單 - 動態測試框架
□ Playwright 整合
  └── 瀏覽器自動化設置
  └── 頁面互動腳本
  
□ 使用者權限模擬
  └── 多角色登入模擬
  └── Session 狀態管理
  
□ DOM 操作監控
  └── 元素可見性檢測
  └── 事件監聽器分析
  
□ 狀態篡改測試
  └── LocalStorage 修改測試
  └── Cookie 篡改檢測
```

**Week 3 驗收檢查項目**:
- [ ] 靜態分析檢測模式 > 15 種
- [ ] AST 解析成功率 > 95%
- [ ] Playwright 自動化穩定運行
- [ ] 多框架支援測試通過

---

## 🎯 Week 4: 客戶端授權繞過 - 完整整合

### Day 1-4: 高級檢測功能 🔴
```python
# 任務清單 - 高級功能開發
□ 權限升級路徑檢測
  └── 權限等級分析
  └── 升級可能性評估
  
□ API 端點權限驗證
  └── 前端 API 調用分析
  └── 授權頭檢查
  
□ 跨域權限洩露檢測
  └── CORS 配置檢查
  └── postMessage 權限檢測
  
□ WebSocket 權限檢查
  └── 連接權限驗證
  └── 消息內容檢查
```

### Day 5-7: 最終整合 🟠
```python
# 任務清單 - 整合與優化
□ 掃描引擎深度整合
  └── 任務調度優化
  └── 結果彙總機制
  
□ 智能誤報過濾
  └── 機器學習分類器
  └── 人工標註數據集
  
□ 詳細修復建議
  └── 代碼修復範例
  └── 最佳實踐推薦
  
□ 最終效能優化
  └── 記憶體洩露檢查
  └── 並發處理優化
```

**Week 4 驗收檢查項目**:
- [ ] 檢測精度 > 85%，誤報率 < 10%
- [ ] 完整端到端測試通過
- [ ] 效能指標達到設計要求
- [ ] 三大模組整合穩定

---

## 🧪 每日檢查腳本

### 開發環境驗證
```bash
# 每日執行檢查
#!/bin/bash
echo "🔍 執行每日開發環境檢查..."

# 1. 補包驗證
python aiva_package_validator.py
if [ $? -eq 0 ]; then
    echo "✅ 補包驗證通過"
else
    echo "❌ 補包驗證失敗"
    exit 1
fi

# 2. 系統通連性
python aiva_system_connectivity_sop_check.py | tail -5

# 3. 模組導入測試
python -c "
try:
    from services.core.aiva_core.execution.attack_plan_mapper import AttackPlanMapper
    from services.features.client_side_auth_bypass import ClientSideAuthBypassWorker
    print('✅ 新模組導入成功')
except ImportError as e:
    print(f'❌ 模組導入失敗: {e}')
    exit(1)
"

# 4. Go 模組編譯
cd services/features/function_ssrf_go
go build ./...
if [ $? -eq 0 ]; then
    echo "✅ Go 模組編譯成功"
else
    echo "❌ Go 模組編譯失敗"
    exit 1
fi

echo "🎉 每日檢查完成！"
```

### 測試執行腳本
```python
# 單元測試執行腳本
import subprocess
import sys

def run_tests():
    """執行所有單元測試"""
    test_commands = [
        "pytest tests/test_attack_plan_mapper.py -v",
        "pytest tests/test_client_auth_bypass.py -v", 
        "pytest tests/test_js_analysis_engine.py -v",
        "python -m unittest discover tests/ -p 'test_*.py'"
    ]
    
    for cmd in test_commands:
        print(f"🧪 執行: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 測試通過")
        else:
            print(f"❌ 測試失敗: {result.stderr}")
            return False
    return True

if __name__ == "__main__":
    if run_tests():
        print("🎉 所有測試通過！")
        sys.exit(0)
    else:
        print("💥 測試失敗，請檢查錯誤")
        sys.exit(1)
```

---

## 📊 進度追蹤表格

| 週次 | 模組 | 完成度 | 關鍵里程碑 | 狀態 |
|------|------|---------|------------|------|
| Week 1 | AI攻擊計畫映射器 | 0% | 核心邏輯實現 | 🔄 準備中 |
| Week 2 | 進階SSRF探測 | 0% | Go模組功能完整 | ⏳ 待開始 |
| Week 3 | 客戶端授權繞過 | 0% | 靜態分析完成 | ⏳ 待開始 |  
| Week 4 | 整合與優化 | 0% | 端到端測試 | ⏳ 待開始 |

## 🚨 風險點檢查清單

### 技術風險監控
- [ ] AI映射邏輯複雜度是否超出預期？
- [ ] Go-Python通信是否穩定？
- [ ] Playwright自動化是否可靠？
- [ ] 記憶體使用是否在可控範圍？
- [ ] 效能指標是否符合預期？

### 時程風險應對
- [ ] 每週進度是否達到 80% 以上？
- [ ] 關鍵功能是否優先完成？
- [ ] 測試覆蓋率是否滿足要求？
- [ ] 整合問題是否及時解決？

---

**執行指南**: 每日更新完成狀態，每週五進行進度檢討  
**責任分配**: 依據團隊規模分配對應任務  
**緊急應對**: 遇到阻塞問題立即啟動應變方案

---

## 🔗 相關資源

### 開發指南
- 📖 [開發快速指南](./DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [開發任務指南](./DEVELOPMENT_TASKS_GUIDE.md)
- 📖 [開發者指南](./DEVELOPER_GUIDE.md)
- 📖 [依賴管理指南](./DEPENDENCY_MANAGEMENT_GUIDE.md)
- 📖 [API 驗證指南](./API_VERIFICATION_GUIDE.md)
- 📖 [Schema 導入指南](./SCHEMA_IMPORT_GUIDE.md)

### 架構指南
- 📖 [Schema 統一指南](../architecture/SCHEMA_GUIDE.md)
- 📖 [架構兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 故障排除
- 📖 [導入問題解決](../troubleshooting/IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)

### 使用者手冊
- 📚 [Core 模組手冊](../../docs/user_guides/01_core/AIVA_CORE_使用者手冊.md)

