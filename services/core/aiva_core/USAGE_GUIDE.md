# AIVA Core - 使用指南

> **🎯 目的**: AIVA Core 新一代AI自主安全代理核心引擎使用教學  
> **👥 適用對象**: Bug Bounty獵人、安全研究員、AI開發者  
> **⚡ 核心特色**: 5M參數BioNeuron + 四種協作模式 + 完全自主執行  
> **📅 版本**: v1.0.1 | **最後更新**: 2025年11月14日

---

## ✅ **系統狀態驗證**

**✅ 已完成專案安裝及驗證** (2025-11-14)

- ✅ Python 虛擬環境: `.venv/` (Python 3.13.9)
- ✅ 套件: `aiva-platform-integrated 1.0.0` + 182個依賴包
- ✅ 可編輯模式安裝完成
- ✅ **系統完整性檢查通過** - 所有核心組件運行正常

**🔍 系統驗證結果**:
- 🧠 **AI引擎**: BioNeuron主控制器、500萬參數神經網路正常
- ⚡ **執行引擎**: 計劃執行器、攻擊編排器、任務轉換器正常  
- 📚 **知識系統**: RAG檢索引擎、知識庫管理、向量存儲正常
- 🎓 **學習系統**: 模型訓練器、學習引擎、經驗管理系統正常

**快速驗證**:
```powershell
# 激活虛擬環境
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1

# 檢查安裝狀態
pip list | Select-String "aiva"
# 預期輸出: aiva-platform-integrated 1.0.0

# 測試核心組件導入
python -c "
import services.core.aiva_core
from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
from services.core.aiva_core.ai_engine.real_bio_net_adapter import create_real_scalable_bionet
from services.core.aiva_core.rag import RAGEngine
print('✅ AIVA Core 系統完整性驗證通過')
"
```

**詳細安裝指南**: [INSTALLATION_GUIDE.md](../../../INSTALLATION_GUIDE.md)

---

## 🧪 **測試驗證指南**

### **ModuleExplorer 測試** ✅ **已驗證通過**

**測試狀態** (2025-11-13):
- ✅ 所有 11 個測試用例通過
- ✅ 修復了 `ModuleName.FEATURES` 枚舉缺失問題  
- ✅ 修復了文件編碼讀取問題 (支援 UTF-8/GBK/Latin1)
- ✅ 修復了相對導入問題

**執行測試的正確方式**:
```powershell
# 1. 激活虛擬環境 (必須!)
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1

# 2. 切換到項目根目錄 (重要!)
cd C:\D\fold7\AIVA-git

# 3. 執行完整測試套件
python -m pytest services/core/tests/test_module_explorer.py -v

# 4. 執行單一測試
python -m pytest services/core/tests/test_module_explorer.py::TestModuleExplorer::test_initialization -v

# 5. 測試導入是否成功
python -c "from services.core.aiva_core.ai_engine.module_explorer import ModuleExplorer; print('ModuleExplorer import successful')"
```

**預期輸出**:
```
============== 11 passed, 20 warnings in 9.22s ===============
```

**已解決的問題**:
1. **枚舉缺失**: 在 `aiva_common/enums/modules.py` 中添加了 `FEATURES = "FeaturesModule"`
2. **編碼問題**: 在 `module_explorer.py` 中添加了多層編碼處理 (UTF-8 → GBK → Latin1)
3. **導入錯誤**: 修改了模組映射使用直接字符串而非枚舉值調用

### **相對導入修復** ✅ **已驗證通過**

**修復狀態** (2025-11-13):
- ✅ 修復了 `ImportError: attempted relative import beyond top-level package`
- ✅ 修復了 `ImportError: attempted relative import with no known parent package`  
- ✅ 修復了 `ModuleNotFoundError: No module named 'services'`

**驗證相對導入修復**:
```powershell
# 從項目根目錄測試導入
cd C:\D\fold7\AIVA-git
python -c "import services.core.aiva_core; print('All imports successful!')"

# 測試 pytest 執行 (應無導入錯誤)
python -m pytest services/core/aiva_core/ -v --tb=short
```

**修復技術**:
- 使用 **try-except 條件導入模式**
- 優先嘗試相對導入，失敗時自動降級到絕對導入  
- 動態添加項目根目錄到 `sys.path`

### **測試最佳實踐**

**⚠️ 常見錯誤避免**:
```powershell
# ❌ 錯誤: 在錯誤目錄執行測試
cd C:\D\fold7\AIVA-git\services\core
python -m pytest tests/test_module_explorer.py  # 找不到文件

# ✅ 正確: 從項目根目錄執行
cd C:\D\fold7\AIVA-git
python -m pytest services/core/tests/test_module_explorer.py
```

**📝 測試記錄模板**:
```bash
# 執行測試並記錄結果
python -m pytest services/core/tests/test_module_explorer.py -v > test_results.log 2>&1

# 檢查測試結果
Get-Content test_results.log | Select-String -Pattern "PASSED|FAILED|ERROR"
```

---

## 📋 **目錄**

- [✅ 安裝狀態](#-安裝狀態)
- [🧪 測試驗證指南](#-測試驗證指南)
- [🚀 快速開始](#-快速開始)
- [🎮 不同模式使用範例](#-不同模式使用範例)
- [⚙️ 高級配置](#️-高級配置)
- [📊 監控和分析](#-監控和分析)
- [🎯 實戰場景範例](#-實戰場景範例)
- [🔧 故障排除](#-故障排除)
- [📚 相關文檔](#-相關文檔)

---

## 🚀 **快速開始**

### **基本初始化**

```python
from aiva_core import BioNeuronMasterController

# 1. 初始化AIVA Core
master = BioNeuronMasterController()
await master.initialize()

# 2. AI自主模式執行
result = await master.process_request(
    request={
        "objective": "測試目標網站的SQL注入漏洞",
        "target": "https://example.com"
    },
    mode="ai"  # 完全自主執行
)

print(f"執行結果: {result}")
```

### **檢查系統狀態**

```python
# 檢查AIVA Core健康狀態
health_status = await master.get_system_health()
print(f"神經網路狀態: {health_status['neural_network_status']}")
print(f"RAG系統狀態: {health_status['rag_system_status']}")
print(f"攻擊引擎狀態: {health_status['attack_engine_status']}")
```

---

## 🎮 **不同模式使用範例**

### **1. AI完全自主模式** 🤖

```python
# AI模式 - 完全自主執行，無需人工干預
result = await master.process_request(
    request={
        "objective": "對目標進行全面安全評估",
        "target": "https://target.com",
        "scope": ["subdomain_enum", "vulnerability_scan", "exploitation"]
    },
    mode="ai"
)

# 自主執行過程中的實時狀態
async for status_update in master.stream_execution_status():
    print(f"當前階段: {status_update['current_phase']}")
    print(f"進度: {status_update['progress']}%")
    print(f"AI信心度: {status_update['ai_confidence']}")
```

### **2. UI協作模式** 👤

```python
# UI模式 - 需要人工確認關鍵決策
result = await master.process_request(
    request="分析目標系統並制定攻擊策略", 
    mode="ui"
)

# 處理需要確認的行動方案
if result.get("requires_confirmation"):
    print("AI建議的攻擊計畫:")
    for step in result["action_plan"]["steps"]:
        print(f"- {step['description']} (風險等級: {step['risk_level']})")
    
    confirmed = input("是否執行建議的攻擊計畫? (y/n): ")
    if confirmed.lower() == 'y':
        final_result = await master.execute_confirmed_action(result["action_plan"])
        print(f"執行完成: {final_result}")
```

### **3. 對話模式** 💬

```python
# Chat模式 - 自然語言交互和知識分享
response = await master.process_request(
    request="請教我關於XSS攻擊的基礎知識",
    mode="chat"
)

print(response["explanation"])  # AI生成的教學內容

# 持續對話
while True:
    user_question = input("您的問題: ")
    if user_question.lower() == 'exit':
        break
    
    answer = await master.process_request(
        request=user_question,
        mode="chat",
        context=response.get("conversation_context")
    )
    print(f"AIVA: {answer['response']}")
```

### **4. 混合智能模式** 🔄

```python
# Hybrid模式 - 智能選擇協作方式
result = await master.process_request(
    request="對新發現的目標進行全面安全評估",
    mode="hybrid"  # 系統會根據複雜度自動選擇UI或AI模式
)

# 系統會自動選擇最適合的模式
print(f"系統選擇的模式: {result['selected_mode']}")
print(f"選擇原因: {result['mode_selection_reasoning']}")
```

---

## ⚙️ **高級配置**

### **AI決策參數調優**

```python
# 初始化時配置AI參數
master = BioNeuronMasterController(
    ai_config={
        "confidence_threshold": 0.8,        # 信心度閾值
        "rag_enhancement": True,            # 啟用RAG增強
        "anti_hallucination": True,         # 啟用抗幻覺機制
        "experience_learning": True,        # 啟用經驗學習
        "max_execution_time": 3600,         # 最大執行時間（秒）
        "neural_network_config": {
            "temperature": 0.7,             # 創造性/保守性平衡
            "top_p": 0.9,                   # 輸出多樣性控制
            "max_tokens": 2048              # 最大生成tokens
        }
    }
)
```

### **攻擊計畫自定義**

```python
# 自定義攻擊流程AST
custom_ast = {
    "metadata": {
        "name": "Advanced Web Application Assessment",
        "description": "全面的Web應用安全評估流程",
        "estimated_duration": "2-4小時"
    },
    "nodes": [
        {
            "id": "reconnaissance", 
            "type": "info_gathering",
            "description": "偵察和信息收集",
            "tools": ["nmap", "gobuster", "subfinder"],
            "parameters": {
                "nmap_options": "-sV -sC --script vuln",
                "wordlist": "common.txt",
                "timeout": 300
            }
        },
        {
            "id": "vulnerability_scan",
            "type": "vulnerability_assessment", 
            "description": "漏洞掃描和識別",
            "depends_on": ["reconnaissance"],
            "tools": ["nuclei", "nikto", "wapiti"],
            "parameters": {
                "nuclei_templates": "critical,high",
                "scan_depth": "deep"
            }
        },
        {
            "id": "exploit",
            "type": "exploitation",
            "description": "漏洞利用驗證",
            "depends_on": ["vulnerability_scan"],
            "tools": ["metasploit", "custom_exploit"],
            "parameters": {
                "auto_exploit": False,  # 需要確認才能執行
                "payload_type": "reverse_shell"
            }
        }
    ]
}

# 執行自定義計畫
result = await master.execute_attack_plan(
    ast_plan=custom_ast,
    target="https://example.com",
    execution_mode="safe"  # safe | normal | aggressive
)
```

### **知識庫自定義**

```python
# 配置RAG知識增強
rag_config = {
    "knowledge_sources": [
        "owasp_top_10",
        "cve_database", 
        "exploit_db",
        "custom_knowledge"
    ],
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "retrieval_settings": {
        "max_results": 10,
        "similarity_threshold": 0.7,
        "rerank_results": True
    }
}

await master.configure_rag_system(rag_config)
```

---

## 📊 **監控和分析**

### **性能監控**

```python
# 獲取實時性能指標
metrics = await master.get_performance_metrics()

print(f"AI決策平均時間: {metrics['ai_decision_time']:.2f}ms")
print(f"攻擊成功率: {metrics['attack_success_rate']:.1%}")
print(f"神經網路吞吐量: {metrics['neural_throughput']} req/s")
print(f"記憶體使用: {metrics['memory_usage']:.1f}MB")

# 設置性能警報
await master.set_performance_alerts({
    "ai_decision_time_threshold": 1000,  # 毫秒
    "memory_usage_threshold": 80,        # 百分比
    "success_rate_threshold": 0.7        # 成功率
})
```

### **經驗分析和學習狀態**

```python
# 查看學習進度
learning_stats = await master.get_learning_statistics()

print(f"累積經驗: {learning_stats['total_experiences']} 次")
print(f"平均執行評分: {learning_stats['avg_execution_score']:.2f}")
print(f"模型改進次數: {learning_stats['model_updates']}")
print(f"知識庫條目: {learning_stats['knowledge_entries']}")

# 導出學習數據
learning_data = await master.export_learning_data(
    format="json",
    include_raw_experiences=True
)
```

### **執行軌跡分析**

```python
# 獲取詳細執行軌跡
execution_trace = await master.get_execution_trace(session_id="latest")

print("執行軌跡分析:")
for step in execution_trace["steps"]:
    print(f"階段: {step['phase']}")
    print(f"工具: {step['tool']}")
    print(f"結果: {step['result']}")
    print(f"AI信心度: {step['ai_confidence']}")
    print(f"執行時間: {step['duration']}秒")
    print("---")
```

---

## 🎯 **實戰場景範例**

### **場景1: Bug Bounty自動化**

```python
# 完整的Bug Bounty自動化流程
async def automated_bug_bounty(target_program):
    """自動化Bug Bounty流程"""
    
    # 1. 初始化針對特定程序的配置
    master = BioNeuronMasterController(
        ai_config={
            "confidence_threshold": 0.85,  # 高信心度
            "conservative_mode": True,     # 保守模式避免破壞
        }
    )
    
    # 2. 加載程序特定規則
    await master.load_program_rules(target_program["rules_file"])
    
    # 3. 執行多階段評估
    phases = ["reconnaissance", "vulnerability_discovery", "exploitation_poc"]
    
    results = {}
    for phase in phases:
        print(f"執行階段: {phase}")
        result = await master.execute_phase(
            phase=phase,
            target=target_program["scope"],
            mode="ai"
        )
        results[phase] = result
        
        # 動態調整策略
        if result["findings"]:
            await master.adapt_strategy(result["findings"])
    
    # 4. 生成報告
    report = await master.generate_bug_bounty_report(results)
    return report

# 使用範例
target_program = {
    "name": "Example Bug Bounty Program",
    "scope": ["*.example.com", "api.example.com"],
    "rules_file": "example_program_rules.json"
}

report = await automated_bug_bounty(target_program)
print(f"發現 {len(report['vulnerabilities'])} 個潛在漏洞")
```

### **場景2: 紅隊演練協助**

```python
# 紅隊演練的AI協助
async def red_team_engagement(engagement_config):
    """AI協助的紅隊演練"""
    
    master = BioNeuronMasterController(
        ai_config={
            "stealth_mode": True,          # 隱匿模式
            "evasion_techniques": True,    # 啟用躲避技術
            "lateral_movement": True       # 啟用橫向移動
        }
    )
    
    # 階段化執行
    engagement_phases = [
        {
            "phase": "external_recon",
            "objective": "外部偵察，收集目標信息",
            "stealth_level": "high"
        },
        {
            "phase": "initial_access",
            "objective": "獲得初始訪問權限",
            "stealth_level": "medium"
        },
        {
            "phase": "privilege_escalation",
            "objective": "提升權限",
            "stealth_level": "high"
        },
        {
            "phase": "lateral_movement",
            "objective": "橫向移動探索",
            "stealth_level": "high"
        }
    ]
    
    timeline = []
    for phase_config in engagement_phases:
        result = await master.execute_red_team_phase(
            config=phase_config,
            target=engagement_config["target"]
        )
        timeline.append({
            "timestamp": result["timestamp"],
            "phase": phase_config["phase"],
            "actions": result["actions_taken"],
            "objectives_met": result["objectives_achieved"]
        })
    
    return timeline
```

### **場景3: 持續安全監控**

```python
# 24/7 持續安全監控
async def continuous_security_monitoring(assets):
    """持續安全監控系統"""
    
    master = BioNeuronMasterController(
        ai_config={
            "continuous_mode": True,
            "alert_threshold": 0.6,
            "auto_response": True
        }
    )
    
    # 啟動持續監控
    monitoring_tasks = []
    
    for asset in assets:
        task = asyncio.create_task(
            master.monitor_asset_continuously(
                asset=asset,
                check_interval=300,  # 5分鐘檢查一次
                threat_models=["web_attacks", "infrastructure_threats"]
            )
        )
        monitoring_tasks.append(task)
    
    # 威脅響應處理
    async def handle_threat_alert(alert):
        print(f"威脅警報: {alert['threat_type']}")
        print(f"資產: {alert['asset']}")
        print(f"風險等級: {alert['risk_level']}")
        
        if alert['risk_level'] >= 0.8:
            # 高風險自動響應
            response = await master.auto_respond_to_threat(alert)
            print(f"自動響應: {response['actions_taken']}")
    
    # 註冊威脅處理器
    master.register_threat_handler(handle_threat_alert)
    
    # 等待監控任務
    await asyncio.gather(*monitoring_tasks)
```

---

## 🔧 **故障排除**

### **測試和導入問題** 🧪

#### **1. ModuleExplorer 測試失敗**

**問題**: `AttributeError: type object 'ModuleName' has no attribute 'FEATURES'`
```powershell
# 解決方案: 檢查枚舉定義
python -c "from aiva_common.enums.modules import ModuleName; print(dir(ModuleName))"

# 如果缺少 FEATURES，需要在 aiva_common/enums/modules.py 中添加:
# FEATURES = "FeaturesModule"
```

#### **2. 文件編碼錯誤**

**問題**: `'utf-8' codec can't decode byte 0xb4 in position 5`
```powershell
# 解決方案已內建在 module_explorer.py 中，會自動嘗試多種編碼:
# UTF-8 → GBK → Latin1
# 如果仍有問題，檢查文件編碼:
file -i path/to/problem_file.py
```

#### **3. 相對導入錯誤**

**問題**: `ImportError: attempted relative import beyond top-level package`
```powershell
# 解決方案: 確保從正確目錄執行
cd C:\D\fold7\AIVA-git  # 必須在項目根目錄

# 測試導入是否成功
python -c "import services.core.aiva_core"

# 如果仍失敗，檢查 sys.path
python -c "import sys; print('\n'.join(sys.path))"
```

#### **4. 測試文件找不到**

**問題**: `ERROR: file or directory not found`
```powershell
# 檢查文件路徑是否正確
ls services/core/tests/test_module_explorer.py

# 確保使用正確的相對路徑
python -m pytest services/core/tests/test_module_explorer.py -v
```

### **開發環境問題**

#### **5. 虛擬環境未激活**
```powershell
# 檢查是否在虛擬環境中
python -c "import sys; print(sys.prefix)"

# 激活虛擬環境
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1

# 驗證激活成功 (應顯示 .venv 路徑)
which python
```

### **AI相關問題**

#### **1. AI決策速度慢**

```python
# 檢查性能瓶頸
diagnostic = await master.diagnose_performance_issues()

if diagnostic["neural_network"]["status"] == "slow":
    # 優化神經網路配置
    await master.optimize_neural_network(
        batch_size=16,
        use_gpu_acceleration=True,
        quantization=True
    )

if diagnostic["rag_system"]["status"] == "slow":
    # 優化RAG檢索
    await master.optimize_rag_performance(
        cache_size=1000,
        parallel_queries=4
    )
```

#### **2. 攻擊執行失敗**

```python
# 診斷攻擊執行問題
execution_log = await master.get_execution_diagnostics()

common_issues = {
    "network_timeout": "網絡超時，建議增加timeout設置",
    "tool_not_found": "工具未安裝，請檢查工具依賴",
    "permission_denied": "權限不足，請檢查執行權限",
    "target_unreachable": "目標不可達，請檢查網絡連接"
}

for issue in execution_log["issues"]:
    if issue["type"] in common_issues:
        print(f"問題: {issue['description']}")
        print(f"建議: {common_issues[issue['type']]}")
```

#### **3. 記憶體使用過高**

```python
# 記憶體使用優化
memory_stats = await master.get_memory_usage()

if memory_stats["usage_percentage"] > 80:
    # 清理緩存
    await master.clear_caches()
    
    # 減少並行執行數
    await master.configure_execution_limits(
        max_parallel_tasks=2,
        memory_limit="4GB"
    )
    
    # 啟用內存優化模式
    await master.enable_memory_optimization()
```

---

## 📚 **相關文檔**

### **核心文檔**
- [AIVA Core README](README.md) - 架構和技術詳細說明
- [API文檔](API_REFERENCE.md) - 完整API參考
- [配置指南](CONFIGURATION_GUIDE.md) - 詳細配置說明

### **開發相關**
- [開發指南](../../../guides/development/README.md) - 開發環境設置
- [模組整合指南](../../../guides/modules/README.md) - 模組開發和整合
- [API驗證指南](../../../guides/development/API_VERIFICATION_GUIDE.md) - API使用驗證

### **架構設計**
- [架構文檔](../../../guides/architecture/README.md) - 系統架構深入說明
- [跨語言Schema指南](../../../guides/architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md) - 跨語言協調

### **示例和模板**
- [使用示例](../../../examples/README.md) - 更多使用範例
- [配置模板](../../../config/templates/README.md) - 配置文件模板

---

## 🎯 **總結**

AIVA Core使用指南涵蓋了從基礎使用到高級配置的完整流程。無論您是Bug Bounty新手還是資深安全研究員，這個5M參數的AI自主安全代理都能為您的安全測試工作提供強大支持。

### **關鍵使用原則**
1. **從簡單開始**: 先使用AI模式體驗自主執行能力
2. **逐步進階**: 根據需要配置高級功能和自定義參數
3. **持續學習**: 利用經驗學習功能讓系統越來越智能
4. **安全為先**: 始終遵守目標程序規則和法律法規

**立即開始使用AIVA Core，體驗AI驅動的安全測試革命！** 🚀

---

## ✅ **快速檢查清單**

### **開發環境檢查** (執行前必檢)
```powershell
# 1. 檢查虛擬環境
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1
python -c "import sys; print('✓ Python:', sys.executable)"

# 2. 檢查工作目錄
cd C:\D\fold7\AIVA-git
pwd  # 應顯示: C:\D\fold7\AIVA-git

# 3. 檢查核心導入
python -c "import services.core.aiva_core; print('✓ AIVA Core 導入成功')"

# 4. 檢查測試環境
python -m pytest --version
python -m pytest services/core/tests/test_module_explorer.py::TestModuleExplorer::test_initialization -v
```

### **故障排除檢查清單**
- [ ] 虛擬環境已激活 (`(.venv)` 顯示在提示符中)
- [ ] 工作目錄為項目根目錄 (`C:\D\fold7\AIVA-git`)
- [ ] Python 版本正確 (>=3.9)
- [ ] 所有依賴已安裝 (`pip list | Select-String aiva`)
- [ ] 核心模組可正常導入
- [ ] 測試文件路徑正確
- [ ] 枚舉定義完整 (包含 `FEATURES`)

### **成功測試的標準輸出**
```
============== 11 passed, 20 warnings in 9.22s ===============
```

---

**📝 文檔版本**: v1.1.0  
**🔄 最後更新**: 2025年11月13日 - 添加測試驗證指南  
**👥 維護團隊**: AIVA Core Development Team  
**📧 支持聯繫**: 請參考主項目文檔聯繫方式