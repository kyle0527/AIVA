# 鄧鄧靶場實測準備檢查清單

**生成時間**: 2025-10-19  
**目標**: 靶場實測驗證  
**參考文檔**: AI_TRAINING_SOP.md  
**執行方式**: 手動執行,AI 僅提供指令

---

## 📋 **Phase 1: 系統通連性檢查** (已完成 ✅)

根據 SOP 第一階段,系統連通性已驗證:

```bash
# 已執行並通過 ✅
python aiva_ai_testing_range.py
```

**結果**: 
- ✅ Core 模組: 100% 通過
- ✅ Scan 模組: 100% 通過  
- ✅ Integration 模組: 100% 通過
- ✅ UI 模組: 100% 通過
- ✅ Reports 模組: 100% 通過

---

## 📋 **Phase 2: AI 核心功能檢查**

### 2.1 檢查 BioNeuronCore 狀態

**執行命令**:
```powershell
# 1. 檢查 AI 核心模組
python -c "from services.ai_core.bio_neuron_core import BioNeuronCore; print('BioNeuronCore 可用')"

# 2. 檢查訓練服務
python -c "from services.ai_core.training_service import IntegratedTrainService; print('TrainingService 可用')"

# 3. 檢查模型管理
python -c "from services.ai_core.model_manager import ModelManager; print('ModelManager 可用')"
```

**預期結果**: 所有模組都能成功導入

**如果失敗**: 檢查依賴安裝和導入路徑

---

### 2.2 測試 AI 攻擊學習能力

**執行命令**:
```powershell
# 測試 AI 核心的攻擊模式學習
python -c "
from services.ai_core.bio_neuron_core import BioNeuronCore
core = BioNeuronCore()
# 測試基本功能
print(f'AI Core initialized: {core is not None}')
print(f'Learning rate: {getattr(core, \"learning_rate\", \"N/A\")}')
"
```

**檢查項目**:
- [ ] AI 核心成功初始化
- [ ] 學習率參數正確
- [ ] 無錯誤輸出

---

## 📋 **Phase 3: 安全掃描模組檢查**

### 3.1 驗證 v2.5 模組可用性

**執行命令**:
```powershell
# 測試所有 v2.5 升級模組
cd C:\D\fold7\AIVA-git

# 1. mass_assignment v2.5
python -c "from services.features.mass_assignment.worker import MassAssignmentWorker; w = MassAssignmentWorker(); print(f'mass_assignment v{w.version}')"

# 2. jwt_confusion v2.5
python -c "from services.features.jwt_confusion.worker import JwtConfusionWorker; w = JwtConfusionWorker(); print(f'jwt_confusion v{w.version}')"

# 3. oauth_confusion v2.5
python -c "from services.features.oauth_confusion.worker import OAuthConfusionWorker; w = OAuthConfusionWorker(); print(f'oauth_confusion v{w.version}')"

# 4. graphql_authz v2.5
python -c "from services.features.graphql_authz.worker import GraphQLAuthzWorker; w = GraphQLAuthzWorker(); print(f'graphql_authz v{w.version}')"

# 5. ssrf_oob v2.5
python -c "from services.features.ssrf_oob.worker import SsrfOobWorker; w = SsrfOobWorker(); print(f'ssrf_oob v{w.version}')"
```

**預期輸出**:
```
mass_assignment v2.5.0
jwt_confusion v2.5.0
oauth_confusion v2.5.0
graphql_authz v2.5.0
ssrf_oob v2.5.0
```

**檢查項目**:
- [ ] 所有模組版本為 2.5.0
- [ ] 無導入錯誤
- [ ] 無語法錯誤

---

### 3.2 測試功能註冊表

**執行命令**:
```powershell
python -c "
from services.features.base.feature_registry import FeatureRegistry
print('已註冊的功能模組:')
for name in FeatureRegistry.list_all():
    print(f'  - {name}')
print(f'總計: {len(FeatureRegistry.list_all())} 個模組')
"
```

**預期結果**:
- [ ] 至少包含 8 個註冊模組
- [ ] 包含所有 v2.5 模組
- [ ] 無重複註冊

---

## 📋 **Phase 4: 靶場環境配置**

### 4.1 準備測試目標

**鄧鄧靶場配置** (需手動設定):

```python
# 創建測試配置文件
# 文件: tests/deng_deng_range_config.py

DENG_DENG_TARGETS = {
    "base_url": "http://localhost:8080",  # 靶場基礎 URL
    "test_accounts": {
        "user": {
            "username": "testuser",
            "password": "testpass123",
            "email": "user@test.local"
        },
        "admin": {
            "username": "admin",
            "password": "adminpass123",
            "email": "admin@test.local"
        }
    },
    "endpoints": {
        "oauth": "/oauth/authorize",
        "graphql": "/graphql",
        "api": "/api/v1",
        "pdf_gen": "/pdf/generate"
    }
}
```

**檢查項目**:
- [ ] 靶場服務運行中
- [ ] 測試帳戶已創建
- [ ] 網路連接正常
- [ ] 端點可訪問

---

### 4.2 OOB 平台設置

**SSRF OOB 測試需求**:

```bash
# 需要外部 OOB 服務
# 選項 1: Burp Collaborator
# 選項 2: interact.sh
# 選項 3: dnslog.cn

# 測試 OOB 連接
curl http://your-collaborator-id.burpcollaborator.net/test
```

**檢查項目**:
- [ ] OOB 服務可用
- [ ] DNS 解析正常
- [ ] HTTP 回調正常
- [ ] Token 追蹤可用

---

## 📋 **Phase 5: 實測執行計劃**

### 5.1 Mass Assignment 測試

**測試命令模板**:
```python
# 文件: tests/test_mass_assignment_range.py
from services.features.mass_assignment.worker import MassAssignmentWorker

worker = MassAssignmentWorker()

params = {
    "target": "http://localhost:8080",
    "endpoint": "/api/users/123",
    "method": "PUT",
    "headers": {"Authorization": "Bearer YOUR_TOKEN"},
    "baseline_request": {
        "name": "Test User",
        "email": "test@example.com"
    },
    "test_fields": [
        {"role": "admin"},
        {"is_staff": True},
        {"is_superuser": True}
    ]
}

result = worker.run(params)
print(f"發現漏洞: {result.ok}")
print(f"發現數量: {len(result.findings)}")
for finding in result.findings:
    print(f"  - {finding.title} ({finding.severity})")
```

**執行步驟**:
1. [ ] 修改 target URL
2. [ ] 設置有效 token
3. [ ] 調整測試端點
4. [ ] 執行測試
5. [ ] 記錄結果

---

### 5.2 JWT Confusion 測試

**測試命令模板**:
```python
# 文件: tests/test_jwt_confusion_range.py
from services.features.jwt_confusion.worker import JwtConfusionWorker

worker = JwtConfusionWorker()

params = {
    "target": "http://localhost:8080",
    "validate_endpoint": "/api/validate",
    "victim_token": "eyJhbGc...",  # 有效的 JWT
    "jwks_url": "http://localhost:8080/.well-known/jwks.json",
    "headers": {},
    "tests": {
        "algorithm_confusion": True,
        "none_algorithm": True,
        "jwk_rotation": True,
        "algorithm_downgrade": True,  # v2.5
        "weak_secret": True            # v2.5
    }
}

result = worker.run(params)
print(f"JWT 漏洞測試結果: {len(result.findings)} 個發現")
```

**執行步驟**:
1. [ ] 獲取有效 JWT token
2. [ ] 確認 JWKS 端點
3. [ ] 執行測試
4. [ ] 驗證 v2.5 新功能
5. [ ] 記錄結果

---

### 5.3 OAuth Confusion 測試

**測試命令模板**:
```python
# 文件: tests/test_oauth_confusion_range.py
from services.features.oauth_confusion.worker import OAuthConfusionWorker

worker = OAuthConfusionWorker()

params = {
    "target": "http://localhost:8080",
    "auth_endpoint": "/oauth/authorize",
    "token_endpoint": "/oauth/token",
    "client_id": "test-client-id",
    "client_secret": "test-secret",
    "redirect_uri": "http://localhost:8080/callback",
    "attacker_redirect": "http://evil.com/steal",
    "scope": "openid profile email",
    "tests": {
        "open_redirect": True,
        "location_header_reflection": True,  # v2.5
        "relaxed_redirect_codes": True,      # v2.5
        "pkce_bypass": True                  # v2.5
    }
}

result = worker.run(params)
print(f"OAuth 漏洞測試結果: {len(result.findings)} 個發現")
```

**執行步驟**:
1. [ ] 配置 OAuth client
2. [ ] 設置重定向 URI
3. [ ] 準備攻擊者端點
4. [ ] 執行測試
5. [ ] 驗證 v2.5 PKCE bypass
6. [ ] 記錄結果

---

### 5.4 GraphQL Authorization 測試

**測試命令模板**:
```python
# 文件: tests/test_graphql_authz_range.py
from services.features.graphql_authz.worker import GraphQLAuthzWorker

worker = GraphQLAuthzWorker()

params = {
    "target": "http://localhost:8080",
    "endpoint": "/graphql",
    "headers_user": {"Authorization": "Bearer USER_TOKEN"},
    "headers_admin": {"Authorization": "Bearer ADMIN_TOKEN"},
    "test_queries": [
        {
            "name": "getUserProfile",
            "query": "{ user(id: 123) { id name email role } }",
            "target_user_id": "123"
        }
    ],
    "tests": {
        "introspection": True,
        "field_level_authz": True,
        "object_level_authz": True,
        "batch_queries": True,        # v2.5
        "error_analysis": True        # v2.5
    },
    "batch_base_query": "{ user(id: $id) { id name email } }"
}

result = worker.run(params)
print(f"GraphQL 漏洞測試結果: {len(result.findings)} 個發現")
```

**執行步驟**:
1. [ ] 獲取 user 和 admin tokens
2. [ ] 準備測試查詢
3. [ ] 執行測試
4. [ ] 驗證 v2.5 批次查詢測試
5. [ ] 檢查欄位權重分析
6. [ ] 記錄結果

---

### 5.5 SSRF OOB 測試

**測試命令模板**:
```python
# 文件: tests/test_ssrf_oob_range.py
from services.features.ssrf_oob.worker import SsrfOobWorker

worker = SsrfOobWorker()

params = {
    "target": "http://localhost:8080",
    "probe_endpoints": ["/fetch", "/pdf/generate", "/preview"],
    "url_params": ["url", "link", "fetch"],
    "json_fields": ["url", "imageUrl"],
    "headers": {},
    "oob_http": "http://your-id.burpcollaborator.net",
    "oob_dns": "your-id.burpcollaborator.net",
    "test_protocols": ["http", "https"],
    "payload_types": ["direct", "encoded"],
    "options": {
        "delay_seconds": 5,
        "auto_discover": True,
        "test_internal": False,
        "test_pdf_injection": True,        # v2.5
        "test_protocol_conversion": True,  # v2.5
        "callback_window": "normal"        # v2.5
    }
}

result = worker.run(params)
print(f"SSRF 漏洞測試結果: {len(result.findings)} 個發現")
print(f"OOB Token: {result.meta['test_token']}")
print("請檢查 OOB 平台確認回調")
```

**執行步驟**:
1. [ ] 設置 OOB 平台
2. [ ] 配置回調 URL
3. [ ] 執行測試
4. [ ] 等待回調
5. [ ] 驗證 v2.5 PDF 注入
6. [ ] 檢查協議轉換
7. [ ] 記錄結果

---

## 📋 **Phase 6: 結果收集與分析**

### 6.1 生成測試報告

**執行命令**:
```python
# 文件: tests/generate_range_test_report.py
import json
from datetime import datetime

results = {
    "test_date": datetime.now().isoformat(),
    "target": "鄧鄧靶場",
    "modules_tested": 5,
    "total_findings": 0,
    "findings_by_module": {},
    "v2_5_features_verified": []
}

# 收集所有測試結果
# 填入實際數據

# 保存報告
with open("deng_deng_range_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("測試報告已生成: deng_deng_range_test_results.json")
```

**檢查項目**:
- [ ] 所有模組測試完成
- [ ] v2.5 新功能驗證
- [ ] 漏洞證據完整
- [ ] 性能數據記錄

---

### 6.2 驗證 v2.5 特性

**v2.5 新功能檢查清單**:

**mass_assignment v2.5**:
- [ ] 欄位矩陣分析運作正常
- [ ] 雙端點驗證執行成功
- [ ] 證據鏈完整記錄
- [ ] 時間戳精確追蹤

**jwt_confusion v2.5**:
- [ ] JWK 輪換窗口檢測
- [ ] 算法降級鏈測試
- [ ] 弱密鑰爆破執行
- [ ] 多階段證據生成

**oauth_confusion v2.5**:
- [ ] Location header 反射檢測
- [ ] 5種重定向碼測試
- [ ] PKCE 繞過鏈執行
- [ ] OAuth 流程時間軸

**graphql_authz v2.5**:
- [ ] 欄位權重矩陣分析
- [ ] 批次查詢測試 (3種模式)
- [ ] 權限矩陣構建
- [ ] 錯誤消息提取

**ssrf_oob v2.5**:
- [ ] PDF 路徑注入 (6種模板)
- [ ] OOB 證據腳手架
- [ ] 協議轉換鏈 (6種)
- [ ] 回調窗口驗證 (4級)

---

## 📋 **Phase 7: 性能基準測試**

### 7.1 掃描速度測試

**執行命令**:
```powershell
# 測試單個模組執行時間
Measure-Command {
    python -c "
from services.features.mass_assignment.worker import MassAssignmentWorker
worker = MassAssignmentWorker()
# 執行快速測試
params = {'target': 'http://localhost:8080', 'endpoint': '/api/test'}
result = worker.run(params)
"
}
```

**性能目標** (根據 SOP):
- [ ] AI 核心通過率: 95%+
- [ ] 掃描時間: <1.0s (目標)
- [ ] 並發能力: 2000+ tasks/s
- [ ] 記憶體使用: <500MB

---

### 7.2 並發測試

**執行命令**:
```python
# 文件: tests/test_concurrent_scan.py
import concurrent.futures
import time

def run_scan(module_name):
    # 執行單個掃描
    start = time.time()
    # ... 掃描邏輯
    duration = time.time() - start
    return module_name, duration

# 並發執行
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(run_scan, f"test_{i}") for i in range(10)]
    results = [f.result() for f in futures]

print(f"並發掃描完成: {len(results)} 個任務")
```

**檢查項目**:
- [ ] 並發執行無錯誤
- [ ] 資源使用合理
- [ ] 結果一致性

---

## 📋 **Phase 8: 問題排查指南**

### 8.1 常見問題

**問題 1: 模組導入失敗**
```bash
# 解決方案
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:/path/to/AIVA-git
```

**問題 2: Token 過期**
```bash
# 重新獲取 token
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"testpass123"}'
```

**問題 3: OOB 無回調**
```bash
# 檢查網路連接
ping your-id.burpcollaborator.net
curl http://your-id.burpcollaborator.net/test
```

**問題 4: 靶場無響應**
```bash
# 檢查靶場狀態
curl http://localhost:8080/health
netstat -ano | findstr :8080
```

---

### 8.2 日誌收集

**啟用詳細日誌**:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='aiva_range_test.log'
)
```

**檢查項目**:
- [ ] 錯誤日誌收集
- [ ] 請求/響應記錄
- [ ] 性能指標記錄

---

## 📋 **Phase 9: 最終檢查清單**

### 準備就緒確認

**系統層面**:
- [ ] Python 環境: 3.9+ ✅
- [ ] 依賴安裝: 完整 ✅
- [ ] 語法檢查: 通過 ✅
- [ ] 系統通連: 100% ✅

**模組層面**:
- [ ] mass_assignment v2.5: 可用
- [ ] jwt_confusion v2.5: 可用
- [ ] oauth_confusion v2.5: 可用
- [ ] graphql_authz v2.5: 可用
- [ ] ssrf_oob v2.5: 可用

**靶場環境**:
- [ ] 靶場服務: 運行中
- [ ] 測試帳戶: 已準備
- [ ] OOB 平台: 已配置
- [ ] 網路連接: 正常

**測試準備**:
- [ ] 測試腳本: 已準備
- [ ] 配置文件: 已創建
- [ ] 日誌系統: 已啟用
- [ ] 報告模板: 已準備

---

## 🚀 **執行命令總覽**

### 快速測試所有模組

**一鍵測試腳本**:
```powershell
# 文件: tests/quick_range_test.ps1

Write-Host "=== AIVA v2.5 靶場快速測試 ===" -ForegroundColor Green

# 1. 檢查模組可用性
Write-Host "`n[1/5] 檢查模組..." -ForegroundColor Yellow
python -c "from services.features.mass_assignment.worker import MassAssignmentWorker; print('✓ mass_assignment')"
python -c "from services.features.jwt_confusion.worker import JwtConfusionWorker; print('✓ jwt_confusion')"
python -c "from services.features.oauth_confusion.worker import OAuthConfusionWorker; print('✓ oauth_confusion')"
python -c "from services.features.graphql_authz.worker import GraphQLAuthzWorker; print('✓ graphql_authz')"
python -c "from services.features.ssrf_oob.worker import SsrfOobWorker; print('✓ ssrf_oob')"

# 2. 驗證版本
Write-Host "`n[2/5] 驗證版本..." -ForegroundColor Yellow
python -c "from services.features.mass_assignment.worker import MassAssignmentWorker; w = MassAssignmentWorker(); print(f'mass_assignment: v{w.version}')"
# ... 其他模組

# 3. 測試基本功能
Write-Host "`n[3/5] 測試基本功能..." -ForegroundColor Yellow
# 執行簡單測試

# 4. 生成報告
Write-Host "`n[4/5] 生成報告..." -ForegroundColor Yellow
# 保存結果

Write-Host "`n[5/5] 測試完成!" -ForegroundColor Green
```

---

## 📊 **預期成果**

### 成功標準

**功能性**:
- ✅ 所有 v2.5 模組正常運作
- ✅ v2.5 新功能完整驗證
- ✅ 漏洞檢測準確率 >90%
- ✅ 無誤報或崩潰

**性能性**:
- ✅ 掃描時間符合目標
- ✅ 並發處理穩定
- ✅ 資源使用合理
- ✅ 時間戳精確記錄

**證據完整性**:
- ✅ 每個漏洞有完整證據
- ✅ 復現步驟清晰
- ✅ 時間軸記錄完整
- ✅ Meta 數據完整

---

## 📝 **注意事項**

### 重要提醒

1. **不要在生產環境測試**: 僅在靶場環境執行
2. **保護敏感信息**: Token、密碼不要提交到 Git
3. **遵守授權範圍**: 僅測試授權的靶場目標
4. **記錄完整日誌**: 便於問題排查和復現
5. **備份測試數據**: 測試前備份靶場狀態

### 合規要求

- [ ] 測試前獲得授權
- [ ] 測試範圍明確
- [ ] 測試後清理數據
- [ ] 報告保密處理

---

## 🎯 **執行順序建議**

1. **Phase 1-2**: 系統檢查 (10分鐘)
2. **Phase 3**: 模組驗證 (15分鐘)
3. **Phase 4**: 環境配置 (30分鐘)
4. **Phase 5**: 實測執行 (60-90分鐘)
5. **Phase 6**: 結果分析 (30分鐘)
6. **Phase 7**: 性能測試 (30分鐘)
7. **Phase 8-9**: 問題排查和最終確認 (視需要)

**總預計時間**: 3-4 小時

---

## ✅ **完成確認**

測試完成後,請確認:

- [ ] 所有檢查項目已執行
- [ ] 測試報告已生成
- [ ] 發現的漏洞已記錄
- [ ] 性能數據已收集
- [ ] v2.5 功能已驗證
- [ ] 問題已記錄並解決
- [ ] 測試環境已清理

---

**準備就緒!請按照此清單逐步執行測試** 🚀

**記住**: AI 僅提供指令,所有操作需人工執行並確認!
