# AIVA 開發環境快速設置指南 ✅ 11/10驗證

> **📋 適用對象**: 新手開發者、團隊成員、快速部署需求  
> **🎯 使用場景**: 開發環境初始化、快速項目設置、環境驗證  
> **⏱️ 預計完成時間**: 10 分鐘  
> **🔧 先決條件**: Python 3.11+、Node.js 18+、Rust 1.70+、Go 1.21+  
> **📅 最後更新**: 2025-11-06 (配合功能模組完成狀況)

---

## 📑 目錄

1. [🚀 立即開始開發](#-立即開始開發)
2. [🔧 環境驗證](#-環境驗證)
3. [📦 依賴安裝](#-依賴安裝)
4. [⚡ 開發工具配置](#-開發工具配置)
5. [🧪 功能測試](#-功能測試)
6. [🔍 問題排查](#-問題排查)
7. [📊 性能檢查](#-性能檢查)
8. [🎯 下一步指引](#-下一步指引)

---

## 🚀 立即開始開發

基於當前系統狀態，功能模組需求文件已完成（6份技術報告涵蓋10個模組），可立即開始功能模組實施開發。

### 1. 環境驗證 (2分鐘)

```bash
# 快速驗證腳本
cd c:\D\fold7\AIVA-git

# 檢查補包狀態
python scripts/utilities/aiva_package_validator.py

# 預期結果: 專案結構驗證 (部分通過即可正常開發)
# 如果出現嚴重錯誤，請檢查以下項目：
# - Python環境是否正確
# - 依賴是否完整安裝
```

### 2. 開發前準備 (5分鐘)

```bash
# 安裝額外依賴 (為 Week 3-4 客戶端檢測準備)
pip install playwright esprima beautifulsoup4 lxml

# 安裝瀏覽器 (僅需要 Chromium)
playwright install chromium

# 驗證 Go 環境 (SSRF 模組需要)
cd services/features/function_ssrf_go
go mod tidy
go build ./...
cd ../../..
```

### 3. 創建開發分支

```bash
# 創建開發分支
git checkout -b phase-i-development
git push -u origin phase-i-development

# 創建功能分支 (建議)
git checkout -b feature/ai-attack-mapper
git checkout -b feature/ssrf-microservice  
git checkout -b feature/client-auth-bypass
```

## 📂 開發目錄結構

```
AIVA-git/
├── 📋 開發規劃文件
│   ├── AIVA_PHASE_I_DEVELOPMENT_PLAN.md      # 詳細開發規劃
│   ├── DEVELOPMENT_TASKS_CHECKLIST.md        # 執行任務清單
│   └── AIVA_PACKAGE_INTEGRATION_COMPLETE.md  # 整合完成報告
│
├── 🧪 測試框架 (需建立)
│   ├── tests/
│   │   ├── test_attack_plan_mapper.py
│   │   ├── test_client_auth_bypass.py
│   │   ├── test_js_analysis_engine.py
│   │   └── test_integration.py
│   │
│   └── test_data/
│       ├── sample_js_files/
│       ├── test_targets.json
│       └── mock_responses/
│
├── 🏗️ 開發模組 (已建立框架)
│   ├── services/core/aiva_core/execution/
│   │   └── attack_plan_mapper.py            # Week 1 主要開發
│   │
│   ├── services/features/client_side_auth_bypass/
│   │   ├── client_side_auth_bypass_worker.py # Week 3-4 主要開發
│   │   └── js_analysis_engine.py
│   │
│   └── services/features/function_ssrf_go/
│       └── internal/detector/               # Week 2 主要開發
│           ├── internal_microservice_probe.go
│           └── cloud_metadata_scanner.go
│
└── 🔧 工具和腳本
    ├── aiva_package_validator.py            # 每日檢查
    ├── aiva_system_connectivity_sop_check.py # 通連性驗證
    └── scripts/ (需建立)
        ├── daily_check.sh                   # 每日檢查腳本
        ├── run_tests.py                     # 測試執行腳本
        └── build_all.sh                     # 建置腳本
```

## 💻 Week 1 立即開始 - AI攻擊計畫映射器

### Step 1: 建立測試框架 (30分鐘)

```bash
# 創建測試目錄
mkdir -p tests test_data/sample_decisions

# 建立測試文件
cat > tests/test_attack_plan_mapper.py << 'EOF'
import unittest
import asyncio
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.core.aiva_core.execution.attack_plan_mapper import AttackPlanMapper
from services.aiva_common.schemas.generated.messaging import AivaMessage

class TestAttackPlanMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = AttackPlanMapper()
    
    def test_mapper_initialization(self):
        """測試映射器初始化"""
        self.assertIsNotNone(self.mapper)
    
    async def async_test_decision_mapping(self):
        """測試決策映射功能"""
        # 創建模擬決策
        mock_decision = Mock()
        mock_decision.payload = {
            'action_type': 'vulnerability_scan',
            'target': {'url': 'http://example.com'},
            'vulnerability_type': 'sqli'
        }
        mock_decision.header.message_id = 'test_001'
        
        # 執行映射
        tasks = await self.mapper.map_decision_to_tasks(
            mock_decision, 
            {'session_id': 'test_session'}
        )
        
        # 驗證結果
        self.assertIsInstance(tasks, list)
        
    def test_vulnerability_module_mapping(self):
        """測試漏洞類型映射"""
        test_cases = [
            ('sqli', 'FUNC_SQLI'),
            ('xss', 'FUNC_XSS'),
            ('ssrf', 'FUNC_SSRF'),
            ('unknown_type', 'FUNC_GENERAL_SCAN')
        ]
        
        for vuln_type, expected_module in test_cases:
            result = self.mapper._map_vulnerability_to_module(vuln_type)
            self.assertEqual(result, expected_module)

if __name__ == '__main__':
    unittest.main()
EOF
```

### Step 2: 開始核心開發 (第一個功能)

```python
# 編輯 services/core/aiva_core/execution/attack_plan_mapper.py
# 擴展 map_decision_to_tasks 方法

# 添加以下代碼到現有檔案中 (在 map_decision_to_tasks 方法內)
```

### Step 3: 執行第一次測試

```bash
# 執行測試
python -m pytest tests/test_attack_plan_mapper.py -v

# 執行系統檢查
python aiva_package_validator.py
```

## ⚡ 開發工具配置

### IDE 性能優化設定
AIVA 專案已針對多語言開發環境進行了語言伺服器優化配置，確保開發時的最佳性能：

📋 **完整優化指南**: [語言伺服器優化設定指南](LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md)

**主要優化效果**：
- ✅ 只分析開啟的檔案，避免全專案掃描
- ✅ 程式碼變更後30秒才開始診斷，減少CPU使用
- ✅ 關閉不必要的索引和即時提示，提升響應速度

**支援語言**：
- 🐍 **Python (Pylance)** - `openFilesOnly` 模式，30秒診斷延遲
- 🦀 **Rust (rust-analyzer)** - 只在儲存時完整檢查
- 🟢 **Go (gopls)** - 30秒延遲觸發，編輯模式

**快速驗證**：
```bash
# 檢查語言伺服器配置
# 注意：verify-language-configs.ps1 腳本不在當前專案中
# 可以手動檢查 .vscode/settings.json 和 AIVA.code-workspace 檔案
```

**如果需要調整設定**，請參考：
- 主要設定檔：`.vscode/settings.json`
- 工作區設定：`AIVA.code-workspace`
- 詳細說明：[語言伺服器優化設定指南](LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md)

---

## 🔄 每日開發流程

### 每日開始 (5分鐘)
```bash
# 1. 環境檢查
python scripts/utilities/aiva_package_validator.py

# 2. 拉取最新代碼
git pull origin main
git merge main  # 如果在feature分支

# 3. 檢查依賴
pip list | grep -E "(pydantic|jinja2|pyyaml)"
```

### 每日結束 (10分鐘)
```bash
# 1. 執行測試
python -m pytest tests/ -v

# 2. 檢查代碼品質
python -m flake8 services/core/aiva_core/execution/
python -m mypy services/core/aiva_core/execution/ --ignore-missing-imports

# 3. 提交代碼
git add .
git commit -m "feat: implement [具體功能描述]"
git push origin feature/[branch-name]

# 4. 更新進度
# 編輯 DEVELOPMENT_TASKS_CHECKLIST.md 更新完成狀態
```

## 🎯 快速啟動命令

### 一鍵環境驗證
```bash
# Windows PowerShell
& {
    Write-Host "🔍 驗證AIVA開發環境..." -ForegroundColor Green
    python scripts/utilities/aiva_package_validator.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 環境就緒，可以開始開發！" -ForegroundColor Green
        Write-Host "📝 請查看 DEVELOPMENT_TASKS_CHECKLIST.md 開始Week 1任務" -ForegroundColor Yellow
    } else {
        Write-Host "❌ 環境檢查失敗，請檢查錯誤信息" -ForegroundColor Red
    }
}
```

### 一鍵測試執行
```bash
# 創建測試執行腳本
cat > run_dev_tests.py << 'EOF'
import subprocess
import sys

def run_command(cmd, description):
    print(f"🧪 {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {description} 成功")
        return True
    else:
        print(f"❌ {description} 失敗:")
        print(result.stderr)
        return False

def main():
    tests = [
        ("python scripts/utilities/aiva_package_validator.py", "補包驗證"),
        ("python -m pytest tests/ -v", "單元測試"),
        ("python -c 'from services.core.aiva_core.execution.attack_plan_mapper import AttackPlanMapper; print(\"導入成功\")'", "模組導入測試"),
    ]
    
    success_count = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n📊 測試結果: {success_count}/{len(tests)} 通過")
    if success_count == len(tests):
        print("🎉 所有檢查通過，可以繼續開發！")
        return 0
    else:
        print("⚠️ 部分檢查失敗，請修復後再繼續")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

python run_dev_tests.py
```

## 📞 問題排解

### 常見問題 Quick Fix

1. **模組導入失敗**
```bash
# 檢查 sys.path
python -c "import sys; print('\n'.join(sys.path))"

# 重新執行路徑修復
python testing/integration/aiva_system_connectivity_sop_check.py
```

2. **Go 編譯失敗**
```bash
cd services/features/function_ssrf_go
go mod tidy
go clean -cache
go build ./...
```

3. **依賴問題**
```bash
pip install --upgrade -r requirements.txt
pip list --outdated
```

## 🎉 準備完成確認

執行以下檢查，確保可以開始開發：

- [ ] `python aiva_package_validator.py` 顯示 🟢 優秀
- [ ] 測試框架建立完成
- [ ] Git 分支策略設置完成  
- [ ] 每日流程腳本準備就緒
- [ ] Week 1 任務清單已檢視

**✅ 全部完成後，即可開始 Phase I Week 1 開發！**

---

**快速開始**: 直接執行 `python run_dev_tests.py` 驗證環境，然後開始編輯 `attack_plan_mapper.py`  
**支援文件**: 參考 `DEVELOPMENT_TASKS_CHECKLIST.md` 了解詳細任務