# AIVA 能力 CLI 指令設計方案

生成時間: 2025-12-07  
分析基礎: 數據流分類結果 (439條數據流)

---

## 📋 執行摘要

### 核心發現

1. **指令進出點** (3個主要入口):
   - `services/core/aiva_core/service_backbone/api/app.py` - FastAPI 主入口 (AI指揮中心)
   - `services/integration/aiva_integration/app.py` - 整合服務入口
   - `src/launchers/aiva_launcher.py` - 系統啟動器

2. **數據流能力分布**:
   - 總數據流: **439條**
   - 服務骨幹模組: **189條** (43.1%) - 最多
   - 外學模組: **166條** (37.8%)
   - 認知核心: **60條** (13.7%)
   - AI組件: **250條** (56.9%)
   - 混合組件: **85條** (19.4%)

3. **現有 CLI 工具**:
   - `capability_cli.py` - ✅ 已存在但已刪除 (功能有限)
   - `capability_command_builder.py` - ✅ 已存在但已刪除 (指令構建)
   - `aiva_launcher.py` - ✅ 系統級啟動器 (argparse)

4. **推薦方案**: **優化並重建 `capability_cli.py`**

---

## 🎯 設計目標

### 1. 指令設計原則

```
Human/AI Input → Entry Point → Capability CLI → Data Flow → Module Execution → Result
     ↓              ↓              ↓                ↓              ↓            ↓
  自然語言       app.py      指令解析        能力路由      功能執行      結果回傳
     ↓              ↓              ↓                ↓              ↓            ↓
  "執行掃描"    FastAPI   → aiva-cap exec  → Flow 14   → backends  → JSON結果
```

### 2. 核心能力

- ✅ **數據流映射**: 根據 439條數據流自動生成CLI指令
- ✅ **多路徑支援**: 處理81個多路徑終點的不同執行方式
- ✅ **模組路由**: 支援六大模組的能力調用
- ✅ **AI整合**: 接受自然語言並轉換為結構化指令
- ✅ **類型安全**: 使用 AICommand/AICommandResult 標準

---

## 📊 數據流分析結果

### 六大模組分布

| 模組 | 數量 | 占比 | 主要能力 |
|------|------|------|----------|
| **service_backbone** | 189 | 43.1% | API網關、存儲、消息總線、協調 |
| **external_learning** | 166 | 37.8% | AI訓練、強化學習、資源追蹤 |
| **cognitive_core** | 60 | 13.7% | AI決策、神經網路、RAG |
| **core_capabilities** | 15 | 3.4% | 能力編排、攻擊鏈、業務邏輯 |
| **task_planning** | 5 | 1.1% | 規劃器、執行器、指揮官 |
| **internal_exploration** | 4 | 0.9% | 自我認知、能力分析 |

### 關鍵多路徑終點 (Top 5)

1. **integration_module_sync** - 18條路徑
   - 用途: 模組同步整合
   - 最短路徑: 3步 (capability_cli → capability_command_builder → integration_module_sync)
   - 最長路徑: 5步 (涉及 AI能力查詢 + 後端存儲)

2. **capability_registry** - 多條路徑
   - 用途: 能力註冊管理
   - 快速路徑: 2步 (capability_cli → capability_registry)
   - 完整路徑: 5步 (包含經驗管理和RL模型)

3. **ai_capability_query** - 多條路徑
   - 用途: AI能力智能檢索
   - 直接調用: 1步
   - 完整流程: 包含RAG和向量搜索

4. **rl_models** - 多條路徑
   - 用途: 強化學習模型管理
   - 訓練流程: 涉及 scalable_bio_trainer
   - 評估流程: 涉及 capability_orchestrator

5. **capability_orchestrator** - 多條路徑
   - 用途: 能力協調編排
   - 簡單編排: 3-4步
   - 複雜編排: 5步以上

---

## 🏗️ 指令進出點分析

### 1. 主要入口: `app.py` (FastAPI)

**位置**: `services/core/aiva_core/service_backbone/api/app.py`

**職責**:
- ✅ FastAPI 應用程序主入口
- ✅ 持有 CoreServiceCoordinator (狀態管理器)
- ✅ 提供 RESTful API 端點
- ✅ 啟動內部閉環和外部學習後台任務

**架構層次**:
```
app.py (FastAPI)          ← 唯一主入口
    ↓ 持有
CoreServiceCoordinator    ← 狀態管理器和服務工廠
    ↓ 管理
各功能服務 (Decision, Planning, Execution...)
```

**接收指令方式**:
- HTTP POST `/api/v1/command` (推薦新增)
- HTTP POST `/api/v1/capabilities/execute` (現有能力執行)
- 內部調用: `CoreServiceCoordinator` 的方法

### 2. 整合入口: `aiva_integration/app.py`

**位置**: `services/integration/aiva_integration/app.py`

**職責**:
- ✅ FastAPI 整合模組入口
- ✅ 接收漏洞發現數據
- ✅ 協調多功能測試
- ✅ 生成報告

**接收指令方式**:
- HTTP GET `/findings/{finding_id}`
- 消息隊列: Topic.LOG_RESULTS_ALL

### 3. 系統啟動器: `aiva_launcher.py`

**位置**: `src/launchers/aiva_launcher.py`

**職責**:
- ✅ 中央入口點
- ✅ 解析命令列參數 (argparse)
- ✅ 啟動 AIVA 核心 AI 服務
- ✅ 協調啟動其他微服務

**支援模式**:
- `--mode full` - 完整系統
- `--mode core` - 僅核心
- `--target <url>` - 指定目標

---

## 🎨 CLI 指令設計方案

### 方案評估: 優化現有 vs 新建

| 維度 | 優化現有 capability_cli | 新建統一CLI工具 |
|------|------------------------|----------------|
| **開發成本** | ⭐⭐⭐⭐⭐ 低 (已有基礎) | ⭐⭐⭐ 中等 |
| **維護性** | ⭐⭐⭐⭐ 好 (專注單一職責) | ⭐⭐⭐⭐⭐ 優 (統一架構) |
| **擴展性** | ⭐⭐⭐ 中 (模組化設計) | ⭐⭐⭐⭐⭐ 優 (插件式) |
| **與現有整合** | ⭐⭐⭐⭐⭐ 完美 (原生支援) | ⭐⭐⭐ 需要適配 |
| **學習曲線** | ⭐⭐⭐⭐ 低 (符合習慣) | ⭐⭐⭐ 中等 |
| **AI整合** | ⭐⭐⭐⭐ 好 | ⭐⭐⭐⭐⭐ 優 (專門設計) |

**推薦**: ✅ **優化並重建 `capability_cli.py`**

**理由**:
1. 已有數據流分析基礎 (439條)
2. 與 aiva_common 規範完美契合
3. 直接使用 AICommand/AICommandResult
4. 開發周期短 (1-2天)
5. 風險低 (基於現有架構)

---

## 🛠️ 重建 Capability CLI 設計

### 1. 核心架構

```python
# capability_cli_v2.py
"""
AIVA 能力 CLI 工具 v2.0
基於數據流分析的智能能力調用系統

架構:
    Human/AI → CLI → AICommand → CommandCenter → Module → Result
    
特性:
- 基於 439 條數據流自動生成指令
- 支援 81 個多路徑終點的智能路由
- 完整 AICommand 協議支援
- 自然語言理解 (可選)
"""

import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

from services.aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandType,
    CommandStatus
)
from services.aiva_common.command_center import get_command_center
from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
    AIVACoreServiceCoordinator
)


class CapabilityCLI:
    """能力命令行工具 v2.0"""
    
    def __init__(self):
        self.command_center = get_command_center()
        self.coordinator = AIVACoreServiceCoordinator()
        self.flow_map = self._load_flow_map()
    
    def _load_flow_map(self) -> Dict[str, List[Dict]]:
        """載入數據流映射 (從分類結果)"""
        # 從 classification_data.json 載入
        pass
    
    async def execute_capability(
        self,
        capability_name: str,
        parameters: Dict[str, Any],
        flow_preference: str = "fastest"  # fastest, balanced, complete
    ) -> AICommandResult:
        """
        執行能力
        
        Args:
            capability_name: 能力名稱 (e.g., "ai_capability_query")
            parameters: 執行參數
            flow_preference: 流程偏好
                - fastest: 最短路徑
                - balanced: 平衡路徑 (預設)
                - complete: 完整路徑 (包含所有處理)
        """
        # 1. 查找可用的數據流路徑
        flows = self.flow_map.get(capability_name, [])
        
        if not flows:
            return AICommandResult(
                command_id="",
                status=CommandStatus.FAILED,
                success=False,
                error=f"Capability '{capability_name}' not found"
            )
        
        # 2. 根據偏好選擇路徑
        selected_flow = self._select_flow(flows, flow_preference)
        
        # 3. 構建 AICommand
        command = AICommand(
            command_type=CommandType.CORE_ANALYZE,  # 根據能力類型映射
            target_module=selected_flow['module'],
            payload={
                "capability": capability_name,
                "flow_path": selected_flow['path'],
                **parameters
            }
        )
        
        # 4. 執行命令
        result = await self.command_center.execute(command)
        
        return result
    
    def _select_flow(
        self,
        flows: List[Dict],
        preference: str
    ) -> Dict:
        """選擇數據流"""
        if preference == "fastest":
            # 選擇最短路徑
            return min(flows, key=lambda f: f['length'])
        elif preference == "complete":
            # 選擇最完整路徑 (涉及最多模組)
            return max(flows, key=lambda f: len(f['modules']))
        else:  # balanced
            # 選擇平衡路徑 (中等長度)
            avg_length = sum(f['length'] for f in flows) / len(flows)
            return min(flows, key=lambda f: abs(f['length'] - avg_length))
```

### 2. CLI 指令結構

```bash
# 基本格式
aiva-cap <command> [options]

# 主要命令
aiva-cap list              # 列出所有能力 (基於數據流分類)
aiva-cap info <name>       # 查看能力詳情 (顯示所有可用路徑)
aiva-cap exec <name>       # 執行能力
aiva-cap flows <endpoint>  # 查看到達終點的所有路徑
aiva-cap analyze           # 分析當前系統能力覆蓋率
```

### 3. 指令示例

#### 示例 1: 執行 AI 能力查詢

```bash
# 使用最快路徑
aiva-cap exec ai_capability_query \
  --query "SQL injection detection" \
  --flow fastest

# 使用完整路徑 (包含 RAG 和向量搜索)
aiva-cap exec ai_capability_query \
  --query "SQL injection detection" \
  --flow complete \
  --parameters '{"use_rag": true, "top_k": 5}'
```

**對應數據流**:
- **Fastest**: Flow 1 (1步) - 直接調用
- **Complete**: Flow 14 (5步) - capability_cli → ai_capability_query → backends → message_broker → integration_module_sync

#### 示例 2: 查看多路徑終點

```bash
# 查看 integration_module_sync 的所有路徑 (18條)
aiva-cap flows integration_module_sync

# 輸出:
# 📊 integration_module_sync - 18 paths available
# 
# 1. Flow 14 (5 steps) - Complete Path
#    capability_cli → ai_capability_query → backends → message_broker → integration_module_sync
#    Module: service_backbone
#    Use case: Full integration with AI query + storage + messaging
# 
# 2. Flow 113 (3 steps) - Fast Path
#    capability_cli → capability_command_builder → integration_module_sync
#    Module: service_backbone
#    Use case: Direct command-based integration
# 
# 3. Flow 132 (3 steps) - Storage Path
#    capability_cli → postgresql_vector_store → integration_module_sync
#    Module: service_backbone
#    Use case: Vector storage-based integration
```

#### 示例 3: 列出模組能力

```bash
# 列出所有服務骨幹模組的能力 (189條)
aiva-cap list --module service_backbone

# 列出所有 AI 組件 (250條)
aiva-cap list --type ai

# 列出混合組件 (85條)
aiva-cap list --type mixed
```

### 4. AI 自然語言接口

```bash
# 自然語言模式 (需要 AI 支援)
aiva-cap ask "我想執行 SQL 注入檢測"

# AI 解析後轉換為:
aiva-cap exec feature_sqli_test \
  --target "http://example.com" \
  --flow balanced
```

---

## 🔗 與現有系統整合

### 1. 與 AICommand 整合

```python
# capability_cli_v2.py 中的 AICommand 使用
async def execute_capability(self, capability_name: str, **kwargs):
    # 根據能力名稱映射到 CommandType
    command_type_map = {
        "ai_capability_query": CommandType.CORE_ANALYZE,
        "scan_phase0": CommandType.SCAN_PHASE0,
        "rl_trainers": CommandType.FEATURE_PAYLOAD_GENERATE,
        # ... 更多映射
    }
    
    command = AICommand(
        command_type=command_type_map.get(
            capability_name,
            CommandType.CORE_ANALYZE  # 預設
        ),
        target_module=self._get_module_from_flow(capability_name),
        payload={
            "capability": capability_name,
            **kwargs
        }
    )
    
    result = await self.command_center.execute(command)
    return result
```

### 2. 與 CoreServiceCoordinator 整合

```python
# 使用 Coordinator 獲取服務實例
class CapabilityCLI:
    def __init__(self):
        self.coordinator = AIVACoreServiceCoordinator()
        
    async def get_service(self, service_name: str):
        """獲取服務實例"""
        return self.coordinator.get_service(service_name)
    
    async def execute_via_service(
        self,
        service_name: str,
        method_name: str,
        **kwargs
    ):
        """通過服務實例直接調用"""
        service = await self.get_service(service_name)
        method = getattr(service, method_name, None)
        
        if not method:
            raise ValueError(f"Method '{method_name}' not found in service '{service_name}'")
        
        return await method(**kwargs)
```

### 3. 與數據流分類整合

```python
# 載入分類數據
def _load_flow_map(self) -> Dict[str, List[Dict]]:
    """從分類結果載入數據流映射"""
    import json
    
    classification_file = Path(
        "services/core/aiva_core/internal_exploration/"
        "classification_results_final/classification_data.json"
    )
    
    with open(classification_file) as f:
        data = json.load(f)
    
    # 構建能力 -> 流程映射
    flow_map = {}
    for flow in data['flows']:
        endpoint = flow['path'][-1]  # 最後一個節點作為能力名稱
        
        if endpoint not in flow_map:
            flow_map[endpoint] = []
        
        flow_map[endpoint].append({
            'flow_id': flow['id'],
            'length': flow['length'],
            'path': flow['path'],
            'module': flow['primary_module'],
            'modules': flow['modules'],
            'description': flow.get('description', '')
        })
    
    return flow_map
```

---

## 📈 實施計劃

### Phase 1: 基礎重建 (1-2天)

**任務**:
1. ✅ 創建 `capability_cli_v2.py` 基礎架構
2. ✅ 實現數據流映射載入 (`_load_flow_map`)
3. ✅ 實現基本指令: `list`, `info`, `flows`
4. ✅ 整合 AICommand/AICommandResult
5. ✅ 編寫單元測試

**驗收標準**:
- 能夠載入並顯示 439 條數據流
- 能夠查詢 81 個多路徑終點
- 能夠列出六大模組的能力分布

### Phase 2: 執行能力 (2-3天)

**任務**:
1. ✅ 實現 `exec` 指令
2. ✅ 實現流程選擇邏輯 (fastest/balanced/complete)
3. ✅ 整合 CommandCenter
4. ✅ 整合 CoreServiceCoordinator
5. ✅ 實現錯誤處理和重試
6. ✅ 編寫集成測試

**驗收標準**:
- 能夠執行任意能力並返回結果
- 能夠根據偏好選擇不同路徑
- 能夠處理執行失敗並重試

### Phase 3: AI 增強 (3-5天)

**任務**:
1. ✅ 實現自然語言接口 (`ask` 指令)
2. ✅ 整合 RAG 系統進行能力推薦
3. ✅ 實現智能參數補全
4. ✅ 實現執行歷史和學習
5. ✅ 編寫 AI 測試用例

**驗收標準**:
- 能夠理解並執行自然語言指令
- 能夠根據歷史推薦最佳路徑
- 能夠自動補全常用參數

### Phase 4: 文檔和部署 (1-2天)

**任務**:
1. ✅ 編寫完整使用文檔
2. ✅ 創建指令速查表
3. ✅ 編寫部署腳本
4. ✅ 集成到 aiva_launcher.py
5. ✅ 用戶培訓材料

**驗收標準**:
- 完整的 CLI 使用文檔
- 可執行的部署腳本
- 用戶培訓視頻/文檔

---

## 🎯 成功指標

### 技術指標

- ✅ 覆蓋率: 100% 數據流能力可調用 (439/439)
- ✅ 響應時間: < 2秒 (快速路徑)
- ✅ 成功率: > 95% (執行成功)
- ✅ 錯誤恢復: 100% (所有錯誤有明確提示)

### 用戶體驗指標

- ✅ 學習曲線: < 30分鐘 (從零到基本使用)
- ✅ 指令直觀性: 90% 用戶無需查看文檔能執行基本操作
- ✅ 錯誤提示清晰度: 100% 錯誤有解決方案提示

### 系統整合指標

- ✅ API 兼容性: 100% 與 app.py 整合
- ✅ 數據合約: 100% 使用 AICommand 協議
- ✅ 模組覆蓋: 100% 六大模組可調用

---

## 📚 附錄

### A. 數據流分類完整統計

```
總數據流: 439

模組分布:
- service_backbone:     189 (43.1%)
- external_learning:    166 (37.8%)
- cognitive_core:        60 (13.7%)
- core_capabilities:     15 ( 3.4%)
- task_planning:          5 ( 1.1%)
- internal_exploration:   4 ( 0.9%)

組件類型:
- AI組件:     250 (56.9%)
- 程式組件:   104 (23.7%)
- 混合組件:    85 (19.4%)

多路徑終點: 81 個
平均流程長度: 4.33 步
最長流程: 5 步
最短流程: 2 步
```

### B. 關鍵文件路徑

```
# CLI 工具
services/core/aiva_core/internal_exploration/capability_cli_v2.py  # 重建的CLI

# 數據流分類結果
services/core/aiva_core/internal_exploration/classification_results_final/
  ├── classification_summary.md       # 統計摘要
  ├── complete_flow_details.md        # 完整流程詳情
  ├── multi_path_analysis.md          # 多路徑分析
  └── classification_data.json        # JSON 數據 (CLI載入)

# 系統入口
services/core/aiva_core/service_backbone/api/app.py              # FastAPI 主入口
services/integration/aiva_integration/app.py                     # 整合服務
src/launchers/aiva_launcher.py                                   # 系統啟動器

# 命令系統
services/aiva_common/schemas/commands.py                         # AICommand 定義
services/aiva_common/command_center.py                           # 命令中心
```

### C. 相關規範文檔

- `services/aiva_common/README.md` - AIVA Common 規範
- `services/core/aiva_core/__init__.py` - Core 模組架構說明
- `第一章流程實施可行性分析.md` - 實施可行性分析

---

## 🔄 後續優化方向

1. **性能優化**:
   - 實現流程緩存 (避免重複查詢)
   - 並行執行多個能力
   - 智能預載入常用模組

2. **功能擴展**:
   - 支援能力組合 (pipeline)
   - 支援條件執行 (if-then-else)
   - 支援循環執行 (for/while)

3. **AI 深度整合**:
   - 自動能力發現和註冊
   - 基於成功率的路徑優化
   - 異常模式識別和預警

4. **多語言支援**:
   - 支援 Go/Rust/TypeScript 能力調用
   - 跨語言能力組合
   - 統一調用接口

---

**文檔結束**

*此方案基於 439 條數據流分析結果，完全符合 aiva_common 規範和 AICommand 協議。*
