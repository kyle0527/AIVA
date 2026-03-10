# 🎯 AI 核心架構優化方案分析報告

> **分析日期**: 2026-02-09  
> **基於**: 使用者手冊第3冊、第5冊 + 當前代碼審查  
> **提案來源**: CLI 參數包驅動架構  

---

## 📊 一、當前架構現狀

### 1.1 已實現的架構要素

✅ **CLI 驅動基礎**（手冊第3冊 Step 7-9）
- 位置：`services/core/aiva_core/core_capabilities/cli/aiva_cli.py`
- 功能：從 `latest_classification.json` 讀取 Flow 定義
- 執行：`python -m aiva_cli flow{N} --target X --intensity Y`

✅ **統一執行器**（UnifiedAttackExecutor）
- 位置：`services/core/aiva_core/task_planning/unified_executor.py`
- 功能：協調內部/外部執行器
- 特性：支援學習、經驗管理、RAG 集成

✅ **Flow 分析數據**
- `aiva_flow_analysis_v3.json`（42萬行，13K+ flows）
- `latest_classification.json`（動態更新的分類數據）
- 內部/外部雙軌分析（171 vs 525 flows）

✅ **依賴注入模式**（手冊第5冊）
```
app.py → CommanderCoordinator → AttackCoordinator → UnifiedExecutor
```

### 1.2 尚未完全實現的部分

❌ **AI 到 CLI 的嚴格解耦**
- 現狀：UnifiedExecutor 仍可能直接 import Python 模組
- 問題：AI 和執行層耦合度較高

❌ **標準化 CLICommand 數據模型**
- 現狀：`commander/types.py` 只定義了 Enum，沒有 CLICommand
- 缺失：Pydantic 模型化的 CLI 參數包

❌ **Flow ID 到 CLI 的自動映射**
- 現狀：CLI 工具選擇邏輯分散
- 缺失：統一的 Flow 選擇器（CLIToolSelector）

---

## 💡 二、提議架構的適用性分析

### 2.1 提議的核心概念

```python
# 步驟 A: AI 產出標準化指令
CLICommand(
    flow_id="flow_8",  
    target="google.com",
    flags={"intensity": 0.8, "mode": "stealth"}
)

# 步驟 B: Planner 讀取 JSON 選擇 Flow
CLIToolSelector.select_flow(intent="scan", type="port") 
  → 返回 "flow_nmap_basic"

# 步驟 C: Executor 純 CLI 調用
subprocess.run(["python", "-m", "aiva_cli", "run", "flow_8", 
                "--target", "google.com", "--intensity", "0.8"])
```

### 2.2 適用性評估

| 層面 | 適用性 | 說明 |
|------|--------|------|
| **技術可行性** | ⭐⭐⭐⭐⭐ | 完全可行，且與現有 CLI 基礎相容 |
| **架構一致性** | ⭐⭐⭐⭐⭐ | 高度契合手冊第3冊 Step 7-9 |
| **解耦效果** | ⭐⭐⭐⭐⭐ | AI 完全不依賴具體實現，只依賴 CLI 接口 |
| **安全沙盒** | ⭐⭐⭐⭐⭐ | AI 無法執行任意代碼，天然安全 |
| **跨語言支援** | ⭐⭐⭐⭐⭐ | Go/Rust/TS 執行完全透明 |
| **可測試性** | ⭐⭐⭐⭐☆ | CLI 命令易於測試，但需模擬 subprocess |
| **性能開銷** | ⭐⭐⭐☆☆ | subprocess 有啟動開銷（可接受） |

### 2.3 與現有架構的契合度

✅ **高度契合點**：
1. CLI 工具已存在（`aiva_cli.py`）
2. Flow 數據已準備好（`latest_classification.json`）
3. 執行器架構支援外部執行（已有 subprocess 邏輯）

⚠️ **需要調整的地方**：
1. `UnifiedExecutor` 需重構為純 CLI 驅動器
2. 需新增 `CLICommand` Pydantic 模型
3. 需實現 `CLIToolSelector` 流程選擇器

---

## 🔧 三、具體實施方案

### 階段 1：定義 CLI 指令結構（1-2 天）

**新增文件**：`services/core/aiva_core/task_planning/commander/cli_command.py`

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class CLICommand(BaseModel):
    """AI 規劃的最終產出：標準化 CLI 執行請求"""
    
    flow_id: str = Field(..., description="對應 flow 的 id 或 name")
    target: str = Field(..., description="攻擊/掃描目標")
    flags: Dict[str, Any] = Field(default_factory=dict, description="CLI 額外參數")
    
    def to_cli_args(self) -> list[str]:
        """將物件轉換為實際命令行參數"""
        args = ["python", "-m", "services.core.aiva_core.core_capabilities.cli.aiva_cli"]
        
        # 提取 flow 編號（如果 flow_id = "flow_8" → "flow8"）
        flow_cmd = self.flow_id.replace("_", "").replace("-", "")
        args.append(flow_cmd)
        
        # 目標參數
        args.extend(["--target", self.target])
        
        # 展開其他標誌
        for key, value in self.flags.items():
            args.append(f"--{key}")
            if value is not True:
                args.append(str(value))
        
        return args
    
    def to_shell_command(self) -> str:
        """轉換為可讀的 shell 命令字串"""
        return " ".join(self.to_cli_args())
```

**整合到 types.py**：

```python
# 在 services/core/aiva_core/task_planning/commander/types.py 新增
from .cli_command import CLICommand

__all__ = ['AITaskType', 'AIComponent', 'CLICommand']
```

---

### 階段 2：實現 Flow 選擇器（2-3 天）

**新增文件**：`services/core/aiva_core/task_planning/planner/cli_tool_selector.py`

```python
import json
from pathlib import Path
from typing import Optional, Dict, List
from aiva_common.utils import get_logger

logger = get_logger(__name__)

class CLIToolSelector:
    """從 latest_classification.json 選擇合適的 Flow"""
    
    def __init__(self, classification_path: Optional[Path] = None):
        self.flows: Dict[str, dict] = {}
        self.categories: Dict[str, List[str]] = {}  # 類別 → flow_ids
        
        if classification_path is None:
            # 自動搜尋（與 aiva_cli.py 邏輯一致）
            classification_path = self._find_classification_file()
        
        self._load_flows(classification_path)
    
    def _find_classification_file(self) -> Path:
        """自動搜尋 classification 文件"""
        possible_paths = [
            Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json"),
            Path("C:/D/fold7/AIVA-git/data/internal_exploration/latest_classification.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ 找到 classification 文件: {path}")
                return path
        
        raise FileNotFoundError("❌ 未找到 latest_classification.json")
    
    def _load_flows(self, path: Path):
        """讀取並索引所有 flows"""
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
                
            flows_list = data.get("flows", [])
            
            for flow in flows_list:
                flow_id = flow.get("id")
                category = flow.get("category", "unknown")
                
                self.flows[flow_id] = flow
                
                if category not in self.categories:
                    self.categories[category] = []
                self.categories[category].append(flow_id)
            
            logger.info(f"✅ 載入 {len(self.flows)} 個 flows，{len(self.categories)} 個類別")
            
        except Exception as e:
            logger.error(f"❌ 載入 flows 失敗: {e}")
            raise
    
    def select_flow(
        self, 
        intent: str, 
        capability_type: Optional[str] = None,
        target_type: Optional[str] = None
    ) -> str:
        """根據意圖選擇最合適的 Flow ID
        
        Args:
            intent: AI 意圖（如 "scan", "exploit", "analyze"）
            capability_type: 能力類型（如 "nmap", "sqli", "xss"）
            target_type: 目標類型（如 "web", "network", "api"）
        
        Returns:
            str: 選中的 flow_id
        
        Examples:
            >>> selector.select_flow("scan", capability_type="port")
            'flow_8'
            
            >>> selector.select_flow("exploit", capability_type="sqli")
            'flow_42'
        """
        # TODO: 這裡可以集成 RAG 或向量搜索
        # 目前使用簡單的規則匹配
        
        # 策略 1: 精確匹配 capability_type
        if capability_type:
            for flow_id, flow_info in self.flows.items():
                if capability_type.lower() in flow_info.get("name", "").lower():
                    logger.info(f"🎯 選中 Flow: {flow_id} (類型匹配: {capability_type})")
                    return flow_id
        
        # 策略 2: 意圖匹配
        intent_map = {
            "scan": "scan",
            "port_scan": "scan",
            "exploit": "exploit",
            "attack": "attack",
            "detect": "detection"
        }
        
        category = intent_map.get(intent.lower())
        if category and category in self.categories:
            flow_id = self.categories[category][0]  # 選第一個
            logger.info(f"🎯 選中 Flow: {flow_id} (意圖匹配: {intent})")
            return flow_id
        
        # 策略 3: 默認回退
        default_flow = "flow_0"
        logger.warning(f"⚠️  無法匹配 intent='{intent}', 使用默認: {default_flow}")
        return default_flow
    
    def get_flow_info(self, flow_id: str) -> Dict:
        """獲取 Flow 的詳細信息"""
        return self.flows.get(flow_id, {})
    
    def list_flows_by_category(self, category: str) -> List[str]:
        """列出指定類別的所有 flows"""
        return self.categories.get(category, [])
```

---

### 階段 3：重構 UnifiedExecutor（3-5 天）

**修改文件**：`services/core/aiva_core/task_planning/unified_executor.py`

關鍵修改：

```python
class UnifiedAttackExecutor:
    """CLI 驅動的統一執行器
    
    新架構：
    - AI 只產出 CLICommand 物件
    - Executor 轉換為 subprocess 調用
    - 完全解耦，不 import 具體模組
    """
    
    def __init__(self, ...):
        # 現有初始化保持不變
        ...
        
        # 新增：CLI 工具選擇器
        self._cli_selector = None
    
    @property
    def cli_selector(self):
        """延遲加載 CLI Tool Selector"""
        if self._cli_selector is None:
            from ..task_planning.planner.cli_tool_selector import CLIToolSelector
            self._cli_selector = CLIToolSelector()
        return self._cli_selector
    
    async def execute(
        self,
        target: str,
        objective: str,
        scenario: Optional[dict] = None,
        constraints: Optional[dict] = None
    ) -> ExecutionResult:
        """執行攻擊（新架構：純 CLI 驅動）"""
        
        logger.info(f"🎯 開始執行攻擊: target={target}, objective={objective}")
        
        # 步驟 1: 選擇合適的 Flow
        flow_id = self.cli_selector.select_flow(
            intent=objective,  # 如 "scan", "exploit"
            target_type=self._infer_target_type(target)
        )
        
        # 步驟 2: 構建 CLI 命令
        from ..task_planning.commander.cli_command import CLICommand
        
        cli_command = CLICommand(
            flow_id=flow_id,
            target=target,
            flags={
                "intensity": constraints.get("intensity", 0.5) if constraints else 0.5,
                "mode": constraints.get("mode", "normal") if constraints else "normal"
            }
        )
        
        # 步驟 3: 執行 CLI 命令
        result = await self._execute_cli_command(cli_command)
        
        # 步驟 4: 解析結果並學習（如果啟用）
        if self.learning_enabled and result["status"] == "success":
            await self._collect_experience(cli_command, result)
        
        return ExecutionResult(
            success=result["status"] == "success",
            vulnerabilities=result.get("vulnerabilities", []),
            execution_details=result
        )
    
    async def _execute_cli_command(self, command: CLICommand) -> Dict[str, Any]:
        """執行 CLI 命令（核心方法）"""
        import asyncio
        
        cli_args = command.to_cli_args()
        cmd_str = command.to_shell_command()
        
        logger.info(f"🤖 AI 指揮官下令執行: {cmd_str}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cli_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent.parent  # 回到 AIVA 根目錄
            )
            
            stdout, stderr = await process.communicate()
            
            # 解析 CLI 輸出（假設 aiva_cli 返回 JSON）
            result = {
                "status": "success" if process.returncode == 0 else "error",
                "command": cmd_str,
                "exit_code": process.returncode,
                "raw_output": stdout.decode().strip(),
                "error": stderr.decode().strip()
            }
            
            # 嘗試解析 JSON 輸出
            try:
                import json
                parsed_output = json.loads(result["raw_output"])
                result.update(parsed_output)
            except json.JSONDecodeError:
                # 如果不是 JSON，保留原始輸出
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 執行失敗: {e}")
            return {
                "status": "crash",
                "error": str(e),
                "command": cmd_str
            }
    
    def _infer_target_type(self, target: str) -> str:
        """推斷目標類型"""
        if target.startswith("http"):
            return "web"
        elif ":" in target and target.split(":")[1].isdigit():
            return "network"
        else:
            return "generic"
```

---

### 階段 4：整合到決策層（2-3 天）

**修改文件**：`services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py`

關鍵整合點：

```python
class EnhancedDecisionAgent:
    async def make_decision(self, task_context: dict) -> dict:
        """AI 決策：產出 CLICommand 而非直接執行"""
        
        # 現有的決策邏輯...
        strategy = await self._select_strategy(task_context)
        
        # 新增：生成 CLI 命令（而非立即執行）
        from ...task_planning.commander.cli_command import CLICommand
        
        cli_command = CLICommand(
            flow_id=self._map_strategy_to_flow(strategy),
            target=task_context["target"],
            flags=self._build_flags_from_strategy(strategy)
        )
        
        return {
            "decision": "execute_attack",
            "cli_command": cli_command,  # 返回結構化命令
            "reasoning": strategy.reasoning
        }
```

---

## ✅ 四、實施後的收益

### 4.1 架構收益

1. **完全解耦**
   - AI 核心（aiva_core）不再依賴具體實現
   - 可以隨時替換底層工具（Go/Rust/TS）而不影響 AI

2. **天然沙盒**
   - AI 無法執行 `rm -rf /` 等危險命令
   - 只能執行 `aiva_cli` 允許的 Flow

3. **易於測試**
   - Mock CLI 命令比 Mock 內部函數簡單
   - 可端到端測試完整流程

4. **跨語言透明**
   - Go/Rust/TS 工具完全透明
   - CLI 接口統一，語言無關

### 4.2 符合手冊規範

✅ 完全符合《手冊第3冊》Step 7-9：命令組裝 → Flow 執行  
✅ 利用《手冊第5冊》的 Flow 分析數據（latest_classification.json）  
✅ 保留現有的依賴注入架構  

---

## 📋 五、風險與緩解

| 風險 | 等級 | 緩解措施 |
|------|------|---------|
| subprocess 性能開銷 | 🟡 中 | 對於簡單任務可保留內部執行器作為快速路徑 |
| CLI 輸出解析失敗 | 🟡 中 | 統一 CLI 返回 JSON 格式，並有 fallback 邏輯 |
| Flow 選擇不準確 | 🟡 中 | 初期使用規則匹配，後期集成 RAG/向量搜索 |
| 重構工作量大 | 🟢 低 | 分階段實施，逐步遷移，保留兼容層 |

---

## 🎯 六、實施建議

### 優先級建議

1. **先實施階段 1 + 階段 2**（CLICommand + CLIToolSelector）
   - 風險低，可立即驗證概念
   - 不影響現有功能

2. **試點重構 UnifiedExecutor**（階段 3）
   - 選擇 1-2 個簡單場景試點
   - 驗證 subprocess 性能

3. **全面推廣**（階段 4）
   - 決策層集成
   - 更新文檔

### 兼容性方案

**建議保留雙執行模式**：

```python
class UnifiedAttackExecutor:
    def __init__(self, execution_mode: str = "cli"):
        """
        execution_mode:
            - "cli": 純 CLI 驅動（新架構）
            - "hybrid": CLI + 內部執行器（過渡期）
            - "legacy": 舊的內部執行器（向後兼容）
        """
        self.execution_mode = execution_mode
    
    async def execute(self, ...):
        if self.execution_mode == "cli":
            return await self._execute_via_cli(...)
        elif self.execution_mode == "hybrid":
            return await self._execute_hybrid(...)
        else:
            return await self._execute_legacy(...)
```

---

## 📊 七、總結

### 提案評估結論

**✅ 強烈建議採用**

- ✅ 技術可行性：100% 可行
- ✅ 架構一致性：完美契合手冊第3冊
- ✅ 解耦效果：顯著提升
- ✅ 安全性：天然沙盒
- ✅ 可維護性：大幅提升

### 下一步行動

1. 審閱本提案
2. 選擇試點場景（建議：port_scan 場景）
3. 實施階段 1 + 2（約 3-5 天）
4. 驗證並迭代
5. 全面推廣

---

> **建立時間**: 2026-02-09  
> **分析者**: GitHub Copilot  
> **參考文檔**: 使用者手冊第3冊、第5冊  
