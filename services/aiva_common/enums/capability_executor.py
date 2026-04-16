"""
AIVA 能力選單執行器 - 選單式操作介面

設計原則：
1. 所有能力都是預先定義的選項 - 無需自然語言解析
2. 只有目標(target)和參數(parameters)需要輸入
3. 使用 5M Decision Engine 進行決策優化（非 LLM）
4. 自動驗證參數完整性
5. **整合現有 CapabilityRegistry 系統**

使用流程：
1. 選擇能力類別 (attack/scan/recon/analysis/forensic/exploit/report)
2. 選擇具體能力 (從枚舉列表選擇)
3. 輸入目標 (URL/IP/Domain/Path)
4. 輸入可選參數
5. 執行

範例：
>>> executor = CapabilityExecutor()
>>> # 列出所有攻擊能力
>>> executor.list_capabilities("attack")
>>> # 執行 SQL 注入測試
>>> result = await executor.execute(
...     capability="sql_injection",
...     target="https://example.com/api/users?id=1"
... )

整合說明：
- 此執行器與 services.integration.capability.CapabilityRegistry 整合
- 能力定義參考 aiva_common.enums.capabilities
- 參考分析資料：cli_commands_db.json, classification_data.json
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .capabilities import (
    AnalysisCapability,
    AttackCapability,
    CapabilityParameter,
    ExploitCapability,
    ForensicCapability,
    ReconCapability,
    ReportCapability,
    ScanCapability,
    get_all_capabilities,
    get_capability_config,
    validate_capability_params,
)

# 避免循環導入
if TYPE_CHECKING:
    from services.integration.capability.registry import CapabilityRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# 執行結果數據結構
# =============================================================================


@dataclass
class ExecutionResult:
    """能力執行結果"""

    success: bool
    capability: str
    target: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # 結果數據
    findings: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""

    # 錯誤信息
    error_message: str | None = None
    error_code: str | None = None

    # 決策引擎信息
    neural_confidence: float = 0.0
    recommended_next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "success": self.success,
            "capability": self.capability,
            "target": self.target,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "findings_count": len(self.findings),
            "findings": self.findings,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "neural_confidence": self.neural_confidence,
            "recommended_next_steps": self.recommended_next_steps,
        }


@dataclass
class CapabilityInfo:
    """能力信息（用於選單顯示）"""

    id: str
    name: str
    description: str
    category: str
    required_params: list[str]
    optional_params: list[str]
    risk_level: str
    default_timeout: int


# =============================================================================
# 能力選單執行器
# =============================================================================


class CapabilityExecutor:
    """能力選單執行器
    
    提供選單式操作介面，所有能力都是預定義選項。
    使用 5M Decision Engine 優化執行策略（非 LLM）。
    
    整合架構：
    - 與 services.integration.capability.CapabilityRegistry 整合
    - 支援現有工具能力和新選單式能力
    - 遵循 aiva_common 單一數據源原則
    """

    def __init__(
        self,
        decision_engine: Any | None = None,
        capability_registry: "CapabilityRegistry | None" = None,
    ):
        """初始化執行器
        
        Args:
            decision_engine: 5M Decision Engine 實例（可選）
            capability_registry: 現有能力註冊中心（可選，延遲載入）
        """
        self._decision_engine = decision_engine
        self._capability_registry = capability_registry
        self._execution_history: list[ExecutionResult] = []

        # 能力執行器映射
        self._capability_handlers: dict[str, Callable] = {}

        # 初始化能力映射
        self._init_capability_handlers()

        logger.info("CapabilityExecutor initialized (menu-based, no NLU/LLM)")

    def _get_registry(self) -> "CapabilityRegistry | None":
        """延遲載入 CapabilityRegistry（避免循環導入）"""
        if self._capability_registry is None:
            try:
                from services.integration.capability.registry import registry
                self._capability_registry = registry
                logger.info("CapabilityRegistry loaded successfully")
            except ImportError as e:
                logger.warning(f"Could not load CapabilityRegistry: {e}")
        return self._capability_registry

    def _init_capability_handlers(self) -> None:
        """初始化能力處理器映射"""
        # 這裡將在整合實際工具時填充
        # 每個能力對應一個具體的執行函數
        pass

    # =========================================================================
    # 選單查詢方法 - 列出可用能力
    # =========================================================================

    def list_categories(self) -> list[str]:
        """列出所有能力類別
        
        Returns:
            類別名稱列表
        """
        return ["attack", "scan", "recon", "analysis", "forensic", "exploit", "report"]

    def list_capabilities(self, category: str | None = None) -> dict[str, list[CapabilityInfo]]:
        """列出能力選單
        
        Args:
            category: 類別過濾（可選）
            
        Returns:
            按類別分組的能力信息
        """
        all_caps = get_all_capabilities()

        if category and category in all_caps:
            all_caps = {category: all_caps[category]}

        result = {}
        for cat, cap_list in all_caps.items():
            result[cat] = []
            for cap_id in cap_list:
                config = get_capability_config(cap_id)
                if config:
                    result[cat].append(CapabilityInfo(
                        id=cap_id,
                        name=config.get("name", cap_id),
                        description=config.get("description", ""),
                        category=cat,
                        required_params=[p.value if isinstance(p, CapabilityParameter) else p
                                        for p in config.get("required_params", [])],
                        optional_params=[p.value if isinstance(p, CapabilityParameter) else p
                                        for p in config.get("optional_params", [])],
                        risk_level=config.get("risk_level", "unknown"),
                        default_timeout=config.get("default_timeout", 60),
                    ))
                else:
                    # 未配置的能力使用默認信息
                    result[cat].append(CapabilityInfo(
                        id=cap_id,
                        name=cap_id.replace("_", " ").title(),
                        description=f"執行 {cap_id}",
                        category=cat,
                        required_params=["target"],
                        optional_params=[],
                        risk_level="unknown",
                        default_timeout=60,
                    ))

        return result

    def get_capability_info(self, capability: str) -> CapabilityInfo | None:
        """獲取單個能力的詳細信息
        
        Args:
            capability: 能力 ID
            
        Returns:
            能力信息，若不存在則返回 None
        """
        config = get_capability_config(capability)
        if not config:
            return None

        # 確定類別
        category = "unknown"
        for cap_enum in [AttackCapability, ScanCapability, ReconCapability,
                        AnalysisCapability, ForensicCapability, ExploitCapability,
                        ReportCapability]:
            try:
                cap_enum(capability)
                category = cap_enum.__name__.replace("Capability", "").lower()
                break
            except ValueError:
                continue

        return CapabilityInfo(
            id=capability,
            name=config.get("name", capability),
            description=config.get("description", ""),
            category=category,
            required_params=[p.value if isinstance(p, CapabilityParameter) else p
                            for p in config.get("required_params", [])],
            optional_params=[p.value if isinstance(p, CapabilityParameter) else p
                            for p in config.get("optional_params", [])],
            risk_level=config.get("risk_level", "unknown"),
            default_timeout=config.get("default_timeout", 60),
        )

    def print_menu(self, category: str | None = None) -> str:
        """生成可讀的能力選單字串
        
        Args:
            category: 類別過濾
            
        Returns:
            格式化的選單字串
        """
        capabilities = self.list_capabilities(category)

        lines = ["=" * 60, "AIVA 能力選單 - 選擇要執行的操作", "=" * 60, ""]

        for cat, cap_list in capabilities.items():
            lines.append(f"【{cat.upper()}】")
            lines.append("-" * 40)

            for i, cap in enumerate(cap_list, 1):
                risk_indicator = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢",
                    "info": "🔵",
                }.get(cap.risk_level, "⚪")

                lines.append(f"  {i:2d}. {risk_indicator} {cap.name}")
                lines.append(f"      ID: {cap.id}")
                if cap.required_params:
                    lines.append(f"      必需參數: {', '.join(cap.required_params)}")

            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # 執行方法
    # =========================================================================

    async def execute(
        self,
        capability: str,
        target: str,
        parameters: dict[str, Any] | None = None,
        use_neural_optimization: bool = True,
    ) -> ExecutionResult:
        """執行選定的能力
        
        Args:
            capability: 能力 ID（從枚舉選擇）
            target: 目標（URL/IP/Domain/Path）
            parameters: 額外參數
            use_neural_optimization: 是否使用 5M 引擎優化策略
            
        Returns:
            執行結果
        """
        start_time = datetime.now(UTC)
        params = parameters or {}

        # 1. 驗證能力是否存在
        cap_info = self.get_capability_info(capability)
        if not cap_info:
            # 嘗試直接使用 - 可能是未配置但有效的能力
            cap_info = CapabilityInfo(
                id=capability,
                name=capability,
                description="",
                category="unknown",
                required_params=["target"],
                optional_params=[],
                risk_level="unknown",
                default_timeout=60,
            )

        # 2. 設置目標參數
        # 自動判斷目標類型並設置對應參數
        params = self._auto_set_target_param(target, params)

        # 3. 驗證參數完整性
        is_valid, missing_params = validate_capability_params(capability, params)
        if not is_valid and missing_params:
            # 檢查是否只是缺少目標參數（已通過 target 提供）
            if not all(p in ["target_url", "target_ip", "target_domain", "target_host", "output_path"]
                      for p in missing_params):
                return ExecutionResult(
                    success=False,
                    capability=capability,
                    target=target,
                    start_time=start_time,
                    end_time=datetime.now(UTC),
                    error_message=f"缺少必需參數: {', '.join(missing_params)}",
                    error_code="MISSING_PARAMS",
                )

        # 4. 使用 5M Decision Engine 優化執行策略（可選）
        neural_confidence = 0.5
        recommended_next = []

        if use_neural_optimization and self._decision_engine:
            try:
                decision = self._decision_engine.generate_decision(
                    target_info={"capability": capability, "target": target},
                    context=params,
                )
                neural_confidence = decision.get("confidence", 0.5)
                recommended_next = decision.get("recommended_tools", [])[:3]
            except Exception as e:
                logger.warning(f"Neural optimization failed: {e}")

        # 5. 執行能力
        try:
            logger.info(f"執行能力: {capability} -> {target}")

            # 檢查是否有註冊的處理器
            if capability in self._capability_handlers:
                handler = self._capability_handlers[capability]
                result_data = await handler(target, params)
            else:
                # 使用通用執行器（待整合具體工具）
                result_data = await self._generic_execute(capability, target, params)

            end_time = datetime.now(UTC)

            result = ExecutionResult(
                success=True,
                capability=capability,
                target=target,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                findings=result_data.get("findings", []),
                raw_output=result_data.get("raw_output", ""),
                neural_confidence=neural_confidence,
                recommended_next_steps=recommended_next,
            )

        except Exception as e:
            end_time = datetime.now(UTC)
            result = ExecutionResult(
                success=False,
                capability=capability,
                target=target,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                neural_confidence=neural_confidence,
            )

        # 6. 記錄執行歷史
        self._execution_history.append(result)

        return result

    def _auto_set_target_param(self, target: str, params: dict[str, Any]) -> dict[str, Any]:
        """自動判斷目標類型並設置對應參數
        
        Args:
            target: 用戶輸入的目標
            params: 現有參數
            
        Returns:
            更新後的參數字典
        """
        params = params.copy()

        # 判斷目標類型
        if target.startswith(("http://", "https://")):
            params["target_url"] = target
        elif self._is_ip(target):
            params["target_ip"] = target
        elif "/" in target or "\\" in target:
            params["output_path"] = target
        elif "." in target:
            params["target_domain"] = target
            params["target_host"] = target
        else:
            params["target_host"] = target

        return params

    @staticmethod
    def _is_ip(value: str) -> bool:
        """檢查是否為 IP 地址"""
        parts = value.split(".")
        if len(parts) == 4:
            try:
                return all(0 <= int(p) <= 255 for p in parts)
            except ValueError:
                pass
        return False

    async def _generic_execute(
        self,
        capability: str,
        target: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """通用執行器 - 整合 CapabilityRegistry
        
        優先順序：
        1. 檢查 CapabilityRegistry 是否有對應的能力
        2. 如果有，使用 registry 的執行機制
        3. 如果沒有，返回佔位符結果
        
        Args:
            capability: 能力 ID
            target: 目標
            params: 參數
            
        Returns:
            執行結果數據
        """
        logger.info(f"[Generic Execute] {capability} on {target}")

        # 嘗試從 CapabilityRegistry 獲取能力定義
        registry = self._get_registry()
        if registry:
            try:
                # 查詢對應的能力記錄
                # 嘗試多種 ID 格式
                cap_ids_to_try = [
                    capability,
                    f"menu.attack.{capability}",
                    f"menu.scan.{capability}",
                    f"menu.recon.{capability}",
                    f"menu.analysis.{capability}",
                    f"menu.forensic.{capability}",
                    f"menu.exploit.{capability}",
                    f"menu.report.{capability}",
                ]

                cap_record = None
                for cap_id in cap_ids_to_try:
                    cap_record = await registry.get_capability(cap_id)
                    if cap_record:
                        break

                if cap_record:
                    logger.info(f"Found capability in registry: {cap_record.id}")

                    # 使用 registry 的執行請求
                    from services.integration.capability.models import (
                        ExecutionRequest as RegistryExecutionRequest,
                    )

                    exec_request = RegistryExecutionRequest(
                        capability_id=cap_record.id,
                        parameters={
                            "target": target,
                            **params
                        },
                        priority=cap_record.priority,
                        timeout_seconds=cap_record.timeout_seconds,
                    )

                    # 執行並返回結果
                    exec_result = await registry.execute_capability(exec_request)

                    return {
                        "findings": exec_result.findings if hasattr(exec_result, 'findings') else [],
                        "raw_output": exec_result.output if hasattr(exec_result, 'output') else str(exec_result),
                        "status": "completed" if exec_result.success else "failed",
                    }

            except Exception as e:
                logger.warning(f"Registry execution failed, using fallback: {e}")

        # 回退：佔位符實現
        await asyncio.sleep(0.1)  # 模擬執行

        return {
            "findings": [],
            "raw_output": f"Capability '{capability}' executed on '{target}' (placeholder - register tool for full functionality)",
            "status": "completed",
        }

    # =========================================================================
    # 歷史記錄方法
    # =========================================================================

    def get_execution_history(
        self,
        limit: int = 50,
        capability_filter: str | None = None,
    ) -> list[ExecutionResult]:
        """獲取執行歷史
        
        Args:
            limit: 返回數量限制
            capability_filter: 能力過濾
            
        Returns:
            執行結果列表
        """
        history = self._execution_history

        if capability_filter:
            history = [r for r in history if r.capability == capability_filter]

        return history[-limit:]

    def clear_history(self) -> None:
        """清除執行歷史"""
        self._execution_history.clear()


# =============================================================================
# 互動式選單介面
# =============================================================================


class InteractiveCapabilityMenu:
    """互動式能力選單 - 命令行介面"""

    def __init__(self, executor: CapabilityExecutor | None = None):
        self.executor = executor or CapabilityExecutor()

    def show_main_menu(self) -> None:
        """顯示主選單"""
        print("\n" + "=" * 60)
        print("AIVA 能力選單系統")
        print("=" * 60)
        print("\n選擇類別:")
        print("  1. attack   - 攻擊能力")
        print("  2. scan     - 掃描能力")
        print("  3. recon    - 偵察能力")
        print("  4. analysis - 分析能力")
        print("  5. forensic - 鑑識能力")
        print("  6. exploit  - 漏洞利用")
        print("  7. report   - 報告生成")
        print("  8. all      - 顯示全部")
        print("  0. exit     - 退出")
        print()

    def show_category_menu(self, category: str) -> list[CapabilityInfo]:
        """顯示類別下的能力選單"""
        capabilities = self.executor.list_capabilities(category)
        cap_list = capabilities.get(category, [])

        print(f"\n【{category.upper()} 能力列表】")
        print("-" * 50)

        for i, cap in enumerate(cap_list, 1):
            risk_badge = f"[{cap.risk_level.upper()}]"
            print(f"  {i:3d}. {cap.name} {risk_badge}")
            print(f"       {cap.description}")
            if cap.required_params:
                print(f"       必需: {', '.join(cap.required_params)}")

        print()
        return cap_list

    async def interactive_execute(self) -> None:
        """互動式執行流程"""
        print("\n" + self.executor.print_menu())

        # 選擇類別
        category = input("輸入類別 (attack/scan/recon/analysis/forensic/exploit/report): ").strip().lower()
        if category not in self.executor.list_categories():
            print(f"❌ 無效類別: {category}")
            return

        # 顯示能力列表
        cap_list = self.show_category_menu(category)
        if not cap_list:
            print("❌ 此類別暫無可用能力")
            return

        # 選擇能力
        try:
            choice = int(input("選擇能力編號: "))
            if 1 <= choice <= len(cap_list):
                selected_cap = cap_list[choice - 1]
            else:
                print("❌ 無效選擇")
                return
        except ValueError:
            print("❌ 請輸入數字")
            return

        # 輸入目標
        print(f"\n選擇的能力: {selected_cap.name}")
        print(f"必需參數: {', '.join(selected_cap.required_params)}")
        target = input("輸入目標 (URL/IP/Domain/Path): ").strip()
        if not target:
            print("❌ 目標不能為空")
            return

        # 可選參數
        params = {}
        if selected_cap.optional_params:
            print(f"\n可選參數: {', '.join(selected_cap.optional_params)}")
            for param in selected_cap.optional_params:
                value = input(f"  {param} (按 Enter 跳過): ").strip()
                if value:
                    params[param] = value

        # 執行
        print(f"\n⏳ 執行中: {selected_cap.name} -> {target}")
        result = await self.executor.execute(
            capability=selected_cap.id,
            target=target,
            parameters=params,
        )

        # 顯示結果
        self._print_result(result)

    def _print_result(self, result: ExecutionResult) -> None:
        """格式化顯示結果"""
        print("\n" + "=" * 60)
        if result.success:
            print("✅ 執行成功")
        else:
            print("❌ 執行失敗")
        print("=" * 60)

        print(f"能力: {result.capability}")
        print(f"目標: {result.target}")
        print(f"耗時: {result.duration_seconds:.2f} 秒")

        if result.error_message:
            print(f"錯誤: {result.error_message}")

        if result.findings:
            print(f"\n發現 {len(result.findings)} 個問題:")
            for i, finding in enumerate(result.findings[:5], 1):
                print(f"  {i}. {finding.get('title', 'Unknown')}")

        if result.recommended_next_steps:
            print(f"\n建議後續步驟: {', '.join(result.recommended_next_steps)}")

        print()
