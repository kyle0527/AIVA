# services/core/aiva_core/execution/attack_plan_mapper.py
"""
AI 攻擊計畫映射器

負責將 AI 生成的抽象攻擊計畫 (來自 BioNeuron) 轉換為
一系列具體的可執行的 FunctionTaskPayload。
"""

import logging
from typing import Dict, Any, List, Optional

# 遵循 aiva_common 單一事實來源原則 - 統一使用標準模組
from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget, FunctionTaskContext
from services.aiva_common.schemas import AivaMessage

import uuid
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

class AttackPlanMapper:
    """
    將 AI 決策映射到具體執行任務。
    """

    def __init__(self):
        logger.info("AttackPlanMapper initialized.")
        # 可能需要初始化資源，例如工具註冊表、目標信息等

    async def map_decision_to_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        將單個 AI 決策步驟轉換為一個或多個 FunctionTaskPayload。

        Args:
            ai_decision: AI 生成的決策對象。
            scan_context: 當前的掃描上下文信息 (例如目標 URL、先前結果等)。

        Returns:
            一個包含具體任務 Payload 的列表。
        """
        tasks: List[FunctionTaskPayload] = []
        logger.info(f"Mapping AI decision: {ai_decision}")

        # --- 核心映射邏輯 ---
        # 1. 解析 ai_decision 中的意圖、目標、使用的工具/技術。
        # 2. 根據意圖選擇合適的 Aiva 功能模組 (Function Module)。
        # 3. 從 scan_context 和 ai_decision 中提取目標信息 (URL, params, headers...)。
        # 4. 構建 FunctionTaskTarget。
        # 5. 構建 FunctionTaskContext (可能包含先前步驟的結果)。
        # 6. 構建 FunctionTaskPayload，指定 module_name、target、context、策略等。
        # 7. 如果一個決策需要多個步驟，則生成多個 Task。

        # 示例：基本映射邏輯
        try:
            # 假設 ai_decision 包含工具選擇和參數
            if hasattr(ai_decision, 'payload') and ai_decision.payload:
                decision_data = ai_decision.payload
                
                # 根據決策類型進行映射
                if decision_data.get('action_type') == 'vulnerability_scan':
                    target_info = decision_data.get('target', {})
                    if target_info.get('url'):
                        # 創建掃描任務
                        target = FunctionTaskTarget(
                            url=target_info.get('url'),
                            method=target_info.get('method', 'GET'),
                            params=target_info.get('params'),
                            data=target_info.get('data'),
                            headers=scan_context.get('base_headers', {})
                        )
                        
                        context = FunctionTaskContext(
                            session_id=scan_context.get('session_id'),
                            parent_task_id=scan_context.get('parent_task_id'),
                            scan_config=scan_context.get('scan_config', {})
                        )
                        
                        # 根據漏洞類型選擇模組
                        vuln_type = decision_data.get('vulnerability_type', 'general')
                        module_name = self._map_vulnerability_to_module(vuln_type)
                        
                        payload = FunctionTaskPayload(
                            task_id=f"task_{ai_decision.header.message_id}_{vuln_type}",
                            module_name=module_name,
                            target=target,
                            context=context,
                            strategy=decision_data.get('strategy', 'NORMAL')
                        )
                        tasks.append(payload)
                        logger.info(f"Generated {vuln_type} task for decision {ai_decision.header.message_id}")
                        
                elif decision_data.get('action_type') == 'information_gathering':
                    # 信息收集任務映射
                    tasks.extend(await self._create_info_gathering_tasks(ai_decision, scan_context))
                    
                elif decision_data.get('action_type') == 'exploitation':
                    # 漏洞利用任務映射
                    tasks.extend(await self._create_exploitation_tasks(ai_decision, scan_context))

        except Exception as e:
            logger.error(f"Error mapping decision {getattr(ai_decision, 'header', {}).get('message_id', 'unknown')}: {e}", exc_info=True)

        if not tasks:
            logger.warning(f"No tasks generated for AI decision: {ai_decision}. Decision might be informational or mapping logic is missing.")

        return tasks

    def _map_vulnerability_to_module(self, vuln_type: str) -> str:
        """將漏洞類型映射到對應的模組"""
        mapping = {
            'sqli': 'FUNC_SQLI',
            'xss': 'FUNC_XSS', 
            'ssrf': 'FUNC_SSRF',
            'client_auth_bypass': 'FUNC_CLIENT_AUTH_BYPASS',
            'rce': 'FUNC_RCE',
            'lfi': 'FUNC_LFI',
            'general': 'FUNC_GENERAL_SCAN'
        }
        return mapping.get(vuln_type, 'FUNC_GENERAL_SCAN')

    async def _create_info_gathering_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        創建信息收集任務
        
        基於 OWASP WSTG 4.1 Information Gathering 和 MITRE ATT&CK TA0007 Discovery 實現
        包含：Web指紋識別、目錄枚舉、技術堆疊探測、端點發現等
        """
        tasks = []
        target_url = scan_context.get("target_url", "")
        
        logger.info(f"Creating information gathering tasks for {target_url}")
        
        # 1. Web Server Fingerprinting (WSTG-INFO-02)
        if target_url:
            fingerprint_task = FunctionTaskPayload(
                task_id=self._generate_task_id("info_fingerprint"),
                function_name="web_fingerprinting",
                target=FunctionTaskTarget(
                    url=target_url,
                    method="GET"
                ),
                strategy="comprehensive",
                config={
                    "techniques": ["headers", "error_pages", "default_files"],
                    "timeout": 30,
                    "follow_redirects": True
                }
            )
            tasks.append(fingerprint_task)
            
        # 2. Directory and File Enumeration (WSTG-INFO-04)
        if target_url:
            directory_task = FunctionTaskPayload(
                task_id=self._generate_task_id("info_directory"),
                function_name="directory_enumeration", 
                target=FunctionTaskTarget(
                    url=target_url,
                    method="GET"
                ),
                strategy="discovery",
                config={
                    "wordlists": ["common", "admin", "backup"],
                    "extensions": [".php", ".asp", ".aspx", ".jsp", ".html"],
                    "max_depth": 3,
                    "threads": 10
                }
            )
            tasks.append(directory_task)
            
        # 3. Technology Stack Detection (WSTG-INFO-08/09)
        if target_url:
            tech_stack_task = FunctionTaskPayload(
                task_id=self._generate_task_id("info_techstack"),
                function_name="technology_detection",
                target=FunctionTaskTarget(
                    url=target_url,
                    method="GET"
                ),
                strategy="passive",
                config={
                    "detect": ["framework", "cms", "javascript", "server", "database"],
                    "passive_only": True,
                    "include_versions": True
                }
            )
            tasks.append(tech_stack_task)
            
        # 4. Entry Points Identification (WSTG-INFO-06)
        if target_url:
            entry_points_task = FunctionTaskPayload(
                task_id=self._generate_task_id("info_entrypoints"),
                function_name="entry_point_discovery",
                target=FunctionTaskTarget(
                    url=target_url,
                    method="GET"
                ),
                strategy="comprehensive",
                config={
                    "scan_forms": True,
                    "scan_parameters": True,
                    "scan_cookies": True,
                    "scan_headers": True,
                    "spider_depth": 2
                }
            )
            tasks.append(entry_points_task)
            
        # 5. Subdomain Enumeration (MITRE ATT&CK T1583.001)
        domain = self._extract_domain(target_url)
        if domain:
            subdomain_task = FunctionTaskPayload(
                task_id=self._generate_task_id("info_subdomains"),
                function_name="subdomain_enumeration",
                target=FunctionTaskTarget(
                    url=domain,
                    method="DNS"
                ),
                strategy="passive",
                config={
                    "techniques": ["certificate_transparency", "dns_brute", "search_engines"],
                    "wordlist_size": "medium",
                    "include_wildcards": True
                }
            )
            tasks.append(subdomain_task)
            
        logger.info(f"Created {len(tasks)} information gathering tasks")
        return tasks

    async def _create_exploitation_tasks(self, ai_decision: AivaMessage, scan_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        創建漏洞利用任務
        
        基於已發現的漏洞信息，創建對應的利用任務
        遵循 OWASP 測試指南和負責任披露原則
        """
        tasks = []
        target_url = scan_context.get("target_url", "")
        findings = scan_context.get("findings", [])
        
        logger.info(f"Creating exploitation tasks for {len(findings)} findings")
        
        for finding in findings:
            vuln_type = finding.get("vulnerability_type", "").lower()
            severity = finding.get("severity", "medium").lower()
            
            # 僅對高危和中危漏洞創建利用任務
            if severity not in ["high", "critical", "medium"]:
                continue
                
            # 1. IDOR 漏洞利用
            if "idor" in vuln_type or "direct_object" in vuln_type:
                idor_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_idor"),
                    function_name="function_idor",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method=finding.get("method", "GET")
                    ),
                    strategy="exploit",
                    config={
                        "resource_id": finding.get("resource_id"),
                        "test_variations": 5,
                        "multi_user_test": True,
                        "privilege_escalation": True
                    }
                )
                tasks.append(idor_task)
                
            # 2. SQL Injection 利用
            elif "sql" in vuln_type or "injection" in vuln_type:
                sqli_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_sqli"),
                    function_name="sql_injection",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method=finding.get("method", "POST")
                    ),
                    strategy="exploit",
                    config={
                        "parameter": finding.get("parameter"),
                        "injection_type": finding.get("injection_type", "boolean"),
                        "database_type": finding.get("database", "mysql"),
                        "payload_level": 2  # 中等侵入性
                    }
                )
                tasks.append(sqli_task)
                
            # 3. XSS 利用
            elif "xss" in vuln_type or "script" in vuln_type:
                xss_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_xss"),
                    function_name="xss_exploitation",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method=finding.get("method", "GET")
                    ),
                    strategy="proof_of_concept",
                    config={
                        "xss_type": finding.get("xss_type", "reflected"),
                        "parameter": finding.get("parameter"),
                        "context": finding.get("context", "html"),
                        "payload_complexity": "medium"
                    }
                )
                tasks.append(xss_task)
                
            # 4. 認證繞過利用
            elif "auth" in vuln_type or "bypass" in vuln_type:
                auth_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_auth"),
                    function_name="function_authn_go",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method="POST"
                    ),
                    strategy="bypass",
                    config={
                        "techniques": ["brute_force", "weak_credentials", "token_manipulation"],
                        "credential_lists": ["common", "default"],
                        "rate_limit_bypass": True
                    }
                )
                tasks.append(auth_task)
                
            # 5. JWT 相關漏洞利用
            elif "jwt" in vuln_type or "token" in vuln_type:
                jwt_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_jwt"),
                    function_name="jwt_confusion",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method="GET"
                    ),
                    strategy="token_manipulation",
                    config={
                        "attacks": ["none_algorithm", "weak_secret", "key_confusion"],
                        "token": finding.get("token"),
                        "target_claims": ["sub", "role", "admin"]
                    }
                )
                tasks.append(jwt_task)
                
            # 6. GraphQL 授權繞過
            elif "graphql" in vuln_type:
                graphql_task = FunctionTaskPayload(
                    task_id=self._generate_task_id("exploit_graphql"),
                    function_name="graphql_authz",
                    target=FunctionTaskTarget(
                        url=finding.get("url", target_url),
                        method="POST"
                    ),
                    strategy="authorization_bypass",
                    config={
                        "query_depth": 5,
                        "introspection": True,
                        "field_suggestions": True,
                        "batch_queries": True
                    }
                )
                tasks.append(graphql_task)
                
        # 7. 通用漏洞掃描 (如果沒有具體發現)
        if not findings and target_url:
            comprehensive_task = FunctionTaskPayload(
                task_id=self._generate_task_id("exploit_comprehensive"),
                function_name="comprehensive_scan",
                target=FunctionTaskTarget(
                    url=target_url,
                    method="GET"
                ),
                strategy="safe_exploitation",
                config={
                    "modules": ["idor", "xss", "sqli", "auth", "jwt"],
                    "intensity": "medium",
                    "safe_mode": True,
                    "proof_of_concept_only": True
                }
            )
            tasks.append(comprehensive_task)
            
        logger.info(f"Created {len(tasks)} exploitation tasks")
        return tasks

    async def map_entire_plan(self, attack_plan: List[AivaMessage], initial_context: Dict[str, Any]) -> List[FunctionTaskPayload]:
        """
        將整個 AI 攻擊計畫 (一系列決策) 映射為任務列表。

        Args:
            attack_plan: AI 生成的包含多個決策步驟的計畫。
            initial_context: 初始掃描上下文。

        Returns:
            一個包含所有計畫步驟對應任務的列表。
        """
        all_tasks: List[FunctionTaskPayload] = []
        current_context = initial_context.copy()

        logger.info(f"Mapping entire attack plan with {len(attack_plan)} steps.")

        for i, decision in enumerate(attack_plan):
            logger.debug(f"Mapping step {i+1}/{len(attack_plan)}")
            step_tasks = await self.map_decision_to_tasks(decision, current_context)
            all_tasks.extend(step_tasks)
            
            # 更新上下文以便後續步驟使用
            if step_tasks:
                current_context['previous_task_ids'] = [task.task_id for task in step_tasks]
                current_context['step_number'] = i + 1

        logger.info(f"Generated {len(all_tasks)} tasks from the attack plan.")
        return all_tasks
    
    def _generate_task_id(self, prefix: str) -> str:
        """生成唯一的任務 ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """從 URL 中提取域名"""
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {e}")
            return None