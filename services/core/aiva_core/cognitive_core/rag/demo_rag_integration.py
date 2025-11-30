"""RAG 與 AI Agent 集成示例

展示如何將 RAG 引擎與 BioNeuronRAGAgent 結合使用
"""

import asyncio
import logging

from services.aiva_common.schemas import AttackPlan, AttackTarget
from services.core.aiva_core.cognitive_core.rag import KnowledgeBase, RAGEngine, VectorStore

logger = logging.getLogger(__name__)


class AIAgentWithRAG:
    """帶 RAG 增強的 AI Agent

    結合向量檢索和 AI 決策
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        ai_model_name: str = "gpt-4",
    ) -> None:
        """初始化

        Args:
            rag_engine: RAG 引擎
            ai_model_name: AI 模型名稱
        """
        self.rag_engine = rag_engine
        self.ai_model_name = ai_model_name

        logger.info(f"AIAgentWithRAG initialized with model {ai_model_name}")

    async def generate_attack_plan(
        self,
        target: AttackTarget,
        objective: str,
    ) -> AttackPlan:
        """生成攻擊計畫（RAG 增強）

        Args:
            target: 攻擊目標
            objective: 攻擊目標

        Returns:
            增強的攻擊計畫
        """
        # 1. 使用 RAG 檢索相關上下文
        rag_context = self.rag_engine.enhance_attack_plan(
            target=target,
            objective=objective,
        )

        # 2. 構建增強的提示詞
        prompt = self._build_prompt_with_context(
            target=target,
            objective=objective,
            context=rag_context,
        )

        # 3. 調用 AI 模型或使用基於規則的生成
        try:
            # 嘗試調用外部LLM (如果配置了)
            plan = await self._call_ai_model(prompt)
        except Exception as e:
            logger.warning(f"AI model unavailable, using rule-based generation: {e}")
            # Fallback: 使用基於RAG上下文的規則生成
            plan = self._generate_plan_from_context(target, objective, rag_context)

        logger.info(
            f"Generated attack plan with RAG context: "
            f"{len(rag_context['similar_techniques'])} techniques, "
            f"{len(rag_context['successful_experiences'])} experiences, "
            f"{len(plan.steps)} steps"
        )

        return plan
    
    async def _call_ai_model(self, prompt: str) -> AttackPlan:
        """調用AI模型生成計劃
        
        這需要外部LLM API配置，如果未配置會拋出異常
        """
        # 檢查是否有可用的LLM配置
        import os
        if not os.getenv('OPENAI_API_KEY') and not os.getenv('AZURE_OPENAI_KEY'):
            raise ValueError("No LLM API key configured")
        
        # TODO: 實際的LLM調用邏輯
        # 這裡需要整合 OpenAI/Azure OpenAI/其他LLM提供商
        raise NotImplementedError("LLM integration requires API key configuration")
    
    def _generate_plan_from_context(
        self,
        target: AttackTarget,
        objective: str,
        context: dict
    ) -> AttackPlan:
        """基於RAG上下文生成攻擊計劃 (Fallback方案)
        
        當外部LLM不可用時，使用基於規則的方法生成計劃
        """
        from services.aiva_common.schemas import AttackStep
        
        steps = []
        expected_results = []
        
        # 1. 偵察階段 (基於best practices)
        steps.append(AttackStep(
            step_id="recon_1",
            name="Target Reconnaissance",
            description="Gather information about the target application",
            tool="nmap,nikto,whatweb",
            parameters={
                "target": target.url,
                "scan_type": "comprehensive"
            },
            expected_time=300,
            priority=1
        ))
        
        # 2. 根據相似技術添加測試步驟
        for i, tech in enumerate(context.get('similar_techniques', [])[:3], start=2):
            technique_name = tech.get('title', 'Unknown Technique')
            steps.append(AttackStep(
                step_id=f"test_{i}",
                name=f"Test: {technique_name}",
                description=tech.get('content', '')[:200],
                tool=self._extract_tool_from_technique(tech),
                parameters={
                    "target": target.url,
                    "technique": technique_name
                },
                expected_time=180,
                priority=i
            ))
        
        # 3. 根據成功經驗添加步驟
        for i, exp in enumerate(context.get('successful_experiences', [])[:2], start=5):
            steps.append(AttackStep(
                step_id=f"exploit_{i}",
                name=f"Apply: {exp.get('title', 'Known Exploit')}",
                description=exp.get('content', '')[:200],
                tool="custom",
                parameters={
                    "target": target.url,
                    "experience_id": exp.get('id', 'unknown')
                },
                expected_time=240,
                priority=i
            ))
        
        # 4. 生成預期結果
        expected_results = [
            f"Comprehensive understanding of {target.url} infrastructure",
            f"Identified vulnerabilities based on {len(context.get('similar_techniques', []))} similar techniques",
            f"Validated attack vectors with {len(context.get('successful_experiences', []))} known successful patterns"
        ]
        
        return AttackPlan(
            target=target,
            objective=objective,
            steps=steps,
            priority=1,
            expected_results=expected_results,
        )
    
    def _extract_tool_from_technique(self, technique: dict) -> str:
        """從技術描述中提取工具名稱"""
        content = technique.get('content', '').lower()
        tags = technique.get('tags', [])
        
        # 根據標籤推斷工具
        tool_mapping = {
            'sqli': 'sqlmap',
            'xss': 'xsstrike',
            'directory_traversal': 'dotdotpwn',
            'file_inclusion': 'fimap',
            'ssrf': 'ssrfmap',
            'xxe': 'xxeinjector',
        }
        
        for tag in tags:
            if tag in tool_mapping:
                return tool_mapping[tag]
        
        # 默認工具
        return 'burpsuite'

    def _build_prompt_with_context(
        self,
        target: AttackTarget,
        objective: str,
        context: dict,
    ) -> str:
        """構建帶 RAG 上下文的提示詞

        Args:
            target: 攻擊目標
            objective: 攻擊目標
            context: RAG 檢索的上下文

        Returns:
            完整提示詞
        """
        prompt_parts = [
            "# Task: Generate an attack plan\n",
            f"Target: {target.url}",
            f"Type: {target.type}",
            f"Objective: {objective}\n",
        ]

        # 添加相似技術
        if context["similar_techniques"]:
            prompt_parts.append("\n## Similar Attack Techniques:")
            for tech in context["similar_techniques"][:3]:
                prompt_parts.append(
                    f"- {tech['title']} " f"(success rate: {tech['success_rate']:.2%})"
                )
                prompt_parts.append(f"  {tech['content'][:200]}...")

        # 添加成功經驗
        if context["successful_experiences"]:
            prompt_parts.append("\n## Successful Experiences:")
            for exp in context["successful_experiences"][:3]:
                prompt_parts.append(f"- {exp['title']}")
                prompt_parts.append(f"  {exp['content'][:200]}...")

        # 添加最佳實踐
        if context["best_practices"]:
            prompt_parts.append("\n## Best Practices:")
            for bp in context["best_practices"]:
                prompt_parts.append(f"- {bp['title']}")

        prompt_parts.append(
            "\n## Instructions:\n"
            "Based on the above context, generate a detailed attack plan "
            "that includes specific steps, tools, and parameters. "
            "Prioritize approaches with high success rates."
        )

        return "\n".join(prompt_parts)


async def demo_rag_integration():
    """RAG 集成演示"""
    print("=" * 60)
    print("RAG Integration Demo")
    print("=" * 60)

    # 1. 初始化 RAG 系統
    print("\n1. Initializing RAG System...")
    vector_store = VectorStore(backend="memory")
    knowledge_base = KnowledgeBase(vector_store=vector_store)
    rag_engine = RAGEngine(knowledge_base=knowledge_base)

    # 2. 添加一些示例知識
    print("\n2. Adding sample knowledge...")

    # 添加攻擊技術
    knowledge_base.add_entry(
        entry_id="tech_sqli_union",
        entry_type="attack_technique",
        title="SQL Injection - UNION Based",
        content="""
UNION based SQL injection technique:
1. Find injection point with ' or "
2. Determine number of columns: ORDER BY
3. Find injectable columns: UNION SELECT
4. Extract data: UNION SELECT column_name FROM table_name
        """,
        tags=["sqli", "union", "database"],
        metadata={"difficulty": "medium", "success_rate": 0.75},
    )

    knowledge_base.add_entry(
        entry_id="tech_xss_stored",
        entry_type="attack_technique",
        title="XSS - Stored Attack",
        content="""
Stored XSS attack technique:
1. Find input fields that store user data
2. Test with basic payload: <script>alert(1)</script>
3. Bypass filters using encoding or alternative tags
4. Inject malicious payload
        """,
        tags=["xss", "stored", "injection"],
        metadata={"difficulty": "medium", "success_rate": 0.68},
    )

    # 添加最佳實踐
    knowledge_base.add_entry(
        entry_id="bp_reconnaissance",
        entry_type="best_practice",
        title="Thorough Reconnaissance",
        content="""
Always start with comprehensive reconnaissance:
- Identify technologies and frameworks
- Map application structure
- Discover hidden endpoints
- Analyze security headers
        """,
        tags=["reconnaissance", "best-practice"],
    )

    print(f"Added {len(knowledge_base.entries)} knowledge entries")

    # 3. 創建帶 RAG 的 AI Agent
    print("\n3. Creating AI Agent with RAG...")
    ai_agent = AIAgentWithRAG(rag_engine=rag_engine)

    # 4. 生成攻擊計畫（RAG 增強）
    print("\n4. Generating attack plan with RAG enhancement...")
    target = AttackTarget(
        url="http://testphp.vulnweb.com",
        type="web_application",
        description="Test PHP vulnerable application",
    )

    plan = await ai_agent.generate_attack_plan(
        target=target,
        objective="Test for SQL injection vulnerabilities",
    )

    # 5. 演示知識檢索
    print("\n5. Demonstrating knowledge retrieval...")
    search_results = knowledge_base.search(
        query="SQL injection techniques",
        top_k=3,
    )

    print("\nSearch Results for 'SQL injection techniques':")
    for i, entry in enumerate(search_results, 1):
        print(f"\n{i}. {entry.title}")
        print(f"   Type: {entry.type.value}")
        print(f"   Tags: {', '.join(entry.tags)}")
        print(f"   Content preview: {entry.content[:100]}...")

    # 6. 統計信息
    print("\n6. RAG System Statistics:")
    stats = rag_engine.get_statistics()
    print(f"Total entries: {stats['knowledge_base']['total_entries']}")
    print("\nBy type:")
    for type_name, type_stats in stats["knowledge_base"]["by_type"].items():
        print(f"  - {type_name}: {type_stats['count']} entries")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    asyncio.run(demo_rag_integration())
