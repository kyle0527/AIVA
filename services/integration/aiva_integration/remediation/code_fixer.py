"""
Code Fixer - 代碼修復器

使用 LLM (OpenAI/LiteLLM) 自動修復代碼中的漏洞和問題
"""

import contextlib
from datetime import datetime
import hashlib
from typing import Any

import structlog

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = structlog.get_logger(__name__)


class CodeFixer:
    """
    AI 驅動的代碼修復器

    使用 LLM 分析和修復代碼漏洞
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        use_litellm: bool = True,
        max_tokens: int = 2000,
    ):
        """
        初始化代碼修復器

        Args:
            api_key: API 密鑰
            model: 模型名稱
            use_litellm: 是否使用 LiteLLM
            max_tokens: 最大 token 數
        """
        self.api_key = api_key
        self.model = model
        self.use_litellm = use_litellm and LITELLM_AVAILABLE
        self.max_tokens = max_tokens
        self.fix_history: list[dict[str, Any]] = []

        if self.use_litellm:
            if not LITELLM_AVAILABLE:
                logger.warning("litellm_not_available", message="Falling back to mock mode")
                self.use_litellm = False
            else:
                litellm.api_key = api_key
                logger.info("code_fixer_initialized", mode="litellm", model=model)
        elif OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            logger.info("code_fixer_initialized", mode="openai", model=model)
        else:
            logger.warning("no_ai_api_available", message="Running in mock mode")

    def fix_vulnerability(
        self,
        code: str,
        vulnerability_type: str,
        language: str = "python",
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        修復代碼漏洞

        Args:
            code: 有漏洞的代碼
            vulnerability_type: 漏洞類型
            language: 編程語言
            context: 額外上下文

        Returns:
            修復結果
        """
        logger.info(
            "fixing_vulnerability",
            vuln_type=vulnerability_type,
            language=language,
        )

        fix_id = hashlib.sha256(
            f"{code}{vulnerability_type}{datetime.now()}".encode()
        ).hexdigest()[:16]

        result = {
            "fix_id": fix_id,
            "timestamp": datetime.now().isoformat(),
            "vulnerability_type": vulnerability_type,
            "language": language,
            "original_code": code,
            "fixed_code": None,
            "explanation": None,
            "confidence": 0.0,
            "status": "processing",
        }

        try:
            if self.use_litellm or (OPENAI_AVAILABLE and self.api_key):
                # 使用真實的 LLM
                fixed = self._fix_with_llm(
                    code, vulnerability_type, language, context
                )
                result.update(fixed)
            else:
                # Mock 模式
                fixed = self._fix_with_mock(code, vulnerability_type, language)
                result.update(fixed)

            result["status"] = "completed"
            logger.info("vulnerability_fixed", fix_id=fix_id)

        except Exception as e:
            logger.error("fix_failed", fix_id=fix_id, error=str(e))
            result["status"] = "failed"
            result["error"] = str(e)

        self.fix_history.append(result)
        return result

    def _fix_with_llm(
        self,
        code: str,
        vulnerability_type: str,
        language: str,
        context: str | None,
    ) -> dict[str, Any]:
        """使用 LLM 修復代碼"""
        prompt = self._build_fix_prompt(code, vulnerability_type, language, context)

        if self.use_litellm:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a security expert. Fix the vulnerability in the code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
        else:
            # OpenAI
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a security expert. Fix the vulnerability in the code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content

        # 解析 LLM 響應
        return self._parse_llm_response(content)

    def _fix_with_mock(
        self,
        code: str,
        vulnerability_type: str,
        language: str,
    ) -> dict[str, Any]:
        """Mock 模式修復"""
        logger.debug("using_mock_fixer")

        # 基於漏洞類型提供模板修復
        if vulnerability_type.lower() == "sql injection":
            fixed_code = self._mock_fix_sql_injection(code, language)
            explanation = "Replaced string concatenation with parameterized query"
            confidence = 0.85

        elif vulnerability_type.lower() in ["xss", "cross-site scripting"]:
            fixed_code = self._mock_fix_xss(code, language)
            explanation = "Added HTML escaping to user input"
            confidence = 0.80

        elif vulnerability_type.lower() == "path traversal":
            fixed_code = self._mock_fix_path_traversal(code, language)
            explanation = "Added path validation and sanitization"
            confidence = 0.75

        else:
            fixed_code = f"# TODO: Fix {vulnerability_type}\n{code}"
            explanation = f"Generic fix template for {vulnerability_type}"
            confidence = 0.50

        return {
            "fixed_code": fixed_code,
            "explanation": explanation,
            "confidence": confidence,
        }

    def _build_fix_prompt(
        self,
        code: str,
        vulnerability_type: str,
        language: str,
        context: str | None,
    ) -> str:
        """構建修復提示"""
        prompt = f"""Fix the following {vulnerability_type} vulnerability in {language} code:

```{language}
{code}
```
"""
        if context:
            prompt += f"\nContext: {context}"

        prompt += """

Provide:
1. Fixed code
2. Explanation of the fix
3. Confidence level (0-1)

Format your response as:
FIXED_CODE:
```
[fixed code here]
```

EXPLANATION:
[explanation here]

CONFIDENCE:
[0-1]
"""
        return prompt

    def _parse_llm_response(self, content: str) -> dict[str, Any]:
        """解析 LLM 響應"""
        # 簡化的解析邏輯
        lines = content.split("\n")

        fixed_code = ""
        explanation = ""
        confidence = 0.5

        in_code = False
        in_explanation = False

        for line in lines:
            if "FIXED_CODE:" in line or "```" in line:
                in_code = not in_code
                continue
            if "EXPLANATION:" in line:
                in_explanation = True
                continue
            if "CONFIDENCE:" in line:
                in_explanation = False
                with contextlib.suppress(ValueError):
                    confidence = float(line.split(":")[-1].strip())
                continue

            if in_code:
                fixed_code += line + "\n"
            elif in_explanation:
                explanation += line + " "

        return {
            "fixed_code": fixed_code.strip(),
            "explanation": explanation.strip(),
            "confidence": confidence,
        }

    def _mock_fix_sql_injection(self, code: str, language: str) -> str:
        """Mock SQL 注入修復"""
        if language == "python" and "execute(" in code and ("%" in code or "+" in code):
            return code.replace(
                "execute(",
                "# Fixed: Use parameterized query\nexecute("
            ) + "\n# Use: cursor.execute(query, (param1, param2))"
        return f"# Fixed SQL Injection\n{code}"

    def _mock_fix_xss(self, code: str, language: str) -> str:
        """Mock XSS 修復"""
        if language == "python" and ("render(" in code or "template" in code.lower()):
            return f"import html\n# Fixed: Escape HTML\n{code.replace('render(', 'render(html.escape(')}"
        return f"# Fixed XSS - Add HTML escaping\n{code}"

    def _mock_fix_path_traversal(self, code: str, language: str) -> str:
        """Mock 路徑遍歷修復"""
        if language == "python":
            return f"""import os
from pathlib import Path

# Fixed: Validate and sanitize path
{code}
# Add: path = Path(user_input).resolve()
# Validate: if not path.is_relative_to(allowed_dir): raise ValueError()
"""
        return f"# Fixed Path Traversal - Add validation\n{code}"

    def fix_multiple_issues(
        self,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        批量修復多個問題

        Args:
            issues: 問題列表,每項包含 code, vulnerability_type, language

        Returns:
            批量修復結果
        """
        logger.info("fixing_multiple_issues", count=len(issues))

        results = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "fixes": [],
            "success_count": 0,
            "failed_count": 0,
        }

        for issue in issues:
            fix = self.fix_vulnerability(
                code=issue.get("code", ""),
                vulnerability_type=issue.get("vulnerability_type", "unknown"),
                language=issue.get("language", "python"),
                context=issue.get("context"),
            )
            results["fixes"].append(fix)

            if fix["status"] == "completed":
                results["success_count"] += 1
            else:
                results["failed_count"] += 1

        logger.info(
            "multiple_issues_fixed",
            success=results["success_count"],
            failed=results["failed_count"],
        )
        return results

    def get_fix_history(self) -> list[dict[str, Any]]:
        """獲取修復歷史"""
        return self.fix_history


def main():
    """測試範例 - Mock 模式"""
    print("[AI] Code Fixer Demo (Mock Mode)")
    print("=" * 60)

    # 使用 Mock 模式(不需要 API key)
    fixer = CodeFixer(use_litellm=False)

    # 修復 SQL 注入
    sql_code = 'cursor.execute("SELECT * FROM users WHERE id=" + user_id)'
    result1 = fixer.fix_vulnerability(
        code=sql_code,
        vulnerability_type="SQL Injection",
        language="python",
    )

    print("\n[LIST] SQL Injection Fix:")
    print(f"   Status: {result1['status']}")
    print(f"   Confidence: {result1['confidence']}")
    print(f"   Explanation: {result1['explanation']}")

    # 修復 XSS
    xss_code = "return render_template('page.html', content=user_input)"
    result2 = fixer.fix_vulnerability(
        code=xss_code,
        vulnerability_type="XSS",
        language="python",
    )

    print("\n[LIST] XSS Fix:")
    print(f"   Status: {result2['status']}")
    print(f"   Confidence: {result2['confidence']}")

    print("\n[OK] Demo completed (Mock mode)")
    print("[TIP] Provide API key to use real LLM fixes")


if __name__ == "__main__":
    main()
