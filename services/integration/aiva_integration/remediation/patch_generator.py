"""
Patch Generator - 補丁生成器

根據發現的漏洞自動生成修復補丁
使用 GitPython 和 Unidiff 進行差異分析和補丁生成
"""

from datetime import datetime
import hashlib
from pathlib import Path
import tempfile
from typing import Any

import git
import structlog

logger = structlog.get_logger(__name__)


class PatchGenerator:
    """
    補丁生成器

    根據漏洞分析結果生成修復補丁
    """

    def __init__(
        self,
        repo_path: str | Path | None = None,
        auto_commit: bool = False,
    ):
        """
        初始化補丁生成器

        Args:
            repo_path: Git 倉庫路徑
            auto_commit: 是否自動提交補丁
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.auto_commit = auto_commit
        self.repo: git.Repo | None = None
        self.patches: list[dict[str, Any]] = []

        if self.repo_path and self.repo_path.exists():
            try:
                self.repo = git.Repo(self.repo_path)
                logger.info(
                    "patch_generator_initialized",
                    repo_path=str(self.repo_path),
                    branch=self.repo.active_branch.name,
                )
            except git.exc.InvalidGitRepositoryError:
                logger.warning(
                    "invalid_git_repo",
                    repo_path=str(self.repo_path),
                )
        else:
            logger.info("patch_generator_initialized", repo_path="None")

    def generate_patch_for_vulnerability(
        self,
        file_path: str | Path,
        vulnerability_type: str,
        line_number: int,
        suggested_fix: str,
    ) -> dict[str, Any]:
        """
        為特定漏洞生成補丁

        Args:
            file_path: 文件路徑
            vulnerability_type: 漏洞類型
            line_number: 行號
            suggested_fix: 建議修復

        Returns:
            補丁信息
        """
        logger.info(
            "generating_patch",
            file=str(file_path),
            vuln_type=vulnerability_type,
            line=line_number,
        )

        patch = {
            "patch_id": hashlib.sha256(
                f"{file_path}{vulnerability_type}{line_number}{datetime.now()}".encode()
            ).hexdigest()[:16],
            "timestamp": datetime.now().isoformat(),
            "file_path": str(file_path),
            "vulnerability_type": vulnerability_type,
            "line_number": line_number,
            "suggested_fix": suggested_fix,
            "status": "generated",
        }

        # 如果有 repo,生成實際的 git diff
        if self.repo:
            try:
                patch["diff"] = self._generate_git_diff(
                    file_path, line_number, suggested_fix
                )
                patch["status"] = "ready"
            except Exception as e:
                logger.error("git_diff_failed", error=str(e))
                patch["error"] = str(e)
                patch["status"] = "failed"
        else:
            # 生成簡單的補丁格式
            patch["diff"] = self._generate_simple_patch(
                file_path, line_number, suggested_fix
            )

        self.patches.append(patch)
        logger.info("patch_generated", patch_id=patch["patch_id"])
        return patch

    def _generate_simple_patch(
        self,
        file_path: str | Path,
        line_number: int,
        suggested_fix: str,
    ) -> str:
        """生成簡單的補丁格式"""
        return f"""--- a/{file_path}
+++ b/{file_path}
@@ -{line_number},1 +{line_number},1 @@
-[Original line would be here]
+{suggested_fix}
"""

    def _generate_git_diff(
        self,
        file_path: str | Path,
        line_number: int,
        suggested_fix: str,
    ) -> str:
        """使用 Git 生成差異"""
        if not self.repo:
            return ""

        # 實際實現會讀取文件,應用修復,然後生成 diff
        # 這裡提供框架
        return f"# Git diff would be generated for {file_path} at line {line_number}"

    def generate_sql_injection_patch(
        self,
        file_path: str | Path,
        line_number: int,
        vulnerable_code: str,
    ) -> dict[str, Any]:
        """
        為 SQL 注入漏洞生成補丁

        Args:
            file_path: 文件路徑
            line_number: 行號
            vulnerable_code: 有漏洞的代碼

        Returns:
            補丁信息
        """
        # 生成參數化查詢建議
        suggested_fix = self._suggest_parameterized_query(vulnerable_code)

        return self.generate_patch_for_vulnerability(
            file_path=file_path,
            vulnerability_type="SQL Injection",
            line_number=line_number,
            suggested_fix=suggested_fix,
        )

    def _suggest_parameterized_query(self, vulnerable_code: str) -> str:
        """建議參數化查詢"""
        # 簡化示例
        if "execute(" in vulnerable_code and "%" in vulnerable_code:
            return vulnerable_code.replace(
                "execute(",
                "execute_with_params("
            ) + " # Use parameterized query"
        return f"{vulnerable_code} # TODO: Use parameterized query"

    def generate_xss_patch(
        self,
        file_path: str | Path,
        line_number: int,
        vulnerable_code: str,
    ) -> dict[str, Any]:
        """
        為 XSS 漏洞生成補丁

        Args:
            file_path: 文件路徑
            line_number: 行號
            vulnerable_code: 有漏洞的代碼

        Returns:
            補丁信息
        """
        # 生成轉義建議
        suggested_fix = self._suggest_html_escape(vulnerable_code)

        return self.generate_patch_for_vulnerability(
            file_path=file_path,
            vulnerability_type="Cross-Site Scripting (XSS)",
            line_number=line_number,
            suggested_fix=suggested_fix,
        )

    def _suggest_html_escape(self, vulnerable_code: str) -> str:
        """建議 HTML 轉義"""
        if "render(" in vulnerable_code or "template" in vulnerable_code.lower():
            return f"html.escape({vulnerable_code.strip()}) # Escape HTML entities"
        return f"html.escape({vulnerable_code.strip()})"

    def apply_patch(self, patch_id: str) -> dict[str, Any]:
        """
        應用補丁

        Args:
            patch_id: 補丁 ID

        Returns:
            應用結果
        """
        patch = next((p for p in self.patches if p["patch_id"] == patch_id), None)

        if not patch:
            return {
                "success": False,
                "error": f"Patch {patch_id} not found",
            }

        logger.info("applying_patch", patch_id=patch_id)

        if not self.repo:
            return {
                "success": False,
                "error": "No Git repository configured",
            }

        try:
            # 這裡會實際應用補丁到文件
            # 並可選擇性地提交
            result = {
                "success": True,
                "patch_id": patch_id,
                "file_path": patch["file_path"],
                "message": "Patch would be applied here",
            }

            if self.auto_commit:
                result["committed"] = True
                result["commit_hash"] = "simulated_commit_hash"

            patch["status"] = "applied"
            logger.info("patch_applied", patch_id=patch_id)
            return result

        except Exception as e:
            logger.error("patch_apply_failed", patch_id=patch_id, error=str(e))
            patch["status"] = "failed"
            return {
                "success": False,
                "error": str(e),
            }

    def export_patches(self, output_dir: str | Path) -> dict[str, Any]:
        """
        導出所有補丁到目錄

        Args:
            output_dir: 輸出目錄

        Returns:
            導出結果
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported = []
        for patch in self.patches:
            patch_file = output_path / f"{patch['patch_id']}.patch"
            try:
                patch_file.write_text(
                    f"""Patch ID: {patch['patch_id']}
File: {patch['file_path']}
Vulnerability: {patch['vulnerability_type']}
Line: {patch['line_number']}
Timestamp: {patch['timestamp']}

Suggested Fix:
{patch['suggested_fix']}

Diff:
{patch.get('diff', 'N/A')}
""",
                    encoding="utf-8",
                )
                exported.append(str(patch_file))
                logger.info("patch_exported", patch_id=patch["patch_id"], file=str(patch_file))
            except Exception as e:
                logger.error("patch_export_failed", patch_id=patch["patch_id"], error=str(e))

        return {
            "success": True,
            "exported_count": len(exported),
            "output_dir": str(output_path),
            "files": exported,
        }

    def get_patches(self) -> list[dict[str, Any]]:
        """獲取所有補丁"""
        return self.patches


def main():
    """測試範例"""
    print("[CONFIG] Patch Generator Demo")
    print("=" * 60)

    # 創建補丁生成器(不連接實際 repo)
    generator = PatchGenerator()

    # 生成 SQL 注入補丁
    sql_patch = generator.generate_sql_injection_patch(
        file_path="app/views.py",
        line_number=42,
        vulnerable_code='cursor.execute("SELECT * FROM users WHERE id=%s" % user_id)',
    )

    print("\n[LIST] Generated SQL Injection Patch:")
    print(f"   Patch ID: {sql_patch['patch_id']}")
    print(f"   File: {sql_patch['file_path']}")
    print(f"   Status: {sql_patch['status']}")

    # 生成 XSS 補丁
    xss_patch = generator.generate_xss_patch(
        file_path="app/templates.py",
        line_number=15,
        vulnerable_code="render(user_input)",
    )

    print("\n[LIST] Generated XSS Patch:")
    print(f"   Patch ID: {xss_patch['patch_id']}")
    print(f"   File: {xss_patch['file_path']}")

    # 導出補丁
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generator.export_patches(tmpdir)
        print(f"\n[SAVE] Exported {result['exported_count']} patches to {result['output_dir']}")

    print("\n[OK] Demo completed")


if __name__ == "__main__":
    main()
