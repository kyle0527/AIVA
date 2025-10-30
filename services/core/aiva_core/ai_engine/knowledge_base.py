"""
Knowledge Base - RAG 知識庫
提供程式碼索引、檢索和上下文增強功能
"""



import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """程式碼知識庫,用於 RAG 檢索."""

    def __init__(self, codebase_path: str) -> None:
        """初始化知識庫.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        self.codebase_path = Path(codebase_path)
        self.chunks: list[dict[str, Any]] = []
        self.index: dict[str, list[int]] = {}  # 關鍵字 -> chunk 索引

    def index_codebase(self) -> None:
        """索引整個程式碼庫."""
        logger.info(f"正在索引程式碼庫: {self.codebase_path}")

        # 排除的目錄
        exclude_dirs = {
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".mypy_cache",
            ".ruff_cache",
        }

        # 遍歷所有 Python 檔案
        for py_file in self.codebase_path.rglob("*.py"):
            # 檢查是否在排除目錄中
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                self._index_file(py_file, content)
            except Exception as e:
                logger.warning(f"  跳過檔案 {py_file}: {e}")

        logger.info(f"索引完成,共 {len(self.chunks)} 個程式碼片段")

    def _index_file(self, file_path: Path, content: str) -> None:
        """索引單個檔案.

        Args:
            file_path: 檔案路徑
            content: 檔案內容
        """
        # 分塊策略: 按函式/類別分塊
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.ClassDef):
                    # 提取函式/類別的程式碼
                    chunk_content = ast.get_source_segment(content, node)
                    if chunk_content:
                        self._add_chunk(
                            file_path=str(file_path.relative_to(self.codebase_path)),
                            content=chunk_content,
                            node_type=type(node).__name__,
                            node_name=node.name,
                        )
        except SyntaxError:
            # 如果無法解析,整個檔案作為一個 chunk
            self._add_chunk(
                file_path=str(file_path.relative_to(self.codebase_path)),
                content=content[:1000],  # 只取前 1000 字元
                node_type="Module",
                node_name=file_path.stem,
            )

    def _add_chunk(
        self, file_path: str, content: str, node_type: str, node_name: str
    ) -> None:
        """添加程式碼片段到知識庫.

        Args:
            file_path: 相對檔案路徑
            content: 程式碼內容
            node_type: 節點類型 (FunctionDef/ClassDef/Module)
            node_name: 節點名稱
        """
        chunk_id = len(self.chunks)
        chunk = {
            "id": chunk_id,
            "path": file_path,
            "content": content,
            "type": node_type,
            "name": node_name,
        }
        self.chunks.append(chunk)

        # 簡單的關鍵字索引
        keywords = self._extract_keywords(content, node_name)
        for keyword in keywords:
            if keyword not in self.index:
                self.index[keyword] = []
            self.index[keyword].append(chunk_id)

    def _extract_keywords(self, content: str, node_name: str) -> set[str]:
        """從內容中提取關鍵字.

        Args:
            content: 程式碼內容
            node_name: 節點名稱

        Returns:
            關鍵字集合
        """
        keywords = set()

        # 添加節點名稱
        keywords.add(node_name.lower())

        # 提取常見的程式碼關鍵字
        common_keywords = [
            "def",
            "class",
            "async",
            "await",
            "scan",
            "detect",
            "analyze",
            "execute",
            "process",
            "vulnerability",
            "xss",
            "sqli",
            "ssrf",
            "idor",
            "core",
            "integration",
        ]

        for keyword in common_keywords:
            if keyword in content.lower():
                keywords.add(keyword)

        return keywords

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """檢索相關的程式碼片段.

        Args:
            query: 查詢字串
            top_k: 返回前 K 個結果

        Returns:
            相關的程式碼片段列表
        """
        # 簡單的關鍵字匹配檢索
        query_keywords = query.lower().split()
        scores: dict[int, int] = {}

        for keyword in query_keywords:
            if keyword in self.index:
                for chunk_id in self.index[keyword]:
                    scores[chunk_id] = scores.get(chunk_id, 0) + 1

        # 按分數排序
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回 top_k 個結果
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            chunk = self.chunks[chunk_id]
            results.append(
                {
                    "path": chunk["path"],
                    "content": chunk["content"],
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "score": score,
                }
            )

        return results

    def get_chunk_count(self) -> int:
        """獲取程式碼片段總數.

        Returns:
            片段數量
        """
        return len(self.chunks)

    def get_file_content(self, file_path: str) -> str | None:
        """獲取完整檔案內容.

        Args:
            file_path: 檔案路徑 (相對於程式碼庫根目錄)

        Returns:
            檔案內容或 None
        """
        try:
            full_path = self.codebase_path / file_path
            return full_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"無法讀取檔案 {file_path}: {e}")
            return None
