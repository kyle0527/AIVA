from __future__ import annotations

import re
from typing import Any

import httpx

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class BackendDbFingerprinter:
    """
    資料庫類型指紋識別器

    通過分析錯誤訊息和回應特徵來識別後端資料庫類型和版本，
    為後續的針對性攻擊提供指導。
    """

    def __init__(self) -> None:
        # 資料庫錯誤模式映射
        self._db_patterns = {
            "MySQL": [
                r"You have an error in your SQL syntax",
                r"mysql_fetch_array\(\)",
                r"mysql_fetch_assoc\(\)",
                r"mysql_fetch_row\(\)",
                r"mysql_num_rows\(\)",
                r"Warning.*mysql_.*",
                r"Unknown column '[^']*' in",
                r"MySQLSyntaxErrorException",
                r"MySQL server version",
            ],
            "Microsoft SQL Server": [
                r"Microsoft OLE DB Provider for ODBC Drivers",
                r"Microsoft OLE DB Provider for SQL Server",
                r"Unclosed quotation mark after the character string",
                r"'80040e14'",
                r"mssql_query\(\)",
                r"Microsoft SQL Native Client error",
                r"SQLSTATE\[.*?\]: Syntax error",
                r"Microsoft SQL Server.*Driver",
            ],
            "Oracle": [
                r"ORA-[0-9]{5}",
                r"Oracle error",
                r"Oracle driver",
                r"Warning.*oci_.*",
                r"Warning.*ora_.*",
                r"Oracle Database.*Error",
            ],
            "PostgreSQL": [
                r"PostgreSQL query failed",
                r"Warning.*pg_.*",
                r"valid PostgreSQL result",
                r"Npgsql\.",
                r"PG::Error",
                r"ERROR:.*syntax error at or near",
                r"PostgreSQL.*ERROR",
            ],
            "SQLite": [
                r"SQLite/JDBCDriver",
                r"SQLite.Exception",
                r"System.Data.SQLite.SQLiteException",
                r"Warning.*sqlite_.*",
                r"SQLITE_ERROR",
                r"SQLite format 3",
            ],
        }

        # 版本提取模式
        self._version_patterns = {
            "MySQL": [
                r"MySQL server version for the right syntax to use near.*?"
                r"(\d+\.\d+\.\d+)",
                r"MySQL.*?(\d+\.\d+\.\d+)",
                r"mysql.*?ver.*?(\d+\.\d+\.\d+)",
            ],
            "Microsoft SQL Server": [
                r"Microsoft SQL Server.*?(\d{4})",
                r"SQL Server.*?(\d+\.\d+\.\d+)",
                r"MSSQL.*?(\d+\.\d+)",
            ],
            "Oracle": [
                r"Oracle Database.*?(\d+c?)",
                r"Oracle.*?(\d+\.\d+\.\d+)",
                r"ORA-.*?Oracle.*?(\d+\.\d+)",
            ],
            "PostgreSQL": [
                r"PostgreSQL.*?(\d+\.\d+(?:\.\d+)?)",
                r"server version.*?(\d+\.\d+)",
                r"PostgreSQL (\d+\.\d+)",
            ],
            "SQLite": [
                r"SQLite version (\d+\.\d+\.\d+)",
                r"SQLite.*?(\d+\.\d+\.\d+)",
            ],
        }

    def fingerprint(self, response: httpx.Response) -> tuple[str, str | None] | None:
        """
        從HTTP回應中識別資料庫類型和版本

        Args:
            response: HTTP回應物件

        Returns:
            (資料庫類型, 版本) 的元組，如果無法識別則返回None
        """
        if response is None:
            return None

        # 組合所有可能的文本源
        text_sources = [
            response.text or "",
            str(response.headers),
            str(response.status_code),
        ]

        combined_text = " ".join(text_sources).lower()

        # 嘗試識別資料庫類型
        for db_name, patterns in self._db_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    logger.info(f"Database fingerprint detected: {db_name}")

                    # 嘗試提取版本資訊
                    version = self._extract_version(combined_text, db_name)

                    return db_name, version

        return None

    def _extract_version(self, text: str, db_name: str) -> str | None:
        """提取資料庫版本資訊"""
        version_patterns = self._version_patterns.get(db_name, [])

        for pattern in version_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                version = match.group(1)
                logger.info(f"Version extracted for {db_name}: {version}")
                return version

        return None

    def get_supported_databases(self) -> list[str]:
        """獲取支援的資料庫類型列表"""
        return list(self._db_patterns.keys())

    def analyze_response_characteristics(
        self, response: httpx.Response
    ) -> dict[str, Any]:
        """
        分析回應特徵以輔助指紋識別

        Args:
            response: HTTP回應物件

        Returns:
            包含各種特徵的字典
        """
        characteristics = {
            "status_code": response.status_code,
            "content_length": len(response.text or ""),
            "headers": dict(response.headers),
            "content_type": response.headers.get("content-type", ""),
            "server_header": response.headers.get("server", ""),
            "has_sql_keywords": self._contains_sql_keywords(response.text or ""),
            "error_signatures": self._extract_error_signatures(response.text or ""),
        }

        return characteristics

    def _contains_sql_keywords(self, text: str) -> list[str]:
        """檢測文本中的SQL關鍵字"""
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "FROM",
            "WHERE",
            "JOIN",
            "UNION",
            "ORDER BY",
            "GROUP BY",
            "HAVING",
            "CREATE",
            "DROP",
            "ALTER",
            "TABLE",
            "DATABASE",
        ]

        found_keywords = []
        text_upper = text.upper()

        for keyword in sql_keywords:
            if keyword in text_upper:
                found_keywords.append(keyword)

        return found_keywords

    def _extract_error_signatures(self, text: str) -> list[str]:
        """提取錯誤簽名"""
        error_signatures = []

        # 通用錯誤模式
        error_patterns = [
            r"syntax error",
            r"unexpected token",
            r"invalid syntax",
            r"parse error",
            r"compilation error",
            r"runtime error",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_signatures.extend(matches)

        return list(set(error_signatures))  # 去重

    def add_custom_pattern(self, db_name: str, pattern: str) -> None:
        """
        添加自定義資料庫識別模式

        Args:
            db_name: 資料庫名稱
            pattern: 正則表達式模式
        """
        if db_name not in self._db_patterns:
            self._db_patterns[db_name] = []

        self._db_patterns[db_name].append(pattern)
        logger.info(f"Added custom pattern for {db_name}: {pattern}")

    def add_custom_version_pattern(self, db_name: str, pattern: str) -> None:
        """
        添加自定義版本提取模式

        Args:
            db_name: 資料庫名稱
            pattern: 正則表達式模式（需包含捕獲組）
        """
        if db_name not in self._version_patterns:
            self._version_patterns[db_name] = []

        self._version_patterns[db_name].append(pattern)
        logger.info(f"Added custom version pattern for {db_name}: {pattern}")
