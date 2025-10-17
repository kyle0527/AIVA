#!/usr/bin/env python3
"""
AIVA Tools CLI - 開發者工具包裝器

提供對 aiva-contracts 工具的便捷訪問
"""
from pathlib import Path
import subprocess
import sys

from ._utils import EXIT_OK, EXIT_SYSTEM, echo


def export_schemas(out: str = "./_out/aiva.schemas.json", fmt: str = "human") -> int:
    """導出 JSON Schema

    Args:
        out: 輸出檔案路徑
        fmt: 輸出格式（human|json）

    Returns:
        退出碼
    """
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "-m", "aiva_contracts_tooling.cli", "export-jsonschema", "--out", out],
            capture_output=True,
            text=True,
            check=True,
        )

        msg = {
            "ok": True,
            "command": "export-schemas",
            "output": str(Path(out).resolve()),
        }
        echo(msg, fmt=fmt)
        return EXIT_OK

    except subprocess.CalledProcessError as e:
        echo({"error": str(e), "stderr": e.stderr}, fmt=fmt)
        return EXIT_SYSTEM
    except Exception as e:
        echo({"error": str(e)}, fmt=fmt)
        return EXIT_SYSTEM


def export_typescript(out: str = "./_out/aiva.d.ts", fmt: str = "human") -> int:
    """導出 TypeScript 型別定義

    Args:
        out: 輸出檔案路徑
        fmt: 輸出格式（human|json）

    Returns:
        退出碼
    """
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "-m", "aiva_contracts_tooling.cli", "export-dts", "--out", out],
            capture_output=True,
            text=True,
            check=True,
        )

        msg = {
            "ok": True,
            "command": "export-typescript",
            "output": str(Path(out).resolve()),
        }
        echo(msg, fmt=fmt)
        return EXIT_OK

    except subprocess.CalledProcessError as e:
        echo({"error": str(e), "stderr": e.stderr}, fmt=fmt)
        return EXIT_SYSTEM
    except Exception as e:
        echo({"error": str(e)}, fmt=fmt)
        return EXIT_SYSTEM


def list_models(fmt: str = "human") -> int:
    """列出所有可用的 Pydantic 模型

    Args:
        fmt: 輸出格式（human|json）

    Returns:
        退出碼
    """
    try:
        result = subprocess.run(
            ["python", "-m", "aiva_contracts_tooling.cli", "list-models"],
            capture_output=True,
            text=True,
            check=True,
        )

        # 直接輸出原始結果
        if fmt == "json":
            # 嘗試解析為 JSON
            try:
                import json
                data = json.loads(result.stdout)
                echo(data, fmt="json")
            except:
                echo({"output": result.stdout}, fmt="json")
        else:
            sys.stdout.write(result.stdout)

        return EXIT_OK

    except subprocess.CalledProcessError as e:
        echo({"error": str(e), "stderr": e.stderr}, fmt=fmt)
        return EXIT_SYSTEM
    except Exception as e:
        echo({"error": str(e)}, fmt=fmt)
        return EXIT_SYSTEM


if __name__ == "__main__":
    # 簡單測試
    print("Tools module loaded successfully")
