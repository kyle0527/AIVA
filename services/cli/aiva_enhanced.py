#!/usr/bin/env python3
"""
AIVA 增強版 CLI - 整合跨模組功能

此檔案展示如何整合：
1. 現有的 aiva_cli.py 功能
2. aiva-contracts 工具（JSON Schema / TypeScript）
3. 參數合併（旗標 > 環境變數 > 設定檔）
4. 統一輸出格式（human / json）
"""
from pathlib import Path
import sys

# 引入 tools 包裝器
from . import tools

# 引入工具函式
from ._utils import EXIT_OK, EXIT_SYSTEM, echo
from .aiva_cli import (
    async_main as base_async_main,
)

# 引入現有的 CLI 邏輯
from .aiva_cli import (
    create_parser as create_base_parser,
)


def create_enhanced_parser():
    """創建增強版的命令解析器，添加跨模組功能"""
    # 獲取基礎解析器
    parser = create_base_parser()

    # 獲取主 subparsers
    # 注意：由於 argparse 的限制，我們需要重新組織
    # 這裡我們添加一個新的頂層指令 "tools"
    subparsers = parser._subparsers._group_actions[0]

    # ========== Tools 命令（跨模組整合）==========
    tools_parser = subparsers.add_parser(
        "tools",
        help="開發者工具（schemas、型別導出、跨語言協定）"
    )
    tools_sub = tools_parser.add_subparsers(dest="tools_action")

    # tools schemas - 導出 JSON Schema
    schemas = tools_sub.add_parser(
        "schemas",
        help="導出 JSON Schema（用於跨語言協定）"
    )
    schemas.add_argument(
        "--out",
        default="./_out/aiva.schemas.json",
        help="輸出檔案路徑"
    )
    schemas.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="輸出格式"
    )
    schemas.set_defaults(func=cmd_tools_schemas)

    # tools typescript - 導出 TypeScript 型別
    typescript = tools_sub.add_parser(
        "typescript",
        help="導出 TypeScript 型別定義（.d.ts）"
    )
    typescript.add_argument(
        "--out",
        default="./_out/aiva.d.ts",
        help="輸出檔案路徑"
    )
    typescript.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="輸出格式"
    )
    typescript.set_defaults(func=cmd_tools_typescript)

    # tools models - 列出所有模型
    models = tools_sub.add_parser(
        "models",
        help="列出所有 Pydantic 模型"
    )
    models.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="輸出格式"
    )
    models.set_defaults(func=cmd_tools_models)

    # tools export-all - 一鍵導出所有
    export_all = tools_sub.add_parser(
        "export-all",
        help="一鍵導出 JSON Schema + TypeScript"
    )
    export_all.add_argument(
        "--out-dir",
        default="./_out",
        help="輸出目錄"
    )
    export_all.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="輸出格式"
    )
    export_all.set_defaults(func=cmd_tools_export_all)

    return parser


# ============================================================================
# Tools 命令實作
# ============================================================================

def cmd_tools_schemas(args):
    """導出 JSON Schema"""
    sys.exit(tools.export_schemas(out=args.out, fmt=args.format))


def cmd_tools_typescript(args):
    """導出 TypeScript 型別"""
    sys.exit(tools.export_typescript(out=args.out, fmt=args.format))


def cmd_tools_models(args):
    """列出所有模型"""
    sys.exit(tools.list_models(fmt=args.format))


def cmd_tools_export_all(args):
    """一鍵導出所有"""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 導出 JSON Schema
    schema_path = out_dir / "aiva.schemas.json"
    exit_code = tools.export_schemas(out=str(schema_path), fmt="json")
    if exit_code == EXIT_OK:
        results.append({"type": "json-schema", "path": str(schema_path.resolve())})
    else:
        echo({"error": "Failed to export JSON Schema"}, fmt=args.format)
        sys.exit(EXIT_SYSTEM)

    # 導出 TypeScript
    ts_path = out_dir / "aiva.d.ts"
    exit_code = tools.export_typescript(out=str(ts_path), fmt="json")
    if exit_code == EXIT_OK:
        results.append({"type": "typescript", "path": str(ts_path.resolve())})
    else:
        echo({"error": "Failed to export TypeScript"}, fmt=args.format)
        sys.exit(EXIT_SYSTEM)

    # 成功訊息
    echo({
        "ok": True,
        "command": "export-all",
        "exports": results,
        "message": f"已導出 {len(results)} 個檔案到 {out_dir.resolve()}"
    }, fmt=args.format)
    sys.exit(EXIT_OK)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """增強版主入口點"""
    parser = create_enhanced_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    # 如果是原有的命令，使用原有的異步處理
    if args.command in ["scan", "detect", "ai", "report", "system"]:
        import asyncio
        asyncio.run(base_async_main())
    else:
        # 新命令（tools）直接執行
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n⚠️  用戶中斷操作")
        except Exception as e:
            echo({"error": str(e)}, fmt="json")
            sys.exit(EXIT_SYSTEM)


if __name__ == "__main__":
    main()
