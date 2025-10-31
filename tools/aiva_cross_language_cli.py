#!/usr/bin/env python3
"""
AIVA 跨語言 CLI 工具
=================

使用 AIVA 現有功能創建的跨語言命令行工具，支援：
- Schema 代碼生成 (Python/Go/Rust/TypeScript)
- 跨語言接口調用
- AI 組件協調
- 安全掃描和分析
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from subprocess import run, PIPE
from typing import Any, Dict, List, Optional

# 添加 AIVA 路徑
sys.path.append(str(Path(__file__).parent.parent))

from services.aiva_common.tools.schema_codegen_tool import SchemaCodeGenerator
from services.aiva_common.tools.cross_language_interface import CrossLanguageSchemaInterface
from services.aiva_common.enums import ProgrammingLanguage
from services.core.aiva_core.multilang_coordinator import MultiLanguageAICoordinator


class AIVACrossLanguageCLI:
    """AIVA 跨語言 CLI 主類"""
    
    def __init__(self):
        self.schema_generator = SchemaCodeGenerator()
        self.cross_lang_interface = CrossLanguageSchemaInterface()
        self.coordinator = MultiLanguageAICoordinator()
    
    def generate_schema(self, language: str, output_dir: Optional[str] = None) -> None:
        """生成指定語言的 Schema 代碼"""
        print(f"🔧 正在生成 {language.upper()} Schema 代碼...")
        
        try:
            if language.lower() == "python":
                files = self.schema_generator.generate_python_schemas(output_dir)
                print(f"✅ Python Schema 生成完成，共 {len(files)} 個文件")
                for file in files:
                    print(f"   📄 {file}")
                    
            elif language.lower() == "go":
                # 使用現有的生成邏輯
                cmd = [
                    sys.executable, 
                    "services/aiva_common/tools/schema_codegen_tool.py",
                    "--lang", "go"
                ]
                if output_dir:
                    cmd.extend(["--output-dir", output_dir])
                
                result = run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Go Schema 生成完成")
                    print(result.stdout)
                else:
                    print(f"❌ Go Schema 生成失敗: {result.stderr}")
                    
            elif language.lower() == "rust":
                cmd = [
                    sys.executable,
                    "services/aiva_common/tools/schema_codegen_tool.py", 
                    "--lang", "rust"
                ]
                if output_dir:
                    cmd.extend(["--output-dir", output_dir])
                
                result = run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Rust Schema 生成完成")
                    print(result.stdout)
                else:
                    print(f"❌ Rust Schema 生成失敗: {result.stderr}")
                    
            elif language.lower() == "typescript":
                cmd = [
                    sys.executable,
                    "services/aiva_common/tools/schema_codegen_tool.py",
                    "--lang", "typescript"
                ]
                if output_dir:
                    cmd.extend(["--output-dir", output_dir])
                
                result = run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ TypeScript Schema 生成完成")
                    print(result.stdout)
                else:
                    print(f"❌ TypeScript Schema 生成失敗: {result.stderr}")
                    
            elif language.lower() == "all":
                print("🚀 生成所有語言的 Schema...")
                languages = ["python", "go", "rust", "typescript"]
                for lang in languages:
                    self.generate_schema(lang, output_dir)
                    
            else:
                print(f"❌ 不支援的語言: {language}")
                print("支援的語言: python, go, rust, typescript, all")
                
        except Exception as e:
            print(f"❌ Schema 生成異常: {e}")
    
    def convert_type(self, source_type: str, target_language: str) -> None:
        """類型轉換"""
        print(f"🔄 將類型 '{source_type}' 轉換為 {target_language.upper()}...")
        
        try:
            result = self.cross_lang_interface.convert_type_to_language(source_type, target_language)
            print(f"✅ 轉換結果: {result}")
        except Exception as e:
            print(f"❌ 類型轉換失敗: {e}")
    
    def generate_code(self, schema_name: str, target_language: str) -> None:
        """生成指定 Schema 的代碼"""
        print(f"📝 為 Schema '{schema_name}' 生成 {target_language.upper()} 代碼...")
        
        try:
            code = self.cross_lang_interface.generate_schema_code(schema_name, target_language)
            print("✅ 代碼生成完成:")
            print("=" * 50)
            print(code)
            print("=" * 50)
        except Exception as e:
            print(f"❌ 代碼生成失敗: {e}")
    
    def list_schemas(self) -> None:
        """列出所有可用的 Schema"""
        print("📋 可用的 Schema 定義:")
        
        try:
            info = self.cross_lang_interface.get_ai_friendly_schema_info()
            
            if "schemas" in info:
                for i, schema in enumerate(info["schemas"], 1):
                    print(f"{i:2d}. {schema['name']}")
                    print(f"    描述: {schema.get('description', 'N/A')}")
                    print(f"    分類: {schema.get('category', 'N/A')}")
                    print(f"    字段數: {len(schema.get('fields', []))}")
                    print()
            else:
                print("❌ 無法獲取 Schema 信息")
                
        except Exception as e:
            print(f"❌ 列出 Schema 失敗: {e}")
    
    async def execute_ai_task(self, task: str, language: Optional[str] = None, **kwargs) -> None:
        """執行 AI 任務"""
        print(f"🤖 執行 AI 任務: {task}")
        if language:
            print(f"🎯 指定語言: {language.upper()}")
        
        try:
            # 轉換語言字符串為枚舉
            lang_enum = None
            if language:
                lang_map = {
                    "python": ProgrammingLanguage.PYTHON,
                    "go": ProgrammingLanguage.GO,
                    "rust": ProgrammingLanguage.RUST,
                    "typescript": ProgrammingLanguage.TYPESCRIPT
                }
                lang_enum = lang_map.get(language.lower())
                if not lang_enum:
                    print(f"❌ 不支援的語言: {language}")
                    return
            
            result = await self.coordinator.execute_task(task, lang_enum, **kwargs)
            
            print("✅ AI 任務執行完成:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ AI 任務執行失敗: {e}")
    
    def validate_schema(self) -> None:
        """驗證 Schema 一致性"""
        print("🔍 驗證 Schema 一致性...")
        
        try:
            issues = self.cross_lang_interface.validate_cross_language_consistency()
            
            if not any(issues.values()):
                print("✅ 所有 Schema 一致性驗證通過")
            else:
                print("⚠️ 發現一致性問題:")
                for category, problems in issues.items():
                    if problems:
                        print(f"  {category}:")
                        for problem in problems:
                            print(f"    - {problem}")
                            
        except Exception as e:
            print(f"❌ Schema 驗證失敗: {e}")
    
    def interactive_mode(self) -> None:
        """交互模式"""
        print("🎯 進入 AIVA 跨語言 CLI 交互模式")
        print("輸入 'help' 查看可用命令，輸入 'quit' 退出")
        print()
        
        while True:
            try:
                command = input("AIVA> ").strip()
                
                if not command:
                    continue
                elif command.lower() in ['quit', 'exit', 'q']:
                    print("👋 再見！")
                    break
                elif command.lower() == 'help':
                    self.show_help()
                elif command.startswith('schema '):
                    # schema generate python
                    # schema list
                    # schema validate
                    parts = command.split()
                    if len(parts) >= 2:
                        if parts[1] == 'generate' and len(parts) >= 3:
                            self.generate_schema(parts[2])
                        elif parts[1] == 'list':
                            self.list_schemas()
                        elif parts[1] == 'validate':
                            self.validate_schema()
                        else:
                            print("❌ 無效的 schema 命令")
                    else:
                        print("❌ schema 命令需要子命令")
                elif command.startswith('convert '):
                    # convert string go
                    parts = command.split()
                    if len(parts) >= 3:
                        self.convert_type(parts[1], parts[2])
                    else:
                        print("❌ convert 命令需要類型和目標語言")
                elif command.startswith('generate '):
                    # generate SecurityFinding rust
                    parts = command.split()
                    if len(parts) >= 3:
                        self.generate_code(parts[1], parts[2])
                    else:
                        print("❌ generate 命令需要 Schema 名稱和目標語言")
                elif command.startswith('ai '):
                    # ai security_scan rust target=example.com
                    parts = command.split()
                    if len(parts) >= 2:
                        task = parts[1]
                        language = parts[2] if len(parts) > 2 else None
                        
                        # 解析額外參數
                        kwargs = {}
                        for part in parts[3:]:
                            if '=' in part:
                                key, value = part.split('=', 1)
                                kwargs[key] = value
                        
                        asyncio.run(self.execute_ai_task(task, language, **kwargs))
                    else:
                        print("❌ ai 命令需要任務名稱")
                else:
                    print(f"❌ 未知命令: {command}")
                    print("輸入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n👋 再見！")
                break
            except Exception as e:
                print(f"❌ 命令執行失敗: {e}")
    
    def show_help(self) -> None:
        """顯示幫助信息"""
        help_text = """
🤖 AIVA 跨語言 CLI 工具幫助

Schema 命令:
  schema generate <language>  - 生成指定語言的 Schema (python/go/rust/typescript/all)
  schema list                 - 列出所有可用的 Schema
  schema validate            - 驗證跨語言 Schema 一致性

類型轉換:
  convert <type> <language>   - 將類型轉換為指定語言格式
  
代碼生成:
  generate <schema> <language> - 為指定 Schema 生成目標語言代碼

AI 任務:
  ai <task> [language] [key=value...] - 執行 AI 任務
  
其他命令:
  help                       - 顯示此幫助信息
  quit/exit/q               - 退出程序

範例:
  schema generate python
  convert "List[str]" go
  generate SecurityFinding rust
  ai security_scan rust target=example.com
        """
        print(help_text)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="AIVA 跨語言 CLI 工具")
    parser.add_argument("--interactive", "-i", action="store_true", help="進入交互模式")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # Schema 命令
    schema_parser = subparsers.add_parser("schema", help="Schema 相關操作")
    schema_subparsers = schema_parser.add_subparsers(dest="schema_action")
    
    generate_parser = schema_subparsers.add_parser("generate", help="生成 Schema 代碼")
    generate_parser.add_argument("language", choices=["python", "go", "rust", "typescript", "all"], help="目標語言")
    generate_parser.add_argument("--output-dir", "-o", help="輸出目錄")
    
    schema_subparsers.add_parser("list", help="列出所有 Schema")
    schema_subparsers.add_parser("validate", help="驗證 Schema 一致性")
    
    # 類型轉換命令
    convert_parser = subparsers.add_parser("convert", help="類型轉換")
    convert_parser.add_argument("source_type", help="源類型")
    convert_parser.add_argument("target_language", choices=["python", "go", "rust", "typescript"], help="目標語言")
    
    # 代碼生成命令
    generate_parser = subparsers.add_parser("generate", help="生成代碼")
    generate_parser.add_argument("schema_name", help="Schema 名稱")
    generate_parser.add_argument("target_language", choices=["python", "go", "rust"], help="目標語言")
    
    # AI 任務命令
    ai_parser = subparsers.add_parser("ai", help="執行 AI 任務")
    ai_parser.add_argument("task", help="任務名稱")
    ai_parser.add_argument("--language", "-l", choices=["python", "go", "rust", "typescript"], help="指定語言")
    ai_parser.add_argument("--params", "-p", help="任務參數 (JSON 格式)")
    
    args = parser.parse_args()
    
    cli = AIVACrossLanguageCLI()
    
    try:
        if args.interactive or not args.command:
            cli.interactive_mode()
        elif args.command == "schema":
            if args.schema_action == "generate":
                cli.generate_schema(args.language, args.output_dir)
            elif args.schema_action == "list":
                cli.list_schemas()
            elif args.schema_action == "validate":
                cli.validate_schema()
        elif args.command == "convert":
            cli.convert_type(args.source_type, args.target_language)
        elif args.command == "generate":
            cli.generate_code(args.schema_name, args.target_language)
        elif args.command == "ai":
            params = {}
            if args.params:
                try:
                    params = json.loads(args.params)
                except json.JSONDecodeError:
                    print("❌ 無效的 JSON 參數格式")
                    return
            asyncio.run(cli.execute_ai_task(args.task, args.language, **params))
            
    except KeyboardInterrupt:
        print("\n👋 程序被中斷")
    except Exception as e:
        print(f"❌ 程序執行失敗: {e}")


if __name__ == "__main__":
    print("🚀 AIVA 跨語言 CLI 工具")
    print("=" * 30)
    main()