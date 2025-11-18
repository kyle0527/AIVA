#!/usr/bin/env python3
"""
AIVA 跨語言 AI 操作示範
====================

這個腳本示範 AIVA 如何使用 AI 進行跨語言操作，包括：
1. 代碼分析和轉換
2. 跨語言調用
3. Schema 生成和驗證
4. 安全性分析
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# 添加 AIVA 路徑
sys.path.append(str(Path(__file__).parent.parent))

from services.aiva_common.ai.cross_language_bridge import CrossLanguageBridge
from services.aiva_common.enums import ProgrammingLanguage
from services.core.aiva_core.multilang_coordinator import MultiLanguageAICoordinator


class CrossLanguageAIDemo:
    """跨語言 AI 操作示範"""
    
    def __init__(self):
        self.coordinator = MultiLanguageAICoordinator()
        self.bridge = CrossLanguageBridge()
        
    async def demo_code_analysis(self):
        """示範跨語言代碼分析"""
        print("🔍 === 跨語言代碼分析示範 ===")
        
        # Python 代碼範例
        python_code = '''
def calculate_risk_score(vulnerabilities):
    """計算風險評分"""
    score = 0
    for vuln in vulnerabilities:
        if vuln["severity"] == "critical":
            score += 10
        elif vuln["severity"] == "high":
            score += 7
        elif vuln["severity"] == "medium":
            score += 4
        else:
            score += 1
    return score
'''
        
        print("🐍 原始 Python 代碼:")
        print(python_code)
        
        # 使用 AI 分析代碼
        analysis_result = await self.coordinator.execute_task(
            "code_analysis",
            language=ProgrammingLanguage.PYTHON,
            code=python_code,
            target_languages=["go", "rust", "typescript"]
        )
        
        print("📊 AI 分析結果:")
        print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
        
        return analysis_result
    
    async def demo_schema_conversion(self):
        """示範 Schema 自動轉換"""
        print("\n🔄 === Schema 跨語言轉換示範 ===")
        
        # 定義一個 Schema
        schema_definition = {
            "name": "SecurityFinding",
            "description": "安全發現",
            "fields": {
                "id": {"type": "string", "required": True},
                "title": {"type": "string", "required": True},
                "severity": {"type": "enum", "values": ["critical", "high", "medium", "low"]},
                "confidence": {"type": "number", "range": [0.0, 1.0]},
                "description": {"type": "string", "required": False},
                "location": {
                    "type": "object",
                    "fields": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "column": {"type": "integer"}
                    }
                }
            }
        }
        
        print("📝 原始 Schema 定義:")
        print(json.dumps(schema_definition, indent=2, ensure_ascii=False))
        
        # 生成多語言代碼
        languages = [ProgrammingLanguage.PYTHON, ProgrammingLanguage.GO, 
                    ProgrammingLanguage.RUST, ProgrammingLanguage.TYPESCRIPT]
        
        generated_code = {}
        for lang in languages:
            result = await self.coordinator.execute_task(
                "schema_generation",
                language=lang,
                schema=schema_definition
            )
            generated_code[lang.value] = result
            
        print("\n🚀 生成的多語言代碼:")
        for lang, code in generated_code.items():
            print(f"\n📄 {lang.upper()} 代碼:")
            if code.get("success"):
                print(code.get("result", "代碼生成完成"))
            else:
                print(f"❌ 生成失敗: {code.get('error')}")
                
        return generated_code
    
    async def demo_cross_language_communication(self):
        """示範跨語言通信"""
        print("\n🌐 === 跨語言通信示範 ===")
        
        # 模擬 Python 調用 Rust 進行高性能安全掃描
        scan_request = {
            "target": "https://example.com",
            "scan_type": "comprehensive",
            "max_depth": 3,
            "timeout": 300
        }
        
        print("📤 Python 向 Rust 發送掃描請求:")
        print(json.dumps(scan_request, indent=2, ensure_ascii=False))
        
        # 調用 Rust AI 模組
        rust_result = await self.coordinator.execute_task(
            "security_scan",
            language=ProgrammingLanguage.RUST,
            **scan_request
        )
        
        print("📥 從 Rust 接收結果:")
        print(json.dumps(rust_result, indent=2, ensure_ascii=False))
        
        # 模擬 Python 調用 Go 進行雲端安全分析
        cloud_request = {
            "cloud_provider": "aws",
            "resources": ["ec2", "s3", "iam"],
            "compliance_standards": ["cis", "nist"]
        }
        
        print("\n📤 Python 向 Go 發送雲端分析請求:")
        print(json.dumps(cloud_request, indent=2, ensure_ascii=False))
        
        go_result = await self.coordinator.execute_task(
            "cloud_security_analysis",
            language=ProgrammingLanguage.GO,
            **cloud_request
        )
        
        print("📥 從 Go 接收結果:")
        print(json.dumps(go_result, indent=2, ensure_ascii=False))
        
        # 模擬 Python 調用 TypeScript 進行動態掃描
        dynamic_request = {
            "url": "https://example.com/login",
            "browser": "chrome",
            "interactions": ["form_fill", "button_click", "navigation"],
            "javascript_enabled": True
        }
        
        print("\n📤 Python 向 TypeScript 發送動態掃描請求:")
        print(json.dumps(dynamic_request, indent=2, ensure_ascii=False))
        
        ts_result = await self.coordinator.execute_task(
            "dynamic_scan",
            language=ProgrammingLanguage.TYPESCRIPT,
            **dynamic_request
        )
        
        print("📥 從 TypeScript 接收結果:")
        print(json.dumps(ts_result, indent=2, ensure_ascii=False))
        
        return {
            "rust_scan": rust_result,
            "go_cloud": go_result,
            "typescript_dynamic": ts_result
        }
    
    async def demo_ai_code_generation(self):
        """示範 AI 代碼生成"""
        print("\n🤖 === AI 代碼生成示範 ===")
        
        # 用自然語言描述需求
        requirements = {
            "description": "創建一個函數來驗證用戶輸入的安全性，檢查 XSS 和 SQL 注入",
            "input_params": ["user_input: string"],
            "return_type": "ValidationResult",
            "security_requirements": ["input_sanitization", "xss_prevention", "sqli_prevention"]
        }
        
        print("📋 需求描述:")
        print(json.dumps(requirements, indent=2, ensure_ascii=False))
        
        # 為每種語言生成代碼
        languages = [ProgrammingLanguage.PYTHON, ProgrammingLanguage.GO, 
                    ProgrammingLanguage.RUST, ProgrammingLanguage.TYPESCRIPT]
        
        generated_functions = {}
        for lang in languages:
            print(f"\n🔧 為 {lang.value.upper()} 生成代碼...")
            
            result = await self.coordinator.execute_task(
                "ai_code_generation",
                language=lang,
                requirements=requirements
            )
            
            generated_functions[lang.value] = result
            
            if result.get("success"):
                print(f"✅ {lang.value.upper()} 代碼生成成功")
            else:
                print(f"❌ {lang.value.upper()} 代碼生成失敗: {result.get('error')}")
                
        return generated_functions
    
    async def demo_security_analysis(self):
        """示範跨語言安全分析"""
        print("\n🛡️ === 跨語言安全分析示範 ===")
        
        # 模擬多語言專案結構
        project_structure = {
            "backend": {
                "language": "python",
                "framework": "fastapi",
                "files": ["main.py", "models.py", "auth.py"]
            },
            "security_scanner": {
                "language": "rust", 
                "framework": "tokio",
                "files": ["main.rs", "scanner.rs", "vulns.rs"]
            },
            "cloud_checker": {
                "language": "go",
                "framework": "gin",
                "files": ["main.go", "aws.go", "compliance.go"]
            },
            "dynamic_scanner": {
                "language": "typescript",
                "framework": "playwright",
                "files": ["index.ts", "crawler.ts", "analyzer.ts"]
            }
        }
        
        print("🏗️ 專案結構:")
        print(json.dumps(project_structure, indent=2, ensure_ascii=False))
        
        # 進行跨語言安全分析
        security_analysis = await self.coordinator.execute_task(
            "cross_language_security_analysis",
            project_structure=project_structure,
            analysis_depth="comprehensive"
        )
        
        print("\n🔍 跨語言安全分析結果:")
        print(json.dumps(security_analysis, indent=2, ensure_ascii=False))
        
        return security_analysis
    
    async def run_full_demo(self):
        """運行完整示範"""
        print("🚀 AIVA 跨語言 AI 操作完整示範")
        print("=" * 50)
        
        results = {}
        
        # 1. 代碼分析
        results["code_analysis"] = await self.demo_code_analysis()
        
        # 2. Schema 轉換
        results["schema_conversion"] = await self.demo_schema_conversion()
        
        # 3. 跨語言通信
        results["cross_language_comm"] = await self.demo_cross_language_communication()
        
        # 4. AI 代碼生成
        results["ai_code_generation"] = await self.demo_ai_code_generation()
        
        # 5. 安全分析
        results["security_analysis"] = await self.demo_security_analysis()
        
        print("\n🎉 === 示範完成 ===")
        print("📊 所有結果已收集，可用於進一步分析")
        
        # 保存結果
        output_file = Path("cross_language_ai_demo_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 結果已保存到: {output_file}")
        
        return results


async def main():
    """主函數"""
    demo = CrossLanguageAIDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    print("🤖 啟動 AIVA 跨語言 AI 操作示範...")
    asyncio.run(main())