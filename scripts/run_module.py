#!/usr/bin/env python3
"""直接調用外部模組 - 不依賴 CLI"""
import sys
import asyncio
from pathlib import Path

# 添加項目路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas.commands import AICommand, CommandType, CommandPriority


async def run_module(module_name: str, target_url: str):
    """直接調用模組的 CommandHandler"""
    
    print(f"[INFO] 載入模組: {module_name}")
    print(f"[INFO] 目標: {target_url}")
    print()
    
    # 模組配置：handler 類和對應的命令類型
    module_config = {
        "xss": ("services.features.function_xss.command_handler", "XSSCommandHandler", CommandType.FEATURE_XSS_TEST),
        "sqli": ("services.features.function_sqli.command_handler", "SQLiCommandHandler", CommandType.FEATURE_SQLI_TEST),
        "ssrf": ("services.features.function_ssrf.command_handler", "SSRFCommandHandler", CommandType.FEATURE_SSRF_TEST),
        "idor": ("services.features.function_idor.command_handler", "IDORCommandHandler", CommandType.FEATURE_IDOR_TEST),
        "bizlogic": ("services.features.function_bizlogic.command_handler", "BizLogicCommandHandler", CommandType.FEATURE_BIZLOGIC_TEST),
    }
    
    if module_name not in module_config:
        print(f"[ERROR] 未知模組: {module_name}")
        print(f"可用模組: {', '.join(module_config.keys())}")
        return None
    
    # 動態導入 CommandHandler
    try:
        module_path, class_name, command_type = module_config[module_name]
        
        # 導入模組
        import importlib
        module = importlib.import_module(module_path)
        handler_class = getattr(module, class_name)
        handler = handler_class()
        
        # 創建命令
        import uuid
        command = AICommand(
            command_id=str(uuid.uuid4()),
            command_type=command_type,
            target_module=f"function_{module_name}",
            priority=CommandPriority.NORMAL,
            payload={
                "target_url": target_url,
                "scan_type": "comprehensive"
            }
        )
        
        print(f"[INFO] 執行掃描...")
        result = await handler.handle_command(command)
        
        print(f"\n[RESULT]")
        print(f"狀態: {result.status}")
        print(f"成功: {result.success}")
        
        if result.data:
            print(f"\n數據:")
            import json
            print(json.dumps(result.data, indent=2, ensure_ascii=False))
        
        if result.error:
            print(f"\n錯誤: {result.error}")
        
        return result
        
    except ImportError as e:
        print(f"[ERROR] 無法載入模組 {module_name}: {e}")
        print(f"[INFO] 該模組可能沒有 CommandHandler")
        return None
    except Exception as e:
        print(f"[ERROR] 執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) < 3:
        print("用法: python run_module.py [模組名稱] [目標URL]")
        print()
        print("可用模組:")
        print("  xss      - XSS 掃描")
        print("  sqli     - SQL 注入掃描")
        print("  ssrf     - SSRF 掃描")
        print("  idor     - IDOR 掃描")
        print("  bizlogic - 業務邏輯掃描")
        print()
        print("範例:")
        print("  python run_module.py xss http://localhost:3000")
        sys.exit(1)
    
    module_name = sys.argv[1]
    target_url = sys.argv[2]
    
    asyncio.run(run_module(module_name, target_url))


if __name__ == "__main__":
    main()
