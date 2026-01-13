#!/usr/bin/env python3
"""測試所有 CommandHandler"""
import asyncio
from services.aiva_common.schemas.commands import AICommand, CommandType, CommandPriority

async def test_xss():
    from services.features.function_xss.command_handler import XSSCommandHandler
    
    cmd = AICommand(
        command_id='test_xss_001',
        command_type=CommandType.FEATURE_XSS_TEST,
        priority=CommandPriority.NORMAL,
        target_module='function_xss',
        payload={
            'target_url': 'http://localhost:3000/rest/products/search',
            'test_type': 'reflected',
            'param_name': 'q'
        }
    )
    handler = XSSCommandHandler()
    result = await handler.handle_command(cmd)
    print(f'\n✅ XSS 測試: {result.success}, 耗時: {result.execution_time_ms}ms')
    return result

async def test_ssrf():
    from services.features.function_ssrf.command_handler import SSRFCommandHandler
    
    cmd = AICommand(
        command_id='test_ssrf_001',
        command_type=CommandType.FEATURE_SSRF_TEST,
        priority=CommandPriority.NORMAL,
        target_module='function_ssrf',
        payload={
            'target_url': 'http://localhost:3000',
            'detection_mode': 'auto'
        }
    )
    handler = SSRFCommandHandler()
    result = await handler.handle_command(cmd)
    print(f'\n✅ SSRF 測試: {result.success}, 耗時: {result.execution_time_ms}ms')
    return result

async def test_idor():
    from services.features.function_idor.command_handler import IDORCommandHandler
    
    cmd = AICommand(
        command_id='test_idor_001',
        command_type=CommandType.FEATURE_IDOR_TEST,
        priority=CommandPriority.NORMAL,
        target_module='function_idor',
        payload={
            'target_url': 'http://localhost:3000/api/Users/1',
            'method': 'auto'
        }
    )
    handler = IDORCommandHandler()
    result = await handler.handle_command(cmd)
    print(f'\n✅ IDOR 測試: {result.success}, 耗時: {result.execution_time_ms}ms')
    return result

async def test_sqli():
    from services.features.function_sqli.command_handler import SQLiCommandHandler
    
    cmd = AICommand(
        command_id='test_sqli_001',
        command_type=CommandType.FEATURE_SQLI_TEST,
        priority=CommandPriority.NORMAL,
        target_module='function_sqli',
        payload={
            'target_url': 'http://localhost:3000/rest/products/search?q=test',
            'param_name': 'q'
        }
    )
    handler = SQLiCommandHandler()
    result = await handler.handle_command(cmd)
    print(f'\n✅ SQLi 測試: {result.success}, 耗時: {result.execution_time_ms}ms')
    return result

async def main():
    print("="*60)
    print("測試所有 CommandHandler 對 Juice Shop 靶場")
    print("="*60)
    
    # 測試 XSS (已知可工作)
    try:
        await test_xss()
    except Exception as e:
        print(f'❌ XSS 測試失敗: {e}')
    
    # 測試 SSRF
    try:
        await test_ssrf()
    except Exception as e:
        print(f'❌ SSRF 測試失敗: {e}')
    
    # 測試 IDOR
    try:
        await test_idor()
    except Exception as e:
        print(f'❌ IDOR 測試失敗: {e}')
    
    # 測試 SQLi
    try:
        await test_sqli()
    except Exception as e:
        print(f'❌ SQLi 測試失敗: {e}')
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)

if __name__ == '__main__':
    asyncio.run(main())
