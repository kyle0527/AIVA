#!/usr/bin/env python3
"""
AIVA 實際靶場掃描執行器 (v2.0)
=============================

使用 v2.0 架構執行實際靶場掃描：
- 無需 RabbitMQ，直接調用 command_handler
- 使用數據合約直接通信
- 支援多種掃描模式

使用範例:
    python live_target_scanner.py --url http://example.com
    python live_target_scanner.py --urls http://site1.com,http://site2.com --strategy deep
    python live_target_scanner.py --url http://example.com --exclude "/admin,/private" --include-subdomains
"""

import argparse
import sys
from typing import List, Optional
from urllib.parse import urlparse
from uuid import uuid4

# 移除 RabbitMQ 依賴
# import pika

# 添加 aiva_common 路徑
sys.path.insert(0, "../../..")

# 新架構使用命令處理器
from services.scan.command_handler import ScanCommandHandler
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class LiveTargetScanner:
    """實際靶場掃描器 (v2.0)"""
    
    def __init__(self):
        self.handler = ScanCommandHandler()
    
    def validate_targets(self, urls: List[str]) -> List[str]:
        """驗證和標準化目標 URL"""
        validated_urls = []
        
        for url in urls:
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            try:
                parsed = urlparse(url)
                if not parsed.netloc:
                    print(f"⚠️ 無效的 URL: {url}")
                    continue
                validated_urls.append(url)
                
            except Exception as e:
                print(f"⚠️ URL 解析錯誤 {url}: {e}")
                continue
        
        return validated_urls
    
    async def execute_scan(self, 
                          urls: List[str],
                          strategy: str = "normal",
                          exclusions: Optional[List[str]] = None,
                          max_depth: int = 3) -> dict:
        """執行掃描 - v2.0 同步架構"""
        
        # 生成掃描 ID
        scan_id = f"scan_{uuid4().hex[:8]}"
        
        # 構建命令
        command_type = CommandType.SCAN_PHASE0 if strategy == "fast" else CommandType.SCAN_COMPREHENSIVE
        
        command = AICommand(
            command_id=f"cmd_{scan_id}",
            command_type=command_type,
            target_module="scan",
            trace_id=scan_id,
            session_id=None,
            parent_command_id=None,
            callback_url=None,
            payload={
                "scan_id": scan_id,
                "targets": urls,
                "max_depth": max_depth,
                "timeout": 300,
            }
        )
        
        print(f"\n🎯 掃描 ID: {scan_id}")
        print(f"   策略: {strategy} ({command_type.value})")
        print(f"   目標數: {len(urls)}")
        print(f"   最大深度: {max_depth}\n")
        
        # 執行掃描
        result = await self.handler.handle_command(command)
        
        return {
            "scan_id": scan_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "result": result.result if result.success else None,
            "error": result.error
        }


def parse_urls(url_string: str) -> List[str]:
    """解析 URL 字符串"""
    return [url.strip() for url in url_string.split(',') if url.strip()]


def parse_exclusions(exclusion_string: str) -> List[str]:
    """解析排除路徑字符串"""
    return [path.strip() for path in exclusion_string.split(',') if path.strip()]


def parse_headers(header_string: str) -> dict:
    """解析自定義標頭字符串"""
    headers = {}
    if not header_string:
        return headers
    
    for header in header_string.split(','):
        if ':' in header:
            key, value = header.split(':', 1)
            headers[key.strip()] = value.strip()
    
    return headers


async def main():
    parser = argparse.ArgumentParser(
        description="AIVA 實際靶場掃描器 (v2.0) - 使用命令處理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
    # 單個目標
    python %(prog)s --url https://example.com
    
    # 多個目標
    python %(prog)s --urls "https://site1.com,https://site2.com"
    
    # 深度掃描並排除特定路徑
    python %(prog)s --url https://example.com --strategy deep --exclude "/admin,/private"
    
    # 不包含子域名的快速掃描
    python %(prog)s --url https://example.com --strategy quick --no-subdomains
    
    # 自定義速率限制和標頭
    python %(prog)s --url https://example.com --rate-limit 5 --delay 2.0 --headers "User-Agent:Custom Bot"

掃描策略:
    quick  - 快速掃描，基本檢查
    normal - 標準掃描（默認）
    deep   - 深度掃描，詳細檢查
    full   - 全面掃描，包含所有測試
    custom - 自定義掃描策略
        """
    )
    
    # URL 參數（互斥）
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument('--url', help='單個目標 URL')
    url_group.add_argument('--urls', help='多個目標 URL，用逗號分隔')
    
    # 掃描策略
    parser.add_argument('--strategy', 
                      choices=['quick', 'normal', 'deep', 'full', 'custom'],
                      default='normal',
                      help='掃描策略 (默認: normal)')
    
    # 範圍配置
    parser.add_argument('--exclude', help='排除的路徑，用逗號分隔 (例: "/admin,/private")')
    parser.add_argument('--max-depth', type=int, default=3, help='最大爬取深度 (默認: 3)')
    
    # 輸出配置
    parser.add_argument('--verbose', action='store_true', help='詳細輸出')
    
    args = parser.parse_args()
    
    try:
        # 解析目標 URL
        if args.url:
            urls = [args.url]
        else:
            urls = parse_urls(args.urls)
        
        if not urls:
            print("❌ 沒有提供有效的目標 URL")
            return 1
        
        # 解析排除路徑
        exclusions = parse_exclusions(args.exclude) if args.exclude else None
        
        # 執行掃描 (v2.0 架構)
        print("=" * 80)
        print("🎯 AIVA 實際靶場掃描器 (v2.0)")
        print("=" * 80)
        
        scanner = LiveTargetScanner()
        
        # 驗證目標
        validated_urls = scanner.validate_targets(urls)
        if not validated_urls:
            print("❌ 沒有有效的目標 URL")
            return 1
        
        print(f"\n📋 掃描配置:")
        print(f"   目標數量: {len(validated_urls)}")
        print(f"   掃描策略: {args.strategy}")
        print(f"   最大深度: {args.max_depth}")
        if exclusions:
            print(f"   排除路徑: {', '.join(exclusions)}")
        
        print("\n🎯 掃描目標:")
        for i, url in enumerate(validated_urls, 1):
            print(f"   [{i}] {url}")
        
        # 執行掃描
        result = await scanner.execute_scan(
            urls=validated_urls,
            strategy=args.strategy,
            exclusions=exclusions,
            max_depth=args.max_depth
        )
        
        if result['success']:
            print(f"\n✅ 掃描完成!")
            print(f"   掃描 ID: {result['scan_id']}")
            print(f"   執行時間: {result['execution_time']:.2f}s")
            
            if args.verbose and result['result']:
                import json
                print("\n📄 掃描結果:")
                print(json.dumps(result['result'], indent=2, ensure_ascii=False))
        else:
            print(f"\n❌ 掃描失敗: {result['error']}")
            return 1
        
        print("\n🔍 v2.0 架構特點:")
        print("   - 無需 RabbitMQ，直接調用引擎")
        print("   - 同步執行，結果即時返回")
        print("   - 數據合約直接通信")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 用戶中斷")
        return 130
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))