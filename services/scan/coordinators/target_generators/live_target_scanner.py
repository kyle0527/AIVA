#!/usr/bin/env python3
"""
AIVA 實際靶場掃描執行器
=============================

支援對實際靶場進行動態目標掃描，遵循 aiva_common 規範
不寫死目標，支援多種掃描模式

使用範例:
    python live_target_scanner.py --url http://example.com
    python live_target_scanner.py --urls http://site1.com,http://site2.com --strategy deep
    python live_target_scanner.py --url http://example.com --exclude "/admin,/private" --include-subdomains
"""

import argparse
import json
import os
import time
import sys
from typing import List, Optional
from urllib.parse import urlparse
from uuid import uuid4

import pika
from pydantic import ValidationError

# 添加 aiva_common 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from services.aiva_common.schemas.testing.tasks import ScanStartPayload
from services.aiva_common.schemas.base import ScanScope, RateLimit, Authentication

# RabbitMQ 配置
RABBITMQ_URL = os.environ.get('RABBITMQ_URL', 'amqp://aiva:aiva_mq_password@localhost:5672/aiva')
TASK_QUEUE = 'tasks.scan.live_targets'

class LiveTargetScanner:
    """實際靶場掃描器"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
    
    def connect_rabbitmq(self) -> bool:
        """建立 RabbitMQ 連接"""
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f"🔗 嘗試連接 RabbitMQ (嘗試 {attempt + 1}/{max_retries})...")
                parameters = pika.URLParameters(RABBITMQ_URL)
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                # 聲明隊列
                self.channel.queue_declare(
                    queue=TASK_QUEUE,
                    durable=True,
                    arguments={'x-message-ttl': 3600000}
                )
                
                print("✅ RabbitMQ 連接成功!")
                return True
                
            except Exception as e:
                print(f"❌ 連接失敗: {e}")
                if attempt < max_retries - 1:
                    print(f"⏳ {retry_delay} 秒後重試...")
                    time.sleep(retry_delay)
                else:
                    print("❌ 無法連接到 RabbitMQ")
                    return False
    
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
    
    def create_scan_payload(self, 
                           urls: List[str],
                           strategy: str = "normal",
                           exclusions: List[str] = None,
                           include_subdomains: bool = True,
                           rate_limit_requests: int = 10,
                           rate_limit_delay: float = 1.0,
                           custom_headers: dict = None) -> ScanStartPayload:
        """創建符合 aiva_common 規範的掃描負載"""
        
        # 生成掃描 ID
        scan_id = f"scan_{uuid4().hex[:8]}"
        
        # 配置掃描範圍
        scope = ScanScope(
            exclusions=exclusions or [],
            include_subdomains=include_subdomains,
            allowed_hosts=[urlparse(url).netloc for url in urls]
        )
        
        # 配置速率限制
        rate_limit = RateLimit(
            requests_per_second=rate_limit_requests,
            delay_between_requests=rate_limit_delay
        )
        
        # 配置身份驗證（如果需要）
        authentication = Authentication()
        
        try:
            payload = ScanStartPayload(
                scan_id=scan_id,
                targets=urls,
                scope=scope,
                authentication=authentication,
                strategy=strategy,
                rate_limit=rate_limit,
                custom_headers=custom_headers or {}
            )
            return payload
            
        except ValidationError as e:
            print(f"❌ 負載驗證失敗: {e}")
            raise
    
    def send_scan_task(self, payload: ScanStartPayload) -> str:
        """發送掃描任務到隊列"""
        
        if not self.channel:
            raise Exception("RabbitMQ 連接未建立")
        
        message = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
        
        self.channel.basic_publish(
            exchange='',
            routing_key=TASK_QUEUE,
            body=message.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=2,  # 持久化
                content_type='application/json',
                headers={
                    'task_type': 'live_scan',
                    'scan_id': payload.scan_id,
                    'target_count': len(payload.targets)
                }
            )
        )
        
        return payload.scan_id
    
    def scan_targets(self, 
                    urls: List[str],
                    strategy: str = "normal",
                    exclusions: List[str] = None,
                    include_subdomains: bool = True,
                    rate_limit_requests: int = 10,
                    rate_limit_delay: float = 1.0,
                    custom_headers: dict = None) -> str:
        """執行對指定目標的掃描"""
        
        print("=" * 80)
        print("🎯 AIVA 實際靶場掃描器")
        print("=" * 80)
        
        # 驗證目標
        validated_urls = self.validate_targets(urls)
        if not validated_urls:
            raise ValueError("❌ 沒有有效的目標 URL")
        
        print(f"\n📋 掃描配置:")
        print(f"   目標數量: {len(validated_urls)}")
        print(f"   掃描策略: {strategy}")
        print(f"   包含子域名: {include_subdomains}")
        print(f"   速率限制: {rate_limit_requests} req/s，延遲 {rate_limit_delay}s")
        if exclusions:
            print(f"   排除路徑: {', '.join(exclusions)}")
        
        # 顯示目標
        print(f"\n🎯 掃描目標:")
        for i, url in enumerate(validated_urls, 1):
            print(f"   [{i}] {url}")
        
        # 建立連接
        if not self.connect_rabbitmq():
            raise Exception("❌ 無法連接到 RabbitMQ")
        
        try:
            # 創建掃描負載
            payload = self.create_scan_payload(
                urls=validated_urls,
                strategy=strategy,
                exclusions=exclusions,
                include_subdomains=include_subdomains,
                rate_limit_requests=rate_limit_requests,
                rate_limit_delay=rate_limit_delay,
                custom_headers=custom_headers
            )
            
            # 發送任務
            scan_id = self.send_scan_task(payload)
            
            print(f"\n✅ 掃描任務已發送!")
            print(f"   掃描 ID: {scan_id}")
            print(f"   隊列: {TASK_QUEUE}")
            print(f"   RabbitMQ 管理界面: http://localhost:15672")
            
            return scan_id
            
        finally:
            if self.connection:
                self.connection.close()
    
    def close(self):
        """關閉連接"""
        if self.connection:
            self.connection.close()


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


def main():
    parser = argparse.ArgumentParser(
        description="AIVA 實際靶場掃描器 - 支援動態目標配置",
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
    parser.add_argument('--no-subdomains', action='store_true', help='不包含子域名')
    
    # 速率限制
    parser.add_argument('--rate-limit', type=int, default=10, 
                      help='每秒請求數 (默認: 10)')
    parser.add_argument('--delay', type=float, default=1.0,
                      help='請求間延遲秒數 (默認: 1.0)')
    
    # 自定義配置
    parser.add_argument('--headers', help='自定義 HTTP 標頭，格式: "Key1:Value1,Key2:Value2"')
    
    # 輸出配置
    parser.add_argument('--verbose', action='store_true', help='詳細輸出')
    parser.add_argument('--dry-run', action='store_true', help='乾運行，不實際發送任務')
    
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
        
        # 解析自定義標頭
        custom_headers = parse_headers(args.headers) if args.headers else None
        
        # 乾運行模式
        if args.dry_run:
            print("🔍 乾運行模式 - 僅驗證配置")
            scanner = LiveTargetScanner()
            payload = scanner.create_scan_payload(
                urls=urls,
                strategy=args.strategy,
                exclusions=exclusions,
                include_subdomains=not args.no_subdomains,
                rate_limit_requests=args.rate_limit,
                rate_limit_delay=args.delay,
                custom_headers=custom_headers
            )
            print(f"\n✅ 配置驗證成功!")
            if args.verbose:
                print(f"\n📄 生成的負載:")
                print(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False))
            return 0
        
        # 實際執行掃描
        scanner = LiveTargetScanner()
        scan_id = scanner.scan_targets(
            urls=urls,
            strategy=args.strategy,
            exclusions=exclusions,
            include_subdomains=not args.no_subdomains,
            rate_limit_requests=args.rate_limit,
            rate_limit_delay=args.delay,
            custom_headers=custom_headers
        )
        
        print(f"\n🔍 監控建議:")
        print(f"   1. 查看 RabbitMQ 隊列狀態: http://localhost:15672")
        print(f"   2. 監控掃描引擎日誌:")
        print(f"      docker logs -f aiva-rust-fast-discovery")
        print(f"      docker logs -f aiva-python-scanner")
        print(f"      docker logs -f aiva-typescript-scanner")
        print(f"   3. 查詢掃描結果:")
        print(f"      python query_scan_results.py --scan-id {scan_id}")
        
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
    exit(main())