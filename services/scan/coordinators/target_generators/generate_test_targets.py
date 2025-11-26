#!/usr/bin/env python3
"""
測試目標生成器 - 用於多目標掃描測試

⚠️ v2.0 架構更新：
- 舊架構: 發送到 RabbitMQ 隊列
- 新架構: 直接調用 command_handler 執行掃描

使用方法:
    python generate_test_targets.py --mode demo
    python generate_test_targets.py --mode full
"""

import argparse
from uuid import uuid4

# 移除 RabbitMQ 依賴
# from services.aiva_common.mq import get_broker

# 新架構使用命令處理器
from services.scan.command_handler import ScanCommandHandler
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

# 測試目標配置
TEST_TARGETS = [
    {
        "name": "Example.com API",
        "url": "https://example.com/api",
        "content": """
        const API_KEY = "AKIAIOSFODNN7EXAMPLE";
        const SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        // Database connection
        const DB_URL = "mongodb://admin:password123@localhost:27017/prod";
        """,
        "expected_findings": ["AWS Access Key", "Generic API Key", "Database Connection String"]
    },
    {
        "name": "GitHub Project",
        "url": "https://github.com/example/project",
        "content": """
        export GITHUB_TOKEN="ghp_1234567890abcdefghijklmnopqrstuv"
        export JWT_SECRET="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        """,
        "expected_findings": ["GitHub Token", "JWT Token"]
    },
    {
        "name": "PHP Application",
        "url": "https://example.com/config.php",
        "content": """
        <?php
        $db_password = "SuperSecret123!";
        $api_key = "sk_test_EXAMPLE1234567890abcdefghijk";
        $email = "admin@example.com";
        define('PRIVATE_KEY', '-----BEGIN RSA PRIVATE KEY-----');
        ?>
        """,
        "expected_findings": ["Password in Code", "Generic API Key", "Email", "Private Key"]
    },
    {
        "name": "React SPA",
        "url": "https://example.com/app.js",
        "content": """
        // React application
        const config = {
            apiEndpoint: '/api/v1/users',
            adminPanel: '/admin/dashboard',
            techStack: 'React + Node.js',
        };
        """,
        "expected_findings": ["API Endpoint", "Admin Interface", "Tech Stack"]
    },
    {
        "name": "Java Backend",
        "url": "https://example.com/config",
        "content": """
        // Java Spring Boot Configuration
        spring.datasource.url=jdbc:postgresql://localhost:5432/prod
        aws.accessKeyId=AKIAIOSFODNN7EXAMPLE
        aws.secretKey=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
        server.port=8080
        """,
        "expected_findings": ["AWS Access Key", "AWS Secret Key", "Database Connection String"]
    },
    {
        "name": ".NET Application",
        "url": "https://example.com/web.config",
        "content": """
        <configuration>
            <appSettings>
                <add key="ApiKey" value="sk_test_1234567890abcdefghij"/>
                <add key="ConnectionString" value="Server=localhost;Database=prod;User=sa;Password=P@ssw0rd!"/>
            </appSettings>
        </configuration>
        """,
        "expected_findings": ["Generic API Key", "Password in Code"]
    },
    {
        "name": "Python Django",
        "url": "https://example.com/settings.py",
        "content": """
        # Django settings
        SECRET_KEY = 'django-insecure-1234567890abcdefghijklmnop'
        DATABASES = {
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'HOST': 'localhost',
                'USER': 'django_user',
                'PASSWORD': 'django_pass_123',
            }
        }
        EMAIL_HOST_USER = 'admin@example.com'
        """,
        "expected_findings": ["Generic API Key", "Password in Code", "Email"]
    },
    {
        "name": "Vue.js Frontend",
        "url": "https://example.com/main.js",
        "content": """
        // Vue.js configuration
        const app = createApp({
            data() {
                return {
                    apiUrl: '/api/v2/data',
                    framework: 'Vue.js 3',
                }
            }
        });
        """,
        "expected_findings": ["API Endpoint", "Tech Stack"]
    },
]


async def main():
    """主函數 - 使用 command_handler 執行掃描"""
    print("=" * 80)
    print("🎯 AIVA 多目標掃描測試生成器 (v2.0)")
    print("=" * 80)
    
    try:
        # 初始化命令處理器
        print("\n📋 初始化掃描命令處理器...")
        handler = ScanCommandHandler()
        
        print(f"\n📋 準備執行 {len(TEST_TARGETS)} 個測試目標掃描...\n")
        
        # 執行所有測試目標
        scan_results = []
        for i, target in enumerate(TEST_TARGETS, 1):
            scan_id = f"test_{uuid4().hex[:8]}"
            
            # 構建命令
            command = AICommand(
                command_id=f"cmd_{scan_id}",
                command_type=CommandType.SCAN_PHASE0,  # Phase 0 快速掃描
                target_module="scan",
                trace_id=scan_id,
                session_id=None,
                parent_command_id=None,
                callback_url=None,
                payload={
                    "scan_id": scan_id,
                    "targets": [target["url"]],
                    "max_depth": 2,
                    "timeout": 60,
                }
            )
            
            print(f"🔍 [{i}/{len(TEST_TARGETS)}] {target['name']}")
            print(f"   Scan ID: {scan_id}")
            print(f"   URL: {target['url']}")
            print(f"   預期發現: {', '.join(target['expected_findings'])}")
            
            # 執行掃描
            result = await handler.handle_command(command)
            
            if result.success:
                print(f"   ✅ 掃描完成 ({result.execution_time:.2f}s)")
                scan_results.append({
                    "scan_id": scan_id,
                    "name": target["name"],
                    "url": target["url"],
                    "expected": target["expected_findings"],
                    "status": "success",
                    "execution_time": result.execution_time,
                    "result": result.result
                })
            else:
                print(f"   ❌ 掃描失敗: {result.error}")
                scan_results.append({
                    "scan_id": scan_id,
                    "name": target["name"],
                    "url": target["url"],
                    "status": "failed",
                    "error": result.error
                })
            print()
        
        # 輸出總結
        print("=" * 80)
        print("✅ 所有測試目標掃描完成!")
        print("=" * 80)
        print(f"\n📊 總結:")
        print(f"   - 已完成掃描: {len(scan_results)}")
        print(f"   - 成功: {sum(1 for r in scan_results if r.get('status') == 'success')}")
        print(f"   - 失敗: {sum(1 for r in scan_results if r.get('status') == 'failed')}")
        
        print("\n🔍 v2.0 架構說明:")
        print("   - 使用同步命令處理器 (command_handler.py)")
        print("   - 無需 RabbitMQ，直接調用引擎")
        print("   - 數據合約直接通信")
        
        print("\n" + "=" * 80)
        
        # 保存任務清單
        import json
        with open('/tmp/scan_results.json', 'w', encoding='utf-8') as f:
            json.dump(scan_results, f, indent=2, ensure_ascii=False)
        print("💾 掃描結果已保存到: /tmp/scan_results.json")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

