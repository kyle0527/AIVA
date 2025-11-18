#!/usr/bin/env python3
"""
測試目標生成器 - 用於多目標掃描測試
將多個測試目標發送到 RabbitMQ 隊列
"""

import json
import os
import time
import pika
from uuid import uuid4

# RabbitMQ 配置
RABBITMQ_URL = os.environ.get('RABBITMQ_URL', 'amqp://aiva:aiva_mq_password@localhost:5672/aiva')
TASK_QUEUE = 'tasks.scan.sensitive_info'

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


def create_rabbitmq_connection():
    """建立 RabbitMQ 連接"""
    max_retries = 10
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"🔗 嘗試連接 RabbitMQ (嘗試 {attempt + 1}/{max_retries})...")
            parameters = pika.URLParameters(RABBITMQ_URL)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # 聲明隊列
            channel.queue_declare(
                queue=TASK_QUEUE,
                durable=True,
                arguments={'x-message-ttl': 3600000}  # 1 小時 TTL
            )
            
            print("✅ RabbitMQ 連接成功!")
            return connection, channel
            
        except Exception as e:
            print(f"❌ 連接失敗: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ {retry_delay} 秒後重試...")
                time.sleep(retry_delay)
            else:
                raise Exception("無法連接到 RabbitMQ")


def send_test_target(channel, target_config):
    """發送單個測試目標到隊列"""
    task_id = f"test_{uuid4().hex[:8]}"
    
    task_payload = {
        "task_id": task_id,
        "content": target_config["content"],
        "source_url": target_config["url"],
        "metadata": {
            "name": target_config["name"],
            "expected_findings": target_config["expected_findings"],
            "timestamp": time.time()
        }
    }
    
    message = json.dumps(task_payload, ensure_ascii=False)
    
    channel.basic_publish(
        exchange='',
        routing_key=TASK_QUEUE,
        body=message.encode('utf-8'),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 持久化
            content_type='application/json',
        )
    )
    
    return task_id


def main():
    print("=" * 80)
    print("🎯 AIVA 多目標掃描測試生成器")
    print("=" * 80)
    
    try:
        # 建立連接
        connection, channel = create_rabbitmq_connection()
        
        print(f"\n📋 準備發送 {len(TEST_TARGETS)} 個測試目標...\n")
        
        # 發送所有測試目標
        sent_tasks = []
        for i, target in enumerate(TEST_TARGETS, 1):
            task_id = send_test_target(channel, target)
            sent_tasks.append({
                "task_id": task_id,
                "name": target["name"],
                "url": target["url"],
                "expected": target["expected_findings"]
            })
            
            print(f"✅ [{i}/{len(TEST_TARGETS)}] {target['name']}")
            print(f"   Task ID: {task_id}")
            print(f"   URL: {target['url']}")
            print(f"   預期發現: {', '.join(target['expected_findings'])}")
            print()
            
            time.sleep(0.5)  # 避免過快發送
        
        # 關閉連接
        connection.close()
        
        # 輸出總結
        print("=" * 80)
        print("✅ 所有測試目標已成功發送!")
        print("=" * 80)
        print(f"\n📊 總結:")
        print(f"   - 已發送任務數: {len(sent_tasks)}")
        print(f"   - 目標隊列: {TASK_QUEUE}")
        print(f"   - RabbitMQ 管理界面: http://localhost:15672")
        print(f"   - 帳號/密碼: aiva / aiva_mq_password")
        
        print("\n🔍 監控建議:")
        print("   1. 查看 RabbitMQ 管理界面確認隊列狀態")
        print("   2. 使用 'docker logs' 查看各引擎處理日誌:")
        print("      docker logs -f aiva-rust-fast-discovery")
        print("      docker logs -f aiva-rust-deep-analysis")
        print("      docker logs -f aiva-rust-focused-verification")
        print("   3. 查看統計指標文件:")
        print("      docker exec aiva-rust-fast-discovery cat /var/log/aiva/metrics/rust_fast_discovery.jsonl")
        
        print("\n" + "=" * 80)
        
        # 保存任務清單
        with open('/tmp/sent_tasks.json', 'w', encoding='utf-8') as f:
            json.dump(sent_tasks, f, indent=2, ensure_ascii=False)
        print("💾 任務清單已保存到: /tmp/sent_tasks.json")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
