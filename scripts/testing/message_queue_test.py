import pika
import json
import time
import threading
from datetime import datetime

print("🐰 AIVA 組件間通信測試 - RabbitMQ")
print("=" * 40)

class MessageQueueTester:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.received_messages = []
        
    def connect(self):
        """連接到 RabbitMQ"""
        try:
            connection_params = pika.ConnectionParameters(
                host='localhost',
                port=5672,
                virtual_host='/',
                credentials=pika.PlainCredentials('guest', 'guest')
            )
            
            self.connection = pika.BlockingConnection(connection_params)
            self.channel = self.connection.channel()
            print("✅ 成功連接到 RabbitMQ")
            return True
            
        except Exception as e:
            print(f"❌ RabbitMQ 連接失敗: {e}")
            return False
    
    def setup_queues(self):
        """設置消息隊列"""
        print("\n📦 設置消息隊列...")
        
        queues = [
            'aiva.scan.requests',
            'aiva.scan.results', 
            'aiva.test.requests',
            'aiva.test.results',
            'aiva.explore.requests',
            'aiva.explore.results'
        ]
        
        try:
            for queue_name in queues:
                self.channel.queue_declare(queue=queue_name, durable=True)
                print(f"✅ 隊列創建: {queue_name}")
                
            # 創建交換機
            self.channel.exchange_declare(
                exchange='aiva.topic',
                exchange_type='topic',
                durable=True
            )
            print("✅ 交換機創建: aiva.topic")
            return True
            
        except Exception as e:
            print(f"❌ 隊列設置失敗: {e}")
            return False
    
    def test_scan_communication(self):
        """測試掃描組件通信"""
        print("\n🔍 測試掃描組件通信...")
        
        # 模擬掃描請求
        scan_request = {
            "task_id": f"scan_{int(time.time())}",
            "type": "sql_injection",
            "target": {
                "url": "http://localhost:3000/rest/user/login",
                "method": "POST"
            },
            "priority": 5,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 發送掃描請求
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.scan.requests',
                body=json.dumps(scan_request),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 持久化消息
                    correlation_id=scan_request['task_id']
                )
            )
            print(f"📤 發送掃描請求: {scan_request['task_id']}")
            
            # 模擬掃描結果
            time.sleep(1)
            scan_result = {
                "task_id": scan_request['task_id'],
                "status": "completed",
                "findings": [
                    {
                        "type": "sql_injection",
                        "severity": "high",
                        "payload": "admin' OR 1=1--",
                        "evidence": "返回狀態碼 200，疑似繞過認證"
                    }
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.scan.results',
                body=json.dumps(scan_result),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=scan_request['task_id']
                )
            )
            print(f"📥 發送掃描結果: {scan_result['status']}")
            return True
            
        except Exception as e:
            print(f"❌ 掃描通信測試失敗: {e}")
            return False
    
    def test_testing_communication(self):
        """測試測試組件通信"""
        print("\n🧪 測試測試組件通信...")
        
        test_request = {
            "task_id": f"test_{int(time.time())}",
            "type": "functional_test",
            "target": {
                "url": "http://localhost:3000",
                "endpoints": ["/api/users", "/rest/user/login"]
            },
            "test_scenarios": ["registration", "login", "search"],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 發送測試請求
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.test.requests',
                body=json.dumps(test_request),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=test_request['task_id']
                )
            )
            print(f"📤 發送測試請求: {test_request['task_id']}")
            
            # 模擬測試結果
            time.sleep(1)
            test_result = {
                "task_id": test_request['task_id'],
                "status": "completed",
                "test_results": {
                    "registration": "pass",
                    "login": "pass", 
                    "search": "pass"
                },
                "vulnerabilities": 5,
                "timestamp": datetime.now().isoformat()
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.test.results',
                body=json.dumps(test_result),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=test_request['task_id']
                )
            )
            print(f"📥 發送測試結果: {test_result['status']}")
            return True
            
        except Exception as e:
            print(f"❌ 測試通信測試失敗: {e}")
            return False
    
    def test_exploration_communication(self):
        """測試探索組件通信"""
        print("\n🔍 測試探索組件通信...")
        
        explore_request = {
            "task_id": f"explore_{int(time.time())}",
            "type": "system_exploration",
            "target": {
                "url": "http://localhost:3000",
                "scope": ["directories", "endpoints", "technologies"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 發送探索請求
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.explore.requests',
                body=json.dumps(explore_request),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=explore_request['task_id']
                )
            )
            print(f"📤 發送探索請求: {explore_request['task_id']}")
            
            # 模擬探索結果
            time.sleep(1)
            explore_result = {
                "task_id": explore_request['task_id'],
                "status": "completed",
                "discoveries": {
                    "directories": 17,
                    "endpoints": 22,
                    "technologies": ["Angular", "Node.js", "Express"],
                    "open_ports": [3000, 5432, 6379]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key='aiva.explore.results',
                body=json.dumps(explore_result),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    correlation_id=explore_request['task_id']
                )
            )
            print(f"📥 發送探索結果: {explore_result['status']}")
            return True
            
        except Exception as e:
            print(f"❌ 探索通信測試失敗: {e}")
            return False
    
    def check_queue_status(self):
        """檢查隊列狀態"""
        print("\n📊 檢查隊列狀態...")
        
        queues = [
            'aiva.scan.requests',
            'aiva.scan.results',
            'aiva.test.requests', 
            'aiva.test.results',
            'aiva.explore.requests',
            'aiva.explore.results'
        ]
        
        total_messages = 0
        
        for queue_name in queues:
            try:
                method = self.channel.queue_declare(queue=queue_name, passive=True)
                message_count = method.method.message_count
                total_messages += message_count
                print(f"📦 {queue_name}: {message_count} 條消息")
                
            except Exception as e:
                print(f"❌ 檢查隊列 {queue_name} 失敗: {e}")
        
        return total_messages
    
    def cleanup(self):
        """清理資源"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            print("✅ 連接已關閉")

def main():
    tester = MessageQueueTester()
    
    try:
        # 1. 連接測試
        if not tester.connect():
            return
        
        # 2. 設置隊列
        if not tester.setup_queues():
            return
        
        # 3. 測試各組件通信
        scan_success = tester.test_scan_communication()
        test_success = tester.test_testing_communication()
        explore_success = tester.test_exploration_communication()
        
        # 4. 檢查隊列狀態
        total_messages = tester.check_queue_status()
        
        # 總結
        print("\n" + "=" * 40)
        print("🐰 組件間通信測試總結:")
        print(f"🔍 掃描組件通信: {'✅ 成功' if scan_success else '❌ 失敗'}")
        print(f"🧪 測試組件通信: {'✅ 成功' if test_success else '❌ 失敗'}")
        print(f"🔍 探索組件通信: {'✅ 成功' if explore_success else '❌ 失敗'}")
        print(f"📨 總消息數量: {total_messages}")
        print(f"🎯 通信狀態: {'🟢 正常' if all([scan_success, test_success, explore_success]) else '🔴 異常'}")
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()