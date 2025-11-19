#!/usr/bin/env python3
"""
AIVA 數據持久化驗證器
測試各組件是否能正確將掃描結果、測試報告存儲到 PostgreSQL，並透過 Redis 進行緩存
"""

import os
import json
import asyncio
import asyncpg
import redis
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# 設置環境變數
os.environ.setdefault('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672')
os.environ.setdefault('POSTGRES_HOST', 'localhost')
os.environ.setdefault('POSTGRES_PORT', '5432')
os.environ.setdefault('POSTGRES_USER', 'postgres')
os.environ.setdefault('POSTGRES_PASSWORD', 'aiva123')
os.environ.setdefault('POSTGRES_DB', 'aiva_db')

class DataPersistenceValidator:
    """數據持久化驗證器"""
    
    def __init__(self):
        # 研發階段直接使用預設配置
        from urllib.parse import urlparse
        db_url = urlparse('postgresql://postgres:postgres@localhost:5432/aiva_db')
        pg_config = {
            'host': db_url.hostname or 'localhost',
            'port': db_url.port or 5432,
            'user': db_url.username or 'postgres',
            'password': db_url.password or 'postgres',
            'database': db_url.path.lstrip('/') or 'aiva_db',
        }
        
        self.redis_config = {
            'url': 'redis://localhost:6379'
        }
        
        self.test_results = {
            'postgres_connection': False,
            'redis_connection': False,
            'table_creation': False,
            'data_insertion': False,
            'data_retrieval': False,
            'redis_caching': False,
            'data_consistency': False
        }

    async def test_postgres_connection(self) -> bool:
        """測試PostgreSQL連接"""
        try:
            print("🐘 測試 PostgreSQL 連接...")
            
            conn = await asyncpg.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                database=self.postgres_config['database']
            )
            
            # 測試基本查詢
            version = await conn.fetchval('SELECT version()')
            print(f"   ✅ PostgreSQL 連接成功")
            print(f"   📋 版本: {version.split(',')[0]}")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ PostgreSQL 連接失敗: {e}")
            return False

    def test_redis_connection(self) -> bool:
        """測試Redis連接"""
        try:
            print("🔴 測試 Redis 連接...")
            
            r = redis.from_url(self.redis_config['url'])
            
            # 測試基本連接
            r.ping()
            
            # 獲取Redis信息
            info = r.info()
            print(f"   ✅ Redis 連接成功")
            print(f"   📋 版本: {info['redis_version']}")
            print(f"   📊 內存使用: {info['used_memory_human']}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Redis 連接失敗: {e}")
            return False

    async def test_table_operations(self) -> bool:
        """測試表格操作"""
        try:
            print("📊 測試數據表操作...")
            
            conn = await asyncpg.connect(**self.postgres_config)
            
            # 創建測試表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS test_scan_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    scan_id VARCHAR(100) NOT NULL,
                    target_url VARCHAR(500) NOT NULL,
                    vulnerability_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    confidence VARCHAR(20) NOT NULL,
                    payload TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            print("   ✅ 測試表創建成功")
            
            # 檢查表結構
            columns = await conn.fetch("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'test_scan_results'")
            print(f"   📋 表結構: {len(columns)} 個欄位")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ 表格操作失敗: {e}")
            return False

    async def test_data_insertion(self) -> bool:
        """測試數據插入"""
        try:
            print("📝 測試數據插入...")
            
            conn = await asyncpg.connect(**self.postgres_config)
            
            # 準備測試數據
            test_data = [
                {
                    'scan_id': f'scan_{uuid.uuid4().hex[:8]}',
                    'target_url': 'http://localhost:3000/login',
                    'vulnerability_type': 'SQL Injection',
                    'severity': 'HIGH',
                    'confidence': 'CERTAIN',
                    'payload': "' OR '1'='1",
                    'description': '登錄頁面存在SQL注入漏洞'
                },
                {
                    'scan_id': f'scan_{uuid.uuid4().hex[:8]}',
                    'target_url': 'http://localhost:3000/search',
                    'vulnerability_type': 'XSS',
                    'severity': 'MEDIUM',
                    'confidence': 'FIRM',
                    'payload': '<script>alert(1)</script>',
                    'description': '搜索頁面存在XSS漏洞'
                }
            ]
            
            # 插入數據
            inserted_count = 0
            for data in test_data:
                await conn.execute('''
                    INSERT INTO test_scan_results 
                    (scan_id, target_url, vulnerability_type, severity, confidence, payload, description)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                ''', data['scan_id'], data['target_url'], data['vulnerability_type'], 
                    data['severity'], data['confidence'], data['payload'], data['description'])
                inserted_count += 1
            
            print(f"   ✅ 成功插入 {inserted_count} 筆測試資料")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ 數據插入失敗: {e}")
            return False

    async def test_data_retrieval(self) -> bool:
        """測試數據檢索"""
        try:
            print("🔍 測試數據檢索...")
            
            conn = await asyncpg.connect(**self.postgres_config)
            
            # 檢索數據
            rows = await conn.fetch('SELECT * FROM test_scan_results ORDER BY created_at DESC LIMIT 10')
            
            print(f"   ✅ 成功檢索 {len(rows)} 筆資料")
            
            # 顯示部分資料
            for i, row in enumerate(rows[:3]):
                print(f"   📋 資料 {i+1}: {row['vulnerability_type']} | {row['severity']} | {row['target_url']}")
            
            # 測試條件查詢
            high_severity = await conn.fetch("SELECT COUNT(*) as count FROM test_scan_results WHERE severity = 'HIGH'")
            print(f"   📊 高危漏洞數量: {high_severity[0]['count']}")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"   ❌ 數據檢索失敗: {e}")
            return False

    def test_redis_caching(self) -> bool:
        """測試Redis緩存功能"""
        try:
            print("💾 測試 Redis 緩存功能...")
            
            r = redis.from_url(self.redis_config['url'])
            
            # 測試基本緩存操作
            test_key = f"aiva:test:{uuid.uuid4().hex[:8]}"
            test_data = {
                'scan_id': 'scan_123',
                'target': 'localhost:3000',
                'findings': ['SQL Injection', 'XSS'],
                'timestamp': datetime.now().isoformat()
            }
            
            # 設置緩存
            r.setex(test_key, 300, json.dumps(test_data))  # 5分鐘過期
            print(f"   ✅ 緩存設置成功: {test_key}")
            
            # 讀取緩存
            cached_data = r.get(test_key)
            if cached_data:
                parsed_data = json.loads(cached_data)
                print(f"   ✅ 緩存讀取成功: {len(parsed_data)} 個字段")
            
            # 測試緩存過期
            ttl = r.ttl(test_key)
            print(f"   ⏱️ 緩存過期時間: {ttl} 秒")
            
            # 測試列表緩存 (用於掃描隊列)
            queue_key = f"aiva:scan_queue:{uuid.uuid4().hex[:8]}"
            r.lpush(queue_key, 'scan_task_1', 'scan_task_2', 'scan_task_3')
            queue_length = r.llen(queue_key)
            print(f"   ✅ 隊列緩存: {queue_length} 個任務")
            
            # 清理測試數據
            r.delete(test_key, queue_key)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Redis 緩存測試失敗: {e}")
            return False

    async def test_data_consistency(self) -> bool:
        """測試數據一致性"""
        try:
            print("🔄 測試數據一致性...")
            
            # PostgreSQL 和 Redis 的數據一致性測試
            conn = await asyncpg.connect(**self.postgres_config)
            r = redis.from_url(self.redis_config['url'])
            
            # 從 PostgreSQL 讀取最新掃描結果
            latest_scans = await conn.fetch('''
                SELECT scan_id, vulnerability_type, severity 
                FROM test_scan_results 
                ORDER BY created_at DESC 
                LIMIT 5
            ''')
            
            print(f"   📊 PostgreSQL 最新掃描: {len(latest_scans)} 筆")
            
            # 將掃描結果同步到 Redis
            for scan in latest_scans:
                cache_key = f"aiva:scan_result:{scan['scan_id']}"
                cache_data = {
                    'vulnerability_type': scan['vulnerability_type'],
                    'severity': scan['severity'],
                    'cached_at': datetime.now().isoformat()
                }
                r.setex(cache_key, 3600, json.dumps(cache_data))  # 1小時過期
            
            print(f"   ✅ 同步到 Redis 緩存成功")
            
            # 驗證一致性
            consistent_count = 0
            for scan in latest_scans:
                cache_key = f"aiva:scan_result:{scan['scan_id']}"
                cached_data = r.get(cache_key)
                if cached_data:
                    parsed_data = json.loads(cached_data)
                    if (parsed_data['vulnerability_type'] == scan['vulnerability_type'] and 
                        parsed_data['severity'] == scan['severity']):
                        consistent_count += 1
            
            consistency_rate = (consistent_count / len(latest_scans)) * 100 if latest_scans else 0
            print(f"   📊 數據一致性: {consistency_rate:.1f}% ({consistent_count}/{len(latest_scans)})")
            
            # 清理測試緩存
            for scan in latest_scans:
                r.delete(f"aiva:scan_result:{scan['scan_id']}")
            
            await conn.close()
            return consistency_rate >= 90
            
        except Exception as e:
            print(f"   ❌ 數據一致性測試失敗: {e}")
            return False

    async def cleanup_test_data(self):
        """清理測試數據"""
        try:
            print("🧹 清理測試數據...")
            
            conn = await asyncpg.connect(**self.postgres_config)
            
            # 刪除測試表
            await conn.execute('DROP TABLE IF EXISTS test_scan_results')
            print("   ✅ 測試表已清理")
            
            await conn.close()
            
        except Exception as e:
            print(f"   ⚠️ 清理過程出現警告: {e}")

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """執行綜合數據持久化測試"""
        print("💾 AIVA 數據持久化驗證器")
        print("="*50)
        
        # 1. 測試連接
        self.test_results['postgres_connection'] = await self.test_postgres_connection()
        self.test_results['redis_connection'] = self.test_redis_connection()
        
        if not (self.test_results['postgres_connection'] and self.test_results['redis_connection']):
            print("❌ 基礎連接失敗，停止測試")
            return self.test_results
        
        # 2. 測試表操作
        self.test_results['table_creation'] = await self.test_table_operations()
        
        # 3. 測試數據操作
        self.test_results['data_insertion'] = await self.test_data_insertion()
        self.test_results['data_retrieval'] = await self.test_data_retrieval()
        
        # 4. 測試緩存
        self.test_results['redis_caching'] = self.test_redis_caching()
        
        # 5. 測試一致性
        self.test_results['data_consistency'] = await self.test_data_consistency()
        
        # 6. 清理
        await self.cleanup_test_data()
        
        return self.test_results

    def print_summary(self, results: Dict[str, Any]):
        """打印測試總結"""
        print("\n" + "="*50)
        print("📊 數據持久化測試總結")
        print("="*50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        success_rate = (passed / total) * 100
        
        status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
        
        print(f"\n{status_icon} 總體狀態: {success_rate:.1f}% ({passed}/{total} 項通過)")
        
        print(f"\n📋 詳細結果:")
        test_names = {
            'postgres_connection': 'PostgreSQL 連接',
            'redis_connection': 'Redis 連接', 
            'table_creation': '表格創建',
            'data_insertion': '數據插入',
            'data_retrieval': '數據檢索',
            'redis_caching': 'Redis 緩存',
            'data_consistency': '數據一致性'
        }
        
        for key, result in results.items():
            icon = "✅" if result else "❌"
            name = test_names.get(key, key)
            print(f"   {icon} {name}")
        
        if success_rate >= 80:
            print(f"\n🎉 數據持久化功能運作正常！")
        elif success_rate >= 60:
            print(f"\n⚠️ 數據持久化功能部分正常，建議檢查失敗項目")
        else:
            print(f"\n❌ 數據持久化功能存在嚴重問題，需要立即修復")

async def main():
    """主函數"""
    validator = DataPersistenceValidator()
    
    try:
        results = await validator.run_comprehensive_test()
        validator.print_summary(results)
        
    except Exception as e:
        print(f"❌ 測試過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())