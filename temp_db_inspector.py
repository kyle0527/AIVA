#!/usr/bin/env python3
"""Temporary script to inspect contract_metrics.db structure."""

import sqlite3
import os

def inspect_database():
    """檢查資料庫結構和內容."""
    db_path = "logs/contract_metrics.db"
    
    if not os.path.exists(db_path):
        print(f"資料庫檔案不存在: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # 檢查表格
        print("=== 資料庫表格 ===")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        for table in tables:
            print(f"表格: {table[0]}")
            
            # 檢查表格結構
            cur.execute(f"PRAGMA table_info({table[0]})")
            columns = cur.fetchall()
            for col in columns:
                print(f"  欄位: {col[1]} ({col[2]})")
            
            # 檢查資料數量
            cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cur.fetchone()[0]
            print(f"  記錄數: {count}")
            
            # 顯示最新的幾筆資料
            if count > 0:
                print("  最新 3 筆資料:")
                cur.execute(f"SELECT * FROM {table[0]} ORDER BY rowid DESC LIMIT 3")
                rows = cur.fetchall()
                for row in rows:
                    print(f"    {row}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"資料庫檢查錯誤: {e}")

if __name__ == "__main__":
    inspect_database()