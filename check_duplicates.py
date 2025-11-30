#!/usr/bin/env python3
"""檢查重複文檔"""
from pathlib import Path
import chromadb
from collections import Counter

def main():
    client = chromadb.PersistentClient(path="data/vector_db/chroma")
    coll = client.get_collection("aiva_capabilities")
    
    print(f"Collection: {coll.name}")
    print(f"總文檔數: {coll.count()}")
    
    # 獲取所有文檔
    result = coll.get(include=["metadatas"])
    
    # 檢查重複
    capabilities = []
    for meta in result["metadatas"]:
        cap_name = meta.get("capability_name", "N/A")
        module = meta.get("module", "N/A")
        capabilities.append(f"{module}::{cap_name}")
    
    counter = Counter(capabilities)
    duplicates = {k: v for k, v in counter.items() if v > 1}
    
    print(f"\n重複能力數: {len(duplicates)}")
    print(f"重複文檔總數: {sum(v - 1 for v in duplicates.values())}")
    
    if duplicates:
        print("\n前10個重複最多的:")
        for cap, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cap}: {count} 次")

if __name__ == "__main__":
    main()
