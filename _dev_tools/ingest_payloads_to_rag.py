import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Make sure ChromaDB is installed
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("[ERROR] 尚未安裝 chromadb。請執行: pip install chromadb")
    sys.exit(1)

def chunk_list(data, chunk_size):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def ingest_payloads():
    """將外部武裝庫的 Payload 注入 ChromaDB 向量庫"""
    print("[INFO] --------- AIVA RAG 知識庫注入工具 ---------")
    
    # 建立 / 取得 ChromaDB Client
    db_path = Path(__file__).resolve().parent.parent / "data" / "chroma"
    os.makedirs(db_path, exist_ok=True)
    
    print(f"[INFO] 初始化 ChromaDB (路徑: {db_path})")
    client = chromadb.PersistentClient(path=str(db_path))
    
    # 使用預設的輕量型 Embedding (MiniLM)
    sentence_transformer_ef = embedding_functions.DefaultEmbeddingFunction()

    # 定義 Payload 來源
    tool_dir = Path(__file__).resolve().parent.parent / "tools" / "external_arsenal"
    
    sources = [
        {
            "collection_name": "knowledge_xss_payloads",
            "file_path": tool_dir / "xss-payload-list" / "Intruder" / "xss-payload-list.txt",
            "type": "XSS"
        },
        {
            "collection_name": "knowledge_sqli_payloads",
            "file_path": tool_dir / "sql-injection-payload-list" / "Intruder" / "exploit" / "Generic-SQLi.txt", # Example
            "type": "SQLi"
        }
    ]
    
    for source in sources:
        print(f"\n[INFO] 正在處理 {source['type']} Payload: {source['file_path']}")
        
        # 尋找所有可能的 txt (如果有資料夾結構)
        target_dir = source["file_path"].parent
        if not target_dir.exists():
            print(f"[WARN] 找不到路徑: {target_dir}，跳過。")
            continue
            
        all_txt_files = list(target_dir.rglob("*.txt"))
        if not all_txt_files:
            print(f"[WARN] 找不到任何 txt 檔案在 {target_dir}，跳過。")
            continue

        collection = client.get_or_create_collection(
            name=source["collection_name"],
            embedding_function=sentence_transformer_ef,
            metadata={"description": f"AI {source['type']} Payload Repository"}
        )
        
        all_payloads = []
        for file_path in all_txt_files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
                    all_payloads.extend(lines)
            except Exception as e:
                print(f"[ERROR] 讀取 {file_path} 發生錯誤: {e}")
                
        # 去首尾去重複
        unique_payloads = list(set(all_payloads))
        print(f"[INFO] 總共載入 {len(unique_payloads)} 筆唯一的 {source['type']} Payloads.")
        
        if unique_payloads:
            # 每 5000 筆批次寫入一次以防止 OOM
            batch_size = 5000
            for i, chunk in enumerate(chunk_list(unique_payloads, batch_size)):
                try:
                    # 建立 ID 和 Meta (為了 RAG 能回溯來源)
                    ids = [f"{source['type'].lower()}_{i * batch_size + j}" for j in range(len(chunk))]
                    metadatas = [{"source": "external_arsenal", "type": source['type']} for _ in chunk]
                    
                    collection.upsert(
                        documents=chunk,
                        metadatas=metadatas,
                        ids=ids
                    )
                    print(f"       ✅ 成功寫入批次 {i+1} ({len(chunk)} 筆資料)")
                except Exception as e:
                    print(f"       ❌ 寫入批次 {i+1} 失敗: {e}")
        
    print("\n[INFO] 所有 Payload 已成功向量化並注入資料庫。")
    print("[INFO] AI 現在可以在掃描時語義檢索最適合目標的 Payload 了！")

if __name__ == "__main__":
    ingest_payloads()
