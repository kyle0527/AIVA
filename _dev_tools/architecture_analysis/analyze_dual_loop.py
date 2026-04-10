"""分析雙閉環系統的數據流和預期產出"""
import asyncio
from services.core.aiva_core.cognitive_core.rag.knowledge_base import KnowledgeBase
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

async def main():
    vs = VectorStore()
    kb = KnowledgeBase(vs)
    
    print("=" * 80)
    print("雙閉環系統數據分析")
    print("=" * 80)
    
    # 查詢不同類型的能力
    queries = [
        ("掃描能力", "scan"),
        ("攻擊能力", "attack exploit"),
        ("分析能力", "analyze parse"),
        ("報告能力", "report generate")
    ]
    for desc, query in queries:
        print(f"\n查詢: {desc} ('{query}')")
        results = kb.search(query=query, top_k=5)
        print(f"找到 {len(results)} 個結果:")
        for i, r in enumerate(results[:5], 1):
            metadata = r.get("metadata", {})
            cap_name = metadata.get("capability_name", "N/A")
            module = metadata.get("module", "N/A")
            language = metadata.get("language", "unknown")
            description = metadata.get("description", "N/A")[:50]
            score = r.get("relevance_score", 0.0)
            print(f"  {i}. [{language}] {module}::{cap_name} (相似度: {score:.3f})")
            print(f"     描述: {description}...")}::{cap_name}")
            print(f"     描述: {description}...")
    
    print("\n" + "=" * 80)
    print("內閉環探索後預計得到的資料:")
    print("=" * 80)
    print("""
1. 能力清單 (ModuleCapability):
   - 782 個能力函數的完整資訊
   - 包含: 名稱、模組路徑、描述、參數、返回值
   - 分類: scanning/attacking/analysis/utility/reporting/integration
   - 健康狀態: health_score, availability, error_rate

2. 能力分類統計 (CapabilitySummary):
   - 總能力數: 782
   - 按類別統計: scan(286), core(207), integration(111), features(98)
   - 按語言統計: Python(495), Rust(123), TypeScript(84), Go(80)
   - 健康度統計: 平均健康分數

3. RAG 知識庫文檔:
   - 782 份向量化文檔
   - 支援語義搜索
   - 包含元數據: 模組、語言、複雜度、標籤

這些資料的用途:
- AI 查詢自己有哪些能力 → 知道「我有什麼」
- 規劃任務時選擇合適的能力 → 知道「該用什麼」
- 評估能力健康度 → 知道「哪些可用、哪些有問題」
- 外閉環執行時調用這些能力 → 實戰應用
- 從實戰結果中學習優化 → 持續進化
""")

if __name__ == "__main__":
    asyncio.run(main())
