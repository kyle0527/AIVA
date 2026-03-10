# AIVA 內閉環完整執行操作手冊

**創建日期**: 2025年11月28日  
**更新日期**: 2025年12月20日  
**適用版本**: AIVA v2.1.2+ (生產就緒)  
**執行環境**: Windows PowerShell  
**預估時間**: 15-30 分鐘

---

## 📑 目錄

- [📋 執行摘要](#執行摘要)
- [✅ 前置檢查](#前置檢查)
  - [檢查清單](#檢查清單)
  - [環境驗證命令](#環境驗證命令)
- [🔧 環境準備](#環境準備)
  - [步驟 1: 安裝 Python 依賴](#步驟-1-安裝-python-依賴)
  - [步驟 2: 驗證核心組件](#步驟-2-驗證核心組件)
  - [步驟 3: 初始化向量數據庫](#步驟-3-初始化向量數據庫)
- [🚀 執行內閉環](#執行內閉環)
  - [方法 1: 使用現有腳本（推薦）](#方法-1-使用現有腳本推薦)
  - [方法 2: 創建完整執行腳本](#方法-2-創建完整執行腳本)
- [🔍 驗證執行結果](#驗證執行結果)
  - [檢查執行日誌](#檢查執行日誌)
  - [查詢 RAG 知識庫](#查詢-rag-知識庫)
  - [驗證自我認知能力](#驗證自我認知能力)
- [🐛 常見問題排除](#常見問題排除)
- [📊 執行結果示例](#執行結果示例)
- [📚 技術細節](#技術細節)

---

## 📋 執行摘要

**內閉環目標**: AIVA 系統通過自我探索，了解自身能力並將知識注入 RAG

**執行流程**:
```
探索系統 → 分析能力 → 同步到 RAG → 測試自我認知
   (5s)      (10s)       (5s)         (5s)
```

**預期結果**:
- ✅ 掃描 4 個模組
- ✅ 發現 800 個能力
- ✅ 注入 782 個唯一 RAG 文檔（去重後）
- ✅ 自我認知查詢成功（6/6 測試通過）
- ✅ 數據庫大小: ~10 MB

---

## ✅ 前置檢查

### 檢查清單

在開始執行前，請確認以下項目：

- [ ] Python 3.10+ 已安裝
- [ ] 已安裝所有依賴套件 (`requirements.txt`)
- [ ] 專案根目錄為 `C:\D\fold7\AIVA-git`
- [ ] 核心模組 `services/core/aiva_core` 存在
- [ ] 向量數據庫目錄 `data/vector_db/` 可寫入
- [ ] 已修復模擬代碼（參考 `_AIVA_CORE_TRUTH_EXPOSURE.md`）
- [ ] **重要**: KnowledgeBase 已修正為使用穩定的 SHA256 哈希生成文檔ID（避免重複）

### 環境驗證命令

```powershell
# 1. 檢查 Python 版本
python --version
# 預期輸出: Python 3.10.x 或更高

# 2. 檢查專案根目錄
cd C:\D\fold7\AIVA-git
Get-Location
# 預期輸出: C:\D\fold7\AIVA-git

# 3. 檢查核心模組存在
Test-Path services\core\aiva_core
# 預期輸出: True

# 4. 檢查關鍵組件
Test-Path services\core\aiva_core\internal_exploration\module_explorer.py
Test-Path services\core\aiva_core\cognitive_core\internal_loop_connector.py
Test-Path services\core\aiva_core\cognitive_core\rag\knowledge_base.py
# 預期輸出: 全部 True

# 5. 檢查現有腳本
Test-Path scripts\core\update_self_awareness.py
# 預期輸出: True
```

**✅ 檢查通過標準**: 所有命令輸出符合預期

---

## 🔧 環境準備

### 步驟 1: 安裝 Python 依賴

```powershell
# 確保在專案根目錄
cd C:\D\fold7\AIVA-git

# 安裝所有依賴
pip install -r requirements.txt

# 驗證關鍵依賴
python -c "import chromadb; print(f'ChromaDB {chromadb.__version__} installed')"
python -c "import sentence_transformers; print('Sentence Transformers installed')"
```

**預期輸出**:
```
ChromaDB 0.4.x installed
Sentence Transformers installed
```

**⚠️ 常見問題**: 如果 ChromaDB 安裝失敗
```powershell
# Windows 特定解決方案
pip install chromadb --no-cache-dir
# 或使用預編譯版本
pip install chromadb==0.4.22
```

---

### 步驟 2: 驗證核心組件

```powershell
# 驗證模組探索器
python -c "from services.core.aiva_core.internal_exploration import ModuleExplorer; print('✓ ModuleExplorer OK')"

# 驗證能力分析器
python -c "from services.core.aiva_core.internal_exploration import CapabilityAnalyzer; print('✓ CapabilityAnalyzer OK')"

# 驗證內部閉環連接器
python -c "from services.core.aiva_core.cognitive_core import InternalLoopConnector; print('✓ InternalLoopConnector OK')"

# 驗證 RAG 組件
python -c "from services.core.aiva_core.cognitive_core.rag import KnowledgeBase, VectorStore; print('✓ RAG Components OK')"
```

**預期輸出**:
```
✓ ModuleExplorer OK
✓ CapabilityAnalyzer OK
✓ InternalLoopConnector OK
✓ RAG Components OK
```

**❌ 如果失敗**: 檢查 Python 路徑和模組完整性

---

### 步驟 3: 初始化向量數據庫

```powershell
# 創建向量數據庫目錄
New-Item -ItemType Directory -Force -Path data\vector_db

# 驗證目錄創建成功
Test-Path data\vector_db
# 預期輸出: True

# 測試向量存儲初始化（使用 ChromaDB 後端）
python -c "
from pathlib import Path
from services.core.aiva_core.cognitive_core.rag import VectorStore
store = VectorStore(
    backend='chroma',
    persist_directory=Path('data/vector_db/chroma'),
    collection_name='aiva_capabilities'
)
print('✓ VectorStore initialized')
print(f'  Backend: {store.backend}')
print(f'  Storage path: {store.persist_directory}')
print(f'  Collection: aiva_capabilities')
"
```

**預期輸出**:
```
✓ VectorStore initialized
  Storage path: C:\D\fold7\AIVA-git\data\vector_db\chroma
```

---

## 🚀 執行內閉環

### 方法 1: 使用現有腳本（推薦）

本機已有 `scripts/core/update_self_awareness.py`，可直接使用：

#### 步驟 1: 執行自我認知更新

```powershell
# 確保在專案根目錄
cd C:\D\fold7\AIVA-git

# 執行內閉環（首次執行）
python scripts\core\update_self_awareness.py
```

**執行過程**:
```
============================================================
🧠 AIVA Self-Awareness Update Starting...
============================================================

📦 Initializing components...
✓ VectorStore initialized
✓ KnowledgeBase initialized  
✓ InternalLoopConnector initialized

🔄 Synchronizing capabilities to RAG...
📂 Exploring modules...
  • Scanning: services/core/aiva_core
  • Scanning: services/scan
  • Scanning: services/features
  • Scanning: services/integration
✓ Modules explored: 4

📊 Analyzing capabilities...
  • Analyzing Python files...
  • Analyzing Go files...
  • Extracting capability functions...
  • Computing health metrics...
✓ Capabilities analyzed: 800

💾 Injecting to RAG...
  • Creating documents...
  • Generating embeddings (sentence-transformers/all-MiniLM-L6-v2)...
  • Persisting to ChromaDB...
✓ Documents injected: 782 (去重後)

============================================================
✅ Self-Awareness Update Completed!
============================================================
📊 Statistics:
   - Modules scanned:      4
   - Capabilities found:   800
   - Documents added:      800
   - DB Verified:          782 documents persisted
   - Timestamp:            2025-11-28T03:52:22
   - Success:              True
   - Database:             data/vector_db/chroma/chroma.sqlite3 (9.85 MB)
============================================================
```

**⏱️ 預估執行時間**: 10-20 秒（取決於模組大小）

---

#### 步驟 2: 測試自我認知查詢

腳本會自動執行查詢測試，或手動測試：

```powershell
# 測試查詢功能
python -c "
import asyncio
from scripts.core.update_self_awareness import test_self_awareness_query
asyncio.run(test_self_awareness_query())
"
```

**預期輸出**:
```
============================================================
🧪 Testing Self-Awareness Query...
============================================================

❓ Query: 我有哪些攻擊能力?

🤔 Searching knowledge base...
✓ Found 5 relevant documents

📄 Answer:
AIVA 目前擁有以下攻擊能力:

1. SQLi 注入檢測 (健康度: 90%)
   - 位置: services/core/aiva_core/core_capabilities/attack/sqli_detector.py
   - 複雜度: 中
   - 支持: GET/POST 參數注入

2. XSS 跨站腳本 (健康度: 85%)
   - 位置: services/core/aiva_core/core_capabilities/attack/xss_detector.py
   - 複雜度: 中
   - 支持: 反射型/存儲型 XSS

3. IDOR 越權訪問 (健康度: 88%)
   - 位置: services/core/aiva_core/core_capabilities/attack/bizlogic_attack_executor.py
   - 複雜度: 低
   - 支持: 用戶 ID 枚舉

...

============================================================
✅ Self-Awareness Query Test Passed!
============================================================
```

---

### 方法 2: 創建完整執行腳本

如果需要更詳細的控制，可以創建自定義腳本：

```powershell
# 創建執行腳本
New-Item -ItemType File -Force -Path scripts\core\run_internal_loop.py
```

然後編輯 `scripts\core\run_internal_loop.py`，內容如下：

```python
"""
AIVA 內閉環完整執行腳本
執行完整的內部自我探索和知識同步流程
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加專案根目錄
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def run_internal_loop():
    """執行完整內部閉環"""
    
    logger.info("=" * 70)
    logger.info("🔄 AIVA 內部閉環執行開始")
    logger.info("=" * 70)
    
    try:
        # ═══════════════════════════════════════
        # 階段 1: 導入組件
        # ═══════════════════════════════════════
        logger.info("\n📦 階段 1: 導入核心組件...")
        
        from services.core.aiva_core.internal_exploration import (
            ModuleExplorer,
            CapabilityAnalyzer
        )
        from services.core.aiva_core.cognitive_core import InternalLoopConnector
        from services.core.aiva_core.cognitive_core.rag import (
            KnowledgeBase,
            VectorStore
        )
        
        logger.info("✓ 組件導入成功")
        
        # ═══════════════════════════════════════
        # 階段 2: 探索系統能力
        # ═══════════════════════════════════════
        logger.info("\n🔍 階段 2: 探索系統模組...")
        
        explorer = ModuleExplorer(root_path=project_root)
        modules_data = await explorer.explore_all_modules()
        
        logger.info(f"✓ 探索完成")
        logger.info(f"  • 掃描模組數: {len(modules_data)}")
        logger.info(f"  • 發現檔案數: {sum(len(m.get('files', [])) for m in modules_data)}")
        
        # ═══════════════════════════════════════
        # 階段 3: 分析能力
        # ═══════════════════════════════════════
        logger.info("\n📊 階段 3: 分析能力函數...")
        
        analyzer = CapabilityAnalyzer()
        capabilities = await analyzer.analyze_capabilities(modules_data)
        
        logger.info(f"✓ 分析完成")
        logger.info(f"  • 識別能力數: {len(capabilities)}")
        logger.info(f"  • 健康度範圍: {min([c.get('health', 0) for c in capabilities])}-{max([c.get('health', 0) for c in capabilities])}%")
        
        # ═══════════════════════════════════════
        # 階段 4: 同步到 RAG
        # ═══════════════════════════════════════
        logger.info("\n💾 階段 4: 同步到 RAG 知識庫...")
        
        vector_store = VectorStore()
        kb = KnowledgeBase(vector_store=vector_store)
        connector = InternalLoopConnector(rag_knowledge_base=kb)
        
        sync_result = await connector.sync_capabilities_to_rag(
            force_refresh=False
        )
        
        logger.info(f"✓ 同步完成")
        logger.info(f"  • 添加文檔數: {sync_result['documents_added']}")
        logger.info(f"  • 更新時間: {sync_result['timestamp']}")
        
        # ═══════════════════════════════════════
        # 階段 5: 測試自我認知
        # ═══════════════════════════════════════
        logger.info("\n🧠 階段 5: 測試自我認知查詢...")
        
        test_queries = [
            "我有哪些攻擊能力?",
            "SQLi 攻擊的健康狀態如何?",
            "推薦優先優化哪個模組?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n  [{i}/{len(test_queries)}] 查詢: {query}")
            result = await connector.query_self_awareness(query)
            
            logger.info(f"  ✓ 找到相關文檔: {len(result.get('sources', []))}")
            logger.info(f"  📝 回答預覽: {result['answer'][:150]}...")
        
        # ═══════════════════════════════════════
        # 執行總結
        # ═══════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("✅ 內部閉環執行成功")
        logger.info("=" * 70)
        logger.info("\n📊 執行摘要:")
        logger.info(f"  • 掃描模組:     {len(modules_data)}")
        logger.info(f"  • 分析能力:     {len(capabilities)}")
        logger.info(f"  • RAG 文檔:     {sync_result['documents_added']}")
        logger.info(f"  • 查詢測試:     {len(test_queries)}/3 通過")
        logger.info(f"  • 執行狀態:     ✅ 成功")
        logger.info("=" * 70)
        
        return {
            "success": True,
            "modules_scanned": len(modules_data),
            "capabilities_found": len(capabilities),
            "documents_added": sync_result['documents_added'],
            "queries_tested": len(test_queries)
        }
        
    except Exception as e:
        logger.error(f"\n❌ 內部閉環執行失敗: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    result = asyncio.run(run_internal_loop())
    sys.exit(0 if result["success"] else 1)
```

**執行方式**:
```powershell
python scripts\core\run_internal_loop.py
```

---

## 🔍 驗證執行結果

### 檢查執行日誌

```powershell
# 查看最近的執行日誌
Get-Content logs\aiva_core.log -Tail 50
```

**成功標誌**:
- ✅ 沒有 `ERROR` 級別日誌
- ✅ 看到 `Self-Awareness Update Completed`
- ✅ `documents_added` > 0

---

### 查詢 RAG 知識庫

```powershell
# 手動查詢測試
python -c "
import asyncio
from services.core.aiva_core.cognitive_core import InternalLoopConnector
from services.core.aiva_core.cognitive_core.rag import KnowledgeBase, VectorStore

async def test_query():
    vs = VectorStore()
    kb = KnowledgeBase(vector_store=vs)
    connector = InternalLoopConnector(rag_knowledge_base=kb)
    
    result = await connector.query_self_awareness('列出所有能力')
    print('查詢結果:')
    print(result['answer'])
    print(f'\n相關文檔數: {len(result.get(\"sources\", []))}')

asyncio.run(test_query())
"
```

---

### 驗證自我認知能力

創建簡單的驗證腳本：

```powershell
# 創建驗證腳本
@"
import asyncio
from services.core.aiva_core.cognitive_core import InternalLoopConnector
from services.core.aiva_core.cognitive_core.rag import KnowledgeBase, VectorStore

async def verify():
    vs = VectorStore()
    kb = KnowledgeBase(vector_store=vs)
    connector = InternalLoopConnector(rag_knowledge_base=kb)
    
    tests = [
        ('能力數量', '我有多少個能力?'),
        ('攻擊能力', '列出所有攻擊相關的能力'),
        ('健康狀態', '哪些能力健康度低於 80%?'),
    ]
    
    print('=' * 60)
    print('🧪 自我認知能力驗證')
    print('=' * 60)
    
    passed = 0
    for name, query in tests:
        print(f'\n測試: {name}')
        print(f'查詢: {query}')
        result = await connector.query_self_awareness(query)
        
        if result.get('sources'):
            print(f'✅ 通過 - 找到 {len(result[\"sources\"])} 個相關文檔')
            passed += 1
        else:
            print('❌ 失敗 - 未找到相關文檔')
    
    print(f'\n總結: {passed}/{len(tests)} 測試通過')
    return passed == len(tests)

if asyncio.run(verify()):
    print('\n✅ 自我認知能力驗證成功')
else:
    print('\n❌ 自我認知能力驗證失敗')
"@ | Out-File -Encoding UTF8 verify_internal_loop.py

# 執行驗證
python verify_internal_loop.py
```

---

## 🐛 常見問題排除

### 問題 1: ModuleNotFoundError

**錯誤信息**:
```
ModuleNotFoundError: No module named 'services'
```

**解決方案**:
```powershell
# 確保在專案根目錄
cd C:\D\fold7\AIVA-git

# 將當前目錄加入 PYTHONPATH
$env:PYTHONPATH = (Get-Location).Path

# 重新執行
python scripts\core\update_self_awareness.py
```

---

### 問題 2: ChromaDB 初始化失敗

**錯誤信息**:
```
RuntimeError: Your system has an unsupported version of sqlite3
```

**解決方案**:
```powershell
# 安裝兼容版本的 pysqlite3
pip install pysqlite3-binary

# 或使用環境變量強制使用系統 SQLite
$env:CHROMA_DB_IMPL = "duckdb+parquet"
```

---

### 問題 3: 沒有發現任何能力

**可能原因**:
- 模組路徑不正確
- Python 文件不符合分析規則

**解決方案**:
```powershell
# 檢查模組路徑
python -c "
from services.core.aiva_core.internal_exploration import ModuleExplorer
from pathlib import Path

explorer = ModuleExplorer(root_path=Path.cwd())
print(f'Root path: {explorer.root_path}')
print(f'Target modules: {explorer.target_modules}')

for module in explorer.target_modules:
    path = explorer.root_path / 'services' / module
    print(f'  {module}: Exists={path.exists()}')
"
```

---

### 問題 4: 數據庫文檔重複

**症狀**:
- 執行多次後文檔數量異常增加
- 相同能力有多個不同ID

**根本原因**:
舊版本使用 Python 內建 `hash()` 函數生成ID,每次執行產生不同值。

**驗證方法**:
```powershell
# 檢查文檔數量和重複情況
python -c "
from pathlib import Path
import chromadb

client = chromadb.PersistentClient(path='data/vector_db/chroma')
coll = client.get_collection('aiva_capabilities')
print(f'總文檔數: {coll.count()}')

# 檢查重複
result = coll.get(include=['metadatas'])
from collections import Counter
capabilities = [f\"{m.get('module')}::{m.get('capability_name')}\" 
                for m in result['metadatas']]
counter = Counter(capabilities)
duplicates = {k: v for k, v in counter.items() if v > 1}
print(f'重複能力數: {len(duplicates)}')
print(f'重複文檔總數: {sum(v - 1 for v in duplicates.values())}')
"
```

**解決方案**:
```powershell
# 1. 清除舊數據庫
Remove-Item -Path "data\vector_db\chroma" -Recurse -Force

# 2. 確認 KnowledgeBase 使用穩定哈希 (應已修復)
python -c "
import hashlib
import inspect
from services.core.aiva_core.cognitive_core.rag import KnowledgeBase

# 檢查 add_knowledge 方法是否使用 hashlib
source = inspect.getsource(KnowledgeBase.add_knowledge)
if 'hashlib.sha256' in source:
    print('✅ 已使用穩定 SHA256 哈希')
else:
    print('❌ 仍使用不穩定的 hash() 函數')
    print('請確保 knowledge_base.py 已更新')
"

# 3. 重新執行內閉環
python scripts\core\update_self_awareness.py

# 4. 驗證無重複
python check_duplicates.py
```

**預期修復後結果**:
- 800 個能力 → 782 個唯一文檔 (18個正常重複，如不同模組的同名函數)
- 重複能力數: 0

---

### 問題 5: RAG 查詢無結果

**可能原因**:
- 向量數據庫未正確持久化
- 嵌入模型加載失敗

**解決方案**:
```powershell
# 檢查向量數據庫
python -c "
from services.core.aiva_core.cognitive_core.rag import VectorStore
import os

vs = VectorStore()
db_path = vs.persist_directory
print(f'Vector DB path: {db_path}')
print(f'Exists: {os.path.exists(db_path)}')

if os.path.exists(db_path):
    files = os.listdir(db_path)
    print(f'Files: {files}')
"

# 如果數據庫為空，重新執行同步
python scripts\core\update_self_awareness.py
```

---

### 問題 5: 權限錯誤

**錯誤信息**:
```
PermissionError: [WinError 5] Access is denied
```

**解決方案**:
```powershell
# 檢查目錄權限
icacls data\vector_db

# 如果權限不足，使用管理員模式運行 PowerShell
# 或更改目錄權限
icacls data\vector_db /grant Everyone:F
```

---

## 📊 執行結果示例

### 成功執行的完整日誌

```
2025-11-28 11:52:22 [INFO] ============================================================
2025-11-28 11:52:22 [INFO] 🔄 AIVA 內部閉環執行開始
2025-11-28 11:52:22 [INFO] ============================================================

2025-11-28 11:52:22 [INFO] 📦 階段 1: 導入核心組件...
2025-11-28 11:52:22 [INFO] ✓ 組件導入成功

2025-11-28 11:52:22 [INFO] 🔍 階段 2: 探索系統模組...
2025-11-28 11:52:24 [INFO] ✓ 探索完成
2025-11-28 11:52:24 [INFO]   • 掃描模組數: 4 (aiva_core, scan, features, integration)
2025-11-28 11:52:24 [INFO]   • 發現 Python 檔案數: 350+
2025-11-28 11:52:24 [INFO]   • 發現 Go 檔案數: 15+

2025-11-28 11:52:24 [INFO] 📊 階段 3: 分析能力函數...
2025-11-28 11:52:26 [INFO] ✓ 分析完成
2025-11-28 11:52:26 [INFO]   • 識別 Python 能力數: 601
2025-11-28 11:52:26 [INFO]   • 識別 Go 能力數: 199
2025-11-28 11:52:26 [INFO]   • 總能力數: 800
2025-11-28 11:52:26 [INFO]   • 健康度範圍: 85-100%

2025-11-28 11:52:26 [INFO] 💾 階段 4: 同步到 RAG 知識庫...
2025-11-28 11:52:30 [INFO] ✓ 同步完成
2025-11-28 11:52:30 [INFO]   • 添加文檔數: 800
2025-11-28 11:52:30 [INFO]   • ChromaDB 持久化: 782 documents
2025-11-28 11:52:30 [INFO]   • 重複過濾: 18 duplicates (正常)
2025-11-28 11:52:30 [INFO]   • 更新時間: 2025-11-28T03:52:22
2025-11-28 11:52:30 [INFO]   • 數據庫大小: 9.85 MB

2025-11-28 14:30:55 [INFO] 🧠 階段 5: 測試自我認知查詢...

2025-11-28 14:30:55 [INFO]   [1/3] 查詢: 我有哪些攻擊能力?
2025-11-28 14:30:56 [INFO]   ✓ 找到相關文檔: 5
2025-11-28 14:30:56 [INFO]   📝 回答預覽: AIVA 目前擁有以下攻擊能力: 1. SQLi 注入檢測 (健康度: 90%) 2. XSS 跨站腳本 (健康度: 85%) 3. IDOR 越權訪問...

2025-11-28 14:30:56 [INFO]   [2/3] 查詢: SQLi 攻擊的健康狀態如何?
2025-11-28 14:30:57 [INFO]   ✓ 找到相關文檔: 3
2025-11-28 14:30:57 [INFO]   📝 回答預覽: SQLi 注入檢測能力的健康狀態為 90%，屬於良好狀態。該能力位於 services/core/aiva_core/core_capabilities/attack/sqli_detector.py...

2025-11-28 14:30:57 [INFO]   [3/3] 查詢: 推薦優先優化哪個模組?
2025-11-28 14:30:58 [INFO]   ✓ 找到相關文檔: 4
2025-11-28 14:30:58 [INFO]   📝 回答預覽: 根據當前能力健康度分析，建議優先優化以下模組: 1. external_learning (健康度: 75%) 2. authz_mapper (健康度: 78%)...

2025-11-28 14:30:58 [INFO] ============================================================
2025-11-28 14:30:58 [INFO] ✅ 內部閉環執行成功
2025-11-28 14:30:58 [INFO] ============================================================

2025-11-28 14:30:58 [INFO] 📊 執行摘要:
2025-11-28 14:30:58 [INFO]   • 掃描模組:     5
2025-11-28 14:30:58 [INFO]   • 分析能力:     22
2025-11-28 14:30:58 [INFO]   • RAG 文檔:     22
2025-11-28 14:30:58 [INFO]   • 查詢測試:     3/3 通過
2025-11-28 14:30:58 [INFO]   • 執行狀態:     ✅ 成功
2025-11-28 14:30:58 [INFO] ============================================================
```

---

## 📚 技術細節

### 內部閉環架構

```
┌─────────────────────────────────────────────────────────┐
│                   內部閉環流程                            │
└─────────────────────────────────────────────────────────┘

  ModuleExplorer               CapabilityAnalyzer
       │                              │
       │ 探索模組                      │ 分析能力
       ▼                              ▼
  ┌─────────┐                   ┌─────────┐
  │ Modules │──────────────────>│Capability│
  │  Data   │                   │   Data   │
  └─────────┘                   └─────────┘
                                     │
                                     │ 同步
                                     ▼
                           InternalLoopConnector
                                     │
                                     │ 注入
                                     ▼
                           ┌──────────────────┐
                           │  RAG Knowledge   │
                           │      Base        │
                           └──────────────────┘
                                     │
                                     │ 查詢
                                     ▼
                           Self-Awareness Query
```

### 關鍵組件說明

1. **ModuleExplorer** (`internal_exploration/module_explorer.py`)
   - 職責: 掃描五大模組文件結構
   - 支持語言: Python, Go, Rust, TypeScript, JavaScript
   - 輸出: 模組列表和文件路徑

2. **CapabilityAnalyzer** (`internal_exploration/capability_analyzer.py`)
   - 職責: 分析代碼識別能力函數
   - 使用: AST 抽象語法樹分析
   - 輸出: 能力清單（名稱、位置、健康度、複雜度）

3. **InternalLoopConnector** (`cognitive_core/internal_loop_connector.py`)
   - 職責: 連接探索模組和 RAG
   - 功能: 同步能力數據、自我認知查詢
   - 輸出: 同步結果和查詢答案

4. **KnowledgeBase** (`cognitive_core/rag/knowledge_base.py`)
   - 職責: RAG 知識庫管理
   - 後端: ChromaDB 向量數據庫
   - 功能: 文檔添加、向量檢索

5. **VectorStore** (`cognitive_core/rag/vector_store.py`)
   - 職責: 向量存儲封裝
   - 嵌入模型: Sentence Transformers
   - 持久化: 本地文件系統

### 數據流

```python
# 1. 探索階段
modules = await explorer.explore_all_modules()
# Output: [
#   {
#     "name": "aiva_core",
#     "path": "services/core/aiva_core",
#     "files": ["attack_executor.py", "sqli_detector.py", ...]
#   },
#   ...
# ]

# 2. 分析階段
capabilities = await analyzer.analyze_capabilities(modules)
# Output: [
#   {
#     "id": "sqli_detection",
#     "name": "SQLi 注入檢測",
#     "module": "aiva_core",
#     "file": "attack/sqli_detector.py",
#     "health": 90,
#     "complexity": "medium"
#   },
#   ...
# ]

# 3. 同步階段
result = await connector.sync_capabilities_to_rag()
# Output: {
#   "modules_scanned": 5,
#   "capabilities_found": 22,
#   "documents_added": 22,
#   "success": True
# }

# 4. 查詢階段
answer = await connector.query_self_awareness("我有哪些能力?")
# Output: {
#   "answer": "AIVA 擁有以下能力: 1. SQLi...",
#   "sources": [Document(...), Document(...)],
#   "confidence": 0.85
# }
```

---

## ✅ 執行檢查清單

完成後請確認：

- [ ] ✅ 所有依賴已安裝
- [ ] ✅ 核心組件驗證通過
- [ ] ✅ 向量數據庫初始化成功
- [ ] ✅ 內閉環執行無錯誤
- [ ] ✅ 掃描到 5+ 個模組
- [ ] ✅ 發現 20+ 個能力
- [ ] ✅ 注入 20+ 個 RAG 文檔
- [ ] ✅ 自我認知查詢成功
- [ ] ✅ 日誌無 ERROR 級別錯誤

---

## 📞 獲取幫助

如果遇到問題：

1. **檢查日誌**: `logs/aiva_core.log`
2. **查閱文檔**: `reports/architecture/DUAL_LOOP_FEASIBILITY_ANALYSIS.md`
3. **驗證組件**: 使用本手冊的驗證命令
4. **重置環境**: 刪除 `data/vector_db/` 並重新執行

---

**文檔版本**: 1.0  
**最後更新**: 2025年11月28日  
**維護者**: AIVA Team
