"""
RAG 經驗同步模組

將 services/integration/data/experiences/*.jsonl 的執行經驗記錄
同步到向量庫，使 RAG 能夠搜索到這些數據。

位置: cognitive_core/rag/sync_experiences.py
版本: v1.0 (2026-01-21 移入 aiva_core)

使用方式:
    from services.core.aiva_core.cognitive_core.rag.sync_experiences import sync_experiences_to_vector_store
    await sync_experiences_to_vector_store(capabilities=["xss", "sqli"])
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

# 設置日誌
logger = logging.getLogger(__name__)


async def sync_experiences_to_vector_store(
    capabilities: Optional[List[str]] = None,
    limit: int = 1000,
):
    """同步執行經驗到向量庫
    
    Args:
        capabilities: 要同步的能力列表（None 表示全部）
        limit: 每個能力同步的記錄數量限制
    """
    try:
        from services.integration.simple_data_manager import get_data_manager
        from .vector_store import VectorStore  # 使用相對導入
    except ImportError as e:
        logger.error(f"導入模組失敗: {e}")
        logger.error("請確保已安裝所有依賴: pip install -r requirements.txt")
        return
    
    # 初始化管理器
    logger.info("初始化數據管理器和向量庫...")
    data_manager = get_data_manager()
    vector_store = VectorStore(
        backend="chroma",
        persist_directory=Path("data/vector_db/chroma")
    )
    
    # 默認要同步的能力
    if capabilities is None:
        capabilities = ["xss", "sqli", "ssrf", "phase0", "phase1"]
    
    logger.info(f"準備同步能力: {capabilities}")
    logger.info(f"每個能力限制: {limit} 條記錄\n")
    
    total_synced = 0
    
    for capability in capabilities:
        logger.info(f"{'='*60}")
        logger.info(f"同步能力: {capability}")
        logger.info(f"{'='*60}")
        
        try:
            # 讀取執行經驗記錄
            logger.info(f"從整合模組讀取 {capability} 記錄...")
            records = data_manager.load_capability_data(
                capability=capability,
                limit=limit
            )
            
            if not records:
                logger.warning(f"⚠️ {capability}: 沒有找到記錄")
                continue
            
            logger.info(f"✅ 讀取到 {len(records)} 條記錄")
            
            # 逐條添加到向量庫
            synced_count = 0
            for i, record in enumerate(records, 1):
                try:
                    # 構建文本描述（用於向量化）
                    text_parts = [
                        f"能力類型: {record.get('capability', 'unknown')}",
                    ]
                    
                    # 目標信息
                    if 'target' in record:
                        target = record['target']
                        if isinstance(target, dict):
                            text_parts.append(f"目標: {target.get('url', target.get('host', 'N/A'))}")
                        else:
                            text_parts.append(f"目標: {target}")
                    
                    # 請求信息
                    if 'request' in record:
                        req = record['request']
                        if isinstance(req, dict):
                            text_parts.append(f"請求方法: {req.get('method', 'N/A')}")
                            if 'parameters' in req:
                                text_parts.append(f"參數: {str(req['parameters'])[:100]}")
                    
                    # 響應信息
                    if 'response' in record:
                        resp = record['response']
                        if isinstance(resp, dict):
                            text_parts.append(f"響應狀態: {resp.get('status_code', 'N/A')}")
                            if 'error_message' in resp:
                                text_parts.append(f"錯誤訊息: {resp['error_message']}")
                            if 'response_type' in resp:
                                text_parts.append(f"響應類型: {resp['response_type']}")
                    
                    # 執行結果
                    if 'result' in record:
                        result = record['result']
                        if isinstance(result, dict):
                            success = result.get('success', False)
                            text_parts.append(f"執行結果: {'成功' if success else '失敗'}")
                            if 'findings' in result:
                                findings_count = len(result['findings'])
                                text_parts.append(f"發現數量: {findings_count}")
                                if findings_count > 0:
                                    # 添加發現摘要
                                    for finding in result['findings'][:2]:  # 只添加前 2 個
                                        if isinstance(finding, dict):
                                            text_parts.append(f"發現: {finding.get('description', 'N/A')}")
                    
                    # 元數據信息
                    if 'metadata' in record:
                        meta = record['metadata']
                        if isinstance(meta, dict) and 'execution_time' in meta:
                            text_parts.append(f"執行時間: {meta['execution_time']}s")
                    
                    text = "\n".join(text_parts)
                    
                    # 生成文檔 ID
                    doc_id = f"exp_{capability}_{record.get('timestamp', i)}"
                    
                    # 準備元數據
                    metadata = {
                        "type": "experience",
                        "capability": capability,
                        "timestamp": record.get('timestamp', ''),
                        "task_id": record.get('task_id', ''),
                        "success": record.get('result', {}).get('success', False) if isinstance(record.get('result'), dict) else False,
                        "source": "integration_module",
                    }
                    
                    # 添加到向量庫
                    await vector_store.add_document(
                        doc_id=doc_id,
                        text=text,
                        metadata=metadata
                    )
                    
                    synced_count += 1
                    
                    # 每 50 條輸出進度
                    if synced_count % 50 == 0:
                        logger.info(f"  進度: {synced_count}/{len(records)} ({synced_count/len(records)*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"  ❌ 添加記錄失敗 (記錄 {i}): {e}")
                    continue
            
            logger.info(f"✅ {capability}: 成功同步 {synced_count}/{len(records)} 條記錄\n")
            total_synced += synced_count
        
        except Exception as e:
            logger.error(f"❌ {capability}: 同步失敗: {e}\n")
            continue
    
    logger.info(f"{'='*60}")
    logger.info(f"同步完成！總計同步 {total_synced} 條記錄")
    logger.info(f"{'='*60}")
    
    # 顯示向量庫統計
    try:
        if vector_store.backend == "chroma" and vector_store.collection:
            count = vector_store.collection.count()
            logger.info(f"\n向量庫總記錄數: {count}")
    except Exception as e:
        logger.warning(f"無法獲取向量庫統計: {e}")


async def sync_knowledge_base_to_vector_store():
    """同步知識庫 Markdown 報告到向量庫"""
    try:
        from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
    except ImportError as e:
        logger.error(f"導入模組失敗: {e}")
        return
    
    logger.info("\n" + "="*60)
    logger.info("同步知識庫分析報告到向量庫")
    logger.info("="*60)
    
    # 初始化向量庫
    vector_store = VectorStore(
        backend="chroma",
        persist_directory=Path("data/vector_db/chroma")
    )
    
    # 知識庫目錄
    knowledge_dir = Path("services/core/aiva_core/cognitive_core/learning_system/knowledge")
    
    if not knowledge_dir.exists():
        logger.warning(f"知識庫目錄不存在: {knowledge_dir}")
        return
    
    # 查找所有 Markdown 分析報告
    md_files = list(knowledge_dir.glob("*_ANALYSIS.md"))
    
    if not md_files:
        logger.warning("沒有找到知識庫分析報告")
        return
    
    logger.info(f"找到 {len(md_files)} 個知識庫報告")
    
    synced_count = 0
    for md_file in md_files:
        try:
            logger.info(f"\n處理: {md_file.name}")
            
            # 讀取內容
            content = md_file.read_text(encoding="utf-8")
            
            # 提取能力類型（從文件名）
            # 例如: XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md → xss
            capability = md_file.stem.split('_')[0].lower()
            
            # 生成文檔 ID
            doc_id = f"kb_{md_file.stem}"
            
            # 準備元數據
            metadata = {
                "type": "knowledge_base",
                "capability": capability,
                "format": "markdown",
                "filename": md_file.name,
                "source": "knowledge_base",
            }
            
            # 添加到向量庫
            await vector_store.add_document(
                doc_id=doc_id,
                text=content,
                metadata=metadata
            )
            
            synced_count += 1
            logger.info(f"✅ 成功添加: {md_file.name}")
        
        except Exception as e:
            logger.error(f"❌ 添加失敗: {md_file.name}: {e}")
            continue
    
    logger.info(f"\n✅ 知識庫同步完成: {synced_count}/{len(md_files)} 個報告")


async def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="同步整合模組數據到向量庫")
    parser.add_argument(
        "--capability",
        type=str,
        help="指定要同步的能力（例如: xss, sqli）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="每個能力同步的記錄數量限制（默認: 1000）"
    )
    parser.add_argument(
        "--sync-knowledge-base",
        action="store_true",
        help="同步知識庫 Markdown 報告"
    )
    
    args = parser.parse_args()
    
    # 同步執行經驗
    if args.capability:
        capabilities = [args.capability]
    else:
        capabilities = None  # 全部
    
    await sync_experiences_to_vector_store(
        capabilities=capabilities,
        limit=args.limit
    )
    
    # 同步知識庫（如果指定）
    if args.sync_knowledge_base:
        await sync_knowledge_base_to_vector_store()


if __name__ == "__main__":
    asyncio.run(main())
