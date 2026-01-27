"""RAG Trigger - RAG 觸發器

當學習系統遇到未知情況時，自動觸發 RAG 搜索外部知識。

觸發條件：
1. 當前數據與三個數據源（當前記錄、歷史數據、知識庫）的相似度都低於閾值
2. 遇到未見過的錯誤訊息或異常響應
3. 檢測到新型攻擊模式或防禦機制

RAG 搜索範圍：
- CVE 資料庫
- 技術文檔和安全研究報告
- 已有知識庫（向量搜索）
- 外部安全資源

用戶通知：
- 當觸發 RAG 時，即時通知用戶系統遇到未知情況
- 提供搜索狀態和發現的相關資源
"""

import logging
from datetime import datetime
from typing import Any, Optional
from difflib import SequenceMatcher

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

logger = logging.getLogger(__name__)


class UnknownSituationAlert:
    """未知情況警報
    
    當觸發 RAG 時生成的警報信息
    """
    
    def __init__(
        self,
        alert_id: str,
        trigger_reason: str,
        data_snapshot: dict[str, Any],
        search_query: str,
        timestamp: datetime,
    ):
        self.alert_id = alert_id
        self.trigger_reason = trigger_reason
        self.data_snapshot = data_snapshot
        self.search_query = search_query
        self.timestamp = timestamp
        self.status = "searching"  # searching, found, not_found
        self.rag_results: list[dict[str, Any]] = []
    
    def to_dict(self) -> dict[str, Any]:
        """轉換為字典"""
        return {
            "alert_id": self.alert_id,
            "trigger_reason": self.trigger_reason,
            "data_snapshot": self.data_snapshot,
            "search_query": self.search_query,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "rag_results_count": len(self.rag_results),
            "rag_results": self.rag_results,
        }


class RAGTrigger:
    """RAG 觸發器
    
    支持對內和對外雙向搜索：
    - 對內：搜索本地向量庫（能力記錄、執行經驗、知識庫）
    - 對外：搜索外部資源（CVE、Exploit-DB、Google、GitHub）
    
    搜索策略：
    - hybrid（默認）：先對內搜索，相似度低時觸發對外搜索
    - internal：只搜索內部向量庫
    - external：只搜索外部資源
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        vector_store=None,  # VectorStore 實例（對內搜索）
        enable_external_search: bool = True,  # 是否啟用對外搜索
        notification_callback=None,  # 用戶通知回調函數
    ):
        """初始化 RAG 觸發器
        
        Args:
            similarity_threshold: 相似度閾值（低於此值觸發外部搜索）
            vector_store: VectorStore 實例（用於內部向量搜索）
            enable_external_search: 是否啟用外部搜索
            notification_callback: 用戶通知回調函數 callback(alert: UnknownSituationAlert)
        """
        self.similarity_threshold = similarity_threshold
        self.vector_store = vector_store  # 內部向量庫
        self.enable_external_search = enable_external_search
        self.notification_callback = notification_callback
        self._alert_history: list[UnknownSituationAlert] = []
        
        logger.info(
            f"RAG Trigger initialized: similarity_threshold={similarity_threshold}, "
            f"has_vector_store={vector_store is not None}, "
            f"external_search={enable_external_search}"
        )
    
    def calculate_similarity(
        self,
        current_data: dict[str, Any],
        reference_data: dict[str, Any],
    ) -> float:
        """計算兩個數據的相似度
        
        Args:
            current_data: 當前數據
            reference_data: 參考數據
            
        Returns:
            相似度分數 (0.0-1.0)
        """
        # 提取關鍵特徵進行比較
        current_features = self._extract_features(current_data)
        reference_features = self._extract_features(reference_data)
        
        # 計算文本相似度
        similarity = SequenceMatcher(
            None,
            current_features,
            reference_features
        ).ratio()
        
        return similarity
    
    def _extract_features(self, data: dict[str, Any]) -> str:
        """提取數據的關鍵特徵
        
        提取：錯誤訊息、響應狀態碼、異常類型、關鍵參數等
        """
        features = []
        
        # 提取響應相關特徵
        if "response" in data:
            resp = data["response"]
            if isinstance(resp, dict):
                features.append(f"status:{resp.get('status_code', '')}")
                features.append(f"error:{resp.get('error_message', '')}")
                features.append(f"type:{resp.get('response_type', '')}")
        
        # 提取請求相關特徵
        if "request" in data:
            req = data["request"]
            if isinstance(req, dict):
                features.append(f"method:{req.get('method', '')}")
                features.append(f"params:{str(req.get('parameters', {}))[:50]}")
        
        # 提取結果特徵
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                features.append(f"success:{result.get('success', False)}")
                features.append(f"findings:{len(result.get('findings', []))}")
        
        return " ".join(features)
    
    def check_if_known_situation(
        self,
        current_data: dict[str, Any],
        current_records: list[dict[str, Any]],
        historical_data: list[dict[str, Any]],
        knowledge_base_data: list[dict[str, Any]],
    ) -> tuple[bool, float, str]:
        """檢查是否為已知情況
        
        與三個數據源比對，判斷是否需要觸發 RAG
        
        Args:
            current_data: 當前數據
            current_records: 當前執行記錄
            historical_data: 歷史數據
            knowledge_base_data: 知識庫數據
            
        Returns:
            (is_known, max_similarity, source_name)
            - is_known: 是否為已知情況
            - max_similarity: 最高相似度
            - source_name: 最相似的數據源名稱
        """
        max_similarity = 0.0
        source_name = "none"
        
        # 1. 與當前記錄比對
        for record in current_records:
            similarity = self.calculate_similarity(current_data, record)
            if similarity > max_similarity:
                max_similarity = similarity
                source_name = "current_records"
        
        # 2. 與歷史數據比對
        for record in historical_data:
            similarity = self.calculate_similarity(current_data, record)
            if similarity > max_similarity:
                max_similarity = similarity
                source_name = "historical_data"
        
        # 3. 與知識庫比對
        for record in knowledge_base_data:
            similarity = self.calculate_similarity(current_data, record)
            if similarity > max_similarity:
                max_similarity = similarity
                source_name = "knowledge_base"
        
        # 判斷是否為已知情況
        is_known = max_similarity >= self.similarity_threshold
        
        logger.info(
            f"Similarity check: max={max_similarity:.3f} from {source_name}, "
            f"is_known={is_known} (threshold={self.similarity_threshold})"
        )
        
        return is_known, max_similarity, source_name
    
    async def trigger_rag_if_needed(
        self,
        current_data: dict[str, Any],
        current_records: list[dict[str, Any]],
        historical_data: list[dict[str, Any]],
        knowledge_base_data: list[dict[str, Any]],
        search_mode: str = "hybrid",  # "internal", "external", "hybrid"
    ) -> Optional[UnknownSituationAlert]:
        """如果需要，觸發 RAG 搜索（支持對內和對外）
        
        Args:
            current_data: 當前數據
            current_records: 當前執行記錄
            historical_data: 歷史數據
            knowledge_base_data: 知識庫數據
            search_mode: 搜索模式
                - "hybrid": 先內部後外部（默認）
                - "internal": 只搜索內部向量庫
                - "external": 只搜索外部資源
            
        Returns:
            如果觸發 RAG，返回警報對象；否則返回 None
        """
        # 檢查是否為已知情況
        is_known, max_similarity, source_name = self.check_if_known_situation(
            current_data,
            current_records,
            historical_data,
            knowledge_base_data,
        )
        
        # 如果是已知情況且不強制外部搜索，不需要觸發 RAG
        if is_known and search_mode != "external":
            logger.info(
                f"Known situation (similarity={max_similarity:.3f} from {source_name}), "
                "no RAG search needed"
            )
            return None
        
        # 未知情況或強制搜索，觸發 RAG
        logger.warning(
            f"⚠️ Triggering RAG search! "
            f"(max_similarity={max_similarity:.3f} < {self.similarity_threshold}, mode={search_mode})"
        )
        
        # 生成搜索查詢
        search_query = self._generate_search_query(current_data)
        
        # 創建警報
        alert = UnknownSituationAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger_reason=(
                f"相似度過低: {max_similarity:.3f} < {self.similarity_threshold} "
                f"(最相似數據源: {source_name})"
            ),
            data_snapshot=current_data,
            search_query=search_query,
            timestamp=datetime.now(),
        )
        
        # 立即通知用戶
        if self.notification_callback:
            try:
                self.notification_callback(alert)
                logger.info(f"✅ User notified about unknown situation: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to notify user: {e}")
        
        # 執行 RAG 搜索（雙向：內部 + 外部）
        try:
            logger.info(f"🔍 Starting RAG search with query: {search_query}")
            alert.status = "searching"
            
            # 執行雙向搜索
            rag_results = await self._perform_rag_search(
                query=search_query,
                current_data=current_data,
                search_mode=search_mode
            )
            
            alert.rag_results = rag_results
            alert.status = "found" if rag_results else "not_found"
            
            logger.info(
                f"✅ RAG search completed: found {len(rag_results)} relevant resources "
                f"(internal + external)"
            )
            
            # 再次通知用戶（更新狀態）
            if self.notification_callback:
                try:
                    self.notification_callback(alert)
                except Exception as e:
                    logger.error(f"Failed to notify user about RAG results: {e}")
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            alert.status = "error"
        
        # 保存警報歷史
        self._alert_history.append(alert)
        
        return alert
    
    def _generate_search_query(self, data: dict[str, Any]) -> str:
        """生成 RAG 搜索查詢
        
        從當前數據中提取關鍵信息構建搜索查詢
        """
        query_parts = []
        
        # 添加能力類型
        if "capability" in data:
            query_parts.append(data["capability"])
        
        # 添加錯誤訊息
        if "response" in data:
            resp = data["response"]
            if isinstance(resp, dict) and "error_message" in resp:
                query_parts.append(resp["error_message"])
        
        # 添加目標類型
        if "target" in data:
            target = data["target"]
            if isinstance(target, dict) and "url" in target:
                # 提取域名或關鍵信息
                query_parts.append(target["url"][:50])
        
        # 添加結果特徵
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                if not result.get("success", False):
                    query_parts.append("failure analysis")
                if "findings" in result and result["findings"]:
                    query_parts.append("unexpected findings")
        
        return " ".join(query_parts) if query_parts else "security analysis"
    
    async def _perform_rag_search(
        self,
        query: str,
        current_data: dict[str, Any],
        search_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """執行 RAG 搜索 - 支持對內和對外雙向搜索
        
        Args:
            query: 搜索查詢
            current_data: 當前數據
            search_mode: 搜索模式
                - "hybrid": 先內部後外部（默認）
                - "internal": 只搜索內部向量庫
                - "external": 只搜索外部資源
        
        Returns:
            搜索結果列表（包含內部和外部結果）
        """
        results = []
        
        # 1. 對內搜索 - 向量庫
        if search_mode in ["internal", "hybrid"]:
            logger.info(f"🔍 [對內搜索] 搜索本地向量庫...")
            internal_results = await self._search_internal_vector_store(query, current_data)
            results.extend(internal_results)
            
            # 計算內部搜索的最高相似度
            max_internal_similarity = max(
                (r.get("relevance_score", 0.0) for r in internal_results),
                default=0.0
            )
            
            logger.info(
                f"✅ 內部搜索完成: 找到 {len(internal_results)} 個結果, "
                f"最高相似度={max_internal_similarity:.3f}"
            )
            
            # 如果內部找到高相關結果，可以跳過外部搜索
            if search_mode == "hybrid" and max_internal_similarity >= self.similarity_threshold:
                logger.info(
                    f"✅ 內部搜索已滿足需求 (similarity >= {self.similarity_threshold}), "
                    "跳過外部搜索"
                )
                return results
        
        # 2. 對外搜索 - 外部資源
        if search_mode in ["external", "hybrid"]:
            if self.enable_external_search:
                logger.info(f"🔍 [對外搜索] 搜索外部資源...")
                external_results = await self._search_external_resources(query, current_data)
                results.extend(external_results)
                
                logger.info(
                    f"✅ 外部搜索完成: 找到 {len(external_results)} 個資源"
                )
            else:
                logger.info("外部搜索已禁用")
        
        # 按相關度排序
        results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        
        logger.info(
            f"✅ RAG 搜索總計: {len(results)} 個結果 "
            f"(內部 + 外部)"
        )
        
        return results
    
    async def _search_internal_vector_store(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """對內搜索 - 本地向量庫
        
        搜索範圍：
        1. 能力註冊表 (782+ 條)
        2. 執行經驗記錄 (JSONL 同步的數據)
        3. 知識庫分析報告 (Markdown)
        4. 外部模組說明
        """
        results = []
        
        if not self.vector_store:
            logger.warning("Vector store not configured, skipping internal search")
            return results
        
        try:
            # 向量相似度搜索（同步調用，VectorStore.search 不是 async）
            vector_results = self.vector_store.search(
                query=query,
                top_k=10  # 返回前 10 個最相關的結果
            )
            
            # 轉換格式（處理不同的返回格式）
            for item in vector_results:
                content = ""
                score = 0.0
                metadata = {}
                
                if isinstance(item, dict):
                    content = item.get("content", item.get("document", ""))
                    score = item.get("score", item.get("distance", 0.0))
                    metadata = item.get("metadata", {})
                elif isinstance(item, tuple) and len(item) >= 2:
                    # (document, metadata, distance) 格式
                    content = item[0] if len(item) > 0 else ""
                    metadata = item[1] if len(item) > 1 else {}
                    score = 1.0 - item[2] if len(item) > 2 else 0.0  # 距離轉相似度
                
                results.append({
                    "type": "internal_knowledge",
                    "source": "vector_store",
                    "content": content,
                    "relevance_score": score,
                    "metadata": metadata,
                })
            
            logger.info(f"內部向量庫返回 {len(results)} 個結果")
        
        except Exception as e:
            logger.error(f"Internal vector store search error: {e}", exc_info=True)
        
        return results
    
    async def _search_external_resources(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """對外搜索 - 外部資源
        
        搜索範圍:
        1. CVE 數據庫 (NVD API)
        2. Exploit-DB
        3. Google 技術搜索
        4. GitHub Security Advisory
        5. 安全研究網站
        
        注意：這是對外 HTTP 請求
        搜索範圍：
        1. CVE 數據庫 (cve.mitre.org, nvd.nist.gov)
        2. Exploit-DB
        3. Google 技術文章搜索
        4. GitHub Security Advisory
        5. 安全研究論文和博客
        
        注意：這是對外搜索，不是內部向量庫！
        """
        results = []
        
        try:
            # 1. 搜索 CVE 數據庫
            logger.info(f"🔍 [外部搜索 1/4] 搜索 CVE 數據庫...")
            cve_results = await self._search_cve_database(query, current_data)
            results.extend(cve_results)
            
            # 2. 搜索 Exploit-DB
            logger.info(f"🔍 [外部搜索 2/4] 搜索 Exploit-DB...")
            exploit_results = await self._search_exploit_db(query, current_data)
            results.extend(exploit_results)
            
            # 3. Google 搜索技術文章
            logger.info(f"🔍 [外部搜索 3/4] Google 搜索...")
            google_results = await self._search_google(query, current_data)
            results.extend(google_results)
            
            # 4. GitHub Security Advisory
            logger.info(f"🔍 [外部搜索 4/4] 搜索 GitHub Security Advisory...")
            github_results = await self._search_github_advisory(query, current_data)
            results.extend(github_results)
            
            # 按相關度排序
            results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
            
            logger.info(f"✅ 外部搜索完成，共找到 {len(results)} 個資源")
            
        except Exception as e:
            logger.error(f"外部搜索錯誤: {e}")
        
        return results
    
    async def _search_cve_database(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """搜索 CVE 數據庫"""
        results = []
        
        try:
            logger.info("CVE 數據庫搜索（模擬）")
            
            # 提取關鍵詞
            capability = current_data.get("capability", "")
            error_msg = current_data.get("response", {}).get("error_message", "")
            
            if capability or error_msg:
                # 生成搜索建議
                results.append({
                    "type": "cve_database",
                    "source": "NVD",
                    "content": f"建議搜索 NVD 數據庫: {capability} {error_msg}",
                    "url": f"https://nvd.nist.gov/vuln/search/results?query={capability}",
                    "relevance_score": 0.7,
                })
            
            # 構建關鍵詞
            keywords = f"{capability} {error_msg}".strip()
            
            if not keywords:
                return results
            
            # 調用 NVD API（需要 API key）
            # 這裡使用簡化版，實際需要註冊 API key
            if aiohttp is None:
                logger.warning("aiohttp not installed, skipping external CVE search")
                return results
            
            url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keywords}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
        
        except Exception as e:
            logger.error(f"CVE database search error: {e}")
        
        return results
    
    async def _search_exploit_db(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """搜索 Exploit-DB（外部 HTTP 請求）"""
        results = []
        
        try:
            capability = current_data.get("capability", "")
            
            # Exploit-DB 搜索 URL（簡化版）
            if capability:
                url = f"https://www.exploit-db.com/search?q={capability}"
                results.append({
                    "type": "exploit_reference",
                    "source": "Exploit-DB",
                    "content": f"查看 Exploit-DB 相關利用代碼: {capability}",
                    "relevance_score": 0.75,
                    "url": url,
                })
        
        except Exception as e:
            logger.error(f"Exploit-DB search error: {e}")
        
        return results
    
    async def _search_google(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Google 搜索技術文章（外部搜索引擎）"""
        results = []
        
        try:
            capability = current_data.get("capability", "")
            error_msg = current_data.get("response", {}).get("error_message", "")
            
            if capability or error_msg:
                search_query = f"{capability} bypass technique"
                url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
                
                results.append({
                    "type": "google_search",
                    "source": "Google",
                    "content": f"建議 Google 搜索: {search_query}",
                    "relevance_score": 0.7,
                    "url": url,
                })
        
        except Exception as e:
            logger.warning(f"Google search failed: {e}")
        
        return results
    
    async def _search_github_advisory(
        self,
        query: str,
        current_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """搜索 GitHub Security Advisory（外部 API）"""
        results = []
        
        try:
            capability = current_data.get("capability", "")
            
            # GitHub GraphQL API 搜索安全公告
            # 需要 GitHub token
            # 簡化版：返回搜索 URL
            url = f"https://github.com/advisories?query={capability}"
            
            results.append({
                "type": "github_advisory",
                "source": "GitHub Security Advisory",
                "content": f"查看 GitHub 安全公告: {capability}",
                "relevance_score": 0.87,
                "url": url,
            })
        
        except Exception as e:
            logger.error(f"GitHub advisory search error: {e}")
        
        return results
    
    def get_alert_history(
        self,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """獲取警報歷史
        
        Args:
            limit: 返回數量限制
            
        Returns:
            警報列表
        """
        return [
            alert.to_dict()
            for alert in self._alert_history[-limit:]
        ]
    
    def clear_alert_history(self) -> None:
        """清空警報歷史"""
        self._alert_history.clear()
        logger.info("Alert history cleared")


# ==================== 自動優化擴展 ====================

class SelfOptimizationTrigger(RAGTrigger):
    """自我優化觸發器 - 基於RAG觸發器的擴展
    
    實現AIVA雙循環自我優化系統的觸發機制：
    - 內部循環：基於系統自我分析觸發優化
    - 外部循環：基於實戰反饋觸發優化
    - 智能決策：基於RAG知識檢索優化方案
    """
    
    def __init__(
        self,
        vector_store=None,
        similarity_threshold: float = 0.4,
        enable_external_search: bool = True,
        notification_callback=None,
        optimization_interval: int = 3600,  # 1小時檢查一次
    ):
        """初始化自我優化觸發器
        
        Args:
            optimization_interval: 優化檢查間隔（秒）
        """
        super().__init__(
            vector_store=vector_store,
            similarity_threshold=similarity_threshold,
            enable_external_search=enable_external_search,
            notification_callback=notification_callback
        )
        
        self.optimization_interval = optimization_interval
        self.last_optimization_check = datetime.now()
        self.internal_loop_data = []
        self.external_loop_data = []
        self.optimization_history = []
        
        logger.info(f"Self-optimization trigger initialized with {optimization_interval}s interval")
    
    async def trigger_internal_analysis(self) -> dict[str, Any]:
        """觸發內部循環分析
        
        基於系統自我探索結果觸發優化分析
        """
        logger.info("🔍 觸發內部循環分析...")
        
        # 收集內部系統數據
        internal_data = {
            "timestamp": datetime.now().isoformat(),
            "trigger_type": "internal_loop",
            "system_health": await self._collect_system_health(),
            "capability_analysis": await self._analyze_capabilities(),
            "code_quality": await self._assess_code_quality(),
            "performance_metrics": await self._collect_performance_metrics()
        }
        
        self.internal_loop_data.append(internal_data)
        
        # 檢查是否需要觸發RAG搜索
        is_known, similarity, source = self.check_if_known_situation(
            internal_data,
            self.internal_loop_data[-5:],  # 最近5次記錄
            self.internal_loop_data[:-5],  # 歷史記錄
            []  # 知識庫數據，可以後續添加
        )
        
        if not is_known:
            # 觸發RAG搜索優化知識
            rag_results = await self.trigger_rag_search(
                current_data=internal_data,
                trigger_reason="Internal loop optimization needed",
                search_query="system optimization self-improvement code quality"
            )
            internal_data["rag_suggestions"] = rag_results
        
        return internal_data
    
    async def trigger_external_feedback(self, battle_results: dict[str, Any]) -> dict[str, Any]:
        """觸發外部循環反饋分析
        
        基於實戰攻擊結果觸發優化分析
        
        Args:
            battle_results: 實戰攻擊結果數據
        """
        logger.info("⚔️ 觸發外部循環反饋分析...")
        
        # 整理外部實戰數據
        external_data = {
            "timestamp": datetime.now().isoformat(),
            "trigger_type": "external_loop",
            "battle_results": battle_results,
            "attack_effectiveness": self._analyze_attack_effectiveness(battle_results),
            "target_patterns": self._identify_target_patterns(battle_results),
            "defense_mechanisms": self._detect_defense_mechanisms(battle_results)
        }
        
        self.external_loop_data.append(external_data)
        
        # 檢查是否遇到新的攻擊場景
        is_known, similarity, source = self.check_if_known_situation(
            external_data,
            self.external_loop_data[-5:],
            self.external_loop_data[:-5], 
            []
        )
        
        if not is_known:
            # 搜索相關攻擊技術和優化方案
            rag_results = await self.trigger_rag_search(
                current_data=external_data,
                trigger_reason="New attack scenario detected",
                search_query="penetration testing optimization attack techniques CVE"
            )
            external_data["rag_suggestions"] = rag_results
        
        return external_data
    
    async def generate_optimization_decisions(self) -> dict[str, Any]:
        """整合雙循環數據生成優化決策
        
        結合內部分析和外部反饋，生成AI驅動的優化決策
        """
        logger.info("🤖 生成AI優化決策...")
        
        # 獲取最新的內外循環數據
        latest_internal = self.internal_loop_data[-1] if self.internal_loop_data else None
        latest_external = self.external_loop_data[-1] if self.external_loop_data else None
        
        if not latest_internal and not latest_external:
            logger.warning("沒有足夠的循環數據來生成優化決策")
            return {"error": "Insufficient data for optimization decisions"}
        
        # 整合決策數據
        decision_data = {
            "timestamp": datetime.now().isoformat(),
            "internal_insights": latest_internal,
            "external_insights": latest_external,
            "optimization_priorities": [],
            "recommended_actions": [],
            "estimated_impact": {}
        }
        
        # 分析內部優化機會
        if latest_internal:
            internal_priorities = self._extract_internal_priorities(latest_internal)
            decision_data["optimization_priorities"].extend(internal_priorities)
        
        # 分析外部優化機會  
        if latest_external:
            external_priorities = self._extract_external_priorities(latest_external)
            decision_data["optimization_priorities"].extend(external_priorities)
        
        # 生成推薦行動
        decision_data["recommended_actions"] = self._generate_action_recommendations(
            decision_data["optimization_priorities"]
        )
        
        # 評估預期影響
        decision_data["estimated_impact"] = self._estimate_optimization_impact(
            decision_data["recommended_actions"]
        )
        
        self.optimization_history.append(decision_data)
        
        # 通知用戶
        if self.notification_callback:
            await self.notification_callback(f"🤖 AI優化決策已生成：{len(decision_data['recommended_actions'])}個建議行動")
        
        return decision_data
    
    # ==================== 內部方法 ====================
    
    async def _collect_system_health(self) -> dict[str, Any]:
        """收集系統健康數據"""
        return {
            "memory_usage": "模擬數據",
            "cpu_usage": "模擬數據", 
            "module_status": "模擬數據",
            "error_rate": "模擬數據"
        }
    
    async def _analyze_capabilities(self) -> dict[str, Any]:
        """分析現有能力"""
        return {
            "available_attacks": ["sqli", "xss", "csrf"],
            "success_rates": {"sqli": 0.78, "xss": 0.62},
            "capability_gaps": ["ssrf_advanced", "xxe_detection"]
        }
    
    async def _assess_code_quality(self) -> dict[str, Any]:
        """評估代碼品質"""
        return {
            "complexity_score": 7.5,
            "test_coverage": 0.85,
            "technical_debt": "medium",
            "refactor_suggestions": ["extract_method", "reduce_complexity"]
        }
    
    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """收集性能指標"""
        return {
            "avg_response_time": 1.2,
            "throughput": 150,
            "error_rate": 0.03,
            "resource_efficiency": 0.78
        }
    
    def _analyze_attack_effectiveness(self, battle_results: dict[str, Any]) -> dict[str, Any]:
        """分析攻擊有效性"""
        return {
            "overall_success_rate": battle_results.get("success_rate", 0.0),
            "best_techniques": ["time_based_sqli", "dom_xss"],
            "failed_techniques": ["basic_ssrf", "simple_csrf"],
            "improvement_areas": ["evasion_techniques", "payload_optimization"]
        }
    
    def _identify_target_patterns(self, battle_results: dict[str, Any]) -> dict[str, Any]:
        """識別目標模式"""
        return {
            "common_tech_stacks": ["nginx+php", "apache+tomcat"],
            "vulnerable_patterns": ["unfiltered_input", "missing_auth"],
            "defense_patterns": ["waf_detection", "rate_limiting"]
        }
    
    def _detect_defense_mechanisms(self, battle_results: dict[str, Any]) -> dict[str, Any]:
        """檢測防禦機制"""
        return {
            "waf_types": ["cloudflare", "aws_waf"],
            "blocking_rules": ["user_agent_filtering", "payload_detection"],
            "bypass_opportunities": ["encoding_bypass", "time_delay_evasion"]
        }
    
    def _extract_internal_priorities(self, internal_data: dict[str, Any]) -> list[dict[str, Any]]:
        """提取內部優化優先級"""
        priorities = []
        
        # 基於代碼品質
        code_quality = internal_data.get("code_quality", {})
        if code_quality.get("complexity_score", 0) > 8.0:
            priorities.append({
                "type": "code_quality",
                "priority": "high",
                "description": "Reduce code complexity",
                "impact": "maintainability"
            })
        
        # 基於性能指標
        performance = internal_data.get("performance_metrics", {})
        if performance.get("avg_response_time", 0) > 2.0:
            priorities.append({
                "type": "performance",
                "priority": "medium", 
                "description": "Optimize response time",
                "impact": "user_experience"
            })
        
        return priorities
    
    def _extract_external_priorities(self, external_data: dict[str, Any]) -> list[dict[str, Any]]:
        """提取外部優化優先級"""
        priorities = []
        
        # 基於攻擊成功率
        effectiveness = external_data.get("attack_effectiveness", {})
        success_rate = effectiveness.get("overall_success_rate", 0.0)
        
        if success_rate < 0.5:
            priorities.append({
                "type": "attack_optimization",
                "priority": "high",
                "description": "Improve attack success rate",
                "impact": "penetration_effectiveness"
            })
        
        # 基於防禦檢測
        defense = external_data.get("defense_mechanisms", {})
        if defense.get("waf_types"):
            priorities.append({
                "type": "evasion",
                "priority": "medium",
                "description": "Develop WAF bypass techniques",
                "impact": "stealth_capability"
            })
        
        return priorities
    
    def _generate_action_recommendations(self, priorities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """生成行動建議"""
        actions = []
        
        for priority in priorities:
            if priority["type"] == "code_quality":
                actions.append({
                    "action": "refactor_high_complexity_modules",
                    "description": "Extract methods and reduce cyclomatic complexity",
                    "effort": "medium",
                    "timeline": "2-3 days"
                })
            elif priority["type"] == "attack_optimization":
                actions.append({
                    "action": "enhance_payload_generation",
                    "description": "Implement AI-driven payload optimization",
                    "effort": "high",
                    "timeline": "1-2 weeks"
                })
            elif priority["type"] == "evasion":
                actions.append({
                    "action": "implement_waf_bypass",
                    "description": "Add advanced evasion techniques",
                    "effort": "medium",
                    "timeline": "3-5 days"
                })
        
        return actions
    
    def _estimate_optimization_impact(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        """評估優化影響"""
        return {
            "performance_improvement": "15-30%",
            "success_rate_boost": "20-40%", 
            "maintainability_gain": "High",
            "estimated_roi": "2.5x",
            "risk_level": "Low"
        }


# ==================== 便捷函數 ====================

async def trigger_internal_optimization():
    """便捷函數：觸發內部優化分析"""
    trigger = SelfOptimizationTrigger()
    return await trigger.trigger_internal_analysis()


async def trigger_external_optimization(battle_results: dict[str, Any]):
    """便捷函數：觸發外部優化分析"""
    trigger = SelfOptimizationTrigger()
    return await trigger.trigger_external_feedback(battle_results)


async def generate_ai_optimization_plan():
    """便捷函數：生成AI優化計畫"""
    trigger = SelfOptimizationTrigger()
    return await trigger.generate_optimization_decisions()
