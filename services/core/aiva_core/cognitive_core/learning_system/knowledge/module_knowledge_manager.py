"""
Module Knowledge Manager - 模組知識庫管理器
==============================================

管理所有外部模組的完整數據流分析報告，提供基於實際執行情況的學習建議。

架構位置：
- cognitive_core/learning_system/knowledge/module_knowledge_manager.py (本文件)
- 整合到 ExternalLearningListener 的決策流程

功能：
1. 載入所有模組的完整分析報告（XSS/SQLi/SSRF/TypeScript等）
2. 接收外部模組執行結果（發出+接收）
3. 比對知識庫，匹配已知情況
4. 提供調整建議（基於因果關係）
5. 遇到未知情況時觸發RAG搜索

數據來源：
- XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
- SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
- SSRF_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md
- TYPESCRIPT_DOM_XSS_COMPLETE_ANALYSIS.md
- ... (未來所有模組的報告)
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """外部模組執行上下文"""
    module_name: str           # 模組名稱（如 function_xss）
    target_url: str            # 目標URL
    sent_data: dict[str, Any]  # 發出的數據（payload, params, headers等）
    received_data: dict[str, Any]  # 接收的數據（response, status, errors等）
    timestamp: str
    execution_id: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'module_name': self.module_name,
            'target_url': self.target_url,
            'sent_data': self.sent_data,
            'received_data': self.received_data,
            'timestamp': self.timestamp,
            'execution_id': self.execution_id
        }


@dataclass
class KnowledgeMatch:
    """知識庫匹配結果"""
    matched: bool              # 是否匹配到已知情況
    scenario_id: str           # 匹配到的情況ID
    scenario_name: str         # 情況名稱
    confidence: float          # 匹配信心度（0-1）
    category: str              # 類別（成功/可疑/失敗）
    adjustment: dict[str, Any] | None  # 建議調整
    causality: str | None   # 因果關係說明
    learning_points: list[str]  # 學習點


@dataclass
class LearningRecommendation:
    """學習建議"""
    recommendation_id: str
    execution_context: ExecutionContext
    knowledge_match: KnowledgeMatch | None
    adjustments: list[dict[str, Any]]  # 具體調整建議
    rationale: str             # 建議理由
    confidence: float          # 建議信心度
    requires_rag: bool         # 是否需要RAG搜索
    rag_query: str | None   # RAG搜索查詢
    timestamp: str


class ModuleKnowledgeManager:
    """模組知識庫管理器
    
    核心職責：
    1. 載入並解析完整分析報告
    2. 建立情況索引（成功/可疑/失敗）
    3. 接收執行結果並匹配已知情況
    4. 生成調整建議
    5. 觸發RAG搜索（未知情況）
    """
    
    def __init__(self, knowledge_base_dir: str, rag_client=None):
        """初始化知識庫管理器
        
        Args:
            knowledge_base_dir: 知識庫目錄（存放所有分析報告）
            rag_client: RAG客戶端（用於未知情況搜索）
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.rag_client = rag_client
        
        # 知識庫結構
        self.module_knowledge = {}  # {module_name: knowledge_data}
        self.scenario_index = {}    # {scenario_id: scenario_data}
        self.causality_map = {}     # {condition: adjustment}
        
        # 統計資訊
        self.total_modules = 0
        self.total_scenarios = 0
        self.match_stats = {
            'total_requests': 0,
            'matched': 0,
            'unmatched_rag': 0,
            'unmatched_no_rag': 0
        }
        
        logger.info("ModuleKnowledgeManager initialized")
    
    def load_all_knowledge(self) -> None:
        """載入所有模組的知識庫"""
        logger.info("Loading all module knowledge bases...")
        
        # 掃描知識庫目錄
        md_files = list(self.knowledge_base_dir.glob("*_COMPLETE_*_ANALYSIS.md"))
        json_files = list(self.knowledge_base_dir.glob("*_knowledge.json"))
        
        if not md_files and not json_files:
            logger.warning(f"No knowledge files found in {self.knowledge_base_dir}")
            return
        
        # 載入Markdown報告（解析為結構化數據）
        for md_file in md_files:
            try:
                module_name = self._extract_module_name(md_file.name)
                knowledge = self._parse_markdown_report(md_file)
                self.module_knowledge[module_name] = knowledge
                self._build_scenario_index(module_name, knowledge)
                logger.info(f"Loaded knowledge for {module_name} from {md_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {md_file}: {e}")
        
        # 載入JSON知識庫（直接加載）
        for json_file in json_files:
            try:
                with open(json_file, encoding='utf-8') as f:
                    data = json.load(f)
                    module_name = data.get('module')
                    if module_name:
                        self.module_knowledge[module_name] = data
                        self._build_scenario_index(module_name, data)
                        logger.info(f"Loaded knowledge for {module_name} from {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        self.total_modules = len(self.module_knowledge)
        self.total_scenarios = len(self.scenario_index)
        
        logger.info(f"Knowledge loading complete: {self.total_modules} modules, "
                   f"{self.total_scenarios} scenarios")
    
    def _extract_module_name(self, filename: str) -> str:
        """從文件名提取模組名稱"""
        # XSS_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md -> function_xss
        # SQLI_MODULE_COMPLETE_DATA_FLOW_ANALYSIS.md -> function_sqli
        # TYPESCRIPT_DOM_XSS_COMPLETE_ANALYSIS.md -> typescript_engine
        
        filename_lower = filename.lower()
        
        if 'xss' in filename_lower and 'typescript' in filename_lower:
            return 'typescript_engine'
        elif 'xss' in filename_lower:
            return 'function_xss'
        elif 'sqli' in filename_lower or 'sql' in filename_lower:
            return 'function_sqli'
        elif 'ssrf' in filename_lower:
            return 'function_ssrf'
        elif 'idor' in filename_lower:
            return 'function_idor'
        elif 'bizlogic' in filename_lower:
            return 'function_bizlogic'
        else:
            return filename.split('_')[0].lower()
    
    def _parse_markdown_report(self, md_file: Path) -> dict[str, Any]:
        """解析Markdown報告為結構化數據
        
        從報告中提取：
        1. 發出的數據類型
        2. 18種接收情況（成功3/可疑5/失敗10）
        3. 可調整參數
        4. 5個因果場景
        5. 學習點
        """
        try:
            content = md_file.read_text(encoding='utf-8')
            result = self._create_empty_knowledge_structure(
                self._extract_module_name(md_file.name), 
                str(md_file)
            )
            
            # 分析 Markdown 結構
            lines = content.split('\n')
            current_section = None
            current_list = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # 識別章節標題
                if line_stripped.startswith('##'):
                    # 儲存前一個章節的內容
                    if current_section and current_list:
                        self._store_section_data(result, current_section, current_list)
                        current_list = []
                    
                    # 更新當前章節
                    section_title = line_stripped.replace('#', '').strip()
                    current_section = self._identify_section_type(section_title)
                
                # 收集列表項目
                elif line_stripped.startswith('-') or line_stripped.startswith('*'):
                    item_text = self._extract_list_item(line_stripped)
                    if current_section and item_text:
                        current_list.append(item_text)
                
                # 收集表格數據（簡化處理）
                elif '|' in line_stripped and current_section == 'params':
                    param_data = self._parse_table_param_line(line_stripped)
                    if param_data:
                        param_name, param_value = param_data
                        result['adjustable_params'][param_name] = param_value
            
            # 儲存最後一個章節
            if current_section and current_list:
                self._store_section_data(result, current_section, current_list)
            
            self._log_parse_success(md_file.name, result)
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 解析 Markdown 報告失敗 {md_file.name}: {e}")
            return self._create_empty_knowledge_structure(
                self._extract_module_name(md_file.name), 
                str(md_file)
            )
    
    def _create_empty_knowledge_structure(self, module_name: str, report_file: str) -> dict[str, Any]:
        """建立空白的知識結構"""
        return {
            'module': module_name,
            'report_file': report_file,
            'scenarios': {
                'success': [],
                'suspicious': [],
                'failure': []
            },
            'adjustable_params': {},
            'causality_scenarios': [],
            'learning_points': []
        }
    
    def _identify_section_type(self, section_title: str) -> str | None:
        """識別章節類型"""
        title_lower = section_title.lower()
        
        if '成功' in title_lower or 'success' in title_lower:
            return 'success'
        elif '可疑' in title_lower or 'suspicious' in title_lower:
            return 'suspicious'
        elif '失敗' in title_lower or 'failure' in title_lower or '錯誤' in title_lower:
            return 'failure'
        elif '參數' in title_lower or 'parameter' in title_lower:
            return 'params'
        elif '因果' in title_lower or 'causality' in title_lower:
            return 'causality'
        elif '學習' in title_lower or 'learning' in title_lower:
            return 'learning'
        else:
            return None
    
    def _extract_list_item(self, line: str) -> str:
        """從Markdown列表行中提取內容"""
        return line.lstrip('-*').strip()
    
    def _parse_table_param_line(self, line: str) -> tuple[str, str] | None:
        """解析表格參數行
        
        Returns:
            (param_name, param_value) 或 None 如果無效
        """
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2 and not parts[0].startswith('-'):
            param_name = parts[0]
            param_value = parts[1] if len(parts) > 1 else ''
            return (param_name, param_value)
        return None
    
    def _log_parse_success(self, filename: str, result: dict[str, Any]) -> None:
        """記錄解析成功資訊"""
        logger.info(f"✅ 成功解析 {filename}: "
                   f"成功={len(result['scenarios']['success'])}, "
                   f"可疑={len(result['scenarios']['suspicious'])}, "
                   f"失敗={len(result['scenarios']['failure'])}")
    
    def _store_section_data(self, result: dict[str, Any], section: str, data: list[str]) -> None:
        """儲存章節數據到結果結構"""
        if section in ['success', 'suspicious', 'failure']:
            result['scenarios'][section].extend(data)
        elif section == 'causality':
            result['causality_scenarios'].extend(data)
        elif section == 'learning':
            result['learning_points'].extend(data)
    
    def _build_scenario_index(self, module_name: str, knowledge: dict[str, Any]) -> None:
        """建立情況索引"""
        scenarios = knowledge.get('scenarios', {})
        
        for category in ['success', 'suspicious', 'failure']:
            for scenario in scenarios.get(category, []):
                scenario_id = f"{module_name}_{category}_{len(self.scenario_index)}"
                self.scenario_index[scenario_id] = {
                    'module': module_name,
                    'category': category,
                    'scenario': scenario,
                    'scenario_id': scenario_id
                }
    
    def match_execution_result(self, context: ExecutionContext) -> KnowledgeMatch:
        """匹配執行結果到知識庫
        
        比對邏輯：
        1. 檢查模組是否有知識庫
        2. 分析 sent_data 和 received_data
        3. 匹配到具體情況
        4. 返回匹配結果和建議
        """
        self.match_stats['total_requests'] += 1
        
        # 檢查模組知識庫
        if context.module_name not in self.module_knowledge:
            logger.warning(f"No knowledge base for module: {context.module_name}")
            return KnowledgeMatch(
                matched=False,
                scenario_id='',
                scenario_name='Unknown module',
                confidence=0.0,
                category='unknown',
                adjustment=None,
                causality=None,
                learning_points=[]
            )
        
        module_kb = self.module_knowledge[context.module_name]
        
        # 特徵提取
        features = self._extract_features(context)
        
        # 匹配邏輯（基於特徵相似度）
        best_match = None
        best_score = 0.0
        
        for scenario_id, scenario_data in self.scenario_index.items():
            if scenario_data['module'] != context.module_name:
                continue
            
            score = self._calculate_similarity(features, scenario_data['scenario'])
            if score > best_score:
                best_score = score
                best_match = scenario_data
        
        # 匹配閾值
        if best_match and best_score >= 0.7:
            self.match_stats['matched'] += 1
            return KnowledgeMatch(
                matched=True,
                scenario_id=best_match['scenario_id'],
                scenario_name=best_match['scenario'].get('name', 'Unknown'),
                confidence=best_score,
                category=best_match['category'],
                adjustment=best_match['scenario'].get('adjustment'),
                causality=best_match['scenario'].get('causality'),
                learning_points=best_match['scenario'].get('learning_points', [])
            )
        else:
            # 未匹配，可能需要RAG
            return KnowledgeMatch(
                matched=False,
                scenario_id='',
                scenario_name='Unknown scenario',
                confidence=best_score,
                category='unknown',
                adjustment=None,
                causality=None,
                learning_points=[]
            )
    
    def _extract_features(self, context: ExecutionContext) -> dict[str, Any]:
        """提取執行上下文的特徵
        
        特徵包括：
        - 發出的payload類型
        - HTTP狀態碼
        - 響應中的關鍵字
        - 錯誤類型
        - 超時情況
        """
        features = {
            'module': context.module_name,
            'sent': {},
            'received': {}
        }
        
        # 提取發出數據特徵
        sent = context.sent_data
        features['sent']['payload_type'] = sent.get('payload_type')
        features['sent']['encoding'] = sent.get('encoding')
        features['sent']['method'] = sent.get('method', 'GET')
        features['sent']['has_custom_headers'] = bool(sent.get('headers'))
        
        # 提取接收數據特徵
        received = context.received_data
        features['received']['status_code'] = received.get('status_code')
        features['received']['has_error'] = bool(received.get('error'))
        features['received']['response_time'] = received.get('response_time')
        features['received']['detected_waf'] = received.get('waf_detected')
        features['received']['xss_triggered'] = received.get('xss_triggered')
        features['received']['sqli_confirmed'] = received.get('sqli_confirmed')
        
        # 提取響應內容特徵
        response_body = received.get('response_body', '')
        features['received']['response_keywords'] = self._extract_keywords(response_body)
        
        return features
    
    def _extract_keywords(self, text: str) -> list[str]:
        """提取響應中的關鍵字"""
        keywords = []
        text_lower = text.lower() if text else ''
        
        # 錯誤關鍵字
        error_keywords = ['error', 'exception', 'denied', 'blocked', 'forbidden']
        for kw in error_keywords:
            if kw in text_lower:
                keywords.append(kw)
        
        # WAF關鍵字
        waf_keywords = ['cloudflare', 'modsecurity', 'wordfence', 'sucuri']
        for kw in waf_keywords:
            if kw in text_lower:
                keywords.append(f'waf_{kw}')
        
        # XSS關鍵字
        xss_keywords = ['<script', 'alert(', 'onerror=', 'javascript:']
        for kw in xss_keywords:
            if kw in text_lower:
                keywords.append(f'xss_{kw}')
        
        return keywords
    
    def _calculate_similarity(self, features: dict[str, Any], scenario: dict[str, Any]) -> float:
        """計算特徵與情況的相似度"""
        # 簡化版：檢查關鍵特徵匹配
        score = 0.0
        total_checks = 0
        
        # 檢查狀態碼
        if 'status_code' in scenario.get('received', {}):
            total_checks += 1
            if features['received']['status_code'] == scenario['received']['status_code']:
                score += 0.3
        
        # 檢查錯誤
        if 'has_error' in scenario.get('received', {}):
            total_checks += 1
            if features['received']['has_error'] == scenario['received']['has_error']:
                score += 0.2
        
        # 檢查WAF
        if 'detected_waf' in scenario.get('received', {}):
            total_checks += 1
            if features['received']['detected_waf'] == scenario['received']['detected_waf']:
                score += 0.3
        
        # 檢查關鍵字
        scenario_keywords = set(scenario.get('keywords', []))
        feature_keywords = set(features['received']['response_keywords'])
        if scenario_keywords and feature_keywords:
            total_checks += 1
            keyword_overlap = len(scenario_keywords & feature_keywords) / len(scenario_keywords)
            score += keyword_overlap * 0.2
        
        return score / max(total_checks, 1) if total_checks > 0 else 0.0
    
    def generate_recommendation(self, context: ExecutionContext) -> LearningRecommendation:
        """生成學習建議
        
        流程：
        1. 匹配知識庫
        2. 如果匹配 -> 提供已知的調整建議
        3. 如果未匹配 -> 觸發RAG搜索
        4. 生成結構化建議
        """
        # 匹配知識庫
        match = self.match_execution_result(context)
        
        adjustments = []
        requires_rag = False
        rag_query = None
        rationale = ""
        
        if match.matched:
            # 已知情況，提供建議
            if match.adjustment:
                adjustments.append(match.adjustment)
            
            rationale = f"匹配到已知情況: {match.scenario_name} ({match.category})\n"
            rationale += f"信心度: {match.confidence:.2f}\n"
            if match.causality:
                rationale += f"因果關係: {match.causality}\n"
            
            logger.info(f"Matched scenario: {match.scenario_id} (confidence: {match.confidence:.2f})")
        
        else:
            # 未知情況，需要RAG
            requires_rag = True
            self.match_stats['unmatched_rag'] += 1
            
            # 構建RAG查詢
            rag_query = self._build_rag_query(context)
            rationale = f"未匹配到已知情況 (最高相似度: {match.confidence:.2f})\n"
            rationale += f"觸發RAG搜索: {rag_query}\n"
            
            logger.warning(f"No match found for {context.module_name}, triggering RAG search")
            
            # 如果有RAG客戶端，執行搜索
            if self.rag_client:
                rag_results = self._query_rag(rag_query, context)
                if rag_results:
                    adjustments.extend(rag_results)
                    rationale += f"RAG搜索返回 {len(rag_results)} 個建議\n"
        
        # 生成建議ID
        recommendation_id = self._generate_recommendation_id(context)
        
        return LearningRecommendation(
            recommendation_id=recommendation_id,
            execution_context=context,
            knowledge_match=match if match.matched else None,
            adjustments=adjustments,
            rationale=rationale,
            confidence=match.confidence,
            requires_rag=requires_rag,
            rag_query=rag_query,
            timestamp=datetime.now().isoformat()
        )
    
    def _build_rag_query(self, context: ExecutionContext) -> str:
        """構建RAG搜索查詢"""
        query_parts = [
            f"Module: {context.module_name}",
            f"Target: {context.target_url}",
        ]
        
        # 添加關鍵特徵
        sent = context.sent_data
        received = context.received_data
        
        if sent.get('payload_type'):
            query_parts.append(f"Payload: {sent['payload_type']}")
        
        if received.get('status_code'):
            query_parts.append(f"Status: {received['status_code']}")
        
        if received.get('error'):
            query_parts.append(f"Error: {received['error']}")
        
        return " | ".join(query_parts)
    
    def _query_rag(self, query: str, context: ExecutionContext) -> list[dict[str, Any]]:
        """執行RAG搜索
        
        Args:
            query: 查詢字符串（包含模組、發送和接收數據）
            context: 執行上下文
            
        Returns:
            RAG搜索結果列表
        """
        try:
            if not self.rag_client:
                logger.debug("RAG client not available, skipping RAG search")
                return []
            
            # 構建查詢上下文
            enhanced_query = f"Module: {context.module_name} | {query}"
            
            # 根據 rag_client 的類型調用不同的方法
            if hasattr(self.rag_client, 'search'):
                # 同步搜索接口
                results = self.rag_client.search(
                    query=enhanced_query,
                    top_k=5,
                    metadata_filter={'module': context.module_name}
                )
            elif hasattr(self.rag_client, 'query'):
                # 替代查詢接口
                results = self.rag_client.query(
                    query=enhanced_query,
                    limit=5
                )
            else:
                logger.warning("RAG client does not have search() or query() method")
                return []
            
            # 解析結果
            parsed_results = self._parse_rag_results(results)
            
            logger.info(f"✅ RAG搜索完成: 找到 {len(parsed_results)} 個相關結果")
            return parsed_results
            
        except Exception as e:
            logger.error(f"❌ RAG查詢失敗: {e}", exc_info=True)
            return []
    
    def _parse_rag_results(self, results: Any) -> list[dict[str, Any]]:
        """解析RAG搜索結果
        
        Args:
            results: 原始RAG結果
            
        Returns:
            標準化的結果列表
        """
        parsed = []
        
        try:
            # 處理不同格式的結果
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        parsed.append({
                            'text': item.get('text', item.get('content', '')),
                            'score': item.get('score', item.get('similarity', 0.0)),
                            'metadata': item.get('metadata', {}),
                            'source': item.get('source', 'unknown')
                        })
                    elif hasattr(item, '__dict__'):
                        # 對象類型
                        parsed.append({
                            'text': getattr(item, 'text', getattr(item, 'content', '')),
                            'score': getattr(item, 'score', getattr(item, 'similarity', 0.0)),
                            'metadata': getattr(item, 'metadata', {}),
                            'source': getattr(item, 'source', 'unknown')
                        })
            elif isinstance(results, dict):
                # 單個結果
                parsed.append({
                    'text': results.get('text', results.get('content', '')),
                    'score': results.get('score', results.get('similarity', 0.0)),
                    'metadata': results.get('metadata', {}),
                    'source': results.get('source', 'unknown')
                })
        
        except Exception as e:
            logger.warning(f"⚠️ 解析RAG結果時出錯: {e}")
        
        return parsed
    
    def _generate_recommendation_id(self, context: ExecutionContext) -> str:
        """生成建議ID"""
        hash_input = f"{context.execution_id}_{context.timestamp}"
        return f"rec_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"
    
    def get_statistics(self) -> dict[str, Any]:
        """獲取統計資訊"""
        return {
            'total_modules': self.total_modules,
            'total_scenarios': self.total_scenarios,
            'match_stats': self.match_stats,
            'modules': list(self.module_knowledge.keys())
        }
    
    def export_knowledge_summary(self, output_file: str) -> None:
        """導出知識庫摘要"""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'modules': {}
        }
        
        for module_name, knowledge in self.module_knowledge.items():
            summary['modules'][module_name] = {
                'report_file': knowledge.get('report_file'),
                'scenario_counts': {
                    'success': len(knowledge['scenarios'].get('success', [])),
                    'suspicious': len(knowledge['scenarios'].get('suspicious', [])),
                    'failure': len(knowledge['scenarios'].get('failure', []))
                }
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge summary exported to {output_file}")
