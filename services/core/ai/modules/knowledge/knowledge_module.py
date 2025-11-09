"""
AIVA Knowledge Module v2.0 - 實用程式設計知識模組
知識模組 - 專注於程式開發實際需要的知識管理功能

實用功能導向的知識模組，只實現程式開發者真正需要的功能：
1. 程式碼分析與建議
2. 增強型RAG知識檢索
3. 上下文相關搜尋
4. 語義文檔查找

Author: AIVA Team
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import ast
import os
import re
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import pickle
from collections import defaultdict
import uuid

# 導入事件系統
from ...core.event_system.event_bus import AIEvent, AIEventBus, EventPriority
from ...core.controller.strangler_fig_controller import StranglerFigController, AIRequest, AIResponse, MessageType

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 程式碼分析引擎 ====================

class CodeAnalyzer(nn.Module):
    """程式碼分析引擎 - 用於程式建議"""
    
    def __init__(self, code_dim: int = 512, suggestion_dim: int = 256):
        super().__init__()
        
        # 程式碼語法編碼器
        self.syntax_encoder = nn.Sequential(
            nn.Linear(code_dim, suggestion_dim * 2),
            nn.LayerNorm(suggestion_dim * 2),
            nn.ReLU(),
            nn.Linear(suggestion_dim * 2, suggestion_dim)
        )
        
        # 模式識別網路
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(suggestion_dim, suggestion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(suggestion_dim, suggestion_dim // 2)
        )
        
        # 建議生成器
        self.suggestion_generator = nn.Linear(suggestion_dim // 2, suggestion_dim)
        
    def analyze_code(self, code_content: str, language: str = "python") -> Dict[str, Any]:
        """分析程式碼並提供建議"""
        
        # 基本語法分析
        syntax_analysis = self._analyze_syntax(code_content, language)
        
        # 程式碼品質評估
        quality_metrics = self._assess_code_quality(code_content, language)
        
        # 效能分析
        performance_analysis = self._analyze_performance(code_content, language)
        
        # 安全性檢查
        security_check = self._check_security(code_content, language)
        
        # 生成改進建議
        improvement_suggestions = self._generate_suggestions(
            syntax_analysis, quality_metrics, performance_analysis, security_check
        )
        
        return {
            'syntax_analysis': syntax_analysis,
            'quality_metrics': quality_metrics,
            'performance_analysis': performance_analysis,
            'security_check': security_check,
            'improvement_suggestions': improvement_suggestions,
            'overall_score': self._calculate_overall_score(quality_metrics),
            'language': language,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """語法分析"""
        if language.lower() == "python":
            return self._analyze_python_syntax(code)
        else:
            return self._analyze_generic_syntax(code)
    
    def _analyze_python_syntax(self, code: str) -> Dict[str, Any]:
        """Python 語法分析"""
        try:
            tree = ast.parse(code)
            
            # 分析 AST 結構
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            # 複雜度分析
            complexity = self._calculate_cyclomatic_complexity(tree)
            
            return {
                'is_valid': True,
                'classes_count': len(classes),
                'functions_count': len(functions),
                'imports_count': len(imports),
                'cyclomatic_complexity': complexity,
                'class_names': [cls.name for cls in classes],
                'function_names': [func.name for func in functions],
                'has_main_block': any(isinstance(node, ast.If) and 
                                     isinstance(node.test, ast.Compare) and
                                     hasattr(node.test.left, 'id') and
                                     node.test.left.id == '__name__' 
                                     for node in ast.walk(tree))
            }
            
        except SyntaxError as e:
            return {
                'is_valid': False,
                'syntax_error': str(e),
                'line_number': e.lineno,
                'error_offset': e.offset
            }
        except Exception as e:
            return {
                'is_valid': False,
                'analysis_error': str(e)
            }
    
    def _analyze_generic_syntax(self, code: str) -> Dict[str, Any]:
        """通用語法分析"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 基本統計
        return {
            'is_valid': True,  # 簡化假設
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'comment_lines': len([line for line in lines if line.strip().startswith(('#', '//', '/*'))]),
            'average_line_length': sum(len(line) for line in non_empty_lines) / max(len(non_empty_lines), 1),
            'max_line_length': max(len(line) for line in lines) if lines else 0
        }
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """計算圈複雜度"""
        complexity = 1  # 基礎複雜度
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _assess_code_quality(self, code: str, language: str) -> Dict[str, Any]:
        """程式碼品質評估"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 命名規範檢查
        naming_score = self._check_naming_conventions(code, language)
        
        # 註解密度
        comment_density = self._calculate_comment_density(lines)
        
        # 函式長度檢查
        function_length_score = self._check_function_length(code, language)
        
        # 巢狀深度檢查
        nesting_depth = self._calculate_max_nesting_depth(code)
        
        return {
            'naming_score': naming_score,
            'comment_density': comment_density,
            'function_length_score': function_length_score,
            'max_nesting_depth': nesting_depth,
            'line_count': len(non_empty_lines),
            'readability_score': (naming_score + comment_density + function_length_score) / 3
        }
    
    def _analyze_performance(self, code: str, language: str) -> Dict[str, Any]:
        """效能分析"""
        
        performance_issues = []
        
        if language.lower() == "python":
            # 檢查常見效能問題
            if 'for' in code and 'in range(len(' in code:
                performance_issues.append({
                    'type': 'inefficient_loop',
                    'description': 'Use direct iteration instead of range(len())',
                    'severity': 'medium'
                })
            
            if '+ str(' in code or 'str(' in code and '+' in code:
                performance_issues.append({
                    'type': 'string_concatenation',
                    'description': 'Consider using f-strings or join() for string concatenation',
                    'severity': 'low'
                })
            
            if 'global ' in code:
                performance_issues.append({
                    'type': 'global_variables',
                    'description': 'Global variables can impact performance and readability',
                    'severity': 'medium'
                })
        
        return {
            'performance_issues': performance_issues,
            'issue_count': len(performance_issues),
            'performance_score': max(0, 1 - len(performance_issues) * 0.2)  # 每個問題-0.2分
        }
    
    def _check_security(self, code: str, language: str) -> Dict[str, Any]:
        """安全性檢查"""
        
        security_issues = []
        
        # 檢查潛在的安全問題
        if 'eval(' in code:
            security_issues.append({
                'type': 'dangerous_eval',
                'description': 'Use of eval() can be dangerous',
                'severity': 'high'
            })
        
        if 'exec(' in code:
            security_issues.append({
                'type': 'dangerous_exec',
                'description': 'Use of exec() can be dangerous',
                'severity': 'high'
            })
        
        if 'os.system(' in code or 'subprocess.call(' in code:
            security_issues.append({
                'type': 'command_execution',
                'description': 'Direct command execution may pose security risks',
                'severity': 'medium'
            })
        
        # 檢查硬編碼密碼等
        password_patterns = ['password', 'passwd', 'pwd', 'secret', 'key', 'token']
        for pattern in password_patterns:
            if re.search(rf'{pattern}\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
                security_issues.append({
                    'type': 'hardcoded_secret',
                    'description': f'Potential hardcoded {pattern} detected',
                    'severity': 'high'
                })
                break
        
        return {
            'security_issues': security_issues,
            'issue_count': len(security_issues),
            'security_score': max(0, 1 - len(security_issues) * 0.3)  # 每個問題-0.3分
        }
    
    def _generate_suggestions(self, syntax_analysis: Dict, quality_metrics: Dict, 
                            performance_analysis: Dict, security_check: Dict) -> List[Dict[str, Any]]:
        """生成改進建議"""
        suggestions = []
        
        # 語法建議
        if not syntax_analysis.get('is_valid', True):
            suggestions.append({
                'category': 'syntax',
                'priority': 'high',
                'suggestion': 'Fix syntax errors before proceeding',
                'details': syntax_analysis.get('syntax_error', 'Unknown syntax issue')
            })
        
        # 品質建議
        if quality_metrics.get('comment_density', 0) < 0.1:
            suggestions.append({
                'category': 'documentation',
                'priority': 'medium',
                'suggestion': 'Add more comments to improve code readability',
                'details': f"Current comment density: {quality_metrics.get('comment_density', 0):.1%}"
            })
        
        if quality_metrics.get('max_nesting_depth', 0) > 4:
            suggestions.append({
                'category': 'complexity',
                'priority': 'medium',
                'suggestion': 'Consider refactoring to reduce nesting depth',
                'details': f"Current max nesting depth: {quality_metrics.get('max_nesting_depth', 0)}"
            })
        
        # 效能建議
        for issue in performance_analysis.get('performance_issues', []):
            suggestions.append({
                'category': 'performance',
                'priority': issue['severity'],
                'suggestion': issue['description'],
                'details': f"Issue type: {issue['type']}"
            })
        
        # 安全建議
        for issue in security_check.get('security_issues', []):
            suggestions.append({
                'category': 'security',
                'priority': issue['severity'],
                'suggestion': issue['description'],
                'details': f"Security risk: {issue['type']}"
            })
        
        return suggestions
    
    def _check_naming_conventions(self, code: str, language: str) -> float:
        """檢查命名規範"""
        if language.lower() == "python":
            # Python PEP 8 命名檢查
            snake_case_pattern = r'^[a-z_][a-z0-9_]*$'
            camel_case_pattern = r'^[A-Z][a-zA-Z0-9]*$'
            
            # 簡化檢查（實際應用可以更完整）
            try:
                tree = ast.parse(code)
                
                # 檢查函式名稱
                function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                valid_functions = sum(1 for name in function_names if re.match(snake_case_pattern, name))
                
                # 檢查類別名稱
                class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                valid_classes = sum(1 for name in class_names if re.match(camel_case_pattern, name))
                
                total_names = len(function_names) + len(class_names)
                valid_names = valid_functions + valid_classes
                
                return valid_names / total_names if total_names > 0 else 1.0
                
            except (SyntaxError, ValueError, TypeError):
                return 0.5  # 無法解析則給予中等分數
        
        return 0.7  # 其他語言給予預設分數
    
    def _calculate_comment_density(self, lines: List[str]) -> float:
        """計算註解密度"""
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith(('#', '//', '/*', '"""', "'''"))]
        
        return len(comment_lines) / max(len(non_empty_lines), 1)
    
    def _check_function_length(self, code: str, language: str) -> float:
        """檢查函式長度"""
        if language.lower() == "python":
            try:
                tree = ast.parse(code)
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                
                if not functions:
                    return 1.0
                
                # 計算平均函式長度（以行為單位）
                total_length = 0
                for func in functions:
                    func_lines = func.end_lineno - func.lineno + 1
                    total_length += func_lines
                
                avg_length = total_length / len(functions)
                
                # 理想長度 <= 20行，超過扣分
                if avg_length <= 20:
                    return 1.0
                elif avg_length <= 50:
                    return 0.8
                else:
                    return 0.6
                    
            except (SyntaxError, ValueError, TypeError):
                return 0.7
        
        return 0.8  # 其他語言預設分數
    
    def _calculate_max_nesting_depth(self, code: str) -> int:
        """計算最大巢狀深度"""
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
                
            # 計算縮排深度
            indent_level = (len(line) - len(stripped)) // 4  # 假設4空格縮排
            
            # 檢查是否為控制結構
            if any(stripped.startswith(keyword) for keyword in ['if', 'for', 'while', 'try', 'with', 'def', 'class']):
                current_depth = indent_level + 1
                max_depth = max(max_depth, current_depth)
        
        return max_depth
    
    def _calculate_overall_score(self, quality_metrics: Dict[str, Any]) -> float:
        """計算總體分數"""
        readability = quality_metrics.get('readability_score', 0.5)
        
        # 綜合評分
        score = readability
        
        # 調整分數
        if quality_metrics.get('max_nesting_depth', 0) > 5:
            score -= 0.2
        
        if quality_metrics.get('line_count', 0) > 500:
            score -= 0.1
        
        return max(0, min(1, score))

# ==================== 增強型 RAG 檢索引擎 ====================

class EnhancedRAGRetriever(nn.Module):
    """增強型 RAG (Retrieval-Augmented Generation) 檢索引擎"""
    
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 512):
        super().__init__()
        
        # 文本編碼器
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 查詢編碼器
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 相關性評分器
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 知識庫
        self.knowledge_base = {}
        self.document_embeddings = {}
        
    def store_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """儲存文檔到知識庫"""
        try:
            # 簡化的文檔處理
            processed_content = self._preprocess_text(content)
            
            # 生成文檔嵌入（簡化實現）
            doc_embedding = self._generate_embedding(processed_content)
            
            # 儲存
            self.knowledge_base[doc_id] = {
                'content': content,
                'processed_content': processed_content,
                'metadata': metadata or {},
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'word_count': len(processed_content.split()),
                'doc_type': self._infer_doc_type(content, metadata)
            }
            
            self.document_embeddings[doc_id] = doc_embedding
            
            return True
            
        except Exception as e:
            logger.error(f"儲存文檔失敗: {str(e)}")
            return False
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5, 
                              min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """檢索相關文檔"""
        
        if not self.knowledge_base:
            return []
        
        # 處理查詢
        processed_query = self._preprocess_text(query)
        query_embedding = self._generate_embedding(processed_query)
        
        # 計算相似度
        doc_scores = []
        
        for doc_id, doc_info in self.knowledge_base.items():
            doc_embedding = self.document_embeddings[doc_id]
            
            # 計算相似度（簡化為餘弦相似度）
            similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
            
            # 上下文相關性加權
            context_boost = self._calculate_context_boost(query, doc_info)
            
            # 最終分數
            final_score = similarity * (1 + context_boost)
            
            if final_score >= min_relevance:
                doc_scores.append({
                    'doc_id': doc_id,
                    'score': final_score,
                    'similarity': similarity,
                    'context_boost': context_boost,
                    'doc_info': doc_info
                })
        
        # 按分數排序並返回 top_k
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return doc_scores[:top_k]
    
    def _preprocess_text(self, text: str) -> str:
        """文本預處理"""
        # 基本清理
        text = re.sub(r'\s+', ' ', text)  # 正規化空白字符
        text = text.lower().strip()
        
        # 移除特殊字符但保留程式碼相關符號
        allowed_chars = r'[^\w\s.,;:()\[\]{}+=<>-]'
        text = re.sub(allowed_chars, '', text)
        
        return text
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """生成文本嵌入（簡化實現）"""
        # 這裡使用簡化的TF-IDF風格嵌入
        words = text.split()
        
        # 創建詞頻向量
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 簡化嵌入：使用固定維度的向量
        embedding = np.zeros(384)  # 固定384維
        
        for i, (word, freq) in enumerate(word_freq.items()):
            if i >= 384:
                break
            # 簡單的雜湊映射到向量維度
            hash_val = hash(word) % 384
            embedding[hash_val] += freq
        
        # 正規化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """計算餘弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except (ZeroDivisionError, ValueError, TypeError):
            return 0.0
    
    def _calculate_context_boost(self, query: str, doc_info: Dict[str, Any]) -> float:
        """計算上下文相關性加權"""
        boost = 0.0
        
        # 文檔類型相關性
        doc_type = doc_info.get('doc_type', 'unknown')
        query_lower = query.lower()
        
        # 文檔類型匹配加權
        doc_type_boost_map = {
            ('code', 'code'): 0.2,
            ('documentation', 'documentation'): 0.15,  # 稍微不同的值避免重複
            ('api', 'api'): 0.25  # API 文檔給予更高權重
        }
        
        for (query_term, target_type), boost_value in doc_type_boost_map.items():
            if query_term in query_lower and doc_type == target_type:
                boost += boost_value
                break
        
        # 新鮮度加權
        timestamp = doc_info.get('timestamp', '')
        if timestamp:
            try:
                doc_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age_days = (datetime.now(timezone.utc) - doc_time).days
                
                # 較新的文檔給予小幅加成
                if age_days < 7:
                    boost += 0.1
                elif age_days < 30:
                    boost += 0.05
            except (ValueError, KeyError, TypeError):
                pass
        
        # 文檔長度適中加成
        word_count = doc_info.get('word_count', 0)
        if 50 <= word_count <= 500:
            boost += 0.05
        
        return min(boost, 0.5)  # 最大加成50%
    
    def _infer_doc_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """推斷文檔類型"""
        if metadata and 'type' in metadata:
            return metadata['type']
        
        content_lower = content.lower()
        
        # 程式碼偵測
        code_indicators = ['def ', 'class ', 'import ', 'function', '():', 'return ', 'if __name__']
        if any(indicator in content_lower for indicator in code_indicators):
            return 'code'
        
        # API 文檔偵測
        api_indicators = ['api', 'endpoint', 'request', 'response', 'parameter', 'http']
        if any(indicator in content_lower for indicator in api_indicators):
            return 'api'
        
        # 文檔偵測
        doc_indicators = ['documentation', 'guide', 'tutorial', 'readme', 'manual']
        if any(indicator in content_lower for indicator in doc_indicators):
            return 'documentation'
        
        return 'general'

# ==================== 語義搜索引擎 ====================

class SemanticSearchEngine:
    """語義搜索引擎 - 用於文檔查找"""
    
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path or "knowledge_index.pkl"
        self.search_index = {}
        self.document_store = {}
        
        # 載入現有索引
        self._load_index()
    
    def index_document(self, doc_id: str, content: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """為文檔建立索引"""
        try:
            # 文本分析
            words = self._extract_keywords(content)
            
            # 建立倒排索引
            for word in words:
                if word not in self.search_index:
                    self.search_index[word] = set()
                self.search_index[word].add(doc_id)
            
            # 儲存文檔資訊
            self.document_store[doc_id] = {
                'content': content,
                'metadata': metadata or {},
                'keywords': words,
                'indexed_at': datetime.now(timezone.utc).isoformat(),
                'word_count': len(content.split())
            }
            
            # 持久化索引
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"文檔索引失敗: {str(e)}")
            return False
    
    def semantic_search(self, query: str, max_results: int = 10, 
                       search_type: str = "fuzzy") -> List[Dict[str, Any]]:
        """語義搜索"""
        
        query_words = self._extract_keywords(query)
        
        if search_type == "exact":
            results = self._exact_search(query_words)
        elif search_type == "fuzzy":
            results = self._fuzzy_search(query_words)
        else:  # semantic
            results = self._semantic_search(query_words)
        
        # 排序和過濾結果
        sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        
        return sorted_results[:max_results]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取關鍵詞"""
        # 基本文本清理
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # 過濾停用詞
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        # 過濾並返回關鍵詞
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def _exact_search(self, query_words: List[str]) -> List[Dict[str, Any]]:
        """精確搜索"""
        results = []
        
        for doc_id, doc_info in self.document_store.items():
            doc_keywords = set(doc_info['keywords'])
            query_set = set(query_words)
            
            # 計算完全匹配
            exact_matches = len(query_set & doc_keywords)
            
            if exact_matches > 0:
                relevance = exact_matches / len(query_set)
                
                results.append({
                    'doc_id': doc_id,
                    'content': doc_info['content'][:200] + "...",  # 預覽
                    'metadata': doc_info['metadata'],
                    'relevance_score': relevance,
                    'match_type': 'exact',
                    'matched_keywords': list(query_set & doc_keywords)
                })
        
        return results
    
    def _fuzzy_search(self, query_words: List[str]) -> List[Dict[str, Any]]:
        """模糊搜索"""
        results = []
        
        for doc_id, doc_info in self.document_store.items():
            doc_keywords = doc_info['keywords']
            
            match_score = 0
            matched_words = []
            
            for query_word in query_words:
                best_match_score = 0
                best_match_word = None
                
                for doc_word in doc_keywords:
                    # 計算字符串相似度
                    similarity = self._calculate_string_similarity(query_word, doc_word)
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_word = doc_word
                
                # 如果相似度超過閾值，計入匹配
                if best_match_score > 0.7:
                    match_score += best_match_score
                    matched_words.append((query_word, best_match_word, best_match_score))
            
            if match_score > 0:
                relevance = match_score / len(query_words)
                
                results.append({
                    'doc_id': doc_id,
                    'content': doc_info['content'][:200] + "...",
                    'metadata': doc_info['metadata'],
                    'relevance_score': relevance,
                    'match_type': 'fuzzy',
                    'matched_words': matched_words
                })
        
        return results
    
    def _semantic_search(self, query_words: List[str]) -> List[Dict[str, Any]]:
        """語義搜索（基於同義詞和相關概念）"""
        
        # 擴展查詢詞（簡化的同義詞擴展）
        expanded_words = query_words.copy()
        
        # 程式設計相關同義詞
        synonyms = {
            'function': ['method', 'procedure', 'routine'],
            'class': ['object', 'type', 'struct'],
            'variable': ['var', 'field', 'property'],
            'error': ['exception', 'bug', 'issue'],
            'test': ['testing', 'unittest', 'spec'],
            'api': ['interface', 'service', 'endpoint'],
            'data': ['information', 'dataset', 'values'],
            'code': ['source', 'script', 'program']
        }
        
        for word in query_words:
            if word in synonyms:
                expanded_words.extend(synonyms[word])
        
        # 使用擴展詞進行搜索
        return self._fuzzy_search(expanded_words)
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """計算字符串相似度（Jaccard相似度）"""
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            index_data = {
                'search_index': {word: list(doc_set) for word, doc_set in self.search_index.items()},
                'document_store': self.document_store,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.index_path, 'wb') as f:
                pickle.dump(index_data, f)
                
        except Exception as e:
            logger.error(f"保存索引失敗: {str(e)}")
    
    def _load_index(self):
        """從文件載入索引"""
        try:
            if os.path.exists(self.index_path):
                with open(self.index_path, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.search_index = {
                    word: set(doc_list) 
                    for word, doc_list in index_data.get('search_index', {}).items()
                }
                self.document_store = index_data.get('document_store', {})
                
                logger.info(f"載入索引完成，包含 {len(self.document_store)} 個文檔")
                
        except Exception as e:
            logger.error(f"載入索引失敗: {str(e)}")
            self.search_index = {}
            self.document_store = {}

# ==================== 知識模組主類 ====================

class KnowledgeModuleV2:
    """知識模組 v2.0 - 實用功能導向"""
    
    def __init__(self, event_bus: Optional[AIEventBus] = None):
        self.module_name = "knowledge"
        self.module_version = "v2.0"
        
        # 事件系統
        self.event_bus = event_bus
        
        # 初始化功能組件
        self.code_analyzer = CodeAnalyzer(512, 256)
        self.rag_retriever = EnhancedRAGRetriever(384, 512)
        self.semantic_engine = SemanticSearchEngine()
        
        # 統計資訊
        self.stats = {
            'code_analyses': 0,
            'document_retrievals': 0,
            'semantic_searches': 0,
            'documents_stored': 0,
            'avg_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info(f"知識模組 {self.module_version} 初始化完成")
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """處理知識模組請求"""
        start_time = time.time()
        
        try:
            operation = request.operation
            payload = request.payload
            
            if operation == 'analyze_code':
                result = self.analyze_code(payload)
            elif operation == 'retrieve_knowledge':
                result = self.retrieve_knowledge(payload)
            elif operation == 'semantic_search':
                result = self.semantic_search(payload)
            elif operation == 'store_document':
                result = self.store_document(payload)
            else:
                raise ValueError(f"不支援的操作: {operation}")
            
            # 發布成功事件
            if self.event_bus:
                await self._publish_event(f'knowledge.{operation}.completed', {
                    'request_id': request.request_id,
                    'operation': operation,
                    'result_summary': self._summarize_result(result),
                    'processing_time': (time.time() - start_time) * 1000
                })
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(operation, processing_time, True)
            
            return AIResponse(
                request_id=request.request_id,
                status="success",
                processed_by=f"{self.module_name}@{self.module_version}",
                execution_time_ms=processing_time,
                result=result,
                metadata={
                    'module': self.module_name,
                    'version': self.module_version,
                    'operation': operation
                }
            )
            
        except Exception as e:
            # 發布錯誤事件
            if self.event_bus:
                await self._publish_event('knowledge.error.occurred', {
                    'request_id': request.request_id,
                    'operation': request.operation,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }, priority=EventPriority.HIGH)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(request.operation, processing_time, False)
            
            logger.error(f"知識模組處理錯誤: {str(e)}")
            
            return AIResponse(
                request_id=request.request_id,
                status="error",
                processed_by=f"{self.module_name}@{self.module_version}",
                execution_time_ms=processing_time,
                error={
                    "type": type(e).__name__,
                    "message": str(e)
                }
            )
    
    def analyze_code(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """程式碼分析"""
        logger.info("開始程式碼分析...")
        
        code_content = payload.get('code', '')
        language = payload.get('language', 'python')
        
        if not code_content:
            raise ValueError("程式碼內容不能為空")
        
        # 執行程式碼分析
        analysis_result = self.code_analyzer.analyze_code(code_content, language)
        
        # 更新統計
        self.stats['code_analyses'] += 1
        
        result = {
            'analysis': analysis_result,
            'recommendations': self._generate_code_recommendations(analysis_result),
            'confidence': analysis_result['overall_score'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"程式碼分析完成，總分: {analysis_result['overall_score']:.2f}")
        
        return result
    
    def retrieve_knowledge(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """知識檢索（RAG）"""
        logger.info("開始知識檢索...")
        
        query = payload.get('query', '')
        top_k = payload.get('top_k', 5)
        min_relevance = payload.get('min_relevance', 0.3)
        
        if not query:
            raise ValueError("檢索查詢不能為空")
        
        # 執行RAG檢索
        relevant_docs = self.rag_retriever.retrieve_relevant_docs(query, top_k, min_relevance)
        
        # 更新統計
        self.stats['document_retrievals'] += 1
        
        result = {
            'query': query,
            'relevant_documents': relevant_docs,
            'total_found': len(relevant_docs),
            'retrieval_quality': self._assess_retrieval_quality(relevant_docs),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"知識檢索完成，找到 {len(relevant_docs)} 個相關文檔")
        
        return result
    
    def semantic_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """語義搜索"""
        logger.info("開始語義搜索...")
        
        query = payload.get('query', '')
        max_results = payload.get('max_results', 10)
        search_type = payload.get('search_type', 'fuzzy')
        
        if not query:
            raise ValueError("搜索查詢不能為空")
        
        # 執行語義搜索
        search_results = self.semantic_engine.semantic_search(query, max_results, search_type)
        
        # 更新統計
        self.stats['semantic_searches'] += 1
        
        result = {
            'query': query,
            'search_type': search_type,
            'results': search_results,
            'total_found': len(search_results),
            'search_quality': self._assess_search_quality(search_results),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"語義搜索完成，找到 {len(search_results)} 個結果")
        
        return result
    
    def store_document(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """儲存文檔"""
        logger.info("開始儲存文檔...")
        
        content = payload.get('content', '')
        doc_id = payload.get('doc_id') or str(uuid.uuid4())
        metadata = payload.get('metadata', {})
        
        if not content:
            raise ValueError("文檔內容不能為空")
        
        # 儲存到RAG系統
        rag_success = self.rag_retriever.store_document(doc_id, content, metadata)
        
        # 建立語義搜索索引
        semantic_success = self.semantic_engine.index_document(doc_id, content, metadata)
        
        if rag_success and semantic_success:
            self.stats['documents_stored'] += 1
            
            result = {
                'doc_id': doc_id,
                'storage_success': True,
                'rag_indexed': rag_success,
                'semantic_indexed': semantic_success,
                'content_length': len(content),
                'word_count': len(content.split()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"文檔儲存成功: {doc_id}")
        else:
            result = {
                'doc_id': doc_id,
                'storage_success': False,
                'error': '文檔儲存或索引失敗',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.error(f"文檔儲存失敗: {doc_id}")
        
        return result
    
    def _generate_code_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基於分析結果生成程式碼建議"""
        recommendations = []
        
        # 基於改進建議生成具體建議
        for suggestion in analysis.get('improvement_suggestions', []):
            recommendation = {
                'category': suggestion['category'],
                'priority': suggestion['priority'],
                'title': suggestion['suggestion'],
                'description': suggestion['details'],
                'actionable': True
            }
            
            # 添加具體的操作建議
            if suggestion['category'] == 'performance':
                recommendation['actions'] = [
                    'Review loops and data structures',
                    'Consider algorithm optimization',
                    'Profile performance bottlenecks'
                ]
            elif suggestion['category'] == 'security':
                recommendation['actions'] = [
                    'Review security vulnerabilities',
                    'Use secure coding practices',
                    'Validate inputs and sanitize outputs'
                ]
            elif suggestion['category'] == 'documentation':
                recommendation['actions'] = [
                    'Add docstrings to functions',
                    'Include inline comments',
                    'Document complex logic'
                ]
            else:
                recommendation['actions'] = ['Review and improve code quality']
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _assess_retrieval_quality(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """評估檢索品質"""
        if not retrieved_docs:
            return {'score': 0.0, 'quality': 'poor'}
        
        # 平均相關性分數
        avg_relevance = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
        
        # 文檔多樣性（簡化評估）
        doc_types = {doc['doc_info'].get('doc_type', 'unknown') for doc in retrieved_docs}
        diversity_score = len(doc_types) / max(len(retrieved_docs), 1)
        
        # 綜合品質分數
        quality_score = (avg_relevance * 0.7 + diversity_score * 0.3)
        
        if quality_score > 0.8:
            quality_level = 'excellent'
        elif quality_score > 0.6:
            quality_level = 'good'
        elif quality_score > 0.4:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'score': quality_score,
            'quality': quality_level,
            'avg_relevance': avg_relevance,
            'diversity_score': diversity_score
        }
    
    def _assess_search_quality(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """評估搜索品質"""
        if not search_results:
            return {'score': 0.0, 'quality': 'poor'}
        
        # 平均相關性分數
        avg_relevance = sum(result['relevance_score'] for result in search_results) / len(search_results)
        
        # 匹配類型分析
        match_types = [result['match_type'] for result in search_results]
        exact_matches = match_types.count('exact')
        exact_ratio = exact_matches / len(match_types) if match_types else 0
        
        # 綜合品質分數
        quality_score = (avg_relevance * 0.8 + exact_ratio * 0.2)
        
        if quality_score > 0.8:
            quality_level = 'excellent'
        elif quality_score > 0.6:
            quality_level = 'good'
        elif quality_score > 0.4:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        return {
            'score': quality_score,
            'quality': quality_level,
            'avg_relevance': avg_relevance,
            'exact_match_ratio': exact_ratio
        }
    
    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """總結結果用於事件"""
        summary = {
            'status': 'completed',
            'data_points': len(str(result))
        }
        
        # 添加特定操作的總結資訊
        if 'analysis' in result:
            summary['overall_score'] = result['analysis'].get('overall_score', 0)
        if 'total_found' in result:
            summary['results_count'] = result['total_found']
        if 'storage_success' in result:
            summary['storage_success'] = result['storage_success']
        
        return summary
    
    async def _publish_event(self, event_type: str, data: Dict[str, Any], 
                           priority: EventPriority = EventPriority.NORMAL):
        """發布事件"""
        if not self.event_bus:
            return
            
        event = AIEvent(
            event_type=event_type,
            source_module=self.module_name,
            source_version=self.module_version,
            data=data,
            priority=priority
        )
        
        await self.event_bus.publish(event)
    
    def _update_stats(self, operation: str, processing_time: float, success: bool):
        """更新統計資訊"""
        # 記錄操作類型統計
        if operation not in self.stats:
            self.stats[f'{operation}_count'] = 0
        self.stats[f'{operation}_count'] += 1
        
        # 更新平均處理時間
        if self.stats['avg_processing_time'] == 0:
            self.stats['avg_processing_time'] = processing_time
        else:
            alpha = 0.1
            self.stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_processing_time']
            )
        
        # 更新成功率
        total_ops = sum([
            self.stats['code_analyses'],
            self.stats['document_retrievals'],
            self.stats['semantic_searches'],
            self.stats['documents_stored']
        ])
        
        if total_ops > 0:
            successful_ops = total_ops - (0 if success else 1)
            self.stats['success_rate'] = successful_ops / total_ops
    
    def get_health_status(self) -> Dict[str, Any]:
        """獲取健康狀態"""
        total_operations = sum([
            self.stats['code_analyses'],
            self.stats['document_retrievals'],
            self.stats['semantic_searches'],
            self.stats['documents_stored']
        ])
        
        if self.stats['success_rate'] > 0.9:
            status = 'healthy'
        elif self.stats['success_rate'] > 0.7:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'module': self.module_name,
            'version': self.module_version,
            'status': status,
            'statistics': self.stats,
            'total_operations': total_operations,
            'knowledge_base': {
                'rag_documents': len(self.rag_retriever.knowledge_base),
                'indexed_documents': len(self.semantic_engine.document_store)
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# ==================== 測試和示例 ====================

async def test_knowledge_module():
    """測試知識模組"""
    
    print("📚 測試知識模組 v2.0 - 實用功能")
    print("=" * 50)
    
    # 創建知識模組
    knowledge = KnowledgeModuleV2()
    
    # 測試程式碼分析
    print("\n🔍 測試程式碼分析...")
    sample_code = '''
def calculate_fibonacci(n):
    """Calculate Fibonacci number using recursion"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_data(self, items):
        for i in range(len(items)):  # 效能問題：應該用 for item in items
            self.data.append(items[i])
    
    def get_result(self):
        return str(self.data[0]) + str(self.data[1])  # 效能問題：應該用 f-string
'''
    
    request = AIRequest(
        message_type=MessageType.COMMAND,
        source_module="test",
        operation="analyze_code",
        payload={
            'code': sample_code,
            'language': 'python'
        }
    )
    
    response = await knowledge.process_request(request)
    print(f"✅ 程式碼分析完成: {response.status}")
    if response.result:
        analysis = response.result['analysis']
        print(f"📊 整體分數: {analysis['overall_score']:.2f}")
        print(f"🔧 改進建議數量: {len(response.result['recommendations'])}")
        print(f"⚠️ 效能問題: {len(analysis['performance_analysis']['performance_issues'])}")
    
    # 測試文檔儲存
    print("\n💾 測試文檔儲存...")
    sample_docs = [
        {
            'content': 'Python函式定義使用 def 關鍵字。函式可以接受參數並返回值。良好的函式應該有明確的文檔字符串。',
            'metadata': {'type': 'documentation', 'topic': 'python_functions', 'language': 'zh-tw'}
        },
        {
            'content': 'def example_function(param1, param2=None): """This is a docstring""" return param1 + (param2 or 0)',
            'metadata': {'type': 'code', 'topic': 'python_functions', 'language': 'python'}
        },
        {
            'content': 'API設計最佳實踐：使用RESTful架構，明確的端點命名，適當的HTTP狀態碼，完整的錯誤處理。',
            'metadata': {'type': 'api', 'topic': 'api_design', 'language': 'zh-tw'}
        }
    ]
    
    stored_docs = 0
    for doc in sample_docs:
        request = AIRequest(
            message_type=MessageType.COMMAND,
            source_module="test",
            operation="store_document",
            payload=doc
        )
        
        response = await knowledge.process_request(request)
        if response.result and response.result.get('storage_success'):
            stored_docs += 1
    
    print(f"✅ 成功儲存 {stored_docs} 個文檔")
    
    # 測試知識檢索 (RAG)
    print("\n🔍 測試知識檢索 (RAG)...")
    request = AIRequest(
        message_type=MessageType.QUERY,
        source_module="test",
        operation="retrieve_knowledge",
        payload={
            'query': 'python function definition parameters',
            'top_k': 3
        }
    )
    
    response = await knowledge.process_request(request)
    print(f"✅ 知識檢索完成: {response.status}")
    if response.result:
        print(f"📄 找到相關文檔: {response.result['total_found']}")
        print(f"⭐ 檢索品質: {response.result['retrieval_quality']['quality']}")
    
    # 測試語義搜索
    print("\n🔍 測試語義搜索...")
    request = AIRequest(
        message_type=MessageType.QUERY,
        source_module="test",
        operation="semantic_search",
        payload={
            'query': 'API design best practices',
            'search_type': 'fuzzy'
        }
    )
    
    response = await knowledge.process_request(request)
    print(f"✅ 語義搜索完成: {response.status}")
    if response.result:
        print(f"🔍 搜索結果: {response.result['total_found']}")
        print(f"⭐ 搜索品質: {response.result['search_quality']['quality']}")
    
    # 獲取健康狀態
    health = knowledge.get_health_status()
    print(f"\n💚 模組健康狀態: {health['status']}")
    print(f"📈 成功率: {health['statistics']['success_rate']:.1%}")
    print(f"📚 知識庫文檔數: RAG={health['knowledge_base']['rag_documents']}, 索引={health['knowledge_base']['indexed_documents']}")

if __name__ == "__main__":
    asyncio.run(test_knowledge_module())