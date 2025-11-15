"""
AI增強代碼分析引擎
基於Tree-sitter AST和神經網路的智能代碼分析系統
"""
import ast
import hashlib
import json
import logging
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import numpy as np

# Tree-sitter 條件導入 - 用於增強解析功能
TREE_SITTER_AVAILABLE = False
tree_sitter = None  # type: ignore

try:
    import tree_sitter  # type: ignore
    TREE_SITTER_AVAILABLE = True
except ImportError:
    # tree_sitter 套件未安裝，使用備用解析方案
    pass

from ..bio_neuron_master import BioNeuronMasterController
from ..ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context

MODULE_NAME = "ai_analysis.analysis_engine"

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """分析類型枚舉"""
    SECURITY = "security"
    VULNERABILITY = "vulnerability" 
    COMPLEXITY = "complexity"
    PATTERNS = "patterns"
    SEMANTIC = "semantic"
    ARCHITECTURE = "architecture"

@dataclass
class IndexingConfig:
    """索引配置（從RAG 1遷移）"""
    batch_size: int = 100  # 批次處理大小
    max_workers: int = 4   # 並行工作線程數
    cache_enabled: bool = True  # 是否啟用緩存
    max_file_size: int = 1024 * 1024  # 1MB 最大檔案大小


class CacheManager:
    """緩存管理器，避免重複索引（從RAG 1遷移）"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = cache_dir / "analysis_cache.json"
        self._load_cache_index()
    
    def _load_cache_index(self):
        """載入緩存索引"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"載入緩存索引失敗: {e}")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def get_file_hash(self, file_path: Path) -> str:
        """計算檔案雜湊值"""
        try:
            stat = file_path.stat()
            hash_str = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_str.encode()).hexdigest()
        except Exception:
            return ""
    
    def is_cached(self, file_path: Path) -> bool:
        """檢查檔案是否已緩存且未過期"""
        file_hash = self.get_file_hash(file_path)
        cached_hash = self.cache_index.get(str(file_path), "")
        return file_hash == cached_hash and file_hash != ""
    
    def update_cache(self, file_path: Path):
        """更新檔案緩存記錄"""
        file_hash = self.get_file_hash(file_path)
        self.cache_index[str(file_path)] = file_hash
        
    def save_cache_index(self):
        """保存緩存索引"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存緩存索引失敗: {e}")


@dataclass
class AIAnalysisResult:
    """AI分析結果數據類"""
    analysis_type: AnalysisType
    confidence: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str
    explanation: str
    metadata: Dict[str, Any]


@dataclass
class CodeChunk:
    """程式碼片段數據類（從RAG 1遷移）"""
    id: int
    path: str
    content: str
    node_type: str  # FunctionDef/ClassDef/Module
    node_name: str
    keywords: Optional[set[str]] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = set()

class AIAnalysisEngine:
    """
    AI驅動的代碼分析引擎
    結合傳統AST分析與神經網路增強
    整合RAG 1的代碼索引和分析功能
    """
    
    def __init__(self, codebase_path: str = "/workspaces/AIVA", config: Optional[IndexingConfig] = None):
        self.bio_controller = None
        self.rag_agent = None
        self.initialized = False
        
        # 從RAG 1遷移的代碼索引功能
        self.codebase_path = Path(codebase_path)
        self.chunks: list[CodeChunk] = []
        self.index: dict[str, list[int]] = {}  # 關鍵字 -> chunk 索引
        
        # 配置和緩存
        self.config = config or IndexingConfig()
        cache_dir = self.codebase_path / ".aiva_analysis_cache"
        self.cache_manager = CacheManager(cache_dir) if self.config.cache_enabled else None
        
        # 性能追蹤
        self.stats = {
            "indexed_files": 0,
            "skipped_files": 0, 
            "cached_files": 0,
            "total_chunks": 0,
            "indexing_time": 0.0
        }
        
    def initialize(self) -> bool:
        """初始化AI分析引擎"""
        try:
            # 初始化生物神經網路控制器
            self.bio_controller = BioNeuronMasterController()
            
            # 創建真實的決策核心
            from ..ai_engine.real_bio_net_adapter import create_real_scalable_bionet, create_real_rag_agent
            try:
                real_decision_core = create_real_scalable_bionet(
                    input_size=1024,
                    num_tools=10,
                    weights_path=None  # 使用隨機初始化
                )
                
                # 初始化RAG代理用於代碼分析
                self.rag_agent = create_real_rag_agent(
                    decision_core=real_decision_core,
                    input_vector_size=1024
                )
            except Exception as e:
                # 降級到無RAG模式
                self.rag_agent = None
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"AI分析引擎初始化失敗: {e}")
            return False
    
    def _extract_code_features(self, source_code: str) -> torch.Tensor:
        """從原始碼中提取特徵向量"""
        try:
            # 解析AST
            tree = ast.parse(source_code)
            
            # 基本特徵統計
            features = []
            
            # 1. 代碼長度特徵
            features.append(len(source_code))
            features.append(len(source_code.splitlines()))
            
            # 2. AST節點統計
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            # 關鍵節點類型統計
            critical_nodes = [
                'FunctionDef', 'ClassDef', 'If', 'For', 'While', 
                'Try', 'Import', 'Call', 'Assign', 'Compare'
            ]
            
            for node_type in critical_nodes:
                features.append(node_counts.get(node_type, 0))
            
            # 3. 複雜度特徵
            features.append(self._calculate_cyclomatic_complexity(tree))
            features.append(self._calculate_nesting_depth(tree))
            
            # 4. 安全特徵
            features.extend(self._extract_security_features(tree, source_code))
            
            # 5. 語義特徵
            features.extend(self._extract_semantic_features(tree))
            
            # 補齊到1024維度
            while len(features) < 1024:
                features.append(0.0)
            
            # 如果超過1024維，截斷
            features = features[:1024]
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"特徵提取失敗: {e}")
            # 返回零向量
            return torch.zeros(1024, dtype=torch.float32)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """計算循環複雜度"""
        complexity = 1  # 基礎複雜度
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, 
                               ast.Try, ast.ExceptHandler, ast.Match)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Compare):
                complexity += len(node.ops)
                
        return complexity
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """計算嵌套深度"""
        max_depth = 0
        
        def visit_node(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, 
                               ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
                
            for child in ast.iter_child_nodes(node):
                visit_node(child, depth)
        
        visit_node(tree)
        return max_depth
    
    def _extract_security_features(self, tree: ast.AST, source_code: str) -> List[float]:
        """提取安全相關特徵"""
        features = []
        
        # 危險函數調用
        dangerous_funcs = ['eval', 'exec', 'input', 'open', 'subprocess', '__import__']
        dangerous_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_funcs:
                    dangerous_count += 1
        
        features.append(dangerous_count)
        
        # SQL注入風險模式
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']
        sql_risk = sum(1 for pattern in sql_patterns if pattern in source_code.upper())
        features.append(sql_risk)
        
        # 硬編碼密碼模式
        password_patterns = ['password', 'passwd', 'pwd', 'secret', 'token']
        hardcoded_secrets = 0
        for line in source_code.lower().splitlines():
            if any(pattern in line and '=' in line for pattern in password_patterns):
                hardcoded_secrets += 1
        features.append(hardcoded_secrets)
        
        return features
    
    def _extract_semantic_features(self, tree: ast.AST) -> List[float]:
        """提取語義特徵"""
        features = []
        
        # 函數參數統計
        func_param_counts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_param_counts.append(len(node.args.args))
        
        features.append(np.mean(func_param_counts) if func_param_counts else 0)
        features.append(np.max(func_param_counts) if func_param_counts else 0)
        
        # 類方法統計
        class_method_counts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for child in node.body if isinstance(child, ast.FunctionDef))
                class_method_counts.append(method_count)
        
        features.append(np.mean(class_method_counts) if class_method_counts else 0)
        
        # 變數名長度統計
        var_name_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name_lengths.append(len(node.id))
        
        features.append(np.mean(var_name_lengths) if var_name_lengths else 0)
        
        return features
    
    def analyze_code(
        self, 
        source_code: str, 
        file_path: str = "",
        analysis_types: Optional[List[AnalysisType]] = None
    ) -> Dict[AnalysisType, AIAnalysisResult]:
        """
        AI增強的代碼分析主函數
        按照業務流程：初始化檢查 -> 特徵提取 -> 分析類型選擇 -> AI分析 -> 風險計算 -> 結果生成
        """
        # 業務流程步驟1: 初始化檢查
        if not self.initialized:
            logger.info("AI分析引擎未初始化，開始初始化流程...")
            if not self.initialize():
                logger.error("AI分析引擎初始化失敗")
                return self._create_failed_results(analysis_types or [AnalysisType.SECURITY], "Engine initialization failed")
            logger.info("AI分析引擎初始化成功")
        
        # 業務流程步驟2: 選擇分析類型
        if analysis_types is None:
            analysis_types = [AnalysisType.SECURITY, AnalysisType.COMPLEXITY, AnalysisType.PATTERNS]
            logger.info(f"使用預設分析類型: {[at.value for at in analysis_types]}")
        
        results = {}
        
        try:
            # 業務流程步驟3: 提取代碼特徵
            logger.info("開始提取代碼特徵...")
            features = self._extract_code_features(source_code)
            logger.info(f"代碼特徵提取完成，特徵向量大小: {len(features)}")
            
            # 業務流程步驟4-6: 逐類型執行AI分析
            for analysis_type in analysis_types:
                logger.info(f"開始執行 {analysis_type.value} 分析...")
                result = self._perform_ai_analysis(
                    source_code, features, analysis_type, file_path
                )
                results[analysis_type] = result
                logger.info(f"{analysis_type.value} 分析完成，風險等級: {result.risk_level}, 信心度: {result.confidence:.2f}")
                
        except Exception as e:
            print(f"AI代碼分析失敗: {e}")
            # 返回空結果
            for analysis_type in analysis_types:
                results[analysis_type] = AIAnalysisResult(
                    analysis_type=analysis_type,
                    confidence=0.0,
                    findings=[],
                    recommendations=[],
                    risk_level="unknown",
                    explanation=f"分析失敗: {e}",
                    metadata={}
                )
        
        return results

    def index_codebase(self) -> dict[str, Any]:
        """索引整個程式碼庫（從RAG 1遷移）"""
        start_time = time.time()
        logger.info(f"開始索引程式碼庫: {self.codebase_path}")
        
        # 收集所有Python檔案
        py_files = self._collect_python_files()
        logger.info(f"找到 {len(py_files)} 個Python檔案")
        
        if not py_files:
            logger.warning("未找到任何Python檔案")
            return self._get_indexing_stats()
            
        # 過濾需要索引的檔案（跳過已緩存的）
        files_to_index = self._filter_files_for_indexing(py_files)
        logger.info(f"需要索引 {len(files_to_index)} 個檔案（{len(py_files) - len(files_to_index)} 個已緩存）")
        
        # 批次並行索引
        self._batch_index_files(files_to_index)
        
        # 保存緩存
        if self.cache_manager:
            self.cache_manager.save_cache_index()
        
        # 統計和記錄
        self.stats["indexing_time"] = time.time() - start_time
        self.stats["total_chunks"] = len(self.chunks)
        
        logger.info(f"索引完成，耗時 {self.stats['indexing_time']:.2f}秒")
        logger.info(f"統計: {self.stats}")
        
        return self._get_indexing_stats()

    def _collect_python_files(self) -> list[Path]:
        """收集所有需要索引的Python檔案"""
        exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".mypy_cache", ".ruff_cache", ".aiva_analysis_cache",
            "build", "dist", ".tox", ".pytest_cache"
        }
        
        py_files = []
        for py_file in self.codebase_path.rglob("*.py"):
            # 檢查檔案大小
            try:
                if py_file.stat().st_size > self.config.max_file_size:
                    logger.debug(f"跳過大檔案: {py_file} ({py_file.stat().st_size} bytes)")
                    continue
            except OSError:
                continue
                
            # 檢查是否在排除目錄中
            if not any(excluded in py_file.parts for excluded in exclude_dirs):
                py_files.append(py_file)
                
        return py_files
    
    def _filter_files_for_indexing(self, py_files: list[Path]) -> list[Path]:
        """過濾出需要索引的檔案（排除已緩存的）"""
        if not self.cache_manager:
            return py_files
            
        files_to_index = []
        for py_file in py_files:
            if self.cache_manager.is_cached(py_file):
                self.stats["cached_files"] += 1
            else:
                files_to_index.append(py_file)
        return files_to_index
    
    def _batch_index_files(self, files_to_index: list[Path]) -> None:
        """批次並行索引檔案"""
        if not files_to_index:
            return
            
        # 分批處理
        batches = [files_to_index[i:i + self.config.batch_size] 
                  for i in range(0, len(files_to_index), self.config.batch_size)]
        
        logger.info(f"分 {len(batches)} 批次處理，每批 {self.config.batch_size} 個檔案")
        
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"處理第 {batch_idx + 1}/{len(batches)} 批次")
            self._process_file_batch(batch)
    
    def _process_file_batch(self, file_batch: list[Path]) -> None:
        """處理一批檔案"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交所有任務
            future_to_file = {
                executor.submit(self._safe_index_file, py_file): py_file 
                for py_file in file_batch
            }
            
            # 收集結果
            for future in as_completed(future_to_file):
                py_file = future_to_file[future]
                try:
                    future.result()
                    self.stats["indexed_files"] += 1
                    if self.cache_manager:
                        self.cache_manager.update_cache(py_file)
                except Exception as e:
                    logger.warning(f"索引檔案失敗 {py_file}: {e}")
                    self.stats["skipped_files"] += 1
    
    def _safe_index_file(self, py_file: Path) -> None:
        """安全地索引單個檔案"""
        try:
            content = py_file.read_text(encoding="utf-8")
            self._index_file_content(py_file, content)
        except UnicodeDecodeError as e:
            # 嘗試其他編碼
            try:
                content = py_file.read_text(encoding="latin-1")
                self._index_file_content(py_file, content)
            except Exception as inner_e:
                raise AIVAError(
                    message=f"編碼錯誤: {inner_e}",
                    error_type=ErrorType.SYSTEM,
                    severity=ErrorSeverity.MEDIUM,
                    context=create_error_context(
                        module=MODULE_NAME,
                        function="_index_python_file",
                        file_path=str(py_file)
                    )
                ) from e
        except (OSError, IOError) as e:
            raise AIVAError(
                message=f"讀取失敗: {e}",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH,
                context=create_error_context(
                    module=MODULE_NAME,
                    function="_index_python_file",
                    file_path=str(py_file)
                )
            ) from e
    
    def _index_file_content(self, file_path: Path, content: str) -> None:
        """索引檔案內容（從RAG 1遷移的核心邏輯）"""
        try:
            tree = ast.parse(content)
            self._extract_chunks_from_ast(tree, content, file_path)
        except SyntaxError:
            self._handle_unparseable_file(content, file_path)
    
    def _extract_chunks_from_ast(self, tree: ast.Module, content: str, file_path: Path) -> None:
        """從AST中提取代碼片段"""
        # 確保 tree 是 ast.Module 類型且有 body 屬性
        if hasattr(tree, 'body'):
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk_content = self._extract_node_content(content, node)
                    if chunk_content:
                        self._add_code_chunk(
                            file_path=str(file_path.relative_to(self.codebase_path)),
                            content=chunk_content,
                            node_type=type(node).__name__,
                            node_name=node.name,
                        )
    
    def _extract_node_content(self, content: str, node: ast.AST) -> Optional[str]:
        """提取AST節點內容"""
        try:
            chunk_content = ast.get_source_segment(content, node)
            if chunk_content and len(chunk_content.strip()) > 10:
                return chunk_content
        except (ValueError, TypeError):
            return self._extract_by_line_numbers(content, node)
        return None
    
    def _extract_by_line_numbers(self, content: str, node: ast.AST) -> Optional[str]:
        """通過行號範圍提取內容"""
        if not (hasattr(node, 'lineno') and hasattr(node, 'end_lineno')):
            return None
            
        lines = content.splitlines()
        start_line = max(0, getattr(node, 'lineno', 1) - 1)
        end_line = min(len(lines), getattr(node, 'end_lineno', len(lines)) or len(lines))
        chunk_content = '\n'.join(lines[start_line:end_line])
        return chunk_content.strip() or None
    
    def _handle_unparseable_file(self, content: str, file_path: Path) -> None:
        """處理無法解析的檔案"""
        max_chunk_size = 2000
        content_preview = content[:max_chunk_size] if len(content) > max_chunk_size else content
        self._add_code_chunk(
            file_path=str(file_path.relative_to(self.codebase_path)),
            content=content_preview,
            node_type="Module",
            node_name=file_path.stem,
        )

    def _add_code_chunk(self, file_path: str, content: str, node_type: str, node_name: str) -> None:
        """添加程式碼片段到分析索引（從RAG 1遷移）"""
        chunk_id = len(self.chunks)
        
        # 提取關鍵字
        keywords = self._extract_analysis_keywords(content, node_name)
        
        chunk = CodeChunk(
            id=chunk_id,
            path=file_path,
            content=content,
            node_type=node_type,
            node_name=node_name,
            keywords=keywords
        )
        self.chunks.append(chunk)

        # 建立關鍵字索引
        for keyword in keywords:
            if keyword not in self.index:
                self.index[keyword] = []
            self.index[keyword].append(chunk_id)

    def _extract_analysis_keywords(self, content: str, node_name: str) -> set[str]:
        """從內容中提取分析關鍵字（從RAG 1遷移並增強）"""
        keywords = set()

        # 添加節點名稱及其變體
        keywords.add(node_name.lower())
        # 分解駝峰命名和底線命名
        import re
        name_parts = re.split(r'[_\s]+|(?<!^)(?=[A-Z])', node_name)
        for part in name_parts:
            if len(part) > 2:
                keywords.add(part.lower())

        # 使用AST分析提取語義關鍵字
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    keywords.add(node.func.id.lower())
                elif isinstance(node, ast.Attribute):
                    keywords.add(node.attr.lower())
                elif isinstance(node, ast.Name):
                    keywords.add(node.id.lower())
        except (SyntaxError, ValueError):
            pass

        # 安全相關關鍵字
        security_keywords = [
            "vulnerability", "exploit", "payload", "injection", "xss", "sqli", 
            "csrf", "ssrf", "rce", "lfi", "rfi", "idor", "authentication",
            "authorization", "encryption", "hash", "token", "session",
            "scan", "detect", "analyze", "validate", "sanitize", "encode"
        ]
        
        content_lower = content.lower()
        for keyword in security_keywords:
            if keyword in content_lower:
                keywords.add(keyword)

        # 過濾關鍵字
        filtered_keywords = {
            kw for kw in keywords 
            if len(kw) > 2 and kw not in {'the', 'and', 'for', 'def', 'if', 'in', 'is', 'to'}
        }
        
        return filtered_keywords

    def search_code_chunks(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """搜尋相關的程式碼片段（從RAG 1遷移）"""
        if not query.strip():
            return []
            
        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return []
            
        chunk_scores = self._calculate_chunk_scores(query_keywords)
        return self._format_search_results(chunk_scores, top_k)
    
    def _extract_query_keywords(self, query: str) -> set[str]:
        """提取查詢關鍵字"""
        import re
        query_keywords = set()
        words = re.split(r'[_\s\-\.]+|(?<!^)(?=[A-Z])', query.lower())
        for word in words:
            if len(word) > 2:
                query_keywords.add(word)
        return query_keywords
    
    def _calculate_chunk_scores(self, query_keywords: set[str]) -> dict[int, float]:
        """計算片段相關性分數 - 重構後複雜度 ≤15"""
        chunk_scores: dict[int, float] = {}
        
        for keyword in query_keywords:
            # 精確匹配 - 提取到獨立方法
            self._apply_exact_matches(keyword, chunk_scores)
            # 部分匹配 - 提取到獨立方法
            self._apply_partial_matches(keyword, chunk_scores)
        
        return chunk_scores
    
    def _apply_exact_matches(self, keyword: str, chunk_scores: dict[int, float]) -> None:
        """應用精確匹配評分 - 複雜度 3"""
        if keyword in self.index:
            for chunk_id in self.index[keyword]:
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 2.0
    
    def _apply_partial_matches(self, keyword: str, chunk_scores: dict[int, float]) -> None:
        """應用部分匹配評分 - 複雜度 5"""
        for index_keyword in self.index:
            if keyword in index_keyword or index_keyword in keyword:
                for chunk_id in self.index[index_keyword]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 0.5
    
    def _format_search_results(self, chunk_scores: dict[int, float], top_k: int) -> list[dict[str, Any]]:
        """格式化搜尋結果"""
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (chunk_id, score) in enumerate(sorted_chunks[:top_k]):
            if score > 0:
                chunk = self.chunks[chunk_id]
                results.append({
                    "path": chunk.path,
                    "content": chunk.content,
                    "type": chunk.node_type,
                    "name": chunk.node_name,
                    "relevance_score": round(score, 2),
                    "rank": i + 1
                })
        
        return results

    def _get_indexing_stats(self) -> dict[str, Any]:
        """獲取索引統計信息"""
        return {
            "indexed_files": self.stats["indexed_files"],
            "cached_files": self.stats["cached_files"],
            "skipped_files": self.stats["skipped_files"],
            "total_chunks": len(self.chunks),
            "indexing_time": self.stats["indexing_time"],
            "avg_chunks_per_file": len(self.chunks) / max(self.stats["indexed_files"], 1)
        }

    def _create_failed_results(self, analysis_types: List[AnalysisType], error_msg: str) -> Dict[AnalysisType, AIAnalysisResult]:
        """創建失敗的分析結果"""
        results = {}
        for analysis_type in analysis_types:
            results[analysis_type] = AIAnalysisResult(
                analysis_type=analysis_type,
                confidence=0.0,
                findings=[],
                recommendations=[f"分析失敗: {error_msg}"],
                risk_level="error",
                explanation=f"{analysis_type.value}分析失敗: {error_msg}",
                metadata={"error": error_msg}
            )
        return results

    def _perform_ai_analysis(
        self,
        source_code: str,
        features: torch.Tensor,
        analysis_type: AnalysisType,
        file_path: str
    ) -> AIAnalysisResult:
        """執行特定類型的AI分析"""
        
        # 構建針對分析類型的提示
        task_prompts = {
            AnalysisType.SECURITY: "分析代碼安全性，識別潛在漏洞和安全風險",
            AnalysisType.VULNERABILITY: "檢測代碼中的已知漏洞模式",
            AnalysisType.COMPLEXITY: "評估代碼複雜度和可維護性",
            AnalysisType.PATTERNS: "識別設計模式和代碼氣味",
            AnalysisType.SEMANTIC: "執行語義分析，理解代碼邏輯",
            AnalysisType.ARCHITECTURE: "分析架構設計和組件關係"
        }
        
        task_description = task_prompts.get(analysis_type, "代碼分析")
        
        try:
            # 使用RAG代理進行AI分析 (如果可用)
            if self.rag_agent is not None:
                rag_result = self.rag_agent.generate(
                    task_description=f"{task_description}\n文件: {file_path}",
                    context=source_code[:2000]  # 限制上下文長度
                )
                confidence = rag_result.get('confidence', 0.5) if isinstance(rag_result, dict) else 0.5
            else:
                confidence = 0.5
            
            # 基於特徵向量和分析類型生成具體發現
            findings = self._generate_findings(features, analysis_type)
            
            # 生成建議
            recommendations = self._generate_recommendations(analysis_type, findings)
            
            # 計算風險等級
            risk_level = self._calculate_risk_level(confidence, findings)
            
            # 生成解釋
            explanation = self._generate_explanation(analysis_type, confidence, findings)
            
            return AIAnalysisResult(
                analysis_type=analysis_type,
                confidence=confidence,
                findings=findings,
                recommendations=recommendations,
                risk_level=risk_level,
                explanation=explanation,
                metadata={
                    'file_path': file_path,
                    'feature_vector_size': len(features),
                    'analysis_timestamp': torch.tensor(0.0).item()  # Use valid torch operation
                }
            )
            
        except Exception as e:
            return AIAnalysisResult(
                analysis_type=analysis_type,
                confidence=0.0,
                findings=[],
                recommendations=[f"分析失敗: {e}"],
                risk_level="error",
                explanation=f"AI分析過程中發生錯誤: {e}",
                metadata={'error': str(e)}
            )
    
    def _generate_findings(self, features: torch.Tensor, analysis_type: AnalysisType) -> List[Dict[str, Any]]:
        """基於特徵向量生成具體發現"""
        findings = []
        
        if analysis_type == AnalysisType.SECURITY:
            # 基於安全特徵生成發現
            dangerous_calls = int(features[12])  # 假設第12個特徵是危險函數調用數
            if dangerous_calls > 0:
                findings.append({
                    'type': 'dangerous_function_calls',
                    'severity': 'high' if dangerous_calls > 3 else 'medium',
                    'count': dangerous_calls,
                    'description': f'發現 {dangerous_calls} 個潛在危險的函數調用'
                })
        
        elif analysis_type == AnalysisType.COMPLEXITY:
            # 基於複雜度特徵生成發現
            complexity = int(features[11])  # 假設第11個特徵是循環複雜度
            if complexity > 10:
                findings.append({
                    'type': 'high_complexity',
                    'severity': 'high' if complexity > 20 else 'medium',
                    'value': complexity,
                    'description': f'循環複雜度過高: {complexity}'
                })
        
        return findings
    
    def _generate_recommendations(self, analysis_type: AnalysisType, findings: List[Dict[str, Any]]) -> List[str]:
        """基於發現生成建議"""
        recommendations = []
        
        for finding in findings:
            if finding['type'] == 'dangerous_function_calls':
                recommendations.append("建議: 避免使用危險函數如eval()、exec()，改用更安全的替代方案")
            elif finding['type'] == 'high_complexity':
                recommendations.append("建議: 重構複雜函數，將其分解為更小的可管理單元")
        
        if not recommendations:
            recommendations.append(f"代碼在{analysis_type.value}方面未發現明顯問題")
        
        return recommendations
    
    def _calculate_risk_level(self, confidence: float, findings: List[Dict[str, Any]]) -> str:
        """計算風險等級"""
        high_severity_count = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity_count = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity_count > 0:
            return "high"
        elif medium_severity_count > 2:
            return "medium"
        elif medium_severity_count > 0 or confidence < 0.3:
            return "low"
        else:
            return "safe"
    
    def _generate_explanation(self, analysis_type: AnalysisType, confidence: float, findings: List[Dict[str, Any]]) -> str:
        """生成分析解釋"""
        explanation = f"{analysis_type.value}分析完成，置信度: {confidence:.2f}\n"
        
        if findings:
            explanation += f"發現 {len(findings)} 個問題:\n"
            for i, finding in enumerate(findings, 1):
                explanation += f"{i}. {finding.get('description', 'Unknown issue')}\n"
        else:
            explanation += "未發現明顯問題"
        
        return explanation

    def get_analysis_summary(self, results: Dict[AnalysisType, AIAnalysisResult]) -> Dict[str, Any]:
        """生成分析摘要"""
        total_findings = sum(len(result.findings) for result in results.values())
        avg_confidence = np.mean([result.confidence for result in results.values()]) if results else 0
        
        risk_levels = [result.risk_level for result in results.values()]
        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            'total_analyses': len(results),
            'total_findings': total_findings,
            'average_confidence': avg_confidence,
            'overall_risk_level': overall_risk,
            'analysis_types': [at.value for at in results.keys()]
        }