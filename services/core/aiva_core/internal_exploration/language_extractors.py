"""Multi-Language Capability Extractors - 多語言能力提取器

使用正則表達式從不同語言的源碼中提取函數/方法簽名

支援語言:
- Go
- Rust  
- TypeScript/JavaScript

遵循 aiva_common 規範
"""

import re
import logging
from pathlib import Path
from typing import Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LanguageExtractor(ABC):
    """語言提取器基類"""
    
    @abstractmethod
    def extract_capabilities(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """提取能力定義
        
        Args:
            content: 文件內容
            file_path: 文件路徑
            
        Returns:
            能力列表
        """
        pass
    
    def _extract_comments_before(self, content: str, start_pos: int, comment_pattern: str) -> list[str]:
        """提取函數前的註釋
        
        Args:
            content: 源碼內容
            start_pos: 函數開始位置
            comment_pattern: 註釋正則模式
            
        Returns:
            註釋行列表
        """
        lines = content[:start_pos].split('\n')
        comments = []
        
        for line in reversed(lines):
            stripped = line.strip()
            if re.match(comment_pattern, stripped):
                # 移除註釋符號
                comment_text = re.sub(comment_pattern, '', stripped).strip()
                comments.insert(0, comment_text)
            elif stripped and not stripped.startswith(('import', 'package', 'use')):
                # 遇到非註釋內容就停止
                break
        
        return comments


class GoExtractor(LanguageExtractor):
    """Go 語言函數提取器"""
    
    # Go 函數定義模式
    FUNCTION_PATTERN = re.compile(
        r'(?:^|\n)'  # 行首
        r'(?://[^\n]*\n)*'  # 可選註釋
        r'func\s+'  # func 關鍵字
        r'(?:\([^)]*\)\s+)?'  # 可選接收者 (receiver)
        r'([A-Z][a-zA-Z0-9_]*)'  # 函數名 (大寫開頭=導出)
        r'\s*\(([^)]*)\)'  # 參數列表
        r'\s*(?:\(([^)]*)\)|([a-zA-Z0-9_\[\].*\s,]*))?',  # 返回類型
        re.MULTILINE
    )
    
    def extract_capabilities(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """從 Go 源碼提取公開函數
        
        Args:
            content: Go 源碼內容
            file_path: 文件路徑
            
        Returns:
            函數能力列表
        """
        capabilities = []
        
        # 提取 package 名稱
        package_match = re.search(r'package\s+(\w+)', content)
        package_name = package_match.group(1) if package_match else "unknown"
        
        # 查找所有導出函數 (大寫開頭)
        for match in self.FUNCTION_PATTERN.finditer(content):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3) or match.group(4) or ""
            
            # 只提取導出函數 (Go 規則: 大寫字母開頭)
            if not func_name[0].isupper():
                continue
            
            # 提取前置註釋
            comments = self._extract_comments_before(
                content, 
                match.start(), 
                r'^//\s*'
            )
            
            # 解析參數
            parsed_params = self._parse_go_params(params)
            
            capability = {
                "name": func_name,
                "language": "go",
                "module": package_name,
                "file_path": file_path,
                "parameters": parsed_params,
                "return_type": return_type.strip() if return_type else None,
                "description": ' '.join(comments) if comments else f"Go function: {func_name}",
                "is_exported": True,  # 已過濾,一定是導出
                "line_number": content[:match.start()].count('\n') + 1
            }
            
            capabilities.append(capability)
        
        logger.debug(f"Extracted {len(capabilities)} Go capabilities from {file_path}")
        return capabilities
    
    def _parse_go_params(self, params_str: str) -> list[dict[str, str]]:
        """解析 Go 函數參數
        
        Args:
            params_str: 參數字串 "name type, name2 type2"
            
        Returns:
            參數列表
        """
        if not params_str.strip():
            return []
        
        params = []
        # 簡化解析: 按逗號分割
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            
            # Go 參數格式: name type 或 name, name2 type
            parts = param.rsplit(None, 1)  # 從右邊分割
            if len(parts) == 2:
                params.append({
                    "name": parts[0],
                    "type": parts[1]
                })
            elif len(parts) == 1:
                # 只有類型,沒有名稱
                params.append({
                    "name": "",
                    "type": parts[0]
                })
        
        return params


class RustExtractor(LanguageExtractor):
    """Rust 語言函數提取器 (增強版 - 支援 impl 區塊)"""
    
    # Rust 公開函數模式 (頂層函數)
    FUNCTION_PATTERN = re.compile(
        r'(?:^|\n)'  # 行首
        r'(?:///[^\n]*\n)*'  # 可選文檔註釋
        r'(?:#\[[^\]]+\]\s*)*'  # 可選屬性 (如 #[pyfunction])
        r'pub\s+'  # pub 關鍵字 (公開)
        r'(?:async\s+)?'  # 可選 async
        r'fn\s+'  # fn 關鍵字
        r'([a-zA-Z_][a-zA-Z0-9_]*)'  # 函數名
        r'\s*(?:<[^>]+>)?'  # 可選泛型
        r'\s*\(([^)]*)\)'  # 參數列表
        r'\s*(?:->\s*([^\{]+))?',  # 可選返回類型
        re.MULTILINE
    )
    
    # ✅ 新增: impl 區塊匹配模式 (簡化版)
    IMPL_PATTERN = re.compile(
        r'impl\s+(?:<[^>]*>\s+)?(\w+)\s*(?:<[^>]*>)?\s*\{',
        re.MULTILINE
    )
    
    # ✅ 新增: impl 內部方法模式
    IMPL_METHOD_PATTERN = re.compile(
        r'(?:///[^\n]*\n)*'  # 可選文檔註釋
        r'(?:#\[[^\]]+\]\s*)*'  # 可選屬性
        r'pub\s+'  # pub 關鍵字
        r'(?:async\s+)?'  # 可選 async
        r'fn\s+'  # fn 關鍵字
        r'([a-zA-Z_][a-zA-Z0-9_]*)'  # 方法名
        r'\s*(?:<[^>]+>)?'  # 可選泛型
        r'\s*\(([^)]*)\)'  # 參數列表
        r'\s*(?:->\s*([^\{]+))?',  # 可選返回類型
        re.MULTILINE
    )
    
    def extract_capabilities(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """從 Rust 源碼提取公開函數和方法 (增強版)
        
        Args:
            content: Rust 源碼內容
            file_path: 文件路徑
            
        Returns:
            函數和方法能力列表
        """
        capabilities = []
        
        # 1. ✅ 提取頂層公開函數
        capabilities.extend(self._extract_top_level_functions(content, file_path))
        
        # 2. ✅ 提取 impl 區塊內的公開方法
        capabilities.extend(self._extract_impl_methods(content, file_path))
        
        logger.debug(f"Extracted {len(capabilities)} Rust capabilities from {file_path}")
        return capabilities
    
    def _extract_top_level_functions(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """提取頂層公開函數
        
        Args:
            content: Rust 源碼內容
            file_path: 文件路徑
            
        Returns:
            函數能力列表
        """
        capabilities = []
        
        for match in self.FUNCTION_PATTERN.finditer(content):
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)
            
            # 提取文檔註釋
            doc_comments = self._extract_comments_before(
                content,
                match.start(),
                r'^///\s*'
            )
            
            # 檢查是否有 #[pyfunction] 屬性
            context_before = content[max(0, match.start() - 100):match.start()]
            is_pyfunction = '#[pyfunction]' in context_before or '#[pyo3::pyfunction]' in context_before
            
            # 解析參數
            parsed_params = self._parse_rust_params(params)
            
            capability = {
                "name": func_name,
                "language": "rust",
                "file_path": file_path,
                "parameters": parsed_params,
                "return_type": return_type.strip() if return_type else None,
                "description": ' '.join(doc_comments) if doc_comments else f"Rust function: {func_name}",
                "is_async": 'async' in match.group(0),
                "is_pyfunction": is_pyfunction,
                "is_method": False,
                "line_number": content[:match.start()].count('\n') + 1
            }
            
            capabilities.append(capability)
        
        return capabilities
    
    def _extract_impl_methods(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """提取 impl 區塊內的公開方法
        
        處理模式:
        impl SensitiveInfoScanner {
            pub fn scan_content(&self, ...) -> Result<...> { ... }
        }
        
        Args:
            content: Rust 源碼內容
            file_path: 文件路徑
            
        Returns:
            方法能力列表
        """
        capabilities = []
        
        # 查找所有 impl 區塊
        for impl_match in self.IMPL_PATTERN.finditer(content):
            struct_name = impl_match.group(1)
            impl_start = impl_match.end()
            
            # 查找對應的結束大括號 (簡化版: 查找下一個 impl 或文件結尾)
            next_impl = self.IMPL_PATTERN.search(content, impl_start)
            impl_end = next_impl.start() if next_impl else len(content)
            
            impl_body = content[impl_start:impl_end]
            
            # 在 impl 區塊內查找 pub fn 方法
            for method_match in self.IMPL_METHOD_PATTERN.finditer(impl_body):
                method_name = method_match.group(1)
                params = method_match.group(2)
                return_type = method_match.group(3)
                
                # 跳過私有方法 (以 _ 開頭)
                if method_name.startswith('_'):
                    continue
                
                # 提取文檔註釋
                doc_comments = self._extract_comments_before(
                    impl_body,
                    method_match.start(),
                    r'^///\s*'
                )
                
                # 解析參數
                parsed_params = self._parse_rust_params(params)
                
                # 計算絕對行號
                absolute_line = content[:impl_start + method_match.start()].count('\n') + 1
                
                capability = {
                    "name": f"{struct_name}::{method_name}",  # 完整路徑
                    "language": "rust",
                    "file_path": file_path,
                    "struct": struct_name,
                    "method": method_name,
                    "parameters": parsed_params,
                    "return_type": return_type.strip() if return_type else None,
                    "description": ' '.join(doc_comments) if doc_comments else f"Rust method: {struct_name}::{method_name}",
                    "is_async": 'async' in method_match.group(0),
                    "is_method": True,
                    "line_number": absolute_line
                }
                
                capabilities.append(capability)
        
        return capabilities
    
    def _parse_rust_params(self, params_str: str) -> list[dict[str, str]]:
        """解析 Rust 函數參數
        
        Args:
            params_str: 參數字串 "name: type, name2: type2"
            
        Returns:
            參數列表
        """
        if not params_str.strip():
            return []
        
        params = []
        # Rust 參數格式: name: type
        for param in params_str.split(','):
            param = param.strip()
            if not param or param == 'self' or param == '&self' or param == '&mut self':
                continue
            
            if ':' in param:
                parts = param.split(':', 1)
                params.append({
                    "name": parts[0].strip(),
                    "type": parts[1].strip()
                })
        
        return params


class TypeScriptExtractor(LanguageExtractor):
    """TypeScript/JavaScript 函數提取器"""
    
    # 多種函數定義模式
    PATTERNS = [
        # export function name(...)
        re.compile(
            r'export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        ),
        # export const name = (...) => 
        re.compile(
            r'export\s+const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
            re.MULTILINE
        ),
        # public/private async methodName(...)
        re.compile(
            r'(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*:\s*\w+',
            re.MULTILINE
        ),
    ]
    
    def extract_capabilities(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """從 TypeScript/JavaScript 源碼提取導出函數
        
        Args:
            content: 源碼內容
            file_path: 文件路徑
            
        Returns:
            函數能力列表
        """
        capabilities = []
        seen_names = set()  # 避免重複
        
        for pattern in self.PATTERNS:
            for match in pattern.finditer(content):
                func_name = match.group(1)
                
                # 避免重複提取
                if func_name in seen_names:
                    continue
                seen_names.add(func_name)
                
                # 提取 JSDoc 註釋
                jsdoc = self._extract_jsdoc(content, match.start())
                
                capability = {
                    "name": func_name,
                    "language": "typescript" if file_path.endswith('.ts') else "javascript",
                    "file_path": file_path,
                    "description": jsdoc.get('description', f"Function: {func_name}"),
                    "parameters": jsdoc.get('params', []),
                    "return_type": jsdoc.get('returns', None),
                    "is_async": 'async' in match.group(0),
                    "is_exported": 'export' in match.group(0),
                    "line_number": content[:match.start()].count('\n') + 1
                }
                
                capabilities.append(capability)
        
        logger.debug(f"Extracted {len(capabilities)} TS/JS capabilities from {file_path}")
        return capabilities
    
    def _extract_jsdoc(self, content: str, start_pos: int) -> dict[str, Any]:
        """提取 JSDoc 註釋
        
        Args:
            content: 源碼內容
            start_pos: 函數開始位置
            
        Returns:
            JSDoc 資訊
        """
        jsdoc_lines = self._extract_jsdoc_lines(content, start_pos)
        return self._parse_jsdoc_lines(jsdoc_lines)
    
    def _extract_jsdoc_lines(self, content: str, start_pos: int) -> list[str]:
        """提取 JSDoc 註釋行
        
        Args:
            content: 源碼內容
            start_pos: 函數開始位置
            
        Returns:
            註釋行列表
        """
        lines = content[:start_pos].split('\n')
        jsdoc_lines = []
        in_jsdoc = False
        
        for line in reversed(lines):
            stripped = line.strip()
            
            if stripped == '*/':
                in_jsdoc = True
            elif stripped.startswith('/**'):
                break
            elif in_jsdoc:
                # 移除 * 符號
                cleaned = re.sub(r'^\s*\*\s?', '', stripped)
                jsdoc_lines.insert(0, cleaned)
        
        return jsdoc_lines
    
    def _parse_jsdoc_lines(self, jsdoc_lines: list[str]) -> dict[str, Any]:
        """解析 JSDoc 註釋行
        
        Args:
            jsdoc_lines: JSDoc 註釋行
            
        Returns:
            解析後的 JSDoc 資訊
        """
        description = []
        params = []
        returns = None
        
        for line in jsdoc_lines:
            if line.startswith('@param'):
                param = self._parse_param_tag(line)
                if param:
                    params.append(param)
            elif line.startswith(('@returns', '@return')):
                returns = self._parse_return_tag(line)
            elif not line.startswith('@'):
                description.append(line)
        
        return {
            "description": ' '.join(description) if description else "",
            "params": params,
            "returns": returns
        }
    
    def _parse_param_tag(self, line: str) -> dict[str, str] | None:
        """解析 @param 標籤
        
        Args:
            line: @param 行
            
        Returns:
            參數資訊
        """
        # @param {type} name - description
        match = re.match(r'@param\s+(?:\{([^}]+)\}\s+)?(\w+)\s*-?\s*(.*)', line)
        if match:
            return {
                "type": match.group(1) or "any",
                "name": match.group(2),
                "description": match.group(3)
            }
        return None
    
    def _parse_return_tag(self, line: str) -> str | None:
        """解析 @returns 標籤
        
        Args:
            line: @returns 行
            
        Returns:
            返回類型
        """
        # @returns {type} description
        match = re.match(r'@returns?\s+(?:\{([^}]+)\}\s*)?(.*)', line)
        if match:
            return match.group(1) or "void"
        return None


# 工廠函數
def get_extractor(language: str) -> LanguageExtractor | None:
    """獲取對應語言的提取器
    
    Args:
        language: 語言名稱 (go, rust, typescript, javascript)
        
    Returns:
        對應的提取器實例,若不支援則返回 None
    """
    extractors = {
        "go": GoExtractor(),
        "rust": RustExtractor(),
        "typescript": TypeScriptExtractor(),
        "javascript": TypeScriptExtractor()
    }
    
    return extractors.get(language.lower())
