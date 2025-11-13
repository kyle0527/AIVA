"""程式碼讀取工具 - 提供安全的檔案讀取能力

從原 tools.py 遷移的 CodeReader 功能，增強了錯誤處理和安全性
支援相對路徑、檔案統計、編碼檢測等功能
"""

from pathlib import Path
from typing import Any

from . import Tool


class CodeReader(Tool):
    """程式碼讀取工具 - 安全讀取程式碼檔案
    
    提供功能：
    - 安全的檔案路徑處理（防止目錄穿越）
    - 自動編碼檢測和處理
    - 檔案基本統計（行數、大小等）
    - 錯誤處理和狀態回報
    """
    
    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼讀取器
        
        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CodeReader",
            description="安全讀取程式碼檔案內容，提供檔案統計和編碼處理"
        )
        self.codebase_path = Path(codebase_path).resolve()
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """讀取檔案內容
        
        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑（相對於程式碼庫根目錄）
                encoding (str): 指定編碼，預設自動檢測
                max_size (int): 最大檔案大小（bytes），預設 10MB
                
        Returns:
            讀取結果字典：
            - status: 'success' | 'error'
            - path: 檔案路徑
            - content: 檔案內容
            - lines: 行數
            - size: 檔案大小（bytes）
            - encoding: 使用的編碼
            - error: 錯誤信息（如果有）
        """
        path = kwargs.get("path", "")
        encoding = kwargs.get("encoding", "utf-8")
        max_size = kwargs.get("max_size", 10 * 1024 * 1024)  # 10MB
        
        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}
        
        try:
            # 構建安全的檔案路徑（防止目錄穿越攻擊）
            requested_path = Path(path)
            if requested_path.is_absolute():
                return {
                    "status": "error",
                    "path": path,
                    "error": "不允許使用絕對路徑，請使用相對路徑"
                }
            
            full_path = (self.codebase_path / path).resolve()
            
            # 確保路徑在程式碼庫範圍內
            try:
                full_path.relative_to(self.codebase_path)
            except ValueError:
                return {
                    "status": "error", 
                    "path": path,
                    "error": "路徑超出程式碼庫範圍，可能存在目錄穿越攻擊"
                }
            
            # 檢查檔案是否存在
            if not full_path.exists():
                return {
                    "status": "error",
                    "path": path, 
                    "error": "檔案不存在"
                }
            
            # 檢查是否為檔案（非目錄）
            if not full_path.is_file():
                return {
                    "status": "error",
                    "path": path,
                    "error": "路徑指向目錄，非檔案"
                }
            
            # 檢查檔案大小
            file_size = full_path.stat().st_size
            if file_size > max_size:
                return {
                    "status": "error",
                    "path": path,
                    "error": f"檔案過大 ({file_size} bytes)，超過限制 ({max_size} bytes)"
                }
            
            # 嘗試讀取檔案
            try:
                content = full_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                # 如果指定編碼失敗，嘗試常見編碼
                for fallback_encoding in ['utf-8', 'gbk', 'big5', 'latin1']:
                    try:
                        content = full_path.read_text(encoding=fallback_encoding)
                        encoding = fallback_encoding
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return {
                        "status": "error",
                        "path": path,
                        "error": "無法解碼檔案，嘗試的編碼均失敗"
                    }
            
            # 計算統計信息
            lines = content.splitlines()
            
            return {
                "status": "success",
                "path": path,
                "content": content,
                "lines": len(lines),
                "size": file_size,
                "encoding": encoding,
                "stats": {
                    "total_lines": len(lines),
                    "non_empty_lines": len([l for l in lines if l.strip()]),
                    "blank_lines": len([l for l in lines if not l.strip()]),
                    "max_line_length": max(len(l) for l in lines) if lines else 0
                }
            }
            
        except PermissionError:
            return {
                "status": "error",
                "path": path,
                "error": "沒有讀取檔案的權限"
            }
        except Exception as e:
            return {
                "status": "error", 
                "path": path,
                "error": f"讀取檔案時發生錯誤: {str(e)}"
            }
    
    def list_files(self, pattern: str = "*", recursive: bool = False) -> dict[str, Any]:
        """列出程式碼庫中的檔案
        
        Args:
            pattern: 檔案匹配模式（glob 格式）
            recursive: 是否遞迴搜索子目錄
            
        Returns:
            檔案列表和統計
        """
        try:
            if recursive:
                files = list(self.codebase_path.rglob(pattern))
            else:
                files = list(self.codebase_path.glob(pattern))
            
            # 過濾出檔案（排除目錄）
            file_paths = [f for f in files if f.is_file()]
            
            # 轉為相對路徑
            relative_paths = []
            for file_path in file_paths:
                try:
                    rel_path = file_path.relative_to(self.codebase_path)
                    relative_paths.append(str(rel_path))
                except ValueError:
                    continue
            
            return {
                "status": "success",
                "pattern": pattern,
                "files": sorted(relative_paths),
                "count": len(relative_paths),
                "recursive": recursive
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"列出檔案時發生錯誤: {str(e)}"
            }
    
    def get_file_info(self, path: str) -> dict[str, Any]:
        """獲取檔案基本信息（不讀取內容）
        
        Args:
            path: 檔案路徑
            
        Returns:
            檔案信息字典
        """
        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}
        
        try:
            full_path = (self.codebase_path / path).resolve()
            
            # 路徑安全檢查
            try:
                full_path.relative_to(self.codebase_path)
            except ValueError:
                return {
                    "status": "error",
                    "path": path, 
                    "error": "路徑超出程式碼庫範圍"
                }
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "path": path,
                    "error": "檔案不存在"
                }
            
            stat_info = full_path.stat()
            
            return {
                "status": "success",
                "path": path,
                "exists": True,
                "is_file": full_path.is_file(),
                "is_dir": full_path.is_dir(),
                "size": stat_info.st_size,
                "modified_time": stat_info.st_mtime,
                "created_time": stat_info.st_ctime,
                "extension": full_path.suffix
            }
            
        except Exception as e:
            return {
                "status": "error",
                "path": path, 
                "error": f"獲取檔案信息時發生錯誤: {str(e)}"
            }


__all__ = ["CodeReader"]