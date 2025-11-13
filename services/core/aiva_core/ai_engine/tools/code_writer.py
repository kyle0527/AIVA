"""程式碼寫入工具 - 提供安全的檔案寫入能力

從原 tools.py 遷移的 CodeWriter 功能，增強了安全性和錯誤處理
支援安全路徑檢查、備份機制、權限驗證等功能
"""

from pathlib import Path
from typing import Any
import shutil
import time

from . import Tool


class CodeWriter(Tool):
    """程式碼寫入工具 - 安全寫入程式碼檔案
    
    提供功能：
    - 安全的檔案路徑處理（防止目錄穿越）
    - 自動目錄建立
    - 檔案備份機制
    - 寫入權限檢查
    - 原子性寫入（先寫臨時檔案再重命名）
    """
    
    def __init__(self, codebase_path: str, enable_backup: bool = True) -> None:
        """初始化程式碼寫入器
        
        Args:
            codebase_path: 程式碼庫根目錄
            enable_backup: 是否啟用檔案備份機制
        """
        super().__init__(
            name="CodeWriter", 
            description="安全寫入程式碼檔案，支援備份和原子性操作"
        )
        self.codebase_path = Path(codebase_path).resolve()
        self.enable_backup = enable_backup
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """寫入檔案內容
        
        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑（相對於程式碼庫根目錄）
                content (str): 要寫入的內容
                encoding (str): 檔案編碼，預設 utf-8
                create_dirs (bool): 是否自動建立目錄，預設 True
                backup (bool): 是否備份原檔案，預設使用全域設定
                
        Returns:
            寫入結果字典：
            - status: 'success' | 'error'
            - path: 檔案路徑
            - bytes_written: 寫入的位元組數
            - backup_path: 備份檔案路徑（如果有）
            - created_dirs: 是否建立了新目錄
            - error: 錯誤信息（如果有）
        """
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        encoding = kwargs.get("encoding", "utf-8")
        create_dirs = kwargs.get("create_dirs", True)
        backup = kwargs.get("backup", self.enable_backup)
        
        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}
        if not isinstance(content, str):
            return {"status": "error", "error": "content 必須為字串"}
        
        try:
            # 構建安全的檔案路徑
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
            
            # 檢查是否嘗試寫入目錄
            if full_path.exists() and full_path.is_dir():
                return {
                    "status": "error",
                    "path": path,
                    "error": "目標是目錄，無法作為檔案寫入"
                }
            
            created_dirs = False
            backup_path = None
            
            # 建立目錄（如果需要）
            if create_dirs and not full_path.parent.exists():
                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    created_dirs = True
                except PermissionError:
                    return {
                        "status": "error",
                        "path": path,
                        "error": "沒有建立目錄的權限"
                    }
            
            # 備份原檔案（如果存在且啟用備份）
            if backup and full_path.exists():
                try:
                    timestamp = str(int(time.time()))
                    backup_filename = f"{full_path.name}.backup.{timestamp}"
                    backup_path = full_path.parent / backup_filename
                    shutil.copy2(full_path, backup_path)
                except Exception as e:
                    return {
                        "status": "error",
                        "path": path, 
                        "error": f"備份檔案失敗: {str(e)}"
                    }
            
            # 原子性寫入：先寫臨時檔案，再重命名
            temp_path = full_path.parent / f".{full_path.name}.tmp"
            try:
                # 寫入臨時檔案
                temp_path.write_text(content, encoding=encoding)
                
                # 原子性重命名
                temp_path.replace(full_path)
                
                bytes_written = len(content.encode(encoding))
                
                result = {
                    "status": "success",
                    "path": path,
                    "bytes_written": bytes_written,
                    "created_dirs": created_dirs,
                    "encoding": encoding
                }
                
                if backup_path:
                    result["backup_path"] = str(backup_path.relative_to(self.codebase_path))
                
                return result
                
            except PermissionError:
                # 清理臨時檔案
                if temp_path.exists():
                    temp_path.unlink()
                return {
                    "status": "error",
                    "path": path,
                    "error": "沒有寫入檔案的權限"
                }
            except Exception as e:
                # 清理臨時檔案
                if temp_path.exists():
                    temp_path.unlink()
                return {
                    "status": "error",
                    "path": path,
                    "error": f"寫入檔案時發生錯誤: {str(e)}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "path": path,
                "error": f"處理寫入請求時發生錯誤: {str(e)}"
            }
    
    def append_content(self, path: str, content: str, **kwargs) -> dict[str, Any]:
        """追加內容到檔案末尾
        
        Args:
            path: 檔案路徑
            content: 要追加的內容
            **kwargs: 其他參數
            
        Returns:
            操作結果
        """
        encoding = kwargs.get("encoding", "utf-8")
        
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
            
            # 如果檔案不存在，建立新檔案
            if not full_path.exists():
                return self.execute(path=path, content=content, **kwargs)
            
            # 讀取現有內容
            existing_content = full_path.read_text(encoding=encoding)
            
            # 合併內容並寫入
            new_content = existing_content + content
            return self.execute(path=path, content=new_content, **kwargs)
            
        except Exception as e:
            return {
                "status": "error",
                "path": path,
                "error": f"追加內容時發生錯誤: {str(e)}"
            }
    
    def delete_file(self, path: str) -> dict[str, Any]:
        """刪除檔案
        
        Args:
            path: 檔案路徑
            
        Returns:
            刪除結果
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
            
            if full_path.is_dir():
                return {
                    "status": "error",
                    "path": path,
                    "error": "目標是目錄，使用專門的目錄刪除功能"
                }
            
            # 備份（如果啟用）
            backup_path = None
            if self.enable_backup:
                timestamp = str(int(time.time()))
                backup_filename = f"{full_path.name}.deleted.{timestamp}"
                backup_path = full_path.parent / backup_filename
                shutil.copy2(full_path, backup_path)
            
            # 刪除檔案
            full_path.unlink()
            
            result = {
                "status": "success",
                "path": path,
                "message": "檔案已刪除"
            }
            
            if backup_path:
                result["backup_path"] = str(backup_path.relative_to(self.codebase_path))
            
            return result
            
        except PermissionError:
            return {
                "status": "error",
                "path": path,
                "error": "沒有刪除檔案的權限"
            }
        except Exception as e:
            return {
                "status": "error",
                "path": path,
                "error": f"刪除檔案時發生錯誤: {str(e)}"
            }


__all__ = ["CodeWriter"]