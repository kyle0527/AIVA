"""
StegX Engine

集成 StegX 工具的隱寫術引擎
"""

import logging
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class StegXEngine:
    """StegX 工具集成引擎"""
    
    def __init__(self):
        """初始化 StegX 引擎"""
        self.stegx_path = None
        self._find_stegx_tool()
        logger.info("初始化 StegX 引擎")
    
    def _find_stegx_tool(self):
        """查找 StegX 工具路徑"""
        # 在 external_tools 目錄中查找
        possible_paths = [
            Path(__file__).parent.parent / "external_tools" / "StegX" / "stegx" / "stegx.py",
            Path(__file__).parent.parent / "external_tools" / "StegX" / "stegx.py"
        ]
        
        for path in possible_paths:
            if path.exists():
                self.stegx_path = str(path)
                logger.info(f"找到 StegX 工具: {self.stegx_path}")
                break
        
        if not self.stegx_path:
            logger.warning("未找到 StegX 工具")
    
    async def embed_data(
        self,
        cover_image: str,
        secret_file: str,
        output_path: str,
        password: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """使用 StegX 嵌入數據
        
        Args:
            cover_image: 載體圖片路徑
            secret_file: 秘密文件路徑
            output_path: 輸出文件路徑
            password: 可選密碼
            **kwargs: 額外參數
            
        Returns:
            嵌入結果字典
        """
        if not self.stegx_path:
            return {
                "success": False,
                "error": "StegX 工具未找到",
                "output_path": None
            }
        
        try:
            # 構建 StegX 命令
            cmd = [
                "python", self.stegx_path,
                "encode",
                "-i", cover_image,
                "-s", secret_file,
                "-o", output_path
            ]
            
            if password:
                cmd.extend(["-p", password])
            
            # 執行 StegX 命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_path": output_path,
                    "method": "stegx",
                    "stdout": result.stdout,
                    "bytes_embedded": Path(secret_file).stat().st_size if Path(secret_file).exists() else 0
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "StegX 執行失敗",
                    "output_path": None
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "StegX 執行超時",
                "output_path": None
            }
        except Exception as e:
            logger.error(f"StegX 嵌入失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_path": None
            }
    
    async def extract_data(
        self,
        stego_image: str,
        output_path: str,
        password: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """使用 StegX 提取數據
        
        Args:
            stego_image: 隱寫圖片路徑
            output_path: 輸出文件路徑
            password: 可選密碼
            **kwargs: 額外參數
            
        Returns:
            提取結果字典
        """
        if not self.stegx_path:
            return {
                "success": False,
                "error": "StegX 工具未找到",
                "extracted_file": None
            }
        
        try:
            # 構建 StegX 命令
            cmd = [
                "python", self.stegx_path,
                "decode",
                "-i", stego_image,
                "-o", output_path
            ]
            
            if password:
                cmd.extend(["-p", password])
            
            # 執行 StegX 命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "extracted_file": output_path,
                    "method": "stegx",
                    "stdout": result.stdout,
                    "bytes_extracted": Path(output_path).stat().st_size if Path(output_path).exists() else 0
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or "StegX 解碼失敗",
                    "extracted_file": None
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "StegX 執行超時",
                "extracted_file": None
            }
        except Exception as e:
            logger.error(f"StegX 提取失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_file": None
            }
    
    async def hide_file(
        self,
        cover_image: str,
        secret_file: str,
        output_path: str,
        password: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """隱藏文件（embed_data 的別名）"""
        return await self.embed_data(cover_image, secret_file, output_path, password, **kwargs)
    
    async def extract_file(
        self,
        stego_image: str,
        output_path: str,
        password: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """提取文件（extract_data 的別名）"""
        return await self.extract_data(stego_image, output_path, password, **kwargs)
    
    async def check_capacity(
        self,
        carrier_image: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """檢查載體容量"""
        try:
            # 基於實際檔案格式與經驗法則計算真實容量 (如 LSB 可用容量)
            if not Path(carrier_image).exists():
                return {
                    "success": False,
                    "error": "載體文件不存在",
                    "capacity": 0
                }
            
            import cv2
            img = cv2.imread(carrier_image)
            if img is None:
                 return {"success": False, "error": "無法讀取圖片", "capacity": 0}

            # LSB 通常能使用 1 bit per channel per pixel, 也就是 (width * height * channels) bits
            height, width, channels = img.shape
            estimated_capacity_bytes = (height * width * channels) // 8
            
            return {
                "success": True,
                "capacity": estimated_capacity_bytes,
                "method": "stegx_estimation"
            }
        except Exception as e:
            logger.error(f"容量檢查失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "capacity": 0
            }
    
    async def analyze_stego_image(
        self,
        image_path: str,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """分析隱寫圖像"""
        try:
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": "圖像文件不存在",
                    "analysis": None
                }
            
            import cv2
            import numpy as np

            # 使用 Chi-square attack 簡化版檢測 LSB 隱寫
            img = cv2.imread(image_path)
            if img is None:
                 return {"success": False, "error": "無法讀取圖片", "analysis": None}

            # 分析圖像底層 LSB 特徵分佈
            # 將最後一 bit 提取出來，看 0 和 1 的分佈，如果是隨機數據，比例接近 1:1
            lsb = img & 1
            ones_ratio = np.sum(lsb) / lsb.size

            # 若比例接近 0.5 (例如 0.49 到 0.51)，則高度懷疑有隨機數據被寫入 (如加密的隱寫)
            suspected = 0.495 <= ones_ratio <= 0.505

            file_size = Path(image_path).stat().st_size
            
            return {
                "success": True,
                "analysis": {
                    "file_size": file_size,
                    "suspected_steganography": suspected,
                    "lsb_ones_ratio": ones_ratio,
                    "method": "lsb_chi_square_estimation"
                }
            }
        except Exception as e:
            logger.error(f"圖像分析失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    async def batch_hide(
        self,
        operations: list,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """批量隱藏操作"""
        results = []
        success_count = 0
        
        for operation in operations:
            try:
                result = await self.hide_file(
                    cover_image=operation.get("cover_image"),
                    secret_file=operation.get("secret_file"),
                    output_path=operation.get("output_path"),
                    password=operation.get("password")
                )
                results.append(result)
                if result.get("success"):
                    success_count += 1
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": success_count > 0,
            "total_operations": len(operations),
            "successful_operations": success_count,
            "results": results
        }