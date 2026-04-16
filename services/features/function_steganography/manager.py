"""
Steganography Manager
整合 AI-Steganography-Detection 和 StegX 引擎
"""

import logging

# 引入新引擎
from .engines.ai_steg_engine import AIStegDetectionEngine
from .engines.stegx_engine import StegXEngine
from .models import DetectionResult, EmbedResult, ExtractResult, SteganographyMethod

logger = logging.getLogger(__name__)


class SteganographyManager:
    """隱寫術管理器
    
    整合兩大引擎:
    - AIStegDetectionEngine: AI 驅動的隱寫檢測 (CNN 模型)
    - StegXEngine: 高級隱寫術 (非線性 LSB + AES-256-GCM)
    """
    
    def __init__(self):
        # 初始化引擎
        self.ai_detection_engine = AIStegDetectionEngine()
        self.stegx_engine = StegXEngine()
        
        logger.info("SteganographyManager initialized")
        logger.info("AI Detection and StegX engines loaded")
    
    async def embed_data(
        self,
        carrier_file: str,
        secret_file: str,
        output_file: str,
        password: str | None = None
    ) -> EmbedResult:
        """嵌入數據到載體"""
        try:
            logger.info(f"Embedding data into {carrier_file}")
            
            # 使用 StegX 引擎進行嵌入
            result = await self.stegx_engine.embed_data(
                cover_image=carrier_file,
                secret_file=secret_file,
                output_path=output_file,
                password=password
            )
            
            return EmbedResult(
                success=result.get("success", False),
                output_file=output_file,
                secret_size=result.get("bytes_embedded", 0),
                error=result.get("error")
            )
            
        except Exception as e:
            logger.error(f"Embed failed: {str(e)}", exc_info=True)
            return EmbedResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def extract_data(
        self,
        stego_file: str,
        output_file: str,
        password: str | None = None
    ) -> ExtractResult:
        """從載體提取數據"""
        try:
            logger.info(f"Extracting data from {stego_file}")
            
            # 使用 StegX 引擎進行提取
            result = await self.stegx_engine.extract_data(
                stego_image=stego_file,
                output_path=output_file,
                password=password
            )
            
            return ExtractResult(
                success=result.get("success", False),
                output_file=output_file,
                extracted_size=result.get("bytes_extracted", 0),
                error=result.get("error"),
            )
            
        except Exception as e:
            logger.error(f"Extract failed: {str(e)}", exc_info=True)
            return ExtractResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def detect_hidden_data(
        self,
        file_path: str
    ) -> DetectionResult:
        """檢測隱藏數據"""
        try:
            logger.info(f"Detecting hidden data in {file_path}")
            
            # 使用 AI 檢測引擎
            result = await self.ai_detection_engine.detect_hidden_data(file_path)
            
            return DetectionResult(
                has_hidden_data=result.get("detected", False),
                confidence=result.get("confidence", 0.0),
                method_detected=None if not result.get("detected") else SteganographyMethod.LSB
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                has_hidden_data=False,
                confidence=0.0
            )
    
    def calculate_capacity(
        self,
        carrier_file: str,
        method: SteganographyMethod = SteganographyMethod.LSB
    ) -> int:
        """計算載體容量"""
        try:
            logger.info(f"Calculating capacity for {carrier_file}")
            
            # 使用 AI 檢測引擎計算容量
            capacity = self.ai_detection_engine.calculate_embedding_capacity(carrier_file)
            
            logger.debug(f"Calculated capacity: {capacity} bytes")
            return capacity
            
        except Exception as e:
            logger.error(f"Capacity calculation failed: {str(e)}", exc_info=True)
            return 0
    
    # ==================== StegX Engine Integration ====================
    
    async def stegx_hide_file(
        self,
        carrier_image: str,
        secret_file: str,
        output_image: str,
        password: str,
        compress: bool = True
    ) -> EmbedResult:
        """使用 StegX 引擎隱藏文件
        
        Args:
            carrier_image: 載體圖片路徑
            secret_file: 要隱藏的文件路徑
            output_image: 輸出圖片路徑
            password: 加密密碼
            compress: 是否壓縮
        
        Returns:
            嵌入結果
        """
        try:
            logger.info(f"StegX: Hiding {secret_file} in {carrier_image}")
            result = await self.stegx_engine.hide_file(
                carrier_image, secret_file, output_image, password
            )
            
            return EmbedResult(
                success=result.get("success", False),
                output_file=output_image,
                secret_size=result.get("bytes_embedded", 0),
                error=result.get("error")
            )
        except Exception as e:
            logger.error(f"StegX hide failed: {str(e)}", exc_info=True)
            return EmbedResult(
                success=False,
                output_file=output_image,
                error=str(e)
            )
    
    async def stegx_extract_file(
        self,
        stego_image: str,
        output_file: str,
        password: str | None = None
    ) -> ExtractResult:
        """使用 StegX 引擎提取隱藏文件
        
        Args:
            stego_image: 含隱藏數據的圖片
            output_file: 輸出文件路徑
            password: 解密密碼
        
        Returns:
            提取結果
        """
        try:
            logger.info(f"StegX: Extracting from {stego_image}")
            result = await self.stegx_engine.extract_file(stego_image, output_file, password)
            
            return ExtractResult(
                success=result.get("success", False),
                output_file=output_file,
                extracted_size=result.get("bytes_extracted", 0),
                error=result.get("error")
            )
        except Exception as e:
            logger.error(f"StegX extract failed: {str(e)}", exc_info=True)
            return ExtractResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def stegx_check_capacity(
        self,
        carrier_image: str
    ) -> dict:
        """檢查載體圖片的隱藏容量
        
        Args:
            carrier_image: 載體圖片路徑
        
        Returns:
            容量信息
        """
        try:
            result = await self.stegx_engine.check_capacity(carrier_image)
            return result
        except Exception as e:
            logger.error(f"Capacity check failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def stegx_analyze_image(
        self,
        image_path: str
    ) -> dict:
        """分析圖片是否包含隱寫數據
        
        Args:
            image_path: 圖片路徑
        
        Returns:
            分析結果 (Chi-Square, Entropy, Histogram)
        """
        try:
            result = await self.stegx_engine.analyze_stego_image(image_path)
            return result
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def stegx_batch_hide(
        self,
        carrier_images: list,
        secret_file: str,
        output_dir: str,
        password: str
    ) -> dict:
        """批量隱藏文件到多張圖片
        
        Args:
            carrier_images: 載體圖片列表
            secret_file: 要隱藏的文件
            output_dir: 輸出目錄
            password: 加密密碼
        
        Returns:
            批量處理結果
        """
        try:
            operations = []
            for img in carrier_images:
                operations.append({
                    "cover_image": img,
                    "secret_file": secret_file,
                    "output_path": f"{output_dir}/{img}",
                    "password": password
                })
            result = await self.stegx_engine.batch_hide(operations)
            return result
        except Exception as e:
            logger.error(f"Batch hide failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    # ==================== AI Detection Engine Integration ====================
    
    async def ai_detect_steganography(
        self,
        image_path: str,
        model_path: str | None = None
    ) -> DetectionResult:
        """使用 AI 模型檢測隱寫
        
        Args:
            image_path: 圖片路徑
            model_path: CNN 模型路徑 (可選)
        
        Returns:
            檢測結果
        """
        try:
            logger.info(f"AI Detection: Analyzing {image_path}")
            
            # 載入模型 (如果提供)
            if model_path:
                load_result = await self.ai_detection_engine.load_model(model_path)
                if not load_result.get("success"):
                    return DetectionResult(
                        has_hidden_data=False,
                        confidence=0.0,
                        method_detected=None
                    )
            
            # 執行檢測
            result = await self.ai_detection_engine.detect_steganography(image_path)
            
            return DetectionResult(
                has_hidden_data=result.get("is_stego", False),
                confidence=result.get("confidence", 0.0),
                method_detected=result.get("method", None),
                statistics=result.get("metadata", {})
            )
        except Exception as e:
            logger.error(f"AI detection failed: {str(e)}", exc_info=True)
            return DetectionResult(
                has_hidden_data=False,
                confidence=0.0
            )
    
    async def ai_batch_scan(
        self,
        directory: str,
        recursive: bool = True,
        extensions: list | None = None
    ) -> dict:
        """批量掃描目錄中的圖片
        
        Args:
            directory: 掃描目錄
            recursive: 是否遞歸掃描
            extensions: 文件擴展名過濾
        
        Returns:
            批量掃描結果
        """
        try:
            logger.info(f"AI Batch Scan: {directory}")
            result = await self.ai_detection_engine.batch_scan(directory, recursive, extensions or [])
            return result
        except Exception as e:
            logger.error(f"Batch scan failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def ai_train_model(
        self,
        training_data_dir: str,
        output_model_path: str,
        epochs: int = 20,
        batch_size: int = 32
    ) -> dict:
        """訓練 AI 檢測模型
        
        Args:
            training_data_dir: 訓練數據目錄 (需包含 clean/ 和 stego/ 子目錄)
            output_model_path: 輸出模型路徑
            epochs: 訓練輪數
            batch_size: 批次大小
        
        Returns:
            訓練結果
        """
        try:
            logger.info(f"Training AI model: {training_data_dir}")
            result = await self.ai_detection_engine.train_model(
                training_data_dir, output_model_path, epochs=epochs, batch_size=batch_size
            )
            return result
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def ai_adjust_threshold(
        self,
        new_threshold: float
    ) -> dict:
        """調整檢測閾值
        
        Args:
            new_threshold: 新閾值 (0.0-1.0)
        
        Returns:
            調整結果
        """
        try:
            result = await self.ai_detection_engine.adjust_threshold(new_threshold)
            return result
        except Exception as e:
            logger.error(f"Threshold adjustment failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
