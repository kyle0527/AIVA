"""
掃描引擎適配器基類 - 適配器模式實現

設計目標：
1. 統一四引擎的接口（Python/TypeScript/Rust/Go）
2. 解耦 Coordinator 與具體引擎實現
3. 支援依賴注入和單元測試
4. 符合開放封閉原則（OCP）- 擴展新引擎時不修改 Coordinator

適配器模式優點：
- 關注點分離：Coordinator 只負責編排，不關心引擎細節
- 可測試性：每個引擎可獨立 Mock 和測試
- 擴展性：新增第 5、6 個引擎不影響現有代碼
- 維護性：引擎特定邏輯集中管理，降低認知負荷
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from services.aiva_common.schemas import Asset


class BaseScannerAdapter(ABC):
    """掃描引擎適配器基類
    
    所有引擎適配器必須實現此接口，確保 Coordinator 可以統一調用。
    
    設計原則：
    1. 異步優先：所有方法都是 async，避免阻塞事件循環
    2. 標準化輸出：統一返回 Asset 列表，元數據另外提供
    3. 優雅降級：引擎不可用時返回空結果而非崩潰
    4. 錯誤透明：將引擎錯誤封裝為標準格式
    """
    
    def __init__(self, logger=None):
        """初始化適配器
        
        Args:
            logger: 日誌記錄器（可選，用於調試）
        """
        from services.aiva_common.utils import get_logger
        self.logger = logger or get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def is_available(self) -> bool:
        """檢查引擎是否可用
        
        實作建議：
        - Python: 嘗試 import 模組
        - Node/Go: 檢查二進制文件或入口腳本是否存在
        - Rust: 檢查 Bridge 模組是否可加載
        
        Returns:
            bool: True 表示引擎可用，False 表示不可用
        """
        pass
    
    @abstractmethod
    async def scan(
        self,
        targets: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行掃描任務的統一入口
        
        這是適配器模式的核心方法。所有引擎適配器都必須實現此方法，
        將引擎特定的調用方式（subprocess/FFI/module import）轉換為
        標準化的輸出格式。
        
        Args:
            targets: 目標 URL 列表
            options: 掃描配置，包含：
                - scan_id: str (掃描 ID)
                - max_depth: int (最大深度，默認 3)
                - max_pages: int (最大頁數，默認 100)
                - timeout: int (超時秒數，默認 60)
                - strategy: str (掃描策略，如 "BALANCED")
                - headers: Dict (自定義 HTTP 頭)
                - 其他引擎特定參數
        
        Returns:
            Dict 包含：
            - assets: List[Asset] (必須是標準化、已驗證的資產列表)
            - metadata: Dict (引擎特定的元數據，如耗時、掃描頁數)
            - error: Optional[str] (如果有非致命錯誤)
        
        Raises:
            Exception: 當發生無法恢復的引擎錯誤時拋出
        
        範例返回值：
        ```python
        {
            "assets": [
                Asset(asset_id="py_1", type="url", value="https://example.com/api"),
                Asset(asset_id="py_2", type="form", value="https://example.com/login")
            ],
            "metadata": {
                "engine": "python",
                "scan_duration": 12.5,
                "pages_scanned": 45,
                "urls_found": 120,
                "forms_found": 8
            },
            "error": None  # 或錯誤訊息字串
        }
        ```
        """
        pass
    
    def _standardize_assets(
        self,
        raw_assets: List[Any],
        engine_name: str
    ) -> List[Asset]:
        """將引擎特定的資產格式轉換為標準 Asset
        
        這是一個輔助方法，幫助適配器實現統一的資產格式轉換。
        
        Args:
            raw_assets: 引擎原始輸出的資產列表
            engine_name: 引擎名稱（用於生成 asset_id）
        
        Returns:
            List[Asset]: 標準化的資產列表
        """
        standardized = []
        
        for idx, raw_asset in enumerate(raw_assets):
            try:
                # 處理不同格式的原始資產
                if isinstance(raw_asset, Asset):
                    standardized.append(raw_asset)
                elif isinstance(raw_asset, dict):
                    # 確保必要字段存在
                    asset = Asset(
                        asset_id=raw_asset.get("asset_id", f"{engine_name}_{idx}"),
                        type=raw_asset.get("type", "url"),
                        value=raw_asset.get("value", ""),
                        parameters=raw_asset.get("parameters", []),
                        has_form=raw_asset.get("has_form", False)
                    )
                    standardized.append(asset)
                else:
                    self.logger.warning(f"無法標準化資產: {raw_asset}")
            except Exception as e:
                self.logger.error(f"資產標準化失敗: {e}")
        
        return standardized
