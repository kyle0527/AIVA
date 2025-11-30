#!/usr/bin/env python3
"""
AI 模組插件基礎接口

定義統一的插件接口，所有 AI 模組必須實現此接口才能被註冊到系統中。
設計參考 Kubernetes Device Plugin Pattern 和 Protocol-based Programming。

使用方式:
    from services.core.aiva_core.plugin_system import AIModulePlugin, AITask, AIResult
    
    class MyPlugin(AIModulePlugin):
        @property
        def module_id(self) -> str:
            return "my_plugin"
        
        @property
        def capabilities(self) -> List[str]:
            return ["capability_a", "capability_b"]
        
        async def initialize(self, config: Dict[str, Any]) -> bool:
            # 初始化邏輯
            return True
        
        async def execute_task(self, task: AITask) -> AIResult:
            # 執行邏輯
            return AIResult(success=True, data={...})
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class AITaskType(Enum):
    """AI 任務類型枚舉"""
    # 掃描相關
    SCAN = "scan"
    PASSIVE_SCAN = "passive_scan"
    ACTIVE_SCAN = "active_scan"
    
    # 分析相關
    ANALYZE = "analyze"
    ANALYZE_CODE = "analyze_code"
    ANALYZE_VULNERABILITIES = "analyze_vulnerabilities"
    
    # 攻擊相關
    EXPLOIT = "exploit"
    GENERATE_PAYLOAD = "generate_payload"
    ATTACK_PLANNING = "attack_planning"
    
    # 防禦相關
    DEFEND = "defend"
    GENERATE_PATCH = "generate_patch"
    RISK_ASSESSMENT = "risk_assessment"
    
    # 學習相關
    TRAIN = "train"
    LEARN = "learn"
    EXPERIENCE_LEARNING = "experience_learning"


@dataclass
class AITask:
    """AI 任務定義
    
    統一的任務封裝，用於模組間通信
    
    Attributes:
        task_type: 任務類型
        target: 目標 (URL、文件路徑等)
        parameters: 任務參數
        description: 任務描述
        priority: 優先級 (1-10)
        timeout: 超時時間 (秒)
        metadata: 額外元數據
    """
    task_type: AITaskType
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    priority: int = 5
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """後處理：確保參數類型正確"""
        if isinstance(self.task_type, str):
            self.task_type = AITaskType(self.task_type)
        
        if self.parameters is None:
            self.parameters = {}
        
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AIResult:
    """AI 任務結果
    
    統一的結果封裝，包含執行狀態、數據、錯誤信息等
    
    Attributes:
        success: 是否成功
        data: 結果數據
        error: 錯誤信息
        execution_time: 執行時間 (秒)
        trace: 執行追蹤信息
        metrics: 性能指標
        feedback: 反饋信息 (用於學習)
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    trace: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "trace": self.trace,
            "metrics": self.metrics,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }


class AIModulePlugin(ABC):
    """AI 模組插件接口
    
    所有 AI 模組必須實現此接口才能被註冊到系統中。
    設計參考 Kubernetes Device Plugin Pattern。
    
    生命週期:
        1. 實例化
        2. initialize() - 初始化配置
        3. load_weights() - 載入權重 (如果需要)
        4. health_check() - 健康檢查
        5. execute_task() - 執行任務 (循環)
        6. shutdown() - 優雅關閉
    
    Example:
        class MyPlugin(AIModulePlugin):
            @property
            def module_id(self) -> str:
                return "my_plugin"
            
            async def initialize(self, config: Dict) -> bool:
                self.config = config
                return True
            
            async def execute_task(self, task: AITask) -> AIResult:
                # 實現任務邏輯
                return AIResult(success=True, data={...})
    """
    
    @property
    @abstractmethod
    def module_id(self) -> str:
        """模組唯一標識符
        
        命名規範: 小寫字母和下劃線，不能包含空格
        
        Example:
            "bio_neuron"
            "scanner"
            "xss_exploiter"
        
        Returns:
            模組 ID 字符串
        """
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """模組支援的能力列表
        
        能力應該是動詞或動詞短語，描述模組可以執行的操作
        
        Example:
            ["scan", "analyze_vulnerabilities", "generate_report"]
        
        Returns:
            能力列表
        """
        pass
    
    @property
    def requires_weights(self) -> bool:
        """是否需要載入 ML 模型權重
        
        Returns:
            True: 此模組需要權重 (如 BioNeuron)
            False: 此模組基於規則或不需要權重 (如 Scanner)
        """
        return False
    
    @property
    def version(self) -> str:
        """模組版本號
        
        使用語義化版本 (Semantic Versioning): MAJOR.MINOR.PATCH
        
        Returns:
            版本號字符串
        """
        return "1.0.0"
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化模組
        
        在模組註冊時調用，用於載入配置、初始化資源等
        
        Args:
            config: 模組配置字典，來自配置文件或註冊時提供
        
        Returns:
            初始化是否成功
        
        Raises:
            可以拋出異常來指示初始化失敗
        """
        pass
    
    async def load_weights(self, weight_path: Path) -> bool:  # type: ignore[misc]
        """載入模型權重 - async保留供未來異步載入擴展 (可選實現)
        
        只有 requires_weights=True 的模組需要實現此方法
        
        Args:
            weight_path: 權重文件路徑，由 WeightManager 提供
        
        Returns:
            載入是否成功
        
        Raises:
            NotImplementedError: 如果 requires_weights=True 但未實現此方法
        """
        if self.requires_weights:
            raise NotImplementedError(
                f"Plugin {self.module_id} requires weights but load_weights() not implemented"
            )
        return True
    
    @abstractmethod
    async def execute_task(self, task: AITask) -> AIResult:
        """執行 AI 任務
        
        這是插件的核心功能，處理具體的業務邏輯
        
        Args:
            task: AI 任務對象，包含任務類型、參數等
        
        Returns:
            任務執行結果
        
        Raises:
            可以拋出異常，會被 AICommander 捕獲並記錄
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康檢查
        
        用於檢測模組是否正常運行，會被定期調用
        
        建議檢查項:
        - 模組是否已初始化
        - 權重是否已載入
        - 依賴服務是否可用
        - 簡單的推理測試
        
        Returns:
            模組是否健康
        """
        pass
    
    async def shutdown(self) -> None:
        """優雅關閉
        
        在系統關閉時調用，用於清理資源、保存狀態等
        
        建議操作:
        - 關閉網絡連接
        - 釋放內存
        - 保存未完成的任務
        - 清理臨時文件
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取模組元數據
        
        Returns:
            包含版本、能力、依賴等信息的字典
        """
        return {
            "module_id": self.module_id,
            "version": self.version,
            "capabilities": self.capabilities,
            "requires_weights": self.requires_weights,
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.module_id}, version={self.version})>"
