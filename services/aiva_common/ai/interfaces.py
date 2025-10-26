"""
AIVA AI 介面定義 - 可插拔 AI 組件介面

此文件定義所有 AI 組件的抽象介面，支援可插拔架構設計。
各模組可以提供不同的實現，透過依賴注入進行替換。

設計模式:
- 抽象工廠模式 (Abstract Factory)
- 策略模式 (Strategy Pattern) 
- 依賴注入 (Dependency Injection)
- 介面分離原則 (Interface Segregation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncContextManager
from datetime import datetime

from ..schemas.ai import (
    AttackPlan,
    PlanExecutionResult,
    ExperienceSample,
    RAGQueryPayload,
    RAGResponsePayload,
)


# ============================================================================
# Core AI Interfaces (核心 AI 介面)
# ============================================================================


class IDialogAssistant(ABC):
    """對話助手介面 - 可插拔的自然語言交互組件"""

    @abstractmethod
    async def process_user_input(
        self, 
        user_input: str, 
        user_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """處理使用者輸入並產生回應
        
        Args:
            user_input: 使用者輸入文字
            user_id: 使用者識別碼
            context: 對話上下文
            
        Returns:
            包含意圖、回應和可執行動作的字典
        """
        pass

    @abstractmethod
    async def identify_intent(
        self, 
        user_input: str
    ) -> tuple[str, Dict[str, Any]]:
        """識別使用者意圖
        
        Args:
            user_input: 使用者輸入
            
        Returns:
            (意圖類型, 提取的參數)
        """
        pass

    @abstractmethod
    def get_conversation_history(
        self, 
        limit: int = 10, 
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """獲取對話歷史
        
        Args:
            limit: 返回記錄數限制
            user_id: 指定使用者ID (None 表示所有使用者)
            
        Returns:
            對話歷史記錄列表
        """
        pass


class IPlanExecutor(ABC):
    """計劃執行器介面 - 可插拔的攻擊計劃執行組件"""

    @abstractmethod
    async def execute_plan(
        self,
        plan: AttackPlan,
        sandbox_mode: bool = True,
        timeout_minutes: int = 30,
    ) -> PlanExecutionResult:
        """執行攻擊計劃
        
        Args:
            plan: 攻擊計劃
            sandbox_mode: 沙箱模式 
            timeout_minutes: 超時分鐘數
            
        Returns:
            執行結果
        """
        pass

    @abstractmethod
    async def wait_for_result(
        self, 
        task_id: str, 
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """等待任務結果 (可插拔的結果訂閱機制)
        
        Args:
            task_id: 任務ID
            timeout: 超時秒數
            
        Returns:
            任務執行結果
        """
        pass

    @abstractmethod
    async def get_execution_status(
        self, 
        plan_id: str
    ) -> Dict[str, Any]:
        """獲取執行狀態
        
        Args:
            plan_id: 計劃ID
            
        Returns:
            執行狀態信息
        """
        pass


class IExperienceManager(ABC):
    """經驗管理器介面 - 可插拔的學習經驗管理組件"""

    @abstractmethod
    async def store_experience(
        self, 
        experience: ExperienceSample
    ) -> bool:
        """存儲經驗樣本
        
        Args:
            experience: 經驗樣本
            
        Returns:
            是否存儲成功
        """
        pass

    @abstractmethod
    async def retrieve_experiences(
        self,
        query_context: Dict[str, Any],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[ExperienceSample]:
        """檢索相關經驗
        
        Args:
            query_context: 查詢上下文
            limit: 返回數量限制
            similarity_threshold: 相似度閾值
            
        Returns:
            相關經驗樣本列表
        """
        pass

    @abstractmethod
    async def create_learning_session(
        self, 
        session_config: Dict[str, Any]
    ) -> str:
        """創建學習會話
        
        Args:
            session_config: 會話配置
            
        Returns:
            會話ID
        """
        pass


class ICapabilityEvaluator(ABC):
    """能力評估器介面 - 可插拔的能力評估組件"""

    @abstractmethod
    async def evaluate_capability(
        self,
        capability_id: str,
        execution_evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        """評估能力表現
        
        Args:
            capability_id: 能力ID
            execution_evidence: 執行證據
            
        Returns:
            評估結果 (包含評分、建議等)
        """
        pass

    @abstractmethod
    async def collect_capability_evidence(
        self,
        capability_id: str,
        time_window_days: int = 7
    ) -> List[Dict[str, Any]]:
        """收集能力證據
        
        Args:
            capability_id: 能力ID
            time_window_days: 時間窗口天數
            
        Returns:
            證據列表
        """
        pass

    @abstractmethod
    async def update_capability_scorecard(
        self,
        capability_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """更新能力記分卡
        
        Args:
            capability_id: 能力ID
            metrics: 指標數據
            
        Returns:
            是否更新成功
        """
        pass


class ICrossLanguageBridge(ABC):
    """跨語言橋接器介面 - 可插拔的多語言支援組件"""

    @abstractmethod
    async def execute_subprocess(
        self,
        language: str,
        executable_path: str,
        args: List[str],
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """執行子進程
        
        Args:
            language: 程式語言 (python/go/rust/node)
            executable_path: 可執行文件路徑
            args: 參數列表
            timeout: 超時秒數
            env: 環境變數
            
        Returns:
            執行結果 (包含 stdout, stderr, exit_code)
        """
        pass

    @abstractmethod
    async def sync_results(
        self,
        source_language: str,
        target_language: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """同步跨語言結果
        
        Args:
            source_language: 來源語言
            target_language: 目標語言 
            data: 要同步的數據
            
        Returns:
            同步後的結果
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """獲取支援的語言列表
        
        Returns:
            支援的程式語言列表
        """
        pass


class IRAGAgent(ABC):
    """RAG 檢索代理介面 - 可插拔的知識檢索組件"""

    @abstractmethod
    async def invoke(
        self, 
        query: RAGQueryPayload
    ) -> RAGResponsePayload:
        """執行 RAG 查詢 (BioNeuronRAGAgent.invoke() 的抽象介面)
        
        Args:
            query: RAG 查詢請求
            
        Returns:
            RAG 查詢響應
        """
        pass

    @abstractmethod
    async def update_knowledge_base(
        self,
        knowledge_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新知識庫
        
        Args:
            knowledge_type: 知識類型
            content: 知識內容
            metadata: 元數據
            
        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    async def search_knowledge(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索知識
        
        Args:
            query_text: 查詢文字
            top_k: 返回數量
            filters: 過濾條件
            
        Returns:
            匹配的知識項目列表
        """
        pass


class ISkillGraphAnalyzer(ABC):
    """技能圖分析器介面 - 可插拔的技能圖分析組件"""

    @abstractmethod
    async def analyze_skill_path(
        self,
        source_capability: str,
        target_capability: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """分析技能路徑 (基於 NetworkX)
        
        Args:
            source_capability: 起始能力
            target_capability: 目標能力
            constraints: 路徑約束
            
        Returns:
            技能路徑列表 (包含路徑步驟和權重)
        """
        pass

    @abstractmethod
    async def build_capability_graph(
        self,
        capabilities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """建構能力圖
        
        Args:
            capabilities: 能力列表
            
        Returns:
            能力圖結構 (節點和邊的信息)
        """
        pass

    @abstractmethod
    async def get_capability_recommendations(
        self,
        current_capabilities: List[str],
        target_scenario: str
    ) -> List[Dict[str, Any]]:
        """獲取能力建議
        
        Args:
            current_capabilities: 當前能力列表
            target_scenario: 目標場景
            
        Returns:
            建議的能力和優先級
        """
        pass


# ============================================================================
# AI Factory Interface (AI 工廠介面)
# ============================================================================


class IAIComponentFactory(ABC):
    """AI 組件工廠介面 - 用於創建可插拔的 AI 組件"""

    @abstractmethod
    def create_dialog_assistant(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> IDialogAssistant:
        """創建對話助手實例"""
        pass

    @abstractmethod
    def create_plan_executor(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> IPlanExecutor:
        """創建計劃執行器實例"""
        pass

    @abstractmethod
    def create_experience_manager(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> IExperienceManager:
        """創建經驗管理器實例"""
        pass

    @abstractmethod
    def create_capability_evaluator(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> ICapabilityEvaluator:
        """創建能力評估器實例"""
        pass

    @abstractmethod
    def create_cross_language_bridge(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> ICrossLanguageBridge:
        """創建跨語言橋接器實例"""
        pass

    @abstractmethod
    def create_rag_agent(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> IRAGAgent:
        """創建 RAG 代理實例"""
        pass

    @abstractmethod
    def create_skill_graph_analyzer(
        self, 
        config: Optional[Dict[str, Any]] = None
    ) -> ISkillGraphAnalyzer:
        """創建技能圖分析器實例"""
        pass


# ============================================================================
# Registry Interface (註冊表介面)
# ============================================================================


class IAIComponentRegistry(ABC):
    """AI 組件註冊表介面 - 管理可插拔組件的註冊和發現"""

    @abstractmethod
    def register_component(
        self,
        component_type: str,
        component_name: str,
        component_class: type,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """註冊 AI 組件
        
        Args:
            component_type: 組件類型 (dialog/executor/rag 等)
            component_name: 組件名稱
            component_class: 組件類別
            config: 預設配置
            
        Returns:
            是否註冊成功
        """
        pass

    @abstractmethod
    def get_component(
        self,
        component_type: str,
        component_name: Optional[str] = None
    ) -> Optional[Any]:
        """獲取 AI 組件實例
        
        Args:
            component_type: 組件類型
            component_name: 組件名稱 (None 表示使用預設)
            
        Returns:
            組件實例
        """
        pass

    @abstractmethod
    def list_components(
        self, 
        component_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """列出可用組件
        
        Args:
            component_type: 組件類型 (None 表示所有類型)
            
        Returns:
            組件類型到組件名稱列表的映射
        """
        pass

    @abstractmethod
    def unregister_component(
        self,
        component_type: str,
        component_name: str
    ) -> bool:
        """取消註冊組件
        
        Args:
            component_type: 組件類型
            component_name: 組件名稱
            
        Returns:
            是否取消成功
        """
        pass


# ============================================================================
# Context Manager Interface (上下文管理器介面)  
# ============================================================================


class IAIContext(AsyncContextManager):
    """AI 上下文管理器介面 - 管理 AI 組件的生命週期"""

    @abstractmethod
    async def initialize(self) -> None:
        """初始化 AI 上下文"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """清理 AI 上下文"""
        pass

    @abstractmethod
    def get_component(self, component_type: str) -> Optional[Any]:
        """從上下文中獲取組件"""
        pass

    @abstractmethod
    def set_component(self, component_type: str, component: Any) -> None:
        """設置組件到上下文"""
        pass