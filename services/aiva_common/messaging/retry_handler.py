"""
統一重試處理機制

解決 poison pill 消息問題，實施重試限制和死信隊列
遵循 AIVA 架構模式和 12-factor app 原則
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetryPolicy(BaseModel):
    """重試策略配置"""
    max_attempts: int = Field(default=3, ge=1, le=10, description="最大重試次數")
    backoff_base: float = Field(default=1.0, gt=0, description="退避基礎時間(秒)")
    backoff_factor: float = Field(default=2.0, gt=1, description="退避倍數")
    max_backoff: float = Field(default=60.0, gt=0, description="最大退避時間(秒)")
    dead_letter_exchange: str = Field(default="aiva.dead_letter", description="死信交換機")
    dead_letter_routing_key: str = Field(default="failed", description="死信路由鍵")


class MessageMetadata(BaseModel):
    """消息元數據"""
    retry_count: int = Field(default=0, ge=0, description="當前重試次數")
    first_attempt: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), description="首次處理時間")
    last_attempt: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), description="最後重試時間")
    error_history: list[str] = Field(default_factory=list, description="錯誤歷史")
    original_routing_key: Optional[str] = Field(default=None, description="原始路由鍵")


class RetryHandler:
    """統一重試處理器
    
    負責管理消息重試邏輯，防止 poison pill 消息無限循環
    """
    
    def __init__(self, policy: RetryPolicy = None):
        """初始化重試處理器
        
        Args:
            policy: 重試策略，如未提供則使用默認策略
        """
        self.policy = policy or RetryPolicy()
        logger.info(f"RetryHandler initialized with policy: max_attempts={self.policy.max_attempts}")
    
    def should_retry(self, message_headers: Dict[str, Any], error: Exception) -> bool:
        """判斷是否應該重試
        
        Args:
            message_headers: 消息頭部信息
            error: 發生的錯誤
            
        Returns:
            True 如果應該重試，False 如果應該放棄
        """
        metadata = self._extract_metadata(message_headers)
        
        # 檢查重試次數
        if metadata.retry_count >= self.policy.max_attempts:
            logger.warning(
                f"消息已達到最大重試次數 {self.policy.max_attempts}，"
                f"將發送到死信隊列: {error}"
            )
            return False
        
        # 記錄錯誤
        metadata.error_history.append(f"{datetime.now(UTC).isoformat()}: {str(error)}")
        
        logger.info(
            f"消息重試 {metadata.retry_count + 1}/{self.policy.max_attempts}: {error}"
        )
        return True
    
    def get_retry_headers(self, original_headers: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """獲取重試消息的頭部信息
        
        Args:
            original_headers: 原始消息頭部
            error: 發生的錯誤
            
        Returns:
            更新後的消息頭部
        """
        metadata = self._extract_metadata(original_headers)
        
        # 更新重試信息
        metadata.retry_count += 1
        metadata.last_attempt = datetime.now(UTC).isoformat()
        metadata.error_history.append(f"{metadata.last_attempt}: {str(error)}")
        
        # 計算退避延遲
        delay = min(
            self.policy.max_backoff,
            self.policy.backoff_base * (self.policy.backoff_factor ** (metadata.retry_count - 1))
        )
        
        # 更新頭部信息
        headers = original_headers.copy()
        headers.update({
            'x-aiva-retry-count': metadata.retry_count,
            'x-aiva-first-attempt': metadata.first_attempt,
            'x-aiva-last-attempt': metadata.last_attempt,
            'x-aiva-error-history': json.dumps(metadata.error_history),
            'x-aiva-retry-delay': int(delay * 1000),  # 毫秒
            'x-death': [],  # 清除 RabbitMQ 的死信信息，使用我們自己的追蹤
        })
        
        if metadata.original_routing_key:
            headers['x-aiva-original-routing-key'] = metadata.original_routing_key
        
        return headers
    
    def get_dead_letter_headers(self, original_headers: Dict[str, Any], final_error: Exception) -> Dict[str, Any]:
        """獲取死信消息的頭部信息
        
        Args:
            original_headers: 原始消息頭部
            final_error: 最終錯誤
            
        Returns:
            死信消息頭部
        """
        metadata = self._extract_metadata(original_headers)
        
        # 標記為最終失敗
        metadata.error_history.append(
            f"{datetime.now(UTC).isoformat()}: FINAL_FAILURE: {str(final_error)}"
        )
        
        headers = original_headers.copy()
        headers.update({
            'x-aiva-final-failure': True,
            'x-aiva-retry-count': metadata.retry_count,
            'x-aiva-error-history': json.dumps(metadata.error_history),
            'x-aiva-failed-at': datetime.now(UTC).isoformat(),
            'x-aiva-dead-letter-reason': 'max_retries_exceeded',
        })
        
        return headers
    
    def _extract_metadata(self, headers: Dict[str, Any]) -> MessageMetadata:
        """從消息頭部提取元數據
        
        Args:
            headers: 消息頭部信息
            
        Returns:
            消息元數據
        """
        try:
            retry_count = int(headers.get('x-aiva-retry-count', 0))
            first_attempt = headers.get('x-aiva-first-attempt')
            last_attempt = headers.get('x-aiva-last-attempt')
            error_history_str = headers.get('x-aiva-error-history', '[]')
            original_routing_key = headers.get('x-aiva-original-routing-key')
            
            error_history = json.loads(error_history_str) if error_history_str else []
            
            return MessageMetadata(
                retry_count=retry_count,
                first_attempt=first_attempt or datetime.now(UTC).isoformat(),
                last_attempt=last_attempt or datetime.now(UTC).isoformat(),
                error_history=error_history,
                original_routing_key=original_routing_key
            )
            
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"無法解析消息元數據，使用默認值: {e}")
            return MessageMetadata()
    
    def calculate_retry_delay(self, retry_count: int) -> float:
        """計算重試延遲時間
        
        Args:
            retry_count: 當前重試次數
            
        Returns:
            延遲時間（秒）
        """
        return min(
            self.policy.max_backoff,
            self.policy.backoff_base * (self.policy.backoff_factor ** retry_count)
        )


# 預定義的重試策略
FAST_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    backoff_base=0.5,
    backoff_factor=2.0,
    max_backoff=5.0
)

STANDARD_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff_base=1.0,
    backoff_factor=2.0,
    max_backoff=30.0
)

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    backoff_base=2.0,
    backoff_factor=1.5,
    max_backoff=60.0
)


# 默認重試處理器實例
default_retry_handler = RetryHandler(STANDARD_RETRY_POLICY)