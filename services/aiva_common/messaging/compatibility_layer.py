"""
AIVA MQ 兼容性層 - V2 統一架構遷移支援
=====================================

支援新舊訊息格式雙軌運行，確保漸進式遷移
- V1 格式: 舊版 routing key + 簡單 payload
- V2 格式: AivaMessage 統一信封 + 增強 header

功能:
- 雙向格式轉換
- 漸進式遷移支援
- 向後兼容性維護
- 自動格式檢測
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from aiva_common.messaging.unified_topic_manager import topic_manager
from aiva_common.schemas.generated.messaging import (
    AivaMessage,
)

logger = logging.getLogger(__name__)


class MessageFormat:
    """訊息格式枚舉"""
    V1_LEGACY = "v1_legacy"      # 舊版格式
    V2_UNIFIED = "v2_unified"    # V2 統一格式
    UNKNOWN = "unknown"          # 未知格式


class CompatibilityLayer:
    """MQ 兼容性層
    
    負責:
    1. 新舊格式雙向轉換
    2. 自動格式檢測
    3. 漸進式遷移支援
    4. 統計與監控
    """

    def __init__(self):
        self._migration_stats = {
            "v1_messages_received": 0,
            "v2_messages_received": 0,
            "v1_to_v2_conversions": 0,
            "v2_to_v1_fallbacks": 0,
            "format_detection_errors": 0
        }
        self._v2_enabled_modules = set()  # 已啟用 V2 的模組

    def detect_message_format(self, message_data: str | bytes | dict) -> str:
        """自動檢測訊息格式"""
        try:
            # 統一轉為字典
            if isinstance(message_data, (str, bytes)):
                data = json.loads(message_data)
            else:
                data = message_data

            # V2 格式檢測（包含完整 AivaMessage 結構）
            if (isinstance(data, dict) and
                'header' in data and
                'topic' in data and
                'schema_version' in data and
                'source_module' in data):
                return MessageFormat.V2_UNIFIED

            # V1 格式檢測（舊版簡單結構）
            if (isinstance(data, dict) and
                ('routing_key' in data or 'exchange' in data or 'queue' in data)):
                return MessageFormat.V1_LEGACY

            return MessageFormat.UNKNOWN

        except Exception as e:
            logger.error(f"❌ 格式檢測錯誤: {e}")
            self._migration_stats["format_detection_errors"] += 1
            return MessageFormat.UNKNOWN

    def convert_v1_to_v2(
        self,
        v1_data: dict[str, Any],
        source_module: str,
        trace_id: str | None = None
    ) -> AivaMessage:
        """V1 格式轉換為 V2 格式"""
        try:
            # 提取 V1 欄位
            routing_key = v1_data.get('routing_key', 'unknown.topic')
            payload = v1_data.get('payload', v1_data.get('body', {}))

            # 映射到 V2 Topic
            normalized_topic = topic_manager.normalize_topic(routing_key)

            # 創建增強版訊息
            v2_message = topic_manager.create_enhanced_message(
                topic=normalized_topic,
                payload=payload,
                source_module=source_module,
                target_module=v1_data.get('target_module'),
                trace_id=trace_id,
                metadata={
                    'converted_from': 'v1_legacy',
                    'original_routing_key': routing_key,
                    'conversion_timestamp': datetime.now().isoformat()
                }
            )

            self._migration_stats["v1_to_v2_conversions"] += 1
            logger.info(f"🔄 V1->V2 轉換: {routing_key} -> {normalized_topic}")

            return v2_message

        except Exception as e:
            logger.error(f"❌ V1->V2 轉換失敗: {e}")
            raise

    def convert_v2_to_v1(self, v2_message: AivaMessage) -> dict[str, Any]:
        """V2 格式轉換為 V1 格式（向後兼容）"""
        try:
            # 構建 V1 格式
            v1_data = {
                'routing_key': v2_message.topic,
                'exchange': 'aiva_exchange',
                'payload': v2_message.payload,
                'timestamp': v2_message.header.timestamp.isoformat(),
                'source': v2_message.source_module,
                'metadata': {
                    'converted_from': 'v2_unified',
                    'original_message_id': v2_message.header.message_id,
                    'trace_id': v2_message.trace_id,
                    'schema_version': v2_message.schema_version
                }
            }

            self._migration_stats["v2_to_v1_fallbacks"] += 1
            logger.info(f"🔙 V2->V1 回退: {v2_message.topic}")

            return v1_data

        except Exception as e:
            logger.error(f"❌ V2->V1 轉換失敗: {e}")
            raise

    def process_incoming_message(
        self,
        raw_message: str | bytes | dict,
        source_module: str
    ) -> tuple[AivaMessage, str]:
        """處理接收到的訊息
        
        Returns:
            (AivaMessage, original_format)
        """
        try:
            # 檢測格式
            format_type = self.detect_message_format(raw_message)

            # 統一轉為字典
            if isinstance(raw_message, (str, bytes)):
                message_data = json.loads(raw_message)
            else:
                message_data = raw_message

            if format_type == MessageFormat.V2_UNIFIED:
                # V2 格式，直接解析
                v2_message = AivaMessage.model_validate(message_data)
                self._migration_stats["v2_messages_received"] += 1
                return v2_message, format_type

            elif format_type == MessageFormat.V1_LEGACY:
                # V1 格式，轉換為 V2
                v2_message = self.convert_v1_to_v2(message_data, source_module)
                self._migration_stats["v1_messages_received"] += 1
                return v2_message, format_type

            else:
                # 未知格式，嘗試最佳猜測
                logger.warning(f"⚠️  未知訊息格式，嘗試 V1 轉換: {raw_message}")
                v2_message = self.convert_v1_to_v2(message_data, source_module)
                return v2_message, MessageFormat.UNKNOWN

        except Exception as e:
            logger.error(f"❌ 訊息處理失敗: {e}")
            raise

    def prepare_outgoing_message(
        self,
        v2_message: AivaMessage,
        target_module: str | None = None
    ) -> AivaMessage | dict[str, Any]:
        """準備發送訊息
        
        根據目標模組是否支援 V2 決定格式
        """
        try:
            # 如果目標模組支援 V2 或未指定目標（廣播），發送 V2
            if not target_module or target_module in self._v2_enabled_modules:
                return v2_message

            # 否則轉換為 V1 格式
            logger.info(f"🔙 為 {target_module} 轉換為 V1 格式")
            return self.convert_v2_to_v1(v2_message)

        except Exception as e:
            logger.error(f"❌ 訊息準備失敗: {e}")
            raise

    def enable_v2_for_module(self, module_name: str):
        """啟用模組的 V2 支援"""
        self._v2_enabled_modules.add(module_name)
        logger.info(f"✅ 模組 {module_name} 已啟用 V2 格式支援")

    def disable_v2_for_module(self, module_name: str):
        """禁用模組的 V2 支援（回退到 V1）"""
        self._v2_enabled_modules.discard(module_name)
        logger.info(f"🔙 模組 {module_name} 已回退到 V1 格式")

    def get_v2_enabled_modules(self) -> set[str]:
        """獲取已啟用 V2 的模組列表"""
        return self._v2_enabled_modules.copy()

    def get_migration_stats(self) -> dict[str, Any]:
        """獲取遷移統計資料"""
        total_messages = (self._migration_stats["v1_messages_received"] +
                         self._migration_stats["v2_messages_received"])

        v2_adoption_rate = (
            self._migration_stats["v2_messages_received"] / total_messages
            if total_messages > 0 else 0
        )

        return {
            **self._migration_stats,
            "total_messages": total_messages,
            "v2_adoption_rate": v2_adoption_rate,
            "v2_enabled_modules": len(self._v2_enabled_modules),
            "v2_enabled_module_list": list(self._v2_enabled_modules)
        }

    def reset_stats(self):
        """重置統計資料"""
        for key in self._migration_stats:
            self._migration_stats[key] = 0
        logger.info("📊 遷移統計資料已重置")


# 全域兼容性層實例
compatibility_layer = CompatibilityLayer()


class UnifiedMessageBroker:
    """統一訊息代理器
    
    整合 Topic 管理器與兼容性層，提供統一的 MQ 介面
    """

    def __init__(self):
        self.topic_manager = topic_manager
        self.compatibility = compatibility_layer

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        source_module: str,
        target_module: str | None = None,
        **kwargs
    ) -> AivaMessage:
        """統一發佈介面"""
        # 創建 V2 訊息
        v2_message = self.topic_manager.create_enhanced_message(
            topic=topic,
            payload=payload,
            source_module=source_module,
            target_module=target_module,
            **kwargs
        )

        # 根據目標準備格式（實際發送邏輯由調用方處理）
        self.compatibility.prepare_outgoing_message(v2_message, target_module)

        logger.info(f"📤 發佈訊息: {topic} ({source_module} -> {target_module or 'broadcast'})")
        return v2_message

    def subscribe_and_process(
        self,
        raw_message: str | bytes | dict,
        source_module: str
    ) -> AivaMessage:
        """統一訂閱處理介面"""
        v2_message, original_format = self.compatibility.process_incoming_message(
            raw_message, source_module
        )

        logger.info(f"📥 接收訊息: {v2_message.topic} (格式: {original_format})")
        return v2_message


# 全域統一訊息代理器實例
message_broker = UnifiedMessageBroker()
