"""
⚠️ DEPRECATED - 此文件已棄用 ⚠️

此文件中的所有類定義與其他模組重複，且未在 services.integration.__init__.py 中導出。
經檢查確認此文件未被任何地方導入使用。

請使用以下替代方案：
- IOCRecord → services.integration.models.EnhancedIOCRecord
- SIEMEvent → services.integration.models.SIEMEvent  
- ThreatIntelPayload → services.aiva_common.schemas.ThreatIntelLookupPayload
- WebhookPayload → services.aiva_common.schemas.WebhookPayload
- ThreatIndicator, SIEMIntegrationConfig, AssetLifecycleEvent 等 → 如需使用請移至 models.py

此文件保留僅供參考，計劃在未來版本中移除。
建議不要從此文件導入任何內容。

最後更新: 2025-10-25
狀態: DEPRECATED (未使用，未導出)
"""

# 此文件已棄用，請勿使用
# 所有功能請使用 services.integration.models 或 services.aiva_common.schemas
