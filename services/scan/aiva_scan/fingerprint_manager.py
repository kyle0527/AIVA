"""
指紋管理器 - 負責收集、合併和管理掃描過程中的指紋信息
"""

from __future__ import annotations

import httpx

from services.aiva_common.schemas import Fingerprints

from .info_gatherer.passive_fingerprinter import PassiveFingerprinter


class FingerprintMerger:
    """類型安全的指紋合併器"""

    def merge(self, existing: Fingerprints | None, new: Fingerprints) -> Fingerprints:
        """合併兩個指紋對象，使用類型安全的邏輯"""
        if existing is None:
            return new

        # 類型安全的合併邏輯 - 優先保留已有的非空值
        return Fingerprints(
            web_server=new.web_server or existing.web_server,
            framework=new.framework or existing.framework,
            language=new.language or existing.language,
            waf_detected=new.waf_detected or existing.waf_detected,
            waf_vendor=new.waf_vendor or existing.waf_vendor,
        )


class FingerprintCollector:
    """指紋信息收集和管理器"""

    def __init__(self):
        self.passive_fp = PassiveFingerprinter()
        self.merger = FingerprintMerger()
        self.collected_fingerprints: Fingerprints | None = None

    async def process_response(self, response: httpx.Response) -> None:
        """處理HTTP回應並收集指紋信息"""
        current_fp = self.passive_fp.from_headers(dict(response.headers))
        if current_fp:
            self.collected_fingerprints = self.merger.merge(
                self.collected_fingerprints, current_fp
            )

    def get_final_fingerprints(self) -> Fingerprints | None:
        """獲取最終合併的指紋信息"""
        return self.collected_fingerprints

    def reset(self) -> None:
        """重置收集器狀態"""
        self.collected_fingerprints = None
