"""
function_postex.detectors
後滲透檢測器模組

注意: 此路徑已統一到 detector/ 路徑。
此處重新匯出以保持向後相容。
"""

from ..detector.postex_detector import PostExDetector

__all__ = ["PostExDetector"]
