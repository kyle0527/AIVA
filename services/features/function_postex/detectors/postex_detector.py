"""
function_postex.detectors.postex_detector
Post-Exploitation 主檢測器 (相容層)

此模組已統一至 detector/postex_detector.py。
保留此檔案以維持向後相容，所有類別從主路徑重新匯出。
"""

# 重新匯出主路徑的 PostExDetector，避免命名衝突
from ..detector.postex_detector import PostExDetector

__all__ = ["PostExDetector"]
