"""
AIVA Scan - 多語言掃描引擎（無協調層架構）

架構原則：每個引擎完全獨立，AI 直接調用

引擎列表：
  ✅ Rust Engine:       端口掃描、服務識別、快速信息收集
  ✅ Go Engine:         SSRF、SCA、CSPM 掃描器
  ✅ TypeScript Engine: DOM XSS、SPA 爬蟲、動態掃描
  ✅ Python Engine:     XXE、反序列化、被動分析

使用方式：
  1. 編譯各語言引擎：cargo build / go build / npm build
  2. AI 通過 subprocess 直接調用 CLI
  3. 無需任何 Python 協調器或中間層

Python 檢測器可直接導入：
  from services.scan.python_engine import XXEDetector, DeserializationDetector, PassiveAnalyzer
"""

# Python 檢測器導出（可選）
from .python_engine import DeserializationDetector, PassiveAnalyzer, XXEDetector

__all__ = [
    "XXEDetector",
    "DeserializationDetector",
    "PassiveAnalyzer",
]

__version__ = "3.1.0"
