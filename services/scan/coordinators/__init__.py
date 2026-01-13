"""Scan Coordinators - 掃描引擎協調層

輕量級協調器，連接 AI 內部任務規劃與外部掃描引擎

架構設計：
    AI Core (task_planning) → MultiEngineCoordinator → CLI Engines (Rust/Go/TS/Python)
    
特點：
    - 🎯 輕量級：專注任務分發與結果收集
    - ⚡ 高效能：保持 CLI 直接調用的效能優勢
    - 🔄 AI 友好：提供結構化輸出和錯誤恢復
    - 🧩 功能兼顧：同時支援掃描任務和功能模組

實現日期: 2026-01-11
版本: v1.0
"""

from .multi_engine_coordinator import (
    MultiEngineCoordinator,
    ScanStrategy,
    EngineStatus,
    EngineConfig,
    ScanResult
)

__all__ = [
    "MultiEngineCoordinator",
    "ScanStrategy",
    "EngineStatus",
    "EngineConfig",
    "ScanResult"
]
