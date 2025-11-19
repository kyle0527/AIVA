"""
Go 專業掃描器集群

重構後的目錄結構：
- cmd/: 命令入口點 (ssrf-scanner, cspm-scanner, sca-scanner)
- internal/: 內部實現邏輯 (ssrf, cspm, sca, common)
- pkg/: 共享模型 (models)
- dispatcher/: Python 協調器
- bin/: 編譯產物

路徑已從 services.scan.go_scanners 更新為 services.scan.engines.go_engine
"""

__all__ = [
    "cmd",
    "internal",
    "pkg",
    "dispatcher",
]