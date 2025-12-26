"""
AIVA Scan - 純 CLI 掃描引擎集合

v3.0 架構：直接 CLI 調用，無需任何 Python 協調層

各引擎完全獨立：
  ✅ Rust:       ./rust_engine/target/release/aiva-info-gatherer.exe
  ✅ Go SSRF:    ./go_engine/bin/ssrf-scanner.exe  
  ✅ Go CSPM:    ./go_engine/bin/cspm-scanner.exe
  ✅ Go SCA:     ./go_engine/bin/sca-scanner.exe
  ✅ TypeScript: node ./typescript_engine/dist/index.js
  ✅ Python:     python ./python_engine/main.py

特點：
  - 🚀 無協調器，直接 CLI 調用
  - 🔧 每個引擎獨立編譯部署
  - 📦 無 Python 依賴管理復雜性
  - ⚡ AI 直接調用，性能最優

部署方式：
  1. 各引擎獨立編譯：cargo build、go build、npm run build
  2. AI 直接通過 subprocess 調用 CLI
  3. 無需啟動任何服務或協調器

這是最簡潔高效的架構！
"""

# CLI 架構下，無需導出任何 Python 模組
__all__ = []
