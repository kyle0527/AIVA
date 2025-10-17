# AIVA 架構圖索引 | Architecture Diagrams Index

**生成時間 Generated**: 2025-10-13 14:40:20
**總計圖表 Total Diagrams**: 14

---

## 圖表列表 | Diagram List

1. [01 Overall Architecture](01_overall_architecture.mmd)
2. [02 Modules Overview](02_modules_overview.mmd)
3. [03 Core Module](03_core_module.mmd)
4. [04 Scan Module](04_scan_module.mmd)
5. [05 Function Module](05_function_module.mmd)
6. [06 Integration Module](06_integration_module.mmd)
7. [07 Sqli Flow](07_sqli_flow.mmd)
8. [08 Xss Flow](08_xss_flow.mmd)
9. [09 Ssrf Flow](09_ssrf_flow.mmd)
10. [10 Idor Flow](10_idor_flow.mmd)
11. [11 Complete Workflow](11_complete_workflow.mmd)
12. [12 Language Decision](12_language_decision.mmd)
13. [13 Data Flow](13_data_flow.mmd)
14. [14 Deployment Architecture](14_deployment_architecture.mmd)

---

## 使用說明 | Usage Guide

### 1. 查看圖表 | View Diagrams

在支援 Mermaid 的環境中查看:
- VS Code + Mermaid 擴展
- GitHub / GitLab
- https://mermaid.live/

### 2. 匯出圖片 | Export Images

```bash
# 匯出 PNG 格式
python tools/generate_complete_architecture.py --export png

# 匯出 SVG 格式
python tools/generate_complete_architecture.py --export svg

# 匯出 PDF 格式
python tools/generate_complete_architecture.py --export pdf
```

### 3. 更新圖表 | Update Diagrams

```bash
# 重新生成所有圖表
python tools/generate_complete_architecture.py

# 生成並匯出
python tools/generate_complete_architecture.py --export png
```

---

**維護者 Maintainer**: AIVA Development Team
