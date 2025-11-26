# Forensic Tools Module

## 模組概述

數位鑑識工具模組，提供證據收集、分析和報告生成能力。

**風險等級**: L0  
**模組版本**: 1.0.0

## 核心能力

### 1. 文件分析
- Autopsy
- 文件恢復
- Metadata 提取

### 2. 網路鑑識
- Wireshark
- 流量分析
- 封包重組

### 3. 記憶體取證
- Volatility
- Process 分析
- Malware 檢測

### 4. 磁碟鑑識
- Disk Imaging
- Bulk Extractor
- 文件雕刻

## 工具整合

- Autopsy
- Wireshark
- Bulk Extractor
- Guymager

## 使用範例

```python
from services.features.function_forensic import ForensicManager

manager = ForensicManager()

# 創建鑑識案件
case = await manager.create_case(
    case_name="Investigation_2025",
    evidence_path="/evidence/disk.img"
)

# 執行分析
result = await manager.analyze_evidence(
    case_id=case.case_id,
    analysis_type="disk_forensics"
)
```
