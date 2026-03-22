# function_forensic - 數位鑑識模組

> **版本**: v1.0.0 | **狀態**: ✅ 核心完成，⬜ CLI 入口待接通 | **語言**: Python | **能力登錄**: ⬜ 待登錄

## 模組概述

數位鑑識模組，提供案件管理、證據採集、磁碟映像分析、記憶體傾印分析與時間線生成能力。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 案件管理 | ✅ 完成 | 建立案件、管理調查人員 |
| 證據採集 | ✅ 完成 | 磁碟/記憶體/檔案/日誌，含雜湊驗證 |
| 磁碟映像分析 | ✅ 完成 | 深度掃描模式 |
| 記憶體傾印分析 | ✅ 完成 | 記憶體取證分析 |
| 時間線生成 | ✅ 完成 | 事件時間序列 |
| CLI 入口接通 | ⬜ 待完成 | aiva_external_executor 尚未對應 |

> ⚠️ `legacy/` 目錄內的舊版程式碼（`forensic_tools_original.py`）已廢棄，**請勿使用**。

## 架構

```
function_forensic/
├── manager.py     # 全部實作（ForensicManager）
├── models.py      # 資料模型（CaseInfo, EvidenceItem 等）
└── legacy/        # ⛔ 廢棄，勿使用
    └── forensic_tools_original.py
```

## 執行方式

### 直接使用

```python
from services.features.function_forensic.manager import ForensicManager

manager = ForensicManager()

# 建立案件
case = manager.create_case(
    case_name="Investigation-001",
    investigator="analyst",
    description="Incident response"
)

# 採集證據
evidence = await manager.acquire_evidence(
    case_id=case.id,
    source_path="/dev/sda",
    evidence_type="DISK",
    acquired_by="analyst"
)

# 分析磁碟映像
await manager.analyze_disk_image(evidence.id, deep_scan=True)

# 分析記憶體傾印
await manager.analyze_memory_dump(evidence.id)

# 生成時間線
timeline = await manager.generate_timeline(evidence.id)
```

## 可調用方法（公開 API）

| 方法 | 說明 |
|------|------|
| `create_case(case_name, investigator, description)` | 建立鑑識案件 |
| `acquire_evidence(case_id, source_path, evidence_type, acquired_by)` | 採集證據（含 MD5/SHA256 驗證） |
| `analyze_disk_image(evidence_id, deep_scan)` | 磁碟映像分析 |
| `analyze_memory_dump(evidence_id)` | 記憶體傾印分析 |
| `generate_timeline(evidence_id)` | 生成事件時間線 |

## 證據類型

| 類型 | 說明 |
|------|------|
| `DISK` | 磁碟映像 |
| `MEMORY` | 記憶體傾印 |
| `FILE` | 個別檔案 |
| `LOG` | 日誌檔案 |

## 待完成工作

- 接通 `aiva_external_executor.py` 的 CLI 入口
- 將 `memory_analysis` / `disk_image` / `timeline_analysis` 補全至 `CAPABILITY_CONFIGS`
- 刪除 `legacy/` 目錄

## 注意事項

- 僅限授權數位鑑識與事故回應使用
- 採集的所有證據自動計算雜湊值（MD5 + SHA256）以確保完整性
