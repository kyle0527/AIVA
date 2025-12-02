# 🔬 數位鑑識模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   數位鑑識分析架構                           │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │Forensic Engine │  Evidence   │
│  Interface    │             │               │   Storage    │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  forensic_mgr  │  evidence   │
│.FORENSIC_ANAL │             │  ─────────────  │  repository │
│      │        │             │   - disk_img   │      │      │
│      └────────┼─────────────┼─  - memory     │      ↓      │
│               │             │   - network    │  Chain of   │
│               ↓             │       ↓        │  Custody    │
│         TaskResult          │   Analysis     │             │
│         (aiva_common)       │   Timeline     │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **證據收集** - 安全獲取數位證據
2. **映像分析** - 磁碟和記憶體映像檢查
3. **時間軸重建** - 重建事件發生時序
4. **報告生成** - 產生法庭可用的鑑識報告

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.FORENSIC_ANALYSIS,
    payload={
        "evidence_sources": ["/dev/sda1", "/path/to/memory.dump"],
        "analysis_type": "comprehensive",  # basic|comprehensive
        "timeline_analysis": True,
        "hash_verification": True,
        "output_format": "legal_report"
    }
)
```

## 🔧 核心能力
- **多源證據**: 磁碟、記憶體、網路封包分析
- **完整性保證**: 雜湊驗證和證據鏈追蹤
- **時間軸重建**: 事件序列分析
- **法庭標準**: 符合法庭證據要求的報告

## ⚖️ 法律合規
- 證據鏈 (Chain of Custody) 維護
- 國際鑑識標準 (ISO 27037) 遵循
- 法庭可採納的證據格式

## 🎯 後續發展
- [ ] **雲端鑑識** - 雲服務和容器環境鑑識
- [ ] **AI 輔助分析** - 異常行為自動識別
- [ ] **行動裝置** - iOS/Android 鑑識支援