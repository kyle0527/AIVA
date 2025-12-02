# 🖼️ 隱寫術分析模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   隱寫術分析架構                            │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │  Stego Engine  │  Detection  │
│  Interface    │             │               │   Algorithms │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  stego_detect  │  lsb_detect │
│.STEGO_ANALYSIS│             │  ─────────────  │  dct_detect │
│      │        │             │   - image      │  freq_anal  │
│      └────────┼─────────────┼─  - audio      │      │      │
│               │             │   - text       │      ↓      │
│               ↓             │       ↓        │  Statistical│
│         FindingPayload      │   Hidden       │  Analysis   │
│         (aiva_common)       │   Data Extract │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **媒體檔案分析** - 檢測圖片、音頻、視頻檔案
2. **隱寫檢測** - 使用多種算法檢測隱藏資訊
3. **資料提取** - 嘗試提取隱藏的資料
4. **結果分析** - 分析提取資料的類型和內容

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.STEGO_ANALYSIS,
    payload={
        "media_files": ["/path/to/image.png", "/path/to/audio.wav"],
        "detection_methods": ["lsb", "dct", "frequency"],
        "extraction_attempts": True,
        "password_list": "/path/to/passwords.txt"  # 可選
    }
)
```

## 🔧 核心能力
- **多媒體支援**: 圖片(PNG/JPEG)、音頻(WAV/MP3)、視頻
- **檢測算法**: LSB, DCT, 頻率分析, 統計分析
- **資料提取**: 自動嘗試提取隱藏資料
- **密碼破解**: 支援密碼保護的隱寫資料

## 🎯 後續發展
- [ ] **深度學習檢測** - 使用 CNN 提升檢測準確性
- [ ] **區塊鏈隱寫** - NFT 和區塊鏈隱藏資訊檢測
- [ ] **即時檢測** - 串流媒體隱寫分析