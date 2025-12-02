# 🔍 逆向工程模組

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                  逆向工程分析架構                            │
├─────────────────────────────────────────────────────────────┤
│  AI Command   │  handler.py  │ Reverse Engine │  Tool       │
│  Interface    │             │               │  Integration │
│      ↓        │      ↓      │       ↓        │     ↓       │
│  CommandType  │ TaskPayload │  reverse_mgr   │  ghidra     │
│ .REVERSE_ANAL │             │  ─────────────  │  jadx       │
│      │        │             │   - disasm     │  androguard │
│      └────────┼─────────────┼─  - decompile  │      │      │
│               │             │   - analysis   │      ↓      │
│               ↓             │       ↓        │  Binary     │
│         TaskResult          │   Pattern      │  Database   │
│         (aiva_common)       │   Recognition  │             │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **檔案類型檢測** - 識別二進位檔案格式
2. **靜態分析** - 反組譯和反編譯
3. **行為分析** - 識別可疑功能和模式
4. **威脅評估** - 產生安全風險報告

## 🚀 支援指令

```python
command = AICommand(
    command_type=CommandType.REVERSE_ANALYSIS,
    payload={
        "target_file": "/path/to/binary",
        "analysis_type": "comprehensive",  # basic|comprehensive
        "file_format": "auto_detect",  # elf|pe|apk|auto_detect
        "output_format": "report"  # report|json|sarif
    }
)
```

## 🔧 核心能力
- **多格式支援**: ELF, PE, APK, DEX, JAR
- **反組譯引擎**: 整合 Ghidra, Radare2, IDA
- **惡意程式檢測**: 行為模式和簽名識別
- **Android 分析**: APK 逆向和漏洞檢測

## 🎯 後續發展
- [ ] **AI 輔助分析** - 機器學習提升識別精度
- [ ] **雲端沙箱** - 安全的動態分析環境
- [ ] **IoT 韌體** - 嵌入式設備分析支援