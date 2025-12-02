# 📝 密碼字典生成模組

**什麼是密碼字典生成？**  
密碼字典生成是針對特定目標創建定制化密碼列表的技術。基於目標的個人資訊、公司資料、常見密碼模式等生成高命中率的密碼字典，用於合法的安全測試和密碼強度評估。

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   智能密碼字典生成架構                         │
├─────────────────────────────────────────────────────────────┤
│ AI Command      │ handler.py     │ GenerationStrategy │ 輸出管理│
│ Interface       │               │                   │        │
│       ↓         │       ↓       │        ↓          │    ↓   │
│ FEATURE_        │ GeneratorConfig│ COMBINATION       │ 文件   │
│ WORDLIST_       │               │ CUPP_PERSONAL     │ 格式   │
│ GENERATE        │       ↓       │ COMPANY_BASED     │ .txt   │
│       │         │ WordlistManager│ PATTERN_BASED     │ .csv   │
│       └─────────┼───────────────┼─ HYBRID          │ .json  │
│                 │               │        ↓          │    ↓   │
│                 ↓               │ 智能組合 +        │ 統計   │
│         GeneratorResult         │ 去重優化          │ 分析   │
│         (models.py)             │                   │        │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ 運作流程
1. **策略選擇** - 根據可用資訊選擇生成策略
2. **數據準備** - 收集目標相關資訊（姓名、生日、公司等）
3. **智能生成** - 執行選定的生成策略：
   - **CUPP 個人化**: 基於個人資訊的密碼預測
   - **組合生成**: 字符集規則組合
   - **公司導向**: 企業環境密碼模式
   - **模式學習**: 基於現有密碼的模式分析
4. **優化輸出** - 去重、排序、統計分析

## 🚀 支援指令

### 實際使用方式
```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# 密碼字典生成命令
command = AICommand(
    command_id="wordlist_gen_001",
    command_type=CommandType.FEATURE_WORDLIST_GENERATE,
    target_module="features.wordlist_generator",
    payload={
        "strategy": "CUPP_PERSONAL",
        "target_info": {
            "first_name": "john",
            "last_name": "doe",
            "birthday": "1990-05-15",
            "company": "acme_corp",
            "spouse": "jane",
            "pets": ["max", "bella"],
            "hobbies": ["gaming", "soccer"]
        },
        "output_file": "/path/to/custom_wordlist.txt",
        "complexity": "medium"  # basic|medium|advanced
    }
)

# 執行生成
result = await command_center.execute(command)
```

### 何時使用？
- ✅ **適用場景**:
  - **滲透測試**: 針對特定目標的合法密碼測試
  - **安全評估**: 企業內部密碼強度評估
  - **紅隊演練**: 模擬攻擊中的密碼暴力破解
  - **教育訓練**: 安全意識培訓的實際演示
  
- ⚠️ **合法使用**:
  - 僅限授權的安全測試環境
  - 不得用於非法密碼破解
  - 遵守當地法律法規
  - 測試後及時刪除生成的字典

### 如何使用？
```python
# 1. CUPP 個人化生成
personal_wordlist = {
    "strategy": "CUPP_PERSONAL",
    "target_info": {
        "first_name": "alice",
        "last_name": "smith",
        "birthday": "1985-12-03",
        "company": "techcorp",
        "significant_years": ["2010", "2015", "2020"],
        "favorite_numbers": ["7", "13", "42"]
    }
}

# 2. 組合規則生成
combination_wordlist = {
    "strategy": "COMBINATION",
    "charset_config": {
        "base_words": ["password", "admin", "test"],
        "numbers": "0123456789",
        "special_chars": "!@#$%",
        "min_length": 8,
        "max_length": 12
    }
}

# 3. 公司環境導向
company_wordlist = {
    "strategy": "COMPANY_BASED",
    "company_info": {
        "name": "TechStartup Inc",
        "founded_year": "2018",
        "industry": "software",
        "locations": ["singapore", "tokyo"],
        "products": ["cloudapp", "mobileapp"]
    }
}

# 4. 混合策略（推薦）
hybrid_wordlist = {
    "strategy": "HYBRID",
    "target_info": {
        "first_name": "bob",
        "company": "megacorp",
        "birthday": "1992-07-20"
    },
    "additional_sources": [
        "/path/to/common_passwords.txt",
        "/path/to/leaked_passwords.txt"
    ],
    "pattern_analysis": True,
    "output_limit": 100000  # 限制輸出數量
}
```

## 🔧 核心能力
- **CUPP 個人化引擎**: 基於心理學的密碼預測算法
- **智能組合生成**: 字符集排列組合與規則應用
- **模式學習**: 從現有密碼中學習常見模式
- **去重優化**: 高效的重複項移除和排序
- **統計分析**: 生成密碼的複雜度和分布統計
- **多格式輸出**: 支援 TXT、CSV、JSON 等格式

## 🎯 後續發展方向
- [ ] **機器學習增強** - 基於大數據的密碼模式學習
- [ ] **多語言支援** - 非英語環境的密碼生成
- [ ] **社交媒體整合** - 從公開社交資料生成密碼
- [ ] **實時統計** - 密碼強度即時評估和優化建議
