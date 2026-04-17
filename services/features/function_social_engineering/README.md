# function_social_engineering - 社交工程模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，實作不完整 | **語言**: Python

## 模組概述

社交工程測試框架，提供釣魚攻擊活動、憑證收集、OSINT 情報蒐集等功能的管理介面。

> ⚠️ **重要警告**：
> - 此模組為框架結構，多數功能尚未完整實作
> - 此為高度敏感的操作，且牽涉複雜的人工互動，目前歸類為「需人工操作」的模組。
> - 僅限獲得明確書面授權的紅隊演練使用。

### 功能清單

| 功能 | 說明 |
|------|------|
| 框架結構 | Manager + Models 架構 |
| 釣魚活動管理 | 啟動與停止釣魚活動 (框架) |
| 憑證收集 | 收集受害者憑證 (框架) |
| OSINT 蒐集 | 開源情報蒐集 (框架) |
| 活動分析 | 匯總活動數據 (框架) |

## 架構設計

```
function_social_engineering/
├── __init__.py   # 模組入口匯出
├── manager.py    # 主入口 (SocialEngineeringManager)
└── models.py     # 資料模型 (CampaignConfig, OSINTResult 等)
```

## 執行方式

### 作為 Python 模組匯入

模組目前僅為 API 框架，可直接實例化 Manager 呼叫。由於缺乏實際實作，多數方法僅回傳 Dummy 資料或未完成狀態。

```python
from services.features.function_social_engineering import SocialEngineeringManager

manager = SocialEngineeringManager()
# 取得情報
osint_results = manager.collect_osint("example.com")
```

## 可調用方法（內部 API）

| 類別 / 方法 | 說明 |
|------|------|
| `SocialEngineeringManager.launch_phishing_campaign(config)` | 啟動釣魚活動 |
| `SocialEngineeringManager.start_credential_harvester(platform, delivery_method, port, custom_template)` | 啟動憑證收集器 |
| `SocialEngineeringManager.collect_osint(target, search_engines, social_media)` | OSINT 情報蒐集 |
| `SocialEngineeringManager.get_campaign_analytics(campaign_id)` | 取得活動分析 |
| `SocialEngineeringManager.get_harvested_credentials(campaign_id)` | 取得收集的憑證 |
| `SocialEngineeringManager.stop_campaign(campaign_id)` | 停止活動 |

## 注意事項

- 此模組需高度人工介入，不適合完全自動化的掃描流程。
- 實際應用前需要接通外部服務（如 GoPhish、OSINT API 等）才能真正運作。
