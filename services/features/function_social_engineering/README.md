# function_social_engineering - 社交工程模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，實作不完整 | **語言**: Python | **能力登錄**: ⬜ 待登錄

## 模組概述

社交工程測試框架，提供釣魚攻擊活動、憑證收集、OSINT 情報蒐集等功能的管理介面。

> ⚠️ **重要警告**：
> - 此模組為框架結構，多數功能尚未完整實作
> - 僅限獲得明確書面授權的紅隊演練使用
> - `legacy/` 目錄內的舊版程式碼已廢棄，**請勿使用**

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 框架結構 | ✅ 完成 | Manager + Models 架構 |
| 釣魚活動管理 | ⬜ 框架 | launch_phishing_campaign() 存在但實作不完整 |
| 憑證收集 | ⬜ 框架 | start_credential_harvester() 存在但實作不完整 |
| OSINT 蒐集 | ⬜ 框架 | collect_osint() 存在但實作不完整 |
| 活動分析 | ⬜ 框架 | get_campaign_analytics() 存在但實作不完整 |

## 架構

```
function_social_engineering/
├── manager.py    # 主入口（SocialEngineeringManager）
├── models.py     # 資料模型
└── legacy/       # ⛔ 廢棄，勿使用
    └── phising_attack_original.py
```

## 可調用方法（公開 API）

| 方法 | 說明 | 實作狀態 |
|------|------|---------|
| `launch_phishing_campaign(config)` | 啟動釣魚活動 | ⬜ 框架 |
| `start_credential_harvester(platform, delivery_method, port, custom_template)` | 啟動憑證收集器 | ⬜ 框架 |
| `collect_osint(target, search_engines, social_media)` | OSINT 情報蒐集 | ⬜ 框架 |
| `get_campaign_analytics(campaign_id)` | 取得活動分析 | ⬜ 框架 |
| `get_harvested_credentials(campaign_id)` | 取得收集的憑證 | ⬜ 框架 |
| `stop_campaign(campaign_id)` | 停止活動 | ⬜ 框架 |

## 待評估工作

- 評估此模組的實際需求與邊界
- 決定是否繼續開發或廢棄
- 若繼續：完成各方法的核心實作，刪除 `legacy/` 目錄

## 注意事項

- **嚴格限制**：僅限持有書面授權的紅隊演練，未授權使用屬違法行為
- 此模組目前多數為框架結構，實際效果有限
