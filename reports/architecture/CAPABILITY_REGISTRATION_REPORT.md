# HackingTool 整合模組能力註冊報告

## 📑 目錄

- [📊 註冊概覽](#註冊概覽)
- [📦 模組詳情](#模組詳情)
  - [function_wireless_attack](#functionwirelessattack)
    - [`scan_networks`](#scannetworks)
    - [`crack_wpa2`](#crackwpa2)
    - [`evil_twin_attack`](#eviltwinattack)
    - [`deauth_attack`](#deauthattack)
  - [function_payload_generator](#functionpayloadgenerator)
    - [`generate_msfvenom_payload`](#generatemsfvenompayload)
    - [`generate_reverse_shell`](#generatereverseshell)
    - [`generate_webshell`](#generatewebshell)
    - [`generate_poc`](#generatepoc)
  - [function_social_engineering](#functionsocialengineering)
    - [`create_phishing_campaign`](#createphishingcampaign)
    - [`clone_website`](#clonewebsite)
    - [`generate_credential_harvester`](#generatecredentialharvester)
  - [function_wordlist_generator](#functionwordlistgenerator)
    - [`generate_wordlist`](#generatewordlist)
    - [`generate_date_based`](#generatedatebased)
    - [`combine_wordlists`](#combinewordlists)
  - [function_forensic](#functionforensic)
    - [`extract_metadata`](#extractmetadata)
    - [`recover_deleted_files`](#recoverdeletedfiles)
    - [`analyze_memory_dump`](#analyzememorydump)
  - [function_steganography](#functionsteganography)
    - [`hide_message`](#hidemessage)
    - [`extract_message`](#extractmessage)
    - [`detect_steganography`](#detectsteganography)
  - [function_exploit_framework](#functionexploitframework)
    - [`search_exploits`](#searchexploits)
    - [`execute_exploit`](#executeexploit)
    - [`generate_shellcode`](#generateshellcode)
  - [function_reverse_engineering](#functionreverseengineering)
    - [`disassemble_binary`](#disassemblebinary)
    - [`decompile_code`](#decompilecode)
    - [`analyze_strings`](#analyzestrings)
- [✅ 下一步](#下一步)

---


**生成時間**: 1143556.3572281

## 📊 註冊概覽

- **模組總數**: 8
- **能力總數**: 26

## 📦 模組詳情

### function_wireless_attack

**管理器類**: `WirelessAttackManager`

**能力列表** (4 個):

#### `scan_networks`

- **描述**: 掃描 WiFi 網路
- **參數**: interface, channel, timeout
- **風險等級**: L2
- **類別**: wireless_attack

#### `crack_wpa2`

- **描述**: 破解 WPA2 加密
- **參數**: target, wordlist, interface
- **風險等級**: L2
- **類別**: wireless_attack

#### `evil_twin_attack`

- **描述**: Evil Twin 攻擊
- **參數**: target, interface, landing_page
- **風險等級**: L2
- **類別**: wireless_attack

#### `deauth_attack`

- **描述**: Deauthentication 攻擊
- **參數**: target, client_mac, interface, count
- **風險等級**: L2
- **類別**: wireless_attack


### function_payload_generator

**管理器類**: `PayloadGeneratorManager`

**能力列表** (4 個):

#### `generate_msfvenom_payload`

- **描述**: 生成 MSFVenom Payload
- **參數**: payload_type, lhost, lport, format
- **風險等級**: L2
- **類別**: payload_generation

#### `generate_reverse_shell`

- **描述**: 生成 Reverse Shell
- **參數**: language, lhost, lport, obfuscate
- **風險等級**: L2
- **類別**: payload_generation

#### `generate_webshell`

- **描述**: 生成 Web Shell
- **參數**: type, password, obfuscate
- **風險等級**: L2
- **類別**: payload_generation

#### `generate_poc`

- **描述**: 生成 PoC 腳本
- **參數**: vulnerability_type, target_url, cve_id
- **風險等級**: L2
- **類別**: payload_generation


### function_social_engineering

**管理器類**: `SocialEngineeringManager`

**能力列表** (3 個):

#### `create_phishing_campaign`

- **描述**: 創建釣魚攻擊活動
- **參數**: target_email, template, landing_page
- **風險等級**: L2
- **類別**: social_engineering

#### `clone_website`

- **描述**: 克隆目標網站
- **參數**: target_url, output_dir
- **風險等級**: L2
- **類別**: social_engineering

#### `generate_credential_harvester`

- **描述**: 生成憑證收集器
- **參數**: target_domain, landing_page
- **風險等級**: L2
- **類別**: social_engineering


### function_wordlist_generator

**管理器類**: `WordlistGeneratorManager`

**能力列表** (3 個):

#### `generate_wordlist`

- **描述**: 生成自定義字典
- **參數**: base_words, min_length, max_length
- **風險等級**: L1
- **類別**: utility

#### `generate_date_based`

- **描述**: 生成基於日期的密碼字典
- **參數**: start_year, end_year, formats
- **風險等級**: L1
- **類別**: utility

#### `combine_wordlists`

- **描述**: 合併多個字典
- **參數**: wordlist_paths, output_path
- **風險等級**: L0
- **類別**: utility


### function_forensic

**管理器類**: `ForensicToolsManager`

**能力列表** (3 個):

#### `extract_metadata`

- **描述**: 提取文件元數據
- **參數**: file_path
- **風險等級**: L0
- **類別**: forensic

#### `recover_deleted_files`

- **描述**: 恢復已刪除文件
- **參數**: drive_path, output_dir
- **風險等級**: L1
- **類別**: forensic

#### `analyze_memory_dump`

- **描述**: 分析內存轉儲
- **參數**: dump_path, profile
- **風險等級**: L1
- **類別**: forensic


### function_steganography

**管理器類**: `SteganographyManager`

**能力列表** (3 個):

#### `hide_message`

- **描述**: 在圖像中隱藏訊息
- **參數**: image_path, message, output_path
- **風險等級**: L0
- **類別**: steganography

#### `extract_message`

- **描述**: 從圖像中提取隱藏訊息
- **參數**: image_path
- **風險等級**: L0
- **類別**: steganography

#### `detect_steganography`

- **描述**: 檢測文件中的隱寫術
- **參數**: file_path
- **風險等級**: L0
- **類別**: steganography


### function_exploit_framework

**管理器類**: `ExploitFrameworkManager`

**能力列表** (3 個):

#### `search_exploits`

- **描述**: 搜索可用的漏洞利用
- **參數**: keyword, platform, type
- **風險等級**: L1
- **類別**: exploitation

#### `execute_exploit`

- **描述**: 執行漏洞利用
- **參數**: exploit_id, target, payload
- **風險等級**: L3
- **類別**: exploitation

#### `generate_shellcode`

- **描述**: 生成 Shellcode
- **參數**: architecture, payload_type
- **風險等級**: L2
- **類別**: exploitation


### function_reverse_engineering

**管理器類**: `ReverseEngineeringManager`

**能力列表** (3 個):

#### `disassemble_binary`

- **描述**: 反彙編二進制文件
- **參數**: binary_path, architecture
- **風險等級**: L0
- **類別**: reverse_engineering

#### `decompile_code`

- **描述**: 反編譯代碼
- **參數**: binary_path, language
- **風險等級**: L0
- **類別**: reverse_engineering

#### `analyze_strings`

- **描述**: 分析二進制文件字串
- **參數**: binary_path, min_length
- **風險等級**: L0
- **類別**: reverse_engineering


## ✅ 下一步

1. 執行內閉環探索: `python run_capability_analysis.py`
2. 驗證能力檢測: 應該新增約 30+ 個能力
3. 檢查 analysis_results/ 中的 JSON 報告

