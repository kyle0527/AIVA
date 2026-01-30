# 高階自動化 Web 漏洞掃描邏輯與判斷架構深度研究報告

## 1. 緒論：AI 驅動的自動化安全檢測新典範

隨著 Web 應用程式架構從傳統的單體式（Monolithic）向微服務（Microservices）與雲原生（Cloud-Native）架構演進，安全漏洞的型態與檢測難度也隨之發生了根本性的轉變。傳統的靜態應用程式安全測試（SAST）與動態應用程式安全測試（DAST）工具，往往依賴於固定的特徵碼（Signatures）與正則表達式（Regex）進行模式匹配。然而，這種基於規則（Rule-based）的方法在面對現代複雜的 Web 應用時，面臨著極高的誤報率（False Positives）與漏報率（False Negatives）。特別是在處理上下文依賴極強的漏洞——如 SQL 注入（SQLi）、跨站腳本攻擊（XSS）、伺服器端請求偽造（SSRF）以及不安全的直接物件參照（IDOR）時，傳統掃描器缺乏對業務邏輯與應用狀態的深層理解。

本研究報告旨在為下一代 AI 驅動的自動化漏洞掃描器構建一套核心判斷邏輯資料庫。透過深度分析四大核心漏洞的觸發機制、響應特徵與邊界條件，我們提取出高置信度的判斷指標。這不僅包含傳統的報錯字串識別，更深入至資料庫底層的解析邏輯、瀏覽器的渲染狀態機、雲端基礎設施的元數據協議，以及業務邏輯層的權限控制模型。目標是產出標準化的 JSON 邏輯模組（function_sqli, function_xss, function_ssrf, function_idor），使 AI 代理（Agent）能夠在掃描過程中進行類似人類專家的決策推理，從而實現精確的漏洞驗證與利用鏈（Exploit Chain）構建。

## 2. SQL Injection (SQLi)：資料庫交互層的深度解析與自動化判定

SQL 注入漏洞依然是 Web 安全中最具破壞性的威脅之一。其本質在於應用程式未能將使用者輸入數據與 SQL 指令代碼進行嚴格隔離，導致攻擊者能夠操控後端資料庫的語法樹（Syntax Tree）。對於自動化掃描器而言，檢測 SQLi 的核心挑戰在於如何從 HTTP 響應中準確區分「無效輸入引發的應用層錯誤」與「成功注入引發的資料庫層錯誤」，以及在無回顯（Blind）場景下如何建立統計學上可靠的時間延遲判定模型。

### 2.1 錯誤型注入（Error-Based）的指紋識別機制

錯誤型注入是自動化掃描中效率最高的檢測方式。當惡意構造的 SQL 片段破壞了原有查詢的語法結構，資料庫管理系統（DBMS）會拋出特定的錯誤訊息。這些訊息不僅證實了注入點的存在，更洩漏了後端資料庫的類型與版本，為後續的攻擊載荷選擇提供關鍵依據。AI 掃描器必須維護一個詳盡的 DBMS 錯誤指紋庫，並理解這些錯誤產生的底層原因。

#### 2.1.1 MySQL 與 MariaDB 的解析器錯誤特徵

MySQL 的錯誤處理機制傾向於將語法解析失敗的具體位置回顯給客戶端。這類錯誤通常源於詞法分析器（Lexer）無法識別注入的特殊字符（如單引號 ' 或註釋符 \#）。

- **語法錯誤指紋：** 最經典的特徵字串為 You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near <sup>1</sup>。這個字串的出現意味著攻擊者成功打破了 SQL 語句的結構，但未能正確閉合或修復後續語句。AI 掃描器應將此視為 100% 置信度的漏洞指標。

- **列數不匹配指紋：** 在使用 UNION 查詢進行數據提取時，若注入的列數與原查詢不一致，MySQL 會報錯 The used SELECT statements have a different number of columns。這對於自動化枚舉列數至關重要。

- **其他特徵：** 包括 Illegal mix of collations（字符集衝突）和 BIGINT value is out of range（數值溢出），這些通常發生在 Boolean-based 盲注的邊緣測試中 <sup>1</sup>。

#### 2.1.2 PostgreSQL 的嚴格類型錯誤特徵

PostgreSQL 以其嚴謹的類型系統著稱。與 MySQL 的隱式類型轉換不同，PostgreSQL 在面對類型不匹配時會直接拋出異常，這為自動化檢測提供了極佳的切入點。

- **語法與引號錯誤：** ERROR: syntax error at or near 是最常見的通用錯誤。更具特異性的是 ERROR: unterminated quoted string at or near，這直接表明注入的單引號未被正確轉義或閉合，是 SQLi 的鐵證 <sup>1</sup>。

- **類型轉換錯誤：** 嘗試將字串注入整數欄位時，會觸發 ERROR: invalid input syntax for type integer。

- **函數執行錯誤：** PostgreSQL 的報錯通常帶有明確的上下文前綴 ERROR:，這使得正則表達式匹配非常高效。AI 模型應專注於提取 at or near "..." 後的內容，以分析注入點的具體語法環境。

#### 2.1.3 Microsoft SQL Server (MSSQL) 的驅動層錯誤特徵

MSSQL 的錯誤訊息經常暴露出底層數據訪問組件（如 ODBC 或 OLE DB）的資訊。利用 convert() 或 cast() 函數進行強制類型轉換錯誤（Error-Based Information Retrieval）是針對 MSSQL 的主要檢測手段。

- **閉合錯誤：** Unclosed quotation mark after the character string 和 Incorrect syntax near 是最基礎的指標 <sup>1</sup>。

- **類型轉換注入：** 攻擊者常使用 convert(int, @@version) 來觸發錯誤。由於 @@version 返回的是字串，無法轉換為整數，MSSQL 會將版本訊息包含在錯誤訊息中回顯：Conversion failed when converting the varchar value 'Microsoft SQL Server...' to data type int <sup>1</sup>。

- **系統函數錯誤：** 諸如 The conversion of the varchar value to data type int resulted in an out-of-range value 也是常見特徵。

#### 2.1.4 Oracle Database 的 ORA 代碼體系

Oracle 的錯誤訊息具有最標準化的格式，均以 ORA- 代碼開頭，這使得自動化識別極為精確。

- **引號與語法：** ORA-01756: quoted string not properly terminated 是注入測試中最常遇到的錯誤，表明單引號注入成功 <sup>1</sup>。

- **關鍵字缺失：** 由於 Oracle 強制要求 SELECT 語句必須包含 FROM 子句（通常是 FROM dual），注入 UNION SELECT 1,2 而遺漏 FROM dual 會觸發 ORA-00923: FROM keyword not found where expected <sup>1</sup>。這是區分 Oracle 與 MySQL/MSSQL 的重要邏輯分支。

- **命令結束錯誤：** ORA-00933: SQL command not properly ended 則常見於註釋符使用不當的情況。

### 2.2 盲注（Blind SQLi）與時間延遲判定模型

在應用程式配置了自定義錯誤頁面（Custom Error Pages）或完全抑制錯誤輸出時，掃描器必須轉向盲注檢測。其中，時間盲注（Time-Based Blind SQLi）是最後的手段，但也最容易受到網路波動（Network Jitter）的影響而產生誤報。AI 掃描器需建立嚴格的統計模型來處理這些信號。

#### 2.2.1 資料庫專屬延遲函數矩陣

不同資料庫實現延遲的機制各異，掃描器需根據指紋識別結果選擇正確的 Payload。

| **資料庫管理系統 (DBMS)** | **延遲技術與函數 Payload** | **機制解析** | **引用來源** |
|----|----|----|----|
| **MySQL / MariaDB** | SLEEP(N) | SLEEP() 函數會讓當前執行緒暫停 N 秒。Payload 如 ' AND SLEEP(10)--。 | <sup>3</sup> |
| **MySQL (替代方案)** | BENCHMARK(N, MD5(1)) | 透過執行 N 次 MD5 雜湊運算消耗 CPU 週期來製造延遲，適用於 SLEEP 被禁用時。 | <sup>5</sup> |
| **PostgreSQL** | pg_sleep(N) | 內建函數，接受秒數作為參數。Payload 如 ; SELECT pg_sleep(10)--。 | <sup>4</sup> |
| **MSSQL** | WAITFOR DELAY '00:00:N' | T-SQL 專用指令，指定暫停的時間長度。Payload 如 '; WAITFOR DELAY '0:0:10'--。 | <sup>4</sup> |
| **Oracle** | dbms_pipe.receive_message(('a'),N) | 利用管道接收訊息的超時機制。這是 Oracle 最可靠的延遲方法，因為它不依賴 PL/SQL 塊權限。 | <sup>4</sup> |
| **Oracle (替代方案)** | dbms_lock.sleep(N) | 需管理員權限，通常僅在擁有高權限時有效。 | <sup>7</sup> |

#### 2.2.2 統計學判斷邏輯與誤報消除

為了區分真實的執行延遲與網路擁塞，AI 掃描器應採用「差異化延遲測試」（Differential Timing Analysis）：

1.  **基準測試 (Baseline):** 發送正常請求 5 次，計算平均響應時間 \$\mu\$ 和標準差 \$\sigma\$。

2.  **觸發測試 (Injection):** 發送帶有 SLEEP(10) 的 Payload。若響應時間 \$T \> \mu + 10s - \delta\$（\$\delta\$ 為容錯值），則進入驗證階段。

3.  **邏輯驗證 (Logic Verification):** 這是消除誤報的關鍵。發送一個邏輯為「假」的延遲請求，例如 ' AND IF(1=2, SLEEP(10), 0)--。

    - **結果判定：** 若此請求**立即返回**（無延遲），而邏輯為「真」的請求（IF(1=1, SLEEP(10), 0)）產生延遲，則可確認漏洞存在。若兩者皆延遲，則極可能是網路問題或 WAF 攔截導致的處理滯後。

### 2.3 Polyglots：通用攻擊載荷的構建

為了減少掃描請求的總量（Request Count），現代掃描器廣泛使用 Polyglots——即能夠同時在多種語境（單引號、雙引號、無引號）和多種資料庫中觸發異常的混合載荷。

- 引號逃逸 Polyglot:  
  攻擊者無法預知後端 SQL 是用單引號還是雙引號包裹輸入。因此，Payload 必須同時嘗試閉合兩者。  
  範例：SLEEP(5) /\*' or SLEEP(5) or '" or SLEEP(5) or "\*/  
  此 Payload 利用了 MySQL 的塊註釋 /\*... \*/ 特性，嘗試在不同引號環境下注入 SLEEP 指令。

- 通用註釋符:  
  MySQL 使用 \# 或 -- （注意空格），而 PostgreSQL/MSSQL/Oracle 均支持 --。  
  最佳實踐 Payload：1' OR '1'='1'/\*  
  在 MySQL 中，/\* 開始註釋直到結尾（或閉合），這可以替代 \#。在其他資料庫中，這可能需要配合 -- 使用。

### 2.4 SQLi 誤報指標與 WAF 識別

掃描器面臨的最大挑戰之一是 Web Application Firewall (WAF) 的攔截頁面被誤認為是應用程式的錯誤或盲注的延遲。

- **WAF 攔截特徵：**

  - **Cloudflare:** 響應中包含 Attention Required!, Cloudflare Ray ID, Error 1020 <sup>9</sup>。

  - **Akamai:** 響應頭或內容中包含 AkamaiGHost, Reference \# <sup>10</sup>。

  - **Imperva (Incapsula):** 包含 Request unsuccessful, Incapsula incident ID, Powered by Imperva <sup>11</sup>。

  - **通用行為：** 狀態碼為 403 或 406，且頁面結構與正常頁面完全不同（DOM 相似度低）。

- 偽 200 OK 錯誤：  
  許多現代應用在發生內部錯誤時仍返回 200 狀態碼，但內容顯示 "System Error" 或 "Processing Failed"。AI 掃描器需具備自然語言處理（NLP）能力或關鍵字庫，識別這類「軟錯誤」，避免將其誤判為注入成功（如在布林盲注中）。

### 2.5 JSON 模組輸出：function_sqli

> JSON

{  
"function_sqli": {  
"detection_logic": {  
"error_based": {  
"mysql":,  
"postgresql":,  
"mssql":,  
"oracle":  
},  
"time_based": {  
"payloads":,  
"threshold_calculation": "baseline_avg + payload_delay - network_jitter_allowance"  
},  
"polyglots":  
},  
"false_positive_indicators":  
}  
}

## 3. Cross-Site Scripting (XSS)：瀏覽器渲染層的動態驗證

XSS 漏洞的檢測場域在於瀏覽器端（Client-Side）。與 SQLi 不同，XSS 的成功取決於瀏覽器的 HTML 解析器、JavaScript 引擎以及 DOM 樹的構建過程。傳統的「發送 Payload -\> 檢查響應體是否包含 Payload」的反射型檢測邏輯（Reflection Check）已嚴重過時，因為現代瀏覽器的 XSS 過濾器（XSS Auditor，雖多已棄用但概念仍存）以及 CSP（Content Security Policy）使得單純的反射並不等同於代碼執行。

### 3.1 上下文感知（Context-Aware）的 Payload 構建

AI 掃描器必須具備解析 HTML 結構的能力，判斷注入點位於 DOM 的哪個具體位置（Context），因為這直接決定了所需的 Payload 結構。

#### 3.1.1 HTML Body 上下文

當使用者輸入直接回顯在 \<body\> 標籤的 PCDATA 區域時，攻擊者需要引入新的 HTML 標籤來觸發腳本執行。

- **基礎 Payload:** \<script\>alert(1)\</script\>。這是最基本的測試，但極易被 WAF 攔截。

- **Bypass Payload:** 現代檢測更傾向於使用事件處理器（Event Handlers）或不常見的標籤。

  - \<img src=x onerror=alert(1)\>：利用圖片加載失敗觸發 onerror 事件。

  - \<svg/onload=alert(1)\>：利用 SVG 向量圖的加載事件 <sup>12</sup>。

  - \<body oninput=alert(1)\>\<input autofocus\>：利用 autofocus 自動聚焦觸發 oninput 事件，無需使用者交互，非常適合自動化掃描。

#### 3.1.2 HTML 屬性上下文

當輸入位於標籤的屬性值中（如 \<input value="USER_INPUT"\>），攻擊者首先需要閉合當前屬性，然後注入新的事件屬性。

- **Payload 策略:** "\>\<script\>alert(1)\</script\> 或 " onmouseover=alert(1) autofocus="。

- **關鍵點:** 若注入點在 href 或 src 屬性中（如 \<a href="USER_INPUT"\>），則可利用偽協議（Pseudo-protocol）進行注入：javascript:alert(1)。AI 掃描器需特別注意 http:// 或 https:// 過濾的繞過，例如利用 HTML 實體編碼 javascript:（'j' 的編碼）<sup>13</sup>。

#### 3.1.3 JavaScript 上下文

最危險的場景是輸入直接位於 \<script\> 區塊內（如 var name = 'USER_INPUT';）。

- **Payload 策略:** 攻擊者需先閉合字串與語句。Payload 如 ';alert(1);//。

- **模板字面量:** 在 ES6 反引號 \` 中，可利用 \${alert(1)} 直接執行表達式，無需閉合引號 <sup>14</sup>。

### 3.2 無頭瀏覽器（Headless Browser）與 DOM XSS 自動化檢測

對於 DOM-Based XSS，後端響應中可能根本不包含 Payload，因為漏洞是由前端 JavaScript 在運行時動態修改 DOM 造成的（從 Source 到 Sink）。因此，靜態分析響應體是無效的。AI 掃描器必須集成無頭瀏覽器（如 Puppeteer 或 Selenium）來模擬真實渲染。

#### 3.2.1 動態驗證邏輯

掃描器通過 Puppeteer 訪問目標頁面，並監聽瀏覽器的特定事件來判斷 XSS 是否成功觸發。最通用的標準是監聽 dialog 事件（Alert/Confirm/Prompt 彈窗）。

> JavaScript

// Puppeteer 監聽邏輯範例  
page.on('dialog', async dialog =\> {  
const message = dialog.message();  
if (message === 'XSS_CONFIRMATION_ID') {  
console.log('XSS Vulnerability Confirmed');  
// 標記漏洞並記錄上下文  
}  
await dialog.dismiss(); // 必須關閉彈窗以免阻塞後續測試  
});

這種方法<sup>15</sup>能達到近乎零誤報的效果：只有當 JavaScript 真正執行並調用 alert() 時，事件才會被觸發。相比之下，正則匹配響應體的方法誤報率極高。

#### 3.2.2 對抗反自動化機制

許多網站會檢測訪問者是否為自動化工具（如檢查 navigator.webdriver 屬性）。為了確保掃描器不被攔截，必須在初始化階段進行特徵隱藏（Stealth Mode）。

- **特徵隱藏代碼:**  
  JavaScript  
  await page.evaluateOnNewDocument(() =\> {  
  Object.defineProperty(navigator, 'webdriver', {  
  get: () =\> undefined, // 隱藏 webdriver 屬性  
  });  
  });  
    
  此外，還需偽造 User-Agent，並確保 chrome.runtime 等物件在 Headless 模式下的表現與正常 Chrome 一致 <sup>17</sup>。

### 3.3 Polyglots 與過濾器繞過

為了應對複雜的過濾規則，AI 掃描器應使用強大的 XSS Polyglots。這些字串被設計為能同時逃逸多種上下文（HTML, Attribute, Script, URL）。

- 經典 Polyglot 範例:  
  javascript:/\*--\>\</title\>\</style\>\</textarea\>\</script\>\</xmp\>\<svg/onload='+/\\/+/onmouseover=1/+/\[\*//+alert(1)//'\> 12

  - **機制解析:**

    - javascript:：適配 URL 上下文。

    - \</title\>\</style\>...：嘗試閉合所有可能的 RAWTEXT 元素。

    - \<svg/onload=...\>：在 HTML 上下文中利用 SVG 事件。

    - //：在 JS 上下文中作為註釋符忽略後續垃圾字符。

### 3.4 誤報指標 (False Positive Indicators)

在使用靜態分析輔助動態檢測時，需排除以下情況：

- **MIME 類型不匹配:** Payload 雖回顯，但響應頭為 Content-Type: application/json 或 text/plain。除非瀏覽器存在 MIME Sniffing 漏洞（現代瀏覽器預設關閉），否則 XSS 無法觸發。

- **轉義回顯:** Payload 被轉義為 HTML 實體（如 \<script\>）。掃描器需解析 DOM，確認 Payload 被解析為 Text Node 而非 Element Node。

- **CSP 攔截:** 雖然 XSS 觸發，但被 Content Security Policy 攔截（瀏覽器控制台會報錯 refused to execute inline script）。這雖然仍是漏洞（CSP 可被繞過），但在利用評級上應有所區別。

### 3.5 JSON 模組輸出：function_xss

> JSON

{  
"function_xss": {  
"contexts": \[  
"html_body",  
"html_attribute",  
"javascript_variable",  
"dom_sink"  
\],  
"payloads": {  
"basic": "\<script\>alert('XSS')\</script\>",  
"attribute_breakout": "\\\>\<script\>alert('XSS')\</script\>",  
"javascript_breakout": "';alert('XSS');//",  
"template_literal": "\${alert('XSS')}",  
"svg_event": "\<svg/onload=alert('XSS')\>",  
"img_event": "\<img src=x onerror=alert('XSS')\>"  
},  
"polyglots": \[  
"javascript:/\*--\>\</title\>\</style\>\</textarea\>\</script\>\</xmp\>\<svg/onload='+/\\/+/onmouseover=1/+/\[\*//+alert(1)//'\>",  
"\\\>\<svg/onload=alert(1)\>",  
"'\>\<svg/onload=alert(1)\>",  
"jaVasCript:/\*-/\*\`/\*\\\`/\*'/\*\\/\*\*/(/\* \*/oNcliCk=alert() )//%0D%0A%0D%0A//\</stYle/\</titLe/\</teXtarEa/\</scRipt/--!\>\\x3csVg/\<sVg/oNloAd=alert()//\>\\x3e"  
\],  
"verification_logic": {  
"method": "headless_browser",  
"event_listener": "dialog",  
"success_condition": "alert_dialog_present",  
"required_action": "dialog.dismiss()"  
},  
"anti_bot_evasion": {  
"navigator_webdriver": "undefined",  
"user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."  
},  
"false_positive_indicators":  
}  
}

## 4. Server-Side Request Forgery (SSRF)：雲端基礎設施的穿透與利用

SSRF 漏洞允許攻擊者強制伺服器向任意目標發起請求。在雲端時代，SSRF 的危害被極大放大，因為攻擊者可以透過訪問雲端服務提供商（CSP）的實例元數據服務（Instance Metadata Service, IMDS）來竊取臨時憑證（IAM Credentials），進而接管整個雲端基礎設施。

### 4.1 雲端 Metadata 服務的指紋與利用路徑

自動化掃描器必須針對主流雲端平台（AWS, GCP, Azure, Oracle, Alibaba Cloud）內建精確的元數據探測邏輯。

#### 4.1.1 Amazon Web Services (AWS)

AWS 的元數據服務位於 169.254.169.254。

- **IMDSv1 (Legacy):** 這是最容易被利用的版本，僅需簡單的 GET 請求。

  - **利用 URL:** http://169.254.169.254/latest/meta-data/

  - **關鍵敏感路徑:** iam/security-credentials/。此目錄下會列出關聯的 IAM Role 名稱，進一步訪問該 Role 名稱路徑（如 .../security-credentials/MyRole）可獲取 AccessKeyId, SecretAccessKey 和 Token <sup>18</sup>。

- **IMDSv2 (Secure):** AWS 引入了基於 Session 的驗證機制，這對傳統 SSRF 掃描器構成了挑戰。利用需要兩個步驟：

  1.  **獲取 Token:** 發送 PUT 請求至 http://169.254.169.254/latest/api/token，並帶上 Header X-aws-ec2-metadata-token-ttl-seconds: 21600。

  2.  **使用 Token:** 在後續請求中帶上 Header X-aws-ec2-metadata-token: \<TOKEN\> <sup>20</sup>。

  - **自動化邏輯:** 掃描器應具備「多步請求鏈」的能力。若發現可以控制 HTTP Method（改為 PUT）和 Headers，則嘗試利用 IMDSv2。

#### 4.1.2 Google Cloud Platform (GCP)

GCP 的防禦機制依賴於自定義 Header。

- **目標地址:** 169.254.169.254 或 DNS 名稱 metadata.google.internal。

- **強制 Header:** 所有的 Metadata 請求必須包含 Metadata-Flavor: Google <sup>21</sup>。若缺乏此 Header，伺服器將拒絕請求，這有效地防禦了無法控制 Header 的簡單 SSRF。

- **高效掃描 URL:** http://metadata.google.internal/computeMetadata/v1/?recursive=true&alt=json。加上 recursive=true 參數可以一次性遞歸獲取所有元數據，這對於掃描器快速判定漏洞是否存在極為有效 <sup>23</sup>。

- **特徵驗證:** 響應 JSON 中應包含 project-id, numeric-project-id, service-accounts 等字段 <sup>24</sup>。

#### 4.1.3 Microsoft Azure

Azure 的元數據服務同樣有嚴格的 Header 要求。

- **目標地址:** 169.254.169.254。

- **強制 Header:** 請求必須包含 Metadata: true。此外，Azure 明確禁止包含 X-Forwarded-For Header 的請求訪問 IMDS，以防止通過代理的誤訪問 <sup>25</sup>。

- **URL:** http://169.254.169.254/metadata/instance?api-version=2021-02-01。注意 Azure API 需要指定 api-version。

- **特徵驗證:** 響應 JSON 包含 compute（計算實例資訊）和 network（網路配置），以及特徵值 "azEnvironment": "AzurePublicCloud" <sup>26</sup>。

### 4.2 繞過技術 (Bypass Techniques) 與過濾器對抗

開發者常使用黑名單（Blacklist）來過濾 169.254.169.254 或 localhost。AI 掃描器需自動嘗試多種編碼變形來繞過正則匹配。

- **IP 地址編碼變形:**

  - **十進制 (Decimal):** http://2852039166/ (即 169.254.169.254 的整數值) <sup>27</sup>。

  - **八進制 (Octal):** http://0251.0376.0251.0376/。

  - **十六進制 (Hex):** http://0xA9FEA9FE/。

  - **混合編碼:** http://169.254.0xa9.0xfe/。

  - **IPv6 映射:** http://\[::ffff:169.254.169.254\]/。

- DNS Rebinding:  
  攻擊者設定一個域名（如 rbnd.attacker.com），第一次解析返回合法 IP（繞過檢查），第二次解析返回 169.254.169.254（Time-of-Check to Time-of-Use 漏洞）。雖然自動化掃描難以實時搭建 DNS 服務，但可利用公共 Rebinding 服務（如 1u.ms）生成的域名進行測試 27。

- 重定向 (Redirection):  
  利用伺服器跟隨 30x 重定向的特性。掃描器發送一個指向自己控制的伺服器的 URL，該伺服器響應 302 Found 並重定向至 Metadata URL。

### 4.3 SSRF 誤報指標

- **自我遞歸 (Self-Recursion/Loop):** 若應用程式只是請求了自身的公開接口，造成無限循環，這通常是邏輯錯誤而非 SSRF 漏洞。

- **圖片/靜態資源代理:** 許多功能（如「加載遠程圖片」）本質上就是 SSRF，但若其嚴格限制了響應的 Content-Type 為圖片，且無法讀取文本（Metadata），則風險較低。

- **通用網路錯誤:** 伺服器返回 Connection refused (連線被拒) 或 Host not found，僅代表伺服器嘗試了連接，但並不意味著可以訪問內網敏感資源。

### 4.4 JSON 模組輸出：function_ssrf

> JSON

{  
"function_ssrf": {  
"cloud_metadata_targets":  
},  
{  
"provider": "GCP",  
"url": "http://metadata.google.internal/computeMetadata/v1/?recursive=true&alt=json",  
"required_headers": {"Metadata-Flavor": "Google"},  
"indicators": \["project-id", "service-accounts", "numeric-project-id"\]  
},  
{  
"provider": "Azure",  
"url": "http://169.254.169.254/metadata/instance?api-version=2021-02-01",  
"required_headers": {"Metadata": "true"},  
"indicators": \["azEnvironment", "AzurePublicCloud", "osProfile"\]  
},  
{  
"provider": "Oracle Cloud",  
"url": "http://169.254.169.254/opc/v1/instance/",  
"required_headers": {"Authorization": "Bearer Oracle"},  
"indicators":  
}  
\],  
"bypass_payloads": \[  
"http://2852039166/",  
"http://0xA9FEA9FE/",  
"http://0251.0376.0251.0376/",  
"http://\[::ffff:169.254.169.254\]/",  
"http://169.254.169.254.nip.io/"  
\],  
"false_positive_indicators":  
}  
}

## 5. Insecure Direct Object References (IDOR)：業務邏輯層的權限漏洞

IDOR 是一種邏輯漏洞，無法像 SQLi 或 XSS 那樣僅憑特定的 Payload 觸發報錯或執行代碼。IDOR 的本質是「授權檢查缺失」，即伺服器驗證了「你是誰」（Authentication），但未驗證「你是否有權訪問此資源」（Authorization）。因此，自動化檢測 IDOR 需要一種全新的策略——差異化分析（Differential Analysis）或成對測試（Pairwise Testing）。

### 5.1 雙帳號成對測試策略 (Pairwise Testing Strategy)

AI 掃描器在進行 IDOR 檢測時，必須配置兩個不同但權限級別相似的帳號：

- **使用者 A (Attacker):** 用於發起越權請求的帳號。

- **使用者 B (Victim):** 資源的合法擁有者。

**自動化判斷邏輯流程：**

1.  **資源枚舉 (Enumeration):** 掃描器在爬取使用者 B 的數據時，識別出 URL 路徑、查詢參數或 JSON Body 中的資源標識符（Identifiers）。例如：/api/users/1001/orders 中的 1001。

2.  **標識符替換 (Parameter Tampering):** 掃描器切換至使用者 A 的會話（Session Token），重放上述請求，但保持資源 ID 為使用者 B 的 ID (1001)。

3.  **響應對比 (Response Comparison):** 這是最關鍵的一步。

    - **基準響應:** 使用者 A 訪問 *自己* 的資源（ID 1002）的響應。

    - **越權響應:** 使用者 A 訪問 *使用者 B* 資源（ID 1001）的響應。

    - **判定:** 若「越權響應」的狀態碼為 200 OK，且響應結構與「基準響應」高度相似（Simhash 算法），且響應內容中包含使用者 B 的私有數據（如 PII），則確認漏洞。

### 5.2 標識符模式識別與變異

AI 掃描器需要能夠識別並針對性地變異不同類型的 ID：

- **順序整數 (Sequential Integers):** 如 1001, 1002。這是最脆弱的設計。自動化邏輯應嘗試 CurrentID + 1 和 CurrentID - 1 進行遍歷 <sup>28</sup>。

- **非順序 ID (Random/UUID):** 如 550e8400-e29b-41d4-a716-446655440000。雖然難以枚舉，但經常發生「ID 洩漏」的情況（例如在列表 API 中返回了 UUID）。掃描器應建立「ID 池」（ID Pool），將在整個爬取過程中發現的所有 UUID 收集起來，嘗試交叉引用 <sup>29</sup>。

- **編碼/雜湊 ID:** 許多應用會將 ID 進行 Base64 編碼（如 MTAwMQ== 對應 1001）。AI 掃描器應具備自動解碼能力，若發現解碼後為整數或特定格式，則構造對應的編碼 ID 進行測試。

### 5.3 消除「軟性失敗」（Soft Failures）誤報

IDOR 檢測中最大的誤報來源是「軟性 403」。即伺服器雖然拒絕了權限，但返回的 HTTP 狀態碼卻是 200 OK，內容顯示為自定義的錯誤頁面或跳轉到登錄頁。

- **權限拒絕指紋庫:** 掃描器需維護一個包含常見拒絕訪問字串的列表，如 Access Denied, Unauthorized, You do not have permission, Login required <sup>30</sup>。

- **模糊哈希與結構對比:** 使用模糊哈希（Fuzzy Hashing，如 ssdeep）對比「越權響應」與「已知的 403/401 響應」或「登錄頁面」。若相似度超過閾值（如 90%），即使狀態碼為 200，仍判定為失敗（無漏洞）。

- **數據存在性檢查 (Data Existence Check):** 這是最可靠的方法。在測試前，預先定義使用者 B 的特徵數據（如 Email: victim@example.com）。若越權請求的響應中包含了該特徵數據，則可無視狀態碼或錯誤訊息，直接判定為漏洞 <sup>31</sup>。

### 5.4 JSON 模組輸出：function_idor

> JSON

{  
"function_idor": {  
"detection_strategy": "pairwise_testing",  
"required_sessions": \["attacker_session", "victim_session"\],  
"id_patterns": {  
"integer": "^\\d+\$",  
"uuid": "^\[0-9a-fA-F\]{8}-\[0-9a-fA-F\]{4}-\[0-9a-fA-F\]{4}-\[0-9a-fA-F\]{4}-\[0-9a-fA-F\]{12}\$",  
"hash_md5": "^\[a-fA-F0-9\]{32}\$",  
"base64": "^(?:\[A-Za-z0-9+/\]{4})\*(?:\[A-Za-z0-9+/\]{2}==\|\[A-Za-z0-9+/\]{3}=)?\$"  
},  
"success_criteria": {  
"status_code_match": true,  
"structure_similarity_threshold": 0.85,  
"sensitive_data_leak": true,  
"keyword_exclusion_check": true  
},  
"false_positive_keywords":,  
"verification_logic": {  
"step1": "identify_resource_id",  
"step2": "replace_id_with_victim_id",  
"step3": "compare_response_with_baseline",  
"step4": "check_for_victim_pii"  
}  
}  
}

## 6. 綜合結論與實施建議

本研究報告詳細闡述了針對 SQLi, XSS, SSRF, IDOR 四大漏洞的自動化檢測邏輯。構建下一代 AI 驅動的掃描器，關鍵在於從「特徵匹配」轉向「行為分析」與「上下文感知」。

1.  **精確度優先:** 對於 SQLi，優先依賴 DBMS 特定的錯誤指紋；對於盲注，必須實施嚴格的統計學延遲分析以消除網路抖動的影響。

2.  **動態驗證:** 對於 XSS，靜態分析已不足以應對現代前端框架，必須引入 Headless Browser 進行基於事件（Event-Based）的動態驗證。

3.  **雲端適配:** 對於 SSRF，掃描器必須識別並適配不同雲端供應商的 Metadata 協議（Header 要求、API 版本），否則將導致大量漏報。

4.  **邏輯推理:** 對於 IDOR，AI 代理必須理解業務實體之間的關係，採用雙帳號差異化測試策略，並具備識別「軟性失敗」的能力。

透過導入本報告定義的 JSON 邏輯模組，安全團隊將能顯著提升自動化掃描的覆蓋率與準確性，實現真正的 DevSecOps 整合。

## 附錄：參考文獻來源索引

- <sup>1</sup> Swissky Repo - SQL Injection Payloads

- <sup>7</sup> Akto - SQL Injection Cheat Sheet

- <sup>2</sup> NetSPI - SQL Injection Wiki

- <sup>3</sup> GitHub - Time-based Blind SQL Injection

- <sup>5</sup> OWASP - Blind SQL Injection

- <sup>4</sup> PortSwigger - SQL Injection Cheat Sheet

- <sup>6</sup> Invicti - SQL Injection Cheat Sheet

- <sup>18</sup> Resecurity - SSRF to AWS Metadata Exposure

- <sup>13</sup> Acunetix - XSS Filter Evasion

- <sup>12</sup> Swissky Repo - XSS Polyglots

- <sup>14</sup> Swissky Repo - XSS Filter Bypass

- <sup>17</sup> Castle.io - Headless Chrome Detection

- <sup>28</sup> Legit Security - IDOR Knowledge Base

- <sup>30</sup> Sycope - IDOR Vulnerability Detection

- <sup>21</sup> Google Cloud - Metadata Overview

- <sup>22</sup> Google Developers - Metadata Server

- <sup>23</sup> Hacking The Cloud - GCP Metadata

- <sup>11</sup> ZenRows - Incapsula Bypass

- <sup>15</sup> BrowserStack - Puppeteer Alerts

- <sup>16</sup> BrowserStack - Selenium Alerts

- <sup>25</sup> Microsoft - Azure Instance Metadata Service

- <sup>26</sup> CyberCX - Azure SSRF Metadata

- <sup>24</sup> Medium - Exploring GCP Metadata Quirks

- <sup>8</sup> Oracle Docs - DBMS_PIPE

- <sup>9</sup> Cloudflare Docs - Gateway Block Page

- <sup>10</sup> Akamai Docs - Customize Error Pages

- <sup>20</sup> Medium - Using User Data and Metadata in AWS

- <sup>19</sup> AWS Docs - IMDS Code Examples

#### 引用的著作

1.  SQL Injection - Payloads All The Things, 檢索日期：1月 18, 2026， [<u>https://swisskyrepo.github.io/PayloadsAllTheThings/SQL%20Injection/</u>](https://swisskyrepo.github.io/PayloadsAllTheThings/SQL%20Injection/)

2.  Error Based - NetSPI SQL Injection Wiki, 檢索日期：1月 18, 2026， [<u>https://sqlwiki.netspi.com/injectionTypes/errorBased/</u>](https://sqlwiki.netspi.com/injectionTypes/errorBased/)

3.  Time-based-Blind-SQL-Injection.md - GitHub, 檢索日期：1月 18, 2026， [<u>https://github.com/Sourabh-Sahu/SQL-Injection/blob/main/Time-based-Blind-SQL-Injection.md</u>](https://github.com/Sourabh-Sahu/SQL-Injection/blob/main/Time-based-Blind-SQL-Injection.md)

4.  SQL injection cheat sheet \| Web Security Academy - PortSwigger, 檢索日期：1月 18, 2026， [<u>https://portswigger.net/web-security/sql-injection/cheat-sheet</u>](https://portswigger.net/web-security/sql-injection/cheat-sheet)

5.  Blind SQL Injection \| OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-community/attacks/Blind_SQL_Injection</u>](https://owasp.org/www-community/attacks/Blind_SQL_Injection)

6.  SQL Injection Cheat Sheet - Invicti, 檢索日期：1月 18, 2026， [<u>https://www.invicti.com/blog/web-security/sql-injection-cheat-sheet</u>](https://www.invicti.com/blog/web-security/sql-injection-cheat-sheet)

7.  SQL Injection Cheat Sheet - Akto, 檢索日期：1月 18, 2026， [<u>https://www.akto.io/blog/sql-injection-cheat-sheet</u>](https://www.akto.io/blog/sql-injection-cheat-sheet)

8.  DBMS_PIPE - Oracle Help Center, 檢索日期：1月 18, 2026， [<u>https://docs.oracle.com/en/database/oracle/oracle-database/18/arpls/DBMS_PIPE.html</u>](https://docs.oracle.com/en/database/oracle/oracle-database/18/arpls/DBMS_PIPE.html)

9.  Block page · Cloudflare One docs, 檢索日期：1月 18, 2026， [<u>https://developers.cloudflare.com/cloudflare-one/reusable-components/custom-pages/gateway-block-page/</u>](https://developers.cloudflare.com/cloudflare-one/reusable-components/custom-pages/gateway-block-page/)

10. Customize error pages - Akamai TechDocs, 檢索日期：1月 18, 2026， [<u>https://techdocs.akamai.com/etp/docs/customize-error-pages</u>](https://techdocs.akamai.com/etp/docs/customize-error-pages)

11. How to Bypass Imperva Incapsula for Web Scraping (2026) - ZenRows, 檢索日期：1月 18, 2026， [<u>https://www.zenrows.com/blog/incapsula-bypass</u>](https://www.zenrows.com/blog/incapsula-bypass)

12. Polyglot XSS - Payloads All The Things, 檢索日期：1月 18, 2026， [<u>https://swisskyrepo.github.io/PayloadsAllTheThings/XSS%20Injection/2%20-%20XSS%20Polyglot/</u>](https://swisskyrepo.github.io/PayloadsAllTheThings/XSS%20Injection/2%20-%20XSS%20Polyglot/)

13. XSS Filter Evasion: How Attackers Bypass XSS Filters – And Why Filtering Alone Isn't Enough \| Acunetix, 檢索日期：1月 18, 2026， [<u>https://www.acunetix.com/blog/articles/xss-filter-evasion-bypass-techniques/</u>](https://www.acunetix.com/blog/articles/xss-filter-evasion-bypass-techniques/)

14. XSS Filter Bypass - Payloads All The Things, 檢索日期：1月 18, 2026， [<u>https://swisskyrepo.github.io/PayloadsAllTheThings/XSS%20Injection/1%20-%20XSS%20Filter%20Bypass/</u>](https://swisskyrepo.github.io/PayloadsAllTheThings/XSS%20Injection/1%20-%20XSS%20Filter%20Bypass/)

15. Handling Alerts and Popups in Puppeteer - BrowserStack, 檢索日期：1月 18, 2026， [<u>https://www.browserstack.com/guide/alerts-and-popups-in-puppeteer</u>](https://www.browserstack.com/guide/alerts-and-popups-in-puppeteer)

16. How to handle Alerts & Popups in Selenium \[2026\] \| BrowserStack, 檢索日期：1月 18, 2026， [<u>https://www.browserstack.com/guide/alerts-and-popups-in-selenium</u>](https://www.browserstack.com/guide/alerts-and-popups-in-selenium)

17. How to detect Headless Chrome bots instrumented with Puppeteer? - The Castle blog, 檢索日期：1月 18, 2026， [<u>https://blog.castle.io/how-to-detect-headless-chrome-bots-instrumented-with-puppeteer-2/</u>](https://blog.castle.io/how-to-detect-headless-chrome-bots-instrumented-with-puppeteer-2/)

18. SSRF to AWS Metadata Exposure: How Attackers Steal Cloud Credentials - Resecurity, 檢索日期：1月 18, 2026， [<u>https://www.resecurity.com/blog/article/ssrf-to-aws-metadata-exposure-how-attackers-steal-cloud-credentials</u>](https://www.resecurity.com/blog/article/ssrf-to-aws-metadata-exposure-how-attackers-steal-cloud-credentials)

19. Examples of retrieving instance metadata using IMDSv1 and IMDSv2 on a Snowball Edge, 檢索日期：1月 18, 2026， [<u>https://docs.aws.amazon.com/snowball/latest/developer-guide/imds-code-examples.html</u>](https://docs.aws.amazon.com/snowball/latest/developer-guide/imds-code-examples.html)

20. Using User Data and Metadata In AWS \| by Augustine Ozor - Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/@augustineozor/using-user-data-and-metadata-in-aws-83c20d1987ff</u>](https://medium.com/@augustineozor/using-user-data-and-metadata-in-aws-83c20d1987ff)

21. About VM metadata \| Compute Engine - Google Cloud Documentation, 檢索日期：1月 18, 2026， [<u>https://docs.cloud.google.com/compute/docs/metadata/overview</u>](https://docs.cloud.google.com/compute/docs/metadata/overview)

22. Storing and Retrieving Metadata - Google Compute Engine - huihoo, 檢索日期：1月 18, 2026， [<u>https://download.huihoo.com/google/gdgdevkit/DVD1/developers.google.com/compute/docs/metadata.html</u>](https://download.huihoo.com/google/gdgdevkit/DVD1/developers.google.com/compute/docs/metadata.html)

23. Metadata in Google Cloud Instances - Hacking The Cloud, 檢索日期：1月 18, 2026， [<u>https://hackingthe.cloud/gcp/general-knowledge/metadata_in_google_cloud_instances/</u>](https://hackingthe.cloud/gcp/general-knowledge/metadata_in_google_cloud_instances/)

24. Exploring the Quirks of GCP's Metadata Server \| by Jared Hatfield \| Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/@jaredhatfield/exploring-the-quirks-of-gcps-metadata-server-c69903632edb</u>](https://medium.com/@jaredhatfield/exploring-the-quirks-of-gcps-metadata-server-c69903632edb)

25. Azure Instance Metadata Service for virtual machines - Microsoft Learn, 檢索日期：1月 18, 2026， [<u>https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service</u>](https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service)

26. Azure SSRF Metadata - CyberCX, 檢索日期：1月 18, 2026， [<u>https://cybercx.com.au/blog/azure-ssrf-metadata/</u>](https://cybercx.com.au/blog/azure-ssrf-metadata/)

27. PayloadsAllTheThings/Server Side Request Forgery/README.md ..., 檢索日期：1月 18, 2026， [<u>https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Server%20Side%20Request%20Forgery/README.md</u>](https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/Server%20Side%20Request%20Forgery/README.md)

28. What Are Insecure Direct Object References (IDOR)? Types and Prevention - Legit Security, 檢索日期：1月 18, 2026， [<u>https://www.legitsecurity.com/aspm-knowledge-base/insecure-direct-object-references/</u>](https://www.legitsecurity.com/aspm-knowledge-base/insecure-direct-object-references/)

29. Insecure Direct Object Reference (IDOR) - A Deep Dive - Hadrian.io, 檢索日期：1月 18, 2026， [<u>https://hadrian.io/blog/insecure-direct-object-reference-idor-a-deep-dive</u>](https://hadrian.io/blog/insecure-direct-object-reference-idor-a-deep-dive)

30. IDOR vulnerability – how to detect an attack on web applications through HTTP traffic analysis - Sycope, 檢索日期：1月 18, 2026， [<u>https://www.sycope.com/post/idor-vulnerability-how-to-detect-an-attack-on-web-applications-through-http-traffic-analysis</u>](https://www.sycope.com/post/idor-vulnerability-how-to-detect-an-attack-on-web-applications-through-http-traffic-analysis)

31. Hunting IDOR: A Deep Dive into Insecure Direct Object References \| by Shah kaif \| InfoSec Write-ups, 檢索日期：1月 18, 2026， [<u>https://infosecwriteups.com/%EF%B8%8F-hunting-idor-a-deep-dive-into-insecure-direct-object-references-b550a9f77333</u>](https://infosecwriteups.com/%EF%B8%8F-hunting-idor-a-deep-dive-into-insecure-direct-object-references-b550a9f77333)
