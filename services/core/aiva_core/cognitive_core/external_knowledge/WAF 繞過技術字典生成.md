# 現代 Web Application Firewall (WAF) 深度滲透與繞過技術分析報告

## 1. 執行摘要與威脅環境概述

隨著數位轉型的加速，Web 應用程式已成為企業與組織的核心資產，同時也成為網路攻擊的主要目標。Web Application Firewall (WAF) 作為防禦應用層攻擊（如 SQL 注入、跨站腳本攻擊 XSS、遠端代碼執行 RCE）的第一道防線，其角色至關重要。然而，2024 年至 2025 年的威脅情報顯示，攻擊者正採用日益複雜的技術來規避 WAF 的檢測，這些技術不再僅限於簡單的特徵碼混淆，而是深入到底層協議的解析差異、編碼轉換的邊界情況，以及針對特定 WAF 架構（如 Cloudflare、AWS WAF、Imperva）的邏輯漏洞 <sup>1</sup>。

本報告旨在提供一份詳盡的深度研究，剖析當前滲透測試中常見且高階的 WAF 繞過技術。報告將首先建立 WAF 運作的理論基礎，解釋「解析差異 (Parser Differential)」如何成為所有繞過技術的核心原理。接著，將系統性地探討編碼混淆（如 IBM037 字符集攻擊）、HTTP 協議層攻擊（如分塊傳輸變異與請求走私），以及針對三大主流 WAF 廠商的特定繞過手法。最後，報告將提供一份結構化的 JSON 繞過字典，供資安專業人員在合法授權的滲透測試與紅隊演練中使用。

分析顯示，儘管 WAF 廠商不斷引入機器學習與行為分析技術，但由於需要在效能（低延遲）與安全性之間取得平衡，WAF 往往無法對所有請求進行與後端應用程式完全一致的深度解析。這種「解析上的不對稱性」正是攻擊者得以利用的關鍵縫隙。例如，AWS WAF 對請求主體 (Body) 的檢查長度限制、Cloudflare 對特定 HTML 屬性的解析忽略，以及 Imperva 對壓縮內容的處理邏輯，都曾被證實存在可被繞過的弱點 <sup>3</sup>。

## 2. WAF 架構解析與檢測邏輯

要成功執行 WAF 繞過，必須先深入理解目標的防禦架構與決策邏輯。現代 WAF 已從單純的規則匹配引擎演變為整合了多種檢測模型的複雜系統。

### 2.1 負面安全模型與特徵碼匹配

早期的 WAF 主要依賴**負面安全模型 (Negative Security Model)**，即所謂的「黑名單」機制。此模型預設允許所有流量通過，僅攔截符合已知攻擊特徵 (Signatures) 的請求。

- **運作機制**：WAF 維護一個龐大的特徵資料庫，其中包含已知的惡意 Payload 片段（如 ' OR 1=1、\<script\>、/etc/passwd 等）。當請求進入時，WAF 會利用正則表達式 (Regular Expressions, Regex) 對請求的各個部分（URL、Header、Body）進行掃描。

- **局限性**：這種模式的弱點在於它依賴於「已知」的威脅。攻擊者只需對 Payload 進行微小的變形（如改變大小寫、插入空白字符、使用等價函數），往往就能避開正則表達式的匹配規則。此外，為了避免誤殺合法流量 (False Positives)，WAF 的正則表達式通常不會寫得過於寬泛，這給了攻擊者可乘之機 <sup>6</sup>。

### 2.2 正面安全模型與異常檢測

為了彌補負面模型的不足，許多 WAF 引入了**正面安全模型 (Positive Security Model)**，即「白名單」機制。

- **運作機制**：此模型定義了什麼是「合法」的流量。例如，定義某個參數必須是整數、長度不超過 10 個字符，或者 Content-Type 必須是特定的格式。任何不符合這些定義的請求都會被視為異常並遭到攔截。

- **異常檢測 (Anomaly Detection)**：基於協議標準（如 RFC 7230）來檢測畸形的 HTTP 請求。例如，檢測是否存在雙重 Content-Length Header、無效的 HTTP 方法或非標準的字符編碼。WAFFLED 的研究指出，許多 WAF 繞過技術正是利用了 WAF 與後端伺服器對「異常」定義的不一致來實現的 <sup>1</sup>。

### 2.3 下一代 WAF：行為分析與 AI

現代的主流 WAF（如 Cloudflare、AWS WAF）已演進為**下一代 WAF (NGWAF)**，整合了行為分析與機器學習技術。

- **行為指紋**：WAF 不再僅看單個請求的內容，而是分析客戶端的行為模式。例如，檢查請求的頻率（Rate Limiting）、TLS 握手時的指紋（JA3/JA4）、瀏覽器指紋（Canvas, User-Agent, Headers 順序）等。

- **威脅情報整合**：利用全球網路的威脅情報，即時封鎖來自惡意 IP、Botnet 或匿名代理的流量。這意味著滲透測試人員若使用標準的掃描工具（如 SQLMap, Burp Suite）而不進行適當的偽裝，很容易在 TCP/TLS 連線建立階段就被識別並封鎖，甚至還沒發送 Payload 就已失敗 <sup>9</sup>。

### 2.4 解析差異：WAF 的阿基里斯之腱

WAF 繞過技術的核心理論基礎是**解析差異 (Parser Differential)**。這是一個根本性的架構問題：WAF 是位於客戶端與後端伺服器之間的中介設備（或服務），它必須模擬後端伺服器的解析邏輯來判斷請求是否惡意。然而，由於後端技術棧的多樣性（PHP, Java, Python,.NET, Node.js 等）以及 WAF 對效能的極致追求，WAF 的解析器永遠無法與後端完全一致。

- **深度不一致**：WAF 為了確保低延遲，可能只解析 JSON 的前幾層，或者只檢查 HTTP Body 的前 8KB。而後端伺服器則會完整解析整個請求。

- **規範寬容度不一致**：後端伺服器為了相容性，往往會接受並處理畸形的請求（如缺少邊界的 Multipart 請求），而 WAF 可能因為無法解析該畸形請求而選擇「放行 (Fail Open)」，導致惡意 Payload 直達後端 <sup>2</sup>。

## 3. 高級編碼與字符集混淆技術

編碼與混淆是試圖將惡意 Payload 轉換為 WAF 無法識別，但後端應用程式能夠還原並執行的形式。這利用了 WAF 與後端在解碼流程上的差異。

### 3.1 字符集編碼攻擊：IBM037 與 EBCDIC

這是一種針對特定後端架構（特別是處理多國語言或傳統大型主機系統）的高階繞過技術。

- **原理**：大多數 WAF 預設將 HTTP 請求視為 ASCII 或 UTF-8 編碼進行檢查。如果攻擊者在 Content-Type Header 中明確指定一個非標準的字符集（如 ibm037, ibm500, cp1025 等 EBCDIC 編碼變體），並將 Payload 依照該字符集進行編碼發送，WAF 的檢測引擎若不支援該字符集，將會看到一串亂碼或無害的字節流，從而放行請求。

- **後端行為**：如果後端應用程式（如 IIS, Java Servlet 容器）支援該字符集並正確處理了 Content-Type，它會將亂碼還原為原始的惡意 Payload（如 SQL 注入指令）並執行。

- **實作**：攻擊者可以使用 Python 腳本將 Payload 轉換為目標編碼。例如，SELECT 指令在 ASCII 中是 0x53 0x45 0x4C 0x45 0x43 0x54，而在 IBM037 中則是完全不同的字節序列。研究表明，這種技術在繞過簽名式檢測時極為有效，且被多種 WAF 解決方案忽略 <sup>11</sup>。

**Python 實現範例**：

> Python

payload = "1 UNION SELECT 1, version()--"  
headers = {'Content-Type': 'application/x-www-form-urlencoded; charset=ibm037'}  
\# 將 Payload 編碼為 IBM037 並進行 URL Quote，確保傳輸過程不被截斷  
encoded_payload = urllib.parse.quote_from_bytes(payload.encode('ibm037'))

### 3.2 雙重 URL 編碼與規範化漏洞

URL 編碼是 Web 傳輸的標準，但 WAF 與後端對解碼次數的處理可能不同。

- **雙重編碼 (Double Encoding)**：將字符進行兩次 URL 編碼。例如，單引號 ' 編碼為 %27，再次編碼為 %2527。若 WAF 僅進行一次解碼（得到 %27），它可能將其視為普通字符串而非 SQL 語法的一部分。然而，若後端應用程式或中間件配置了自動解碼機制（有時為了容錯會進行多次解碼），最終將還原出單引號，觸發注入 <sup>7</sup>。

- **路徑遍歷 (Path Traversal)**：對於路徑遍歷攻擊（如 ../），攻擊者可使用 %252e%252e%252f。WAF 解碼一次得到 %2e%2e%2f（即 ..%2f），這看起來是安全的字符串，但後端再次解碼後變成 ../，導致目錄穿越 <sup>13</sup>。

### 3.3 Unicode 變體與同形字攻擊

Unicode 是一個龐大的字符集，其中包含許多「看起來不一樣」但在規範化 (Normalization) 後代表相同意義的字符。

- **兼容性分解 (Compatibility Decomposition)**：某些 Unicode 字符在 NFKC 或 NFKD 規範化後會分解為 ASCII 字符。例如，全形字符 ＜ (U+FF1C) 在規範化後會變成 ASCII 的 \< (U+003C)。如果 WAF 在規範化之前進行檢查，它可能無法識別 ＜script＞ 為惡意標籤，但後端在處理時若進行了規範化，就會將其轉換為可執行的 \<script\> <sup>7</sup>。

- **空白字符變體**：SQL 語法通常允許使用空格分隔關鍵字。WAF 可能攔截標準空格 (%20)，但攻擊者可以使用其他 Unicode 空白字符（如 U+00A0 Non-breaking space, U+200B Zero-width space）來分隔 SQL 關鍵字。若 WAF 僅過濾 ASCII 空格，這些變體將繞過檢測，而資料庫解析器可能視其為有效分隔符 <sup>14</sup>。

### 3.4 SQL 注入專用混淆技術

SQL 注入是 WAF 防禦的重點，因此針對 SQLi 的混淆技術也最為豐富。

- **Case Toggling (大小寫變換)**：利用 SQL 不區分大小寫的特性。例如 SeLeCt 代替 SELECT。雖然現代 WAF 多已支援不區分大小寫匹配，但在某些特定規則或舊版 WAF 中仍可能有效 <sup>6</sup>。

- **註解符號插入 (Comment Injection)**：在 SQL 關鍵字中間插入註解，打斷 WAF 的正則匹配，但 SQL 解析器會忽略註解，還原關鍵字。例如 UN/\*\*/ION 或 SE/\*\*/LECT。在 MySQL 中，/\*!50000UNION\*/ 是一種特殊的版本註解，只有當版本高於 5.00.00 時才會執行其中的語句，這可用於隱藏 Payload <sup>12</sup>。

- **科學記號與數學運算**：在需要輸入數字的地方，使用科學記號（如 1.0、1e0）或數學表達式（如 1+0）來避開整數型別的特徵檢測。

- **HTTP 參數污染 (HPP)**：發送多個同名參數，如 id=1&id=1' OR '1'='1。WAF 可能只檢查第一個 id（合法），但後端應用程式（如 ASP.NET, IIS）可能會將所有同名參數串聯，或取最後一個參數值，從而執行惡意代碼 <sup>6</sup>。

### 3.5 XSS 專用混淆與 HTML 實體

跨站腳本攻擊 (XSS) 的繞過重點在於隱藏 JavaScript 代碼與 HTML 標籤。

- **HTML 實體編碼**：在 HTML 屬性中，可以使用十進位 (e.g., \<) 或十六進位實體來代替 \<、\>、" 等字符。瀏覽器在渲染時會自動解碼這些實體。

- **JavaScript 編碼**：使用 \uXXXX Unicode 轉義序列來表示字符串中的字符。例如 alert(1) 可寫為 \u0061lert(1)。

- **畸形標籤與屬性**：瀏覽器具有強大的容錯能力，能解析許多不符合標準的 HTML。例如 \<svg/onload=alert(1)\>（利用 / 代替空格）或 \<img src=x onerror=alert(1)\>。WAF 若嚴格依照標準解析，可能會漏掉這些畸形但有效的 Payload <sup>12</sup>。

- **Attribute Overloading (Cloudflare 案例)**：攻擊者利用大量無效屬性或特定構造的屬性來「推擠」惡意 Payload，使其超出 WAF 的檢查視窗或擾亂解析邏輯 <sup>4</sup>。

## 4. HTTP 協議層規避與走私技術

此類技術不直接對 Payload 進行變形，而是操縱 HTTP 請求的結構，利用 WAF 與後端伺服器對 HTTP 協議理解的不一致，使 WAF 無法正確解析請求內容，進而放行請求。

### 4.1 分塊傳輸編碼 (Chunked Transfer Encoding) 變異

HTTP/1.1 定義了 Transfer-Encoding: chunked 機制，允許將訊息主體分割成多個區塊傳輸。這對 WAF 來說是一個巨大的挑戰，因為它必須緩存並重組所有區塊才能進行檢查，這極消耗資源。

- **基本繞過**：簡單地使用 Chunked 編碼有時就能繞過不支援此功能的 WAF。

- **分塊變異 (Chunked Mutations)**：WAFFLED 研究團隊發現了多種透過變異 Transfer-Encoding Header 來繞過 WAF 的技術。例如：

  - **Header 變形**：使用 Transfer-Encoding: xchunked、Transfer-Encoding : chunked（冒號前有空格）或雙重 Header。WAF 可能因為看不懂這些變形而將請求視為普通請求（使用 Content-Length），只檢查 Body 的第一部分。但後端（如某些版本的 Tomcat, Jetty, Go）可能具有更強的容錯性，識別出 Chunked 編碼並處理後續的惡意區塊 <sup>1</sup>。

  - **分塊結構干擾**：在分塊長度與數據之間插入分號、註解或特定的控制字符，擾亂 WAF 的解析器，使其無法正確識別 Payload 的起始位置。

### 4.2 HTTP 請求走私 (Request Smuggling)

請求走私是協議層攻擊的極致表現，它利用前端（WAF/Load Balancer）與後端對 Content-Length (CL) 與 Transfer-Encoding (TE) 優先級處理的不一致，將一個惡意請求「走私」到下一個合法請求的開頭。

- **CL.TE 漏洞**：前端優先看 CL，後端優先看 TE。攻擊者構造一個請求，前端認為只有一個請求，將其全部轉發。後端根據 TE 解析，發現請求在中間結束了，剩下的部分（惡意 Payload）被留在緩衝區，並被視為下一個請求的開頭。

- **TE.CL 漏洞**：前端優先看 TE，後端優先看 CL。原理類似，只是方向相反。

- **WAF 繞過效應**：在走私攻擊中，惡意 Payload 實際上隱藏在第一個請求的 Body 中（對於前端 WAF 而言）。WAF 檢查第一個請求時，可能看不出異狀（因為 Payload 看起來像是一堆數據）。但當這些數據在後端被解釋為第二個獨立的 HTTP 請求時，WAF 的防護就完全被繞過了，因為 WAF 從未單獨檢查過這個「走私」進去的請求 <sup>18</sup>。

### 4.3 HTTP Header 欺騙與注入

WAF 常利用特定的 HTTP Header 來判斷客戶端來源、IP 信譽或連線屬性。攻擊者可偽造這些 Header 來誤導 WAF。

- **IP 偽造**：WAF 通常會信任來自內部網路或特定可信代理的流量。攻擊者可在請求中加入 X-Forwarded-For: 127.0.0.1、X-Originating-IP: 127.0.0.1 或 Client-IP: 127.0.0.1。若 WAF 配置不當，可能會認為請求來自本地或受信設備，從而降低檢查等級或直接放行 <sup>21</sup>。

- **URL 覆蓋**：某些框架（如 Symfony, ASP.NET, Drupal）支援 X-Original-URL 或 X-Rewrite-URL Header，允許客戶端覆蓋請求行中的 URL。

  - **攻擊場景**：WAF 檢查請求行 GET /public-page，認為是合法的。但請求中包含 X-Original-URL: /admin/delete-user。後端應用程式收到後，根據 Header 將路徑重寫為 /admin/delete-user 並執行。由於 WAF 只檢查了 /public-page，攻擊者成功繞過了對 /admin 路徑的存取控制 <sup>1</sup>。

### 4.4 HTTP 方法與參數污染

- **HTTP 方法覆蓋**：許多 WAF 規則僅針對 GET 與 POST 方法。攻擊者可嘗試使用 PUT、DELETE 甚至 TRACE、OPTIONS 方法發送 Payload。某些應用程式框架會自動將 Body 中的參數映射到變數，而不區分 HTTP 方法 <sup>1</sup>。

- **參數污染**：如前所述，利用 HPP 技術，攻擊者可以發送多個同名參數。WAF 可能只清洗第一個參數，而後端使用未清洗的第二個參數。這在繞過 SQLi 與 XSS 過濾時非常有效 <sup>6</sup>。

## 5. 特定廠商 WAF 繞過深度分析

不同的 WAF 廠商有其獨特的架構與檢測邏輯。針對特定廠商的繞過技術往往利用了該產品特有的實作細節或已知限制。

### 5.1 Cloudflare WAF

Cloudflare 是基於邊緣網路 (Edge Network) 的 WAF，擁有強大的分佈式架構與機器學習能力。

#### 5.1.1 架構與防禦機制

Cloudflare 在全球邊緣節點進行流量清洗。它結合了託管規則集 (Managed Ruleset)、OWASP 核心規則集 (CRS) 以及瀏覽器完整性檢查 (Browser Integrity Check)。它非常依賴 TLS 指紋與 JavaScript 挑戰（如 Turnstile）來過濾機器人 <sup>24</sup>。

#### 5.1.2 屬性重載 (Attribute Overloading) 與 DOM 事件

在 2024-2025 年間，研究人員發現 Cloudflare 在處理極長的 HTML 屬性列表時存在解析限制。

- **屬性重載**：攻擊者在 HTML 標籤中注入大量的無效屬性（例如 100 個 data-junk="..."），將真正的惡意屬性（如 onmouseover）推到標籤的末尾。Cloudflare 的解析器為了效能可能在處理一定數量的屬性後停止檢查，導致末尾的惡意屬性被漏過 <sup>4</sup>。

- **DOM 事件**：Cloudflare 對常見的 onload, onclick, onerror 封鎖較嚴，但對新興或較冷門的 HTML5 事件支援較慢。例如 ontoggle (用於 \<details\> 元素) 和 onbeforetoggle (用於 Popover API) 曾被證實可繞過其 XSS 過濾規則 <sup>4</sup>。

#### 5.1.3 源站 IP 暴露與直接攻擊

這不是技術上的 WAF 漏洞，而是架構上的繞過。如果攻擊者能找到後端源站的真實 IP 地址 (Origin IP)，就可以修改本地 hosts 文件，直接向源站發送 HTTP 請求，完全繞過 Cloudflare 的防護網路。

- **偵測方法**：利用 Censys, Shodan 掃描 SSL 證書、查詢歷史 DNS 記錄 (SecurityTrails)、或觸發源站主動發出連線（如發送郵件、SSRF）來獲取真實 IP <sup>26</sup>。

#### 5.1.4 自動化挑戰與無頭瀏覽器對抗

針對 Cloudflare 的 "I'm Under Attack" 模式，攻擊者使用高度客製化的無頭瀏覽器工具。

- **工具**：**FlareSolverr** 和 **Undetected-Chromedriver**。這些工具不僅能執行 JavaScript 來解決挑戰，還能修補 Selenium/Puppeteer 的特徵（如 navigator.webdriver 屬性），並模擬真實瀏覽器的 TLS Client Hello 指紋 (JA3)，從而騙過 Cloudflare 的機器人檢測 <sup>24</sup>。

### 5.2 AWS WAF

AWS WAF 緊密整合於 AWS 生態系統（ALB, API Gateway, CloudFront），以高度可配置與按量計費著稱。

#### 5.2.1 請求主體檢查限制 (Body Inspection Limits)

這是 AWS WAF 最著名的架構限制。為了保證雲端服務的低延遲，AWS WAF 對請求 Body 的檢查有硬性限制。

- **限制詳情**：對於大多數服務，預設檢查前 **8KB** 的 Body 內容。對於 CloudFront 和 API Gateway，此限制可配置提升至 **64KB**，但這仍是一個有限的視窗 <sup>3</sup>。

- **繞過手法 (Oversized Body Padding)**：攻擊者在 JSON 或 Form 表單的開頭填充大量的垃圾數據（Padding），使其長度超過 8KB (或 64KB)。將真正的惡意 Payload（如 SQLi 或 RCE 指令）放在這些填充數據之後。AWS WAF 檢查完前段數據判定無害後放行，而後端應用程式（如 Lambda, EC2 上的 Web Server）則會完整接收並解析整個 Body，執行位於檢查視窗之外的惡意代碼。這在 JSON 請求中特別有效，因為 JSON 允許在鍵值對之間插入大量空白或無意義的欄位 <sup>3</sup>。

#### 5.2.2 JSON 解析差異與繞過

AWS WAF 的 JSON 解析器與後端常見的解析器（如 Java Jackson, Node.js JSON.parse）存在行為差異。

- **深度巢狀**：AWS WAF 可能無法正確解析深度巢狀 (Deeply Nested) 的 JSON 結構。

- **鍵值重複**：當 JSON 中存在重複的 Key 時，AWS WAF 可能檢查第一個，而後端可能使用最後一個。這種差異可用於繞過針對特定 JSON 欄位的安全規則 <sup>1</sup>。

#### 5.2.3 特定事件處理器漏洞

AWS WAF 的託管規則集 (Managed Ruleset) 有時對新興 Web 標準的更新存在滯後。研究人員曾發現利用實驗性的 DOM 事件如 onbeforetoggle 可以繞過 AWS WAF 的 XSS 過濾規則。這顯示了 WAF 規則庫維護與瀏覽器功能演進之間的時間差漏洞 <sup>28</sup>。

### 5.3 Imperva (Incapsula) WAF

Imperva 提供企業級的 WAF 保護，強調精細的存取控制與應用層防護。

#### 5.3.1 壓縮編碼繞過 (Compression Bypass)

這是一個歷史上著名的漏洞，雖然已被修補，但其變體仍值得測試。

- **原理**：Imperva WAF 曾存在一個邏輯漏洞，當請求 Header 包含 Content-Encoding: gzip，但請求 Body 實際上並未壓縮（Raw Data）時，WAF 會嘗試解壓縮失敗，然後錯誤地選擇「放行」該請求而不進行進一步檢查。若後端伺服器能夠容忍這種 Header 與 Body 不一致的情況（或者忽略 Header 直接解析 Body），攻擊者就能成功繞過 WAF 發送任意 Payload。

- **現狀**：雖然原漏洞已修補，但測試其他壓縮格式（如 deflate, br, zstd）或構造畸形的壓縮流仍是滲透測試的標準步驟 <sup>5</sup>。

#### 5.3.2 內部 Header 注入與 Session 操縱

Imperva 在其 Proxy 與客戶端/後端之間使用特定的 HTTP Header 來傳遞狀態資訊。

- **Header 操縱**：Header 如 X-Iinfo、incap_ses、visid_incap 用於追蹤 Session 與客戶端信譽。如果攻擊者能夠在請求中偽造這些 Header，可能會擾亂 WAF 的 Session 追蹤邏輯，導致 WAF 誤判請求來源，或者重用一個高信譽的 Session ID 來發送惡意流量 <sup>9</sup>。

#### 5.3.3 TLS 指紋識別與規避

Imperva 非常依賴 TLS 指紋 (JA3/JA4) 來識別非瀏覽器流量。

- **對抗技術**：使用標準的 Python requests 或 curl 發出的請求會因為 TLS Client Hello 的特徵（如 Cipher Suites 順序、Extensions 排列）而被識別為機器人並攔截。繞過此檢測需要使用 **Cipher Stunting** 技術，或使用支援修改 TLS 指紋的工具（如 ja3transport, CycleTLS），將攻擊工具的 TLS 指紋偽裝成標準的 Chrome 或 Firefox 瀏覽器 <sup>9</sup>。

## 6. 自動化工具與測試框架

在滲透測試中，手動構造上述所有繞過 Payload 極耗時，因此專業人員常依賴自動化工具。

- **SQLMap Tamper Scripts**: SQLMap 內建了大量 tamper 腳本，可自動對 Payload 進行編碼混淆（如 space2comment.py, charencode.py, ibm037.py）。這些腳本能動態地將注入語句轉換為 WAF 難以識別的形式 <sup>6</sup>。

- **Burp Suite Extensions**:

  - **Bypass WAF**: 自動對請求添加偽造的 IP Header (X-Forwarded-For 等)。

  - **Turbo Intruder**: 用於發送高併發請求，測試競爭條件 (Race Conditions) 或暴力破解 WAF 的速率限制。

  - **Hackvertor**: 強大的編碼轉換工具，支援多層編碼與特殊字符集轉換 <sup>7</sup>。

- **WAFW00F**: 用於指紋識別，判斷目標使用何種 WAF，這對於選擇正確的繞過策略至關重要 <sup>32</sup>。

- **FlareSolverr**: 專門用於繞過 Cloudflare 等基於挑戰的 WAF 的代理伺服器，自動處理 JS 挑戰與 Cookie 獲取 <sup>24</sup>。

## 7. 繞過字典 (Bypass Dictionary)

基於上述深度研究，以下整理了一份結構化的 JSON 繞過字典。此字典分類列出了針對不同 WAF 與攻擊場景的有效 Payload 模式與技術說明，可直接整合進自動化測試工具中。

> JSON

{  
"waf_bypass_dictionary": {  
"meta": {  
"description": "Comprehensive WAF Bypass Payloads and Techniques 2024-2025",  
"targets":,  
"version": "1.0",  
"author": "Security Research Analysis"  
},  
"techniques":,  
"tooling_reference": "Python script utilizing 'payload.encode(\\ibm037\\)'"  
},  
{  
"category": "Encoding & Obfuscation",  
"method": "Double URL Encoding",  
"target": "Generic",  
"description": "對特殊字符進行兩次 URL 編碼，繞過僅解碼一次的 WAF。",  
"payloads":  
},  
{  
"category": "Encoding & Obfuscation",  
"method": "SQL Injection Obfuscation",  
"target": "Generic / Older WAFs",  
"description": "混合大小寫、SQL 註解符號、科學記號與參數污染以打斷特徵碼匹配。",  
"payloads":  
},  
{  
"category": "HTTP Protocol Evasion",  
"method": "Chunked Transfer Encoding Mutation",  
"target": "Generic / ModSecurity / AWS WAF",  
"description": "利用畸形的分塊傳輸編碼 Header 導致 WAF 解析失敗並放行 (Fail Open)。",  
"payloads":  
},  
{  
"category": "HTTP Protocol Evasion",  
"method": "HTTP Header Spoofing",  
"target": "Generic",  
"description": "偽造來源 IP 相關 Header 以欺騙 WAF 信任白名單機制。",  
"payloads":  
},  
{  
"category": "Resource Limit Exploitation",  
"method": "Oversized Body Padding",  
"target": "AWS WAF",  
"description": "填充超過 8KB (或 64KB) 的垃圾數據，將惡意 Payload 推至 WAF 檢查視窗之外。",  
"payloads": \[  
"{\\padding\\: \\\[8192+ bytes of garbage\]...\\, \\malicious\\: \\' OR 1=1 --\\}",  
"POST body with 8KB+ prefix before SQLi/RCE payload"  
\]  
},  
{  
"category": "Client-Side Evasion",  
"method": "Attribute Overloading & New Events",  
"target": "Cloudflare / AWS WAF",  
"description": "利用 WAF 尚未支援的 HTML5 屬性或事件處理器執行 XSS，或利用屬性過載耗盡解析資源。",  
"payloads": \[  
"\<details ontoggle=alert(1)\>",  
"\<div onbeforetoggle=alert(1)\>",  
"\<button popovertarget=x\>click me\</button\>\<test onbeforetoggle=alert(1) popover id=x\>",  
"\<tag data-junk1='...'... data-junk100='...' onmouseover=alert(1)\> (Attribute Overloading)"  
\]  
},  
{  
"category": "Vendor Specific",  
"method": "Gzip Content-Encoding Bypass",  
"target": "Imperva (Historical/Variants)",  
"description": "發送宣告為 Gzip 但實際未壓縮的 Payload，誘騙 WAF 因解壓失敗而跳過檢查。",  
"payloads":  
},  
{  
"category": "Vendor Specific",  
"method": "Internal Header Injection",  
"target": "Imperva",  
"description": "偽造 Imperva 內部使用的 Header 來操縱 Session 或繞過檢查。",  
"payloads": \[  
"X-Iinfo: \<forged_value\>",  
"incap_ses: \<forged_value\>"  
\]  
}  
\]  
}  
}

## 8. 結論與防禦建議

本報告詳盡分析了針對主流 WAF 的多種繞過技術。從分析中可以得出一個核心洞察：**WAF 並非萬能的防護盾牌，而是一個基於概率與性能權衡的過濾器**。攻擊者總能找到 WAF 與後端應用程式之間的「解析差異」來進行突破。

- **解析差異的必然性**：只要 WAF 與後端使用不同的解析引擎（例如 WAF 用 C++ 編寫，後端用 Java），解析差異就永遠存在。IBM037 編碼攻擊與 AWS WAF 的 JSON 解析漏洞就是最佳例證。

- **自動化對抗的升級**：隨著 WAF 引入 AI 識別，攻擊者也轉向使用自動化工具（如無頭瀏覽器）來模擬真實用戶行為，這使得單純基於特徵或簡單行為的防禦越來越難以奏效。

**防禦建議**：

1.  **縱深防禦 (Defense in Depth)**：企業不應過度依賴 WAF。後端應用程式必須實施嚴格的輸入驗證 (Input Validation) 與參數化查詢 (Parameterized Queries)，假設所有傳入的流量（即使經過 WAF）都是不可信的。

2.  **標準化與規範化**：在 WAF 層強制執行嚴格的 HTTP 協議規範，拒絕任何模糊不清、畸形或非標準編碼的請求，減少解析差異的攻擊面。

3.  **主動威脅獵捕**：利用 WAF 提供的日誌功能，主動尋找那些試圖探測 WAF 邊界的異常行為（如頻繁變換編碼、特殊的 Header 組合），而不僅僅關注被攔截的請求。

這場 WAF 防禦與繞過的競賽將持續進行，唯有深入理解底層技術原理，才能在攻防對抗中保持優勢。

References:

<sup>1</sup>

#### 引用的著作

1.  Web Application Firewall (WAF) Bypass Techniques that Work in 2025 \| by Karthikeyan Nagaraj \| Infosec Matrix \| Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/infosecmatrix/web-application-firewall-waf-bypass-techniques-that-work-in-2025-b11861b2767b</u>](https://medium.com/infosecmatrix/web-application-firewall-waf-bypass-techniques-that-work-in-2025-b11861b2767b)

2.  WAFFLED: Exploiting Parsing Discrepancies to Bypass Web Application Firewalls - arXiv, 檢索日期：1月 18, 2026， [<u>https://arxiv.org/html/2503.10846v1</u>](https://arxiv.org/html/2503.10846v1)

3.  Bypassing WAFs Using Oversized Requests - Black Hills Information ..., 檢索日期：1月 18, 2026， [<u>https://www.blackhillsinfosec.com/bypassing-wafs-using-oversized-requests/</u>](https://www.blackhillsinfosec.com/bypassing-wafs-using-oversized-requests/)

4.  WAF Release - 2025-07-14 · Changelog - Cloudflare Docs, 檢索日期：1月 18, 2026， [<u>https://developers.cloudflare.com/changelog/2025-07-14-waf-release/</u>](https://developers.cloudflare.com/changelog/2025-07-14-waf-release/)

5.  BishopFox/Imperva_gzip_WAF_Bypass - GitHub, 檢索日期：1月 18, 2026， [<u>https://github.com/BishopFox/Imperva_gzip_WAF_Bypass</u>](https://github.com/BishopFox/Imperva_gzip_WAF_Bypass)

6.  Top 10 Ways to Bypass a WAF - BugBase Blogs, 檢索日期：1月 18, 2026， [<u>https://bugbase.ai/blog/top-10-ways-to-bypass-waf</u>](https://bugbase.ai/blog/top-10-ways-to-bypass-waf)

7.  When WAFs Go Awry: Common Detection & Evasion Techniques for Web Application Firewalls - MDSec, 檢索日期：1月 18, 2026， [<u>https://www.mdsec.co.uk/2024/10/when-wafs-go-awry-common-detection-evasion-techniques-for-web-application-firewalls/</u>](https://www.mdsec.co.uk/2024/10/when-wafs-go-awry-common-detection-evasion-techniques-for-web-application-firewalls/)

8.  (PDF) WAFFLED: Exploiting Parsing Discrepancies to Bypass Web Application Firewalls, 檢索日期：1月 18, 2026， [<u>https://www.researchgate.net/publication/389894787_WAFFLED_Exploiting_Parsing_Discrepancies_to_Bypass_Web_Application_Firewalls</u>](https://www.researchgate.net/publication/389894787_WAFFLED_Exploiting_Parsing_Discrepancies_to_Bypass_Web_Application_Firewalls)

9.  How to Bypass Imperva Incapsula for Web Scraping (2026) - ZenRows, 檢索日期：1月 18, 2026， [<u>https://www.zenrows.com/blog/incapsula-bypass</u>](https://www.zenrows.com/blog/incapsula-bypass)

10. WAFFLED: Exploiting Parsing Discrepancies to Bypass Web Application Firewalls - arXiv, 檢索日期：1月 18, 2026， [<u>https://arxiv.org/html/2503.10846v3</u>](https://arxiv.org/html/2503.10846v3)

11. Request encoding to bypass web application firewalls \| by NCC ..., 檢索日期：1月 18, 2026， [<u>https://medium.com/keylogged/request-encoding-to-bypass-web-application-firewalls-71ffec97b80b</u>](https://medium.com/keylogged/request-encoding-to-bypass-web-application-firewalls-71ffec97b80b)

12. How to Bypass WAF. HackenProof Cheat Sheet - Hacken.io, 檢索日期：1月 18, 2026， [<u>https://hacken.io/discover/how-to-bypass-waf-hackenproof-cheat-sheet/</u>](https://hacken.io/discover/how-to-bypass-waf-hackenproof-cheat-sheet/)

13. Double Encoding \| OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-community/Double_Encoding</u>](https://owasp.org/www-community/Double_Encoding)

14. Introducing the URL validation bypass cheat sheet \| PortSwigger ..., 檢索日期：1月 18, 2026， [<u>https://portswigger.net/research/introducing-the-url-validation-bypass-cheat-sheet</u>](https://portswigger.net/research/introducing-the-url-validation-bypass-cheat-sheet)

15. SQL Injection Bypassing WAF - OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-community/attacks/SQL_Injection_Bypassing_WAF</u>](https://owasp.org/www-community/attacks/SQL_Injection_Bypassing_WAF)

16. XSS Filter Evasion - OWASP Cheat Sheet Series, 檢索日期：1月 18, 2026， [<u>https://cheatsheetseries.owasp.org/cheatsheets/XSS_Filter_Evasion_Cheat_Sheet.html</u>](https://cheatsheetseries.owasp.org/cheatsheets/XSS_Filter_Evasion_Cheat_Sheet.html)

17. How XSS Payloads Work with Code Examples, and How to Prevent Them \| HackerOne, 檢索日期：1月 18, 2026， [<u>https://www.hackerone.com/knowledge-center/how-xss-payloads-work-code-examples-and-how-prevent-them</u>](https://www.hackerone.com/knowledge-center/how-xss-payloads-work-code-examples-and-how-prevent-them)

18. The ultimate Bug Bounty guide to HTTP request smuggling vulnerabilities - YesWeHack, 檢索日期：1月 18, 2026， [<u>https://www.yeswehack.com/learn-bug-bounty/http-request-smuggling-guide-vulnerabilities</u>](https://www.yeswehack.com/learn-bug-bounty/http-request-smuggling-guide-vulnerabilities)

19. What is HTTP Request Smuggling? Exploitations and Security Best Practices - Vaadata, 檢索日期：1月 18, 2026， [<u>https://www.vaadata.com/blog/what-is-http-request-smuggling-exploitations-and-security-best-practices/</u>](https://www.vaadata.com/blog/what-is-http-request-smuggling-exploitations-and-security-best-practices/)

20. What Is HTTP Request Smuggling? \| Attack Examples - Imperva, 檢索日期：1月 18, 2026， [<u>https://www.imperva.com/learn/application-security/http-request-smuggling/</u>](https://www.imperva.com/learn/application-security/http-request-smuggling/)

21. X-Forwarded-For header - HTTP - MDN Web Docs, 檢索日期：1月 18, 2026， [<u>https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/X-Forwarded-For</u>](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/X-Forwarded-For)

22. What Is X-Forwarded-For Spoofing? \| SKUDONET, 檢索日期：1月 18, 2026， [<u>https://www.skudonet.com/blog/what-is-x-forwarded-for-spoofing/</u>](https://www.skudonet.com/blog/what-is-x-forwarded-for-spoofing/)

23. Bypass WAF - PortSwigger, 檢索日期：1月 18, 2026， [<u>https://portswigger.net/bappstore/ae2611da3bbc4687953a1f4ba6a4e04c</u>](https://portswigger.net/bappstore/ae2611da3bbc4687953a1f4ba6a4e04c)

24. 5 Working Methods to Bypass Cloudflare (August 2025 Updated) - Scrape.do, 檢索日期：1月 18, 2026， [<u>https://scrape.do/blog/bypass-cloudflare/</u>](https://scrape.do/blog/bypass-cloudflare/)

25. WAF Release - 2025-08-11 · Changelog - Cloudflare Docs, 檢索日期：1月 18, 2026， [<u>https://developers.cloudflare.com/changelog/2025-08-11-waf-release/</u>](https://developers.cloudflare.com/changelog/2025-08-11-waf-release/)

26. How to Bypass WAF in 2026: Challenges and Solutions - ZenRows, 檢索日期：1月 18, 2026， [<u>https://www.zenrows.com/blog/waf-bypass</u>](https://www.zenrows.com/blog/waf-bypass)

27. How to Bypass Cloudflare in 2026: Top Methods & Scripts - Bright Data, 檢索日期：1月 18, 2026， [<u>https://brightdata.com/blog/web-data/bypass-cloudflare</u>](https://brightdata.com/blog/web-data/bypass-cloudflare)

28. Fuzzing and Bypassing the AWS WAF - Sysdig, 檢索日期：1月 18, 2026， [<u>https://www.sysdig.com/blog/fuzzing-and-bypassing-the-aws-waf</u>](https://www.sysdig.com/blog/fuzzing-and-bypassing-the-aws-waf)

29. 0xhaggis/Imperva_gzip_bypass: Exploit for CVE-2021-45468, an Imperva WAF bypass., 檢索日期：1月 18, 2026， [<u>https://github.com/0xhaggis/Imperva_gzip_bypass</u>](https://github.com/0xhaggis/Imperva_gzip_bypass)

30. How to Bypass Imperva Incapsula when Web Scraping in 2026 - Scrapfly, 檢索日期：1月 18, 2026， [<u>https://scrapfly.io/blog/posts/how-to-bypass-imperva-incapsula-anti-scraping</u>](https://scrapfly.io/blog/posts/how-to-bypass-imperva-incapsula-anti-scraping)

31. Mastering SQLMap and Ghauri: A Practical Guide to WAF Bypass Techniques, 檢索日期：1月 18, 2026， [<u>https://infosecwriteups.com/mastering-sqlmap-and-ghauri-a-practical-guide-to-waf-bypass-techniques-1aaa9eee9d32</u>](https://infosecwriteups.com/mastering-sqlmap-and-ghauri-a-practical-guide-to-waf-bypass-techniques-1aaa9eee9d32)

32. WAF bypass technique — Part 1. Introduction \| by yee-yore - Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/@yee-yore/waf-bypass-technique-part-1-cc01e5639313</u>](https://medium.com/@yee-yore/waf-bypass-technique-part-1-cc01e5639313)

33. Transfer-Encoding \| Fastly Documentation, 檢索日期：1月 18, 2026， [<u>https://www.fastly.com/documentation/reference/http/http-headers/Transfer-Encoding/</u>](https://www.fastly.com/documentation/reference/http/http-headers/Transfer-Encoding/)
