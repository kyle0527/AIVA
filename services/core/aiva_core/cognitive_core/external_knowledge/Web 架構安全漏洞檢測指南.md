# 現代 Web 架構安全性深度剖析：針對 GraphQL、REST API、WebSocket 與 JWT 之自動化漏洞檢測與防禦機制研究報告

## 1. 執行摘要與架構典範轉移

隨著 Web 應用程式架構從傳統的單體式（Monolithic）、伺服器端渲染（Server-Side Rendering, SSR）模式，演進至現代化的前後端分離（Headless）、微服務（Microservices）與分散式架構，網路安全的威脅邊界已發生了根本性的轉移。現代 Web 架構高度依賴 API（Application Programming Interfaces）作為數據交換的核心管道，並採用無狀態（Stateless）的驗證機制以及即時雙向通訊協議。這種架構變革雖然極大提升了開發效率與使用者體驗，但也引入了傳統 Web 弱點掃描器難以觸及的新型攻擊面 <sup>1</sup>。

本研究報告旨在針對現代 Web 架構中的四大核心技術——GraphQL、REST API、WebSocket 以及 JSON Web Tokens (JWT)——進行詳盡的安全分析。研究重點在於剖析五種高風險漏洞：GraphQL 內省機制暴露 (Introspection)、REST API 的大量賦值 (Mass Assignment)、物件層級授權失效 (Broken Object Level Authorization, BOLA/IDOR)、JWT 的 "None" 演算法攻擊，以及跨站 WebSocket 劫持 (Cross-Site WebSocket Hijacking, CSWSH)。

本報告的核心目標是為構建下一代 AI 安全代理（AI Security Agents）提供理論基礎與實踐邏輯，使其能夠自動檢測非傳統 HTML 頁面的邏輯漏洞，並輸出標準化的 JSON 格式報告，以利於安全編排與自動化響應（SOAR）系統的整合。

## 2. 現代 API 安全格局與 AI 檢測的必要性

### 2.1 從結構化掃描到行為分析

傳統的安全掃描工具主要依賴於爬取 HTML 頁面中的超連結（Tags）並對表單輸入進行模糊測試（Fuzzing），這種方法在面對單頁應用程式（SPA）與移動端後端時顯得力不從心。現代應用程式的前端往往只是一個 JavaScript 殼層，真正的業務邏輯與數據流隱藏在背後的 API 調用中 <sup>1</sup>。

此外，現代漏洞更多表現為「邏輯缺陷」而非單純的「語法錯誤」。例如，BOLA 漏洞並非源於輸入驗證不足導致的注入，而是源於伺服器端對請求者權限驗證的缺失；大量賦值漏洞則源於框架對數據綁定的過度便利化 <sup>3</sup>。這些漏洞需要檢測引擎具備對業務上下文（Context）的理解能力，這正是 AI 技術介入的關鍵切入點。

### 2.2 AI 在漏洞檢測中的角色

AI 代理在此情境下的角色不再是簡單的規則匹配，而是進行高階的行為分析：

1.  **語義理解**：理解 API 文檔（如 Swagger/OpenAPI）與 GraphQL Schema，識別敏感欄位（如 isAdmin, balance）。

2.  **上下文關聯**：關聯不同用戶的會話（Session），以測試橫向越權（Horizontal Privilege Escalation）。

3.  **協議解析**：深入解析 WebSocket 握手過程與 JWT 的加密結構，而非僅將其視為隨機字串。

## 3. GraphQL 內省機制與架構偵查

GraphQL 作為一種查詢語言，允許客戶端精確指定所需的數據結構，解決了 REST API 中的過度獲取（Over-fetching）與獲取不足（Under-fetching）問題。然而，其靈活性也將資料庫的關聯結構暴露給了前端，若配置不當，將成為攻擊者的「藏寶圖」<sup>1</sup>。

### 3.1 內省機制 (Introspection) 的運作原理

內省是 GraphQL 的一項元功能（Meta-feature），允許開發者查詢 Schema 本身以獲取類型系統的詳細資訊。這在開發階段對於構建像 GraphiQL 這樣的 IDE 工具至關重要，但在生產環境中，它賦予了攻擊者極大的偵查優勢 <sup>5</sup>。

透過向 GraphQL 端點（通常是 /graphql）發送特定的查詢，攻擊者可以獲取：

- **Queries (查詢)**：所有可讀取的數據入口。

- **Mutations (變更)**：所有可寫入的操作，往往包含敏感的 deleteUser, updateRole 等功能 <sup>6</sup>。

- **Types (類型)**：後端數據模型，包括欄位名稱、類型描述，甚至廢棄欄位（Deprecated Fields）<sup>7</sup>。

### 3.2 漏洞檢測邏輯與 Payload 設計

為了檢測內省機制是否開啟，AI 代理需要構造一個標準的內省查詢 Payload。

#### 3.2.1 基礎檢測 Payload

AI 代理應向目標端點發送以下 JSON 負載：

> JSON

{  
"query": "query IntrospectionQuery { \_\_schema { queryType { name } mutationType { name } subscriptionType { name } types { kind name description fields(includeDeprecated: true) { name description args { name type { kind name } } } } } }"  
}

此查詢請求伺服器返回 Schema 的根類型（Query, Mutation, Subscription）以及所有定義的類型詳情 <sup>5</sup>。

#### 3.2.2 響應分析與判定

- **漏洞存在**：若伺服器返回 HTTP 200 狀態碼，且回應的 JSON 結構中包含 "data": { "\_\_schema": {... } }，則表示內省機制完全開啟。攻擊者可據此繪製完整的 API 地圖 <sup>6</sup>。

- **防禦生效（硬性禁用）**：若伺服器返回錯誤訊息，如 "message": "GraphQL introspection has been disabled" 或 "Cannot query field '\_\_schema'"，則表示該功能已被禁用 <sup>8</sup>。

- **防禦不全（軟性禁用與欄位建議）**：即使內省被禁用，許多 GraphQL 引擎（如 Apollo Server）預設開啟了「欄位建議」（Field Suggestions）功能。當查詢包含拼寫錯誤的欄位時，伺服器會回應 "Did you mean 'users'?"。這允許攻擊者利用工具（如 Clairvoyance）透過模糊測試來「盲測」並重建 Schema，這是一種高階的偵查技術 <sup>9</sup>。

### 3.3 自動化檢測之 JSON 輸出規範

當 AI 代理檢測到內省機制開啟時，應輸出包含風險等級、證據片段與修復建議的結構化報告。

**表 1：GraphQL 內省漏洞 JSON 輸出規範**

> JSON

{  
"vulnerability_report": {  
"type": "GraphQL Introspection Exposure",  
"risk_level": "Medium",  
"confidence_score": 1.0,  
"target_info": {  
"url": "https://api.target.com/graphql",  
"method": "POST"  
},  
"detection_evidence": {  
"payload_sent": "query { \_\_schema { queryType { name } } }",  
"response_snippet": "{\\data\\:{\\\_\_schema\\:{\\queryType\\:{\\name\\:\\Query\\}}}}",  
"indicator": "Presence of '\_\_schema' key in valid JSON response"  
},  
"impact_analysis": {  
"description": "Exposed schema allows attackers to map internal API structure, identify hidden mutations, and enumerate sensitive fields without authentication.",  
"attack_vectors":  
},  
"remediation": {  
"action": "Disable introspection in production environments.",  
"configuration_example": "app.use('/graphql', graphqlHTTP({ schema: mySchema, graphiql: false, validationRules: }));"  
}  
}  
}

## 4. REST API 漏洞分析：大量賦值 (Mass Assignment)

大量賦值（又稱自動綁定 Auto-Binding）是現代 Web 框架（如 Ruby on Rails, Spring Boot, Laravel, Node.js）為了簡化開發而提供的功能，允許將 HTTP 請求參數直接映射到內部物件模型或資料庫欄位 <sup>3</sup>。

### 4.1 漏洞成因與業務邏輯風險

在典型的開發場景中，當使用者提交註冊表單時，前端發送如下 JSON：

{ "username": "alice", "email": "alice@example.com" }

後端框架可能會自動將這些鍵值對綁定到 User 物件並存入資料庫。然而，如果 User 模型中還包含如 isAdmin, role, account_status 等敏感欄位，且開發者未明確設置「允許清單」（Allow-list）或「禁止清單」（Block-list），攻擊者便可透過在請求中注入 { "isAdmin": true } 來篡改這些屬性，從而實現權限提升 <sup>3</sup>。

這類漏洞特別隱蔽，因為從 HTTP 協議層面看，請求完全合法，且通常不會觸發語法錯誤，屬於純粹的業務邏輯缺陷 <sup>3</sup>。

### 4.2 AI 驅動的差異化檢測策略

檢測大量賦值需要 AI 代理具備「屬性探測」（Property Probing）的能力，這通常分為三個階段：

#### 4.2.1 階段一：模型偵查 (Reconnaissance)

AI 首先透過合法的讀取操作（如 GET /api/profile）來獲取目標物件的完整結構。回應中可能包含前端未顯示但後端存在的欄位，例如：

> JSON

{  
"id": 101,  
"username": "user",  
"role": "customer",  
"created_at": "2023-01-01",  
"is_verified": true  
}

在此階段，AI 需識別出潛在的敏感欄位（如 role, is_verified）<sup>2</sup>。

#### 4.2.2 階段二：屬性注入 (Injection)

AI 構造一個更新請求（如 PUT /api/profile 或 PATCH /api/users/101），在包含合法參數的同時，混入在階段一發現或基於字典猜測的敏感參數。

- Payload 範例：{ "username": "user_mod", "role": "admin", "isAdmin": true } <sup>12</sup>。

#### 4.2.3 階段三：狀態驗證 (Verification)

這是最關鍵的一步。伺服器可能返回 200 OK，但實際上忽略了注入的欄位。AI 必須：

1.  **直接反射檢查**：檢查 PUT 的回應是否包含 "role": "admin"。

2.  **二次讀取檢查**：再次發送 GET /api/profile，驗證資料庫中的狀態是否確實被改變 <sup>12</sup>。

3.  **錯誤推斷**：有時注入錯誤的數據類型（如將字串注入布林欄位）會觸發 500 錯誤或特定的框架錯誤，這也間接證實了該欄位是可綁定的 <sup>14</sup>。

### 4.3 自動化檢測之 JSON 輸出規範

**表 2：Mass Assignment 漏洞 JSON 輸出規範**

> JSON

{  
"vulnerability_report": {  
"type": "Mass Assignment",  
"risk_level": "High",  
"confidence_score": 0.9,  
"target_info": {  
"url": "https://api.target.com/api/v1/users/update",  
"method": "PATCH"  
},  
"detection_evidence": {  
"injected_parameters": \[  
{ "key": "is_admin", "value": true },  
{ "key": "role", "value": "administrator" }  
\],  
"verification_method": "State Comparison",  
"state_before": { "is_admin": false, "role": "user" },  
"state_after": { "is_admin": true, "role": "administrator" }  
},  
"impact_analysis": {  
"description": "The API endpoint allows modification of sensitive object properties not intended for user access, leading to privilege escalation.",  
"affected_object": "UserModel"  
},  
"remediation": {  
"action": "Implement Data Transfer Objects (DTOs) or field allow-lists.",  
"framework_specific": "In Laravel use \$fillable; in Spring Boot use @JsonView or separate DTO classes."  
}  
}  
}

## 5. 物件層級授權失效 (BOLA/IDOR)

物件層級授權失效（Broken Object Level Authorization, BOLA），舊稱不安全的直接物件參照（Insecure Direct Object Reference, IDOR），在 OWASP API Security Top 10 中長期位居榜首 <sup>4</sup>。

### 5.1 漏洞機制與分類

BOLA 發生在伺服器端驗證了使用者的「身份」（Authentication），卻未驗證使用者是否擁有訪問特定「資料物件」（Object）的「權限」（Authorization）<sup>16</sup>。

- **基於 ID 的利用**：API 透過 URL 路徑（如 /api/orders/1001）或查詢參數（如 ?userId=55）來識別資源。攻擊者只需遍歷 ID（如將 1001 改為 1002），即可訪問他人資料 <sup>4</sup>。

- **水平越權 (Horizontal)**：訪問同級別其他用戶的資源。

- **垂直越權 (Vertical)**：普通用戶訪問管理員資源 <sup>17</sup>。

### 5.2 AI 驅動的矩陣式檢測邏輯

檢測 BOLA 需要 AI 維護多個用戶上下文（Contexts），進行交叉請求測試，這被稱為「矩陣測試法」<sup>18</sup>。

#### 5.2.1 測試前置準備

AI 需建立兩個低權限使用者會話：

- **User A**：擁有資源 ID Res_A。

- **User B**：擁有資源 ID Res_B。

#### 5.2.2 交叉攻擊執行

AI 使用 User A 的身份憑證（如 JWT 或 Cookie），嘗試請求 Res_B 17。

請求範例：GET /api/invoices/Res_B with Authorization: Bearer Token_A。

#### 5.2.3 結果判定邏輯

- **漏洞確認**：伺服器返回 HTTP 200 OK 且回應內容包含 User B 的私有數據。這表明授權檢查缺失 <sup>16</sup>。

- **安全回應**：伺服器返回 HTTP 403 Forbidden（表示已認證但無權限）或 HTTP 404 Not Found（為防止 ID 枚舉，有時會隱藏無權限資源的存在）<sup>4</sup>。

#### 5.2.4 ID 熵值分析 (Entropy Analysis)

AI 應分析資源 ID 的格式。若是連續整數（Sequential Integers），則極易受到大規模枚舉攻擊；若是 UUID/GUID，雖然難以猜測，但若 ID 在其他端點（如列表頁面）洩露，BOLA 風險依然存在 <sup>15</sup>。

### 5.3 自動化檢測之 JSON 輸出規範

**表 3：BOLA 漏洞 JSON 輸出規範**

> JSON

{  
"vulnerability_report": {  
"type": "Broken Object Level Authorization (BOLA)",  
"risk_level": "Critical",  
"confidence_score": 0.95,  
"target_info": {  
"url": "https://api.target.com/api/orders/{order_id}",  
"method": "GET"  
},  
"detection_evidence": {  
"attacker_context": "User_A",  
"victim_resource_id": "Order_1002 (Belongs to User_B)",  
"response_status": 200,  
"data_leakage_detected": true,  
"response_sample": "{\\order_id\\: \\1002\\, \\owner\\: \\User_B\\, \\cc_last4\\: \\4242\\}"  
},  
"impact_analysis": {  
"description": "Authenticated users can access resources belonging to other users by manipulating the resource ID.",  
"severity_vector": "Data Breach / Privacy Violation"  
},  
"remediation": {  
"action": "Enforce object-level permission checks in controllers.",  
"logic": "Ensure 'resource.owner_id == current_user.id' before returning data."  
}  
}  
}

## 6. JWT 安全缺陷：None 演算法與弱密鑰

JSON Web Tokens (JWT) 是現代無狀態架構中最常見的驗證載體。JWT 由 Header、Payload 和 Signature 三部分組成，並以 Base64Url 編碼，中間以點號（.）分隔 <sup>19</sup>。

### 6.1 "None" 演算法攻擊 (The "None" Algorithm Bypass)

#### 6.1.1 歷史背景與機制

JWT 規範（RFC 7519）定義了一個名為 none 的簽名演算法，本意是用於不需要簽名的調試場景 20。在 none 模式下，JWT 僅包含 Header 和 Payload，Signature 部分為空。

早期或配置不當的 JWT 庫在驗證 Token 時，如果 Header 中指定 alg: none，會直接跳過簽名驗證並信任 Payload 中的內容 19。

#### 6.1.2 攻擊構造流程

AI 代理檢測此漏洞的步驟如下：

1.  **解碼**：獲取合法 Token，解碼 Header。

2.  **篡改 Header**：將 alg 欄位改為 none。AI 應測試變體以繞過簡單的字串過濾，如 None, NONE, nOnE <sup>22</sup>。

3.  **篡改 Payload**：將 admin: false 改為 true，或修改 user_id。

4.  **重組 Token**：將新的 Header 和 Payload 進行 Base64Url 編碼。

5.  **簽名剝離**：將編碼後的 Header 和 Payload 以點號連接，並在最後加上一個點號（表示簽名為空），格式為 header.payload. <sup>19</sup>。

6.  **發送測試**：將偽造的 Token 置於 Authorization Header 發送請求。若伺服器接受，則漏洞存在。

### 6.2 弱密鑰爆破 (Weak Secret Keys)

對於使用 HMAC-SHA256 (HS256) 對稱加密的 JWT，其安全性完全依賴於伺服器端的 Secret Key。

#### 6.2.1 離線攻擊邏輯

與其他漏洞不同，此檢測可完全在「離線」狀態下進行。AI 代理提取 Token 的簽名部分，使用常見弱密碼字典（如 "secret", "123456", "key"）進行暴力破解 20。

公式：Signature = HMAC-SHA256(Base64(Header) + "." + Base64(Payload), Secret)

若計算出的簽名與 Token 中的簽名一致，則密鑰即被破解，攻擊者隨後可任意偽造 Token。

### 6.3 密鑰混淆攻擊 (Algorithm Confusion)

這是一種更高階的攻擊。當伺服器支援非對稱加密（如 RS256）但也支援對稱加密（HS256）時，攻擊者可將 Header 中的 alg 改為 HS256，並使用伺服器的**公鑰**（Public Key）作為 HMAC 的 Secret 進行簽名 <sup>20</sup>。由於公鑰通常是公開的，伺服器若未強制驗證演算法類型，可能會用公鑰去解密（驗證）HMAC 簽名，導致驗證通過。

### 6.4 自動化檢測之 JSON 輸出規範

**表 4：JWT None 演算法漏洞 JSON 輸出規範**

> JSON

{  
"vulnerability_report": {  
"type": "JWT None Algorithm Bypass",  
"risk_level": "Critical",  
"confidence_score": 1.0,  
"target_info": {  
"component": "Authorization Header",  
"token_type": "Bearer"  
},  
"detection_evidence": {  
"original_algorithm": "HS256",  
"manipulated_header": "{\\alg\\:\\none\\,\\typ\\:\\JWT\\}",  
"forged_token_structure": "eyJhbGciOiJub25l....eyJ1c2VyIjoiYWRtaW4i....",  
"server_response": "200 OK (Access Granted)"  
},  
"impact_analysis": {  
"description": "Server accepts unsigned tokens, allowing arbitrary claim manipulation (Identity Spoofing / Privilege Escalation).",  
"exploitability": "Trivial"  
},  
"remediation": {  
"action": "Explicitly reject the 'none' algorithm in JWT library validation.",  
"code_fix": "jwt.verify(token, secret, { algorithms: })"  
}  
}  
}

## 7. WebSocket 安全：跨站 WebSocket 劫持 (CSWSH)

WebSocket 提供了持久化的全雙工通訊，是即時聊天、股票報價等應用的核心。然而，WebSocket 的握手過程（Handshake）本質上是一個 HTTP 請求，這使其繼承了 HTTP 的部分弱點，特別是跨站請求偽造（CSRF）的變體——跨站 WebSocket 劫持（CSWSH）<sup>26</sup>。

### 7.1 握手協議與漏洞機制

WebSocket 連線始於一個 HTTP Upgrade 請求：

> HTTP

GET /chat HTTP/1.1  
Host: target.com  
Connection: Upgrade  
Upgrade: websocket  
Cookie: session=xyz  
Origin: https://target.com

瀏覽器會自動在請求中帶上該域名的 Cookies（除非設定了 SameSite 屬性）。如果使用者登入了 target.com，然後訪問了攻擊者的網站 evil.com，evil.com 上的 JavaScript 可以發起向 ws://target.com/chat 的連線請求。此時，瀏覽器會發送帶有 Origin: https://evil.com 但包含 target.com Cookies 的請求 <sup>26</sup>。

如果伺服器僅依賴 Cookie 進行驗證，而沒有檢查 Origin Header，連線將會建立。攻擊者隨後可透過此 WebSocket 管道發送和接收數據，實現雙向劫持 <sup>27</sup>。

### 7.2 AI 檢測邏輯與狀態碼分析

AI 代理需模擬一個惡意的跨域握手請求。

#### 7.2.1 握手模擬

構造請求，刻意將 Origin 設置為第三方域名（如 https://attacker.com）<sup>26</sup>。

#### 7.2.2 響應狀態碼的深層解讀

- **HTTP 101 Switching Protocols**：這是最危險的信號。表示伺服器接受了協議升級，建立了 WebSocket 連線，漏洞存在 <sup>26</sup>。

- **HTTP 403 Forbidden**：表示伺服器在 HTTP 握手階段就檢查了 Origin 並拒絕了請求，這是安全的表現 <sup>30</sup>。

- **WebSocket Close Code 1008 (Policy Violation)**：這是一種細微的情況。伺服器可能先建立連線（返回 101），但隨即在 WebSocket 協議層發送一個 Close Frame，代碼為 1008。這表示伺服器進行了延遲檢查（Post-handshake check）。雖然連線曾短暫建立，但通常無法進行有效數據交換，屬於相對安全的防禦，但也可能暗示存在時間競爭（Race Condition）的風險 <sup>32</sup>。

### 7.3 自動化檢測之 JSON 輸出規範

**表 5：CSWSH 漏洞 JSON 輸出規範**

> JSON

{  
"vulnerability_report": {  
"type": "Cross-Site WebSocket Hijacking (CSWSH)",  
"risk_level": "High",  
"confidence_score": 0.9,  
"target_info": {  
"endpoint": "wss://api.target.com/chat",  
"handshake_method": "GET"  
},  
"detection_evidence": {  
"injected_origin": "https://random-attacker.com",  
"auth_mechanism": "Cookie-based",  
"handshake_response_code": 101,  
"connection_state": "ESTABLISHED",  
"message_exchange_possible": true  
},  
"impact_analysis": {  
"description": "Attacker can initiate a WebSocket connection from a malicious origin using the victim's credentials, allowing two-way communication hijacking.",  
"prerequisites": "Cookie-based authentication without SameSite=Strict or CSRF tokens."  
},  
"remediation": {  
"action": "Validate the 'Origin' header during the handshake.",  
"code_fix": "if (request.getHeader('Origin')!== 'https://trusted.com') return 403;"  
}  
}  
}

## 8. 結論與戰略展望

本報告詳細分析了現代 Web 架構中五種最具代表性的安全漏洞。這些漏洞的共同特徵在於它們多發生在**邏輯層**與**配置層**，而非傳統的輸入過濾層。

1.  **GraphQL Introspection** 顯示了開發便利性與安全隱私之間的矛盾，AI 檢測需關注 Schema 的可重建性。

2.  **Mass Assignment** 與 **BOLA** 揭示了自動化框架與微服務架構在權限控制上的盲點，強調了上下文感知（Context-Aware）測試的重要性。

3.  **JWT 漏洞** 提醒我們，加密協議的安全性不僅取決於演算法本身，更取決於實作的嚴謹性與庫的預設配置。

4.  **CSWSH** 則凸顯了在持久化連線中，傳統 HTTP 安全策略（如 SOP）的邊界模糊問題。

對於 AI 安全代理而言，未來的檢測能力必須從靜態的 Payload 發送，進化為動態的、多步驟的行為分析。這包括對 API 文檔的自然語言理解、對業務邏輯的狀態追蹤，以及對加密結構的深度解析。唯有如此，才能在快速迭代的現代 Web 開發流程中，建立起有效的安全防線。

### 9. 附錄：檢測邏輯摘要表

| **漏洞類型** | **核心檢測邏輯** | **關鍵識別特徵** | **AI 代理所需能力** |
|----|----|----|----|
| **GraphQL Introspection** | 發送 \_\_schema 查詢 | 回應含完整 Schema JSON | JSON 解析、遞歸查詢構造 |
| **Mass Assignment** | 差異化屬性注入 | 響應反射或狀態改變 | 上下文理解、屬性預測 |
| **BOLA (IDOR)** | 多帳號交叉請求 | 跨用戶讀取成功 (200 OK) | 多會話管理、ID 規律分析 |
| **JWT None Algo** | 修改 Header alg: none | 接受無簽名 Token | Base64Url 編解碼、結構重組 |
| **CSWSH** | 偽造 Origin Header | 握手返回 101 Switching Protocols | WebSocket 協議握手模擬 |

透過上述標準化的檢測邏輯與 JSON 輸出規範，安全團隊可將此分析模組整合至 CI/CD 流程或自動化滲透測試平台中，實現對現代 Web 應用程式的持續性安全監控。

#### 引用的著作

1.  Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks - arXiv, 檢索日期：1月 18, 2026， [<u>https://arxiv.org/html/2508.11711v2</u>](https://arxiv.org/html/2508.11711v2)

2.  API testing \| Web Security Academy - PortSwigger, 檢索日期：1月 18, 2026， [<u>https://portswigger.net/web-security/api-testing</u>](https://portswigger.net/web-security/api-testing)

3.  What is Mass Assignment? \| Glossary - A10 Networks, 檢索日期：1月 18, 2026， [<u>https://www.a10networks.com/glossary/what-is-mass-assignment/</u>](https://www.a10networks.com/glossary/what-is-mass-assignment/)

4.  What is Broken Object Level Authorization? \| Indusface Blog, 檢索日期：1月 18, 2026， [<u>https://www.indusface.com/blog/owasp-api1-2019-broken-object-level-authorization/</u>](https://www.indusface.com/blog/owasp-api1-2019-broken-object-level-authorization/)

5.  Introspection queries - Tyk API Management, 檢索日期：1月 18, 2026， [<u>https://tyk.io/docs/5.0/graphql/introspection/introspection-queries/</u>](https://tyk.io/docs/5.0/graphql/introspection/introspection-queries/)

6.  Exploiting GraphQL Introspection: Mapping the API Like an Insider - DEV Community, 檢索日期：1月 18, 2026， [<u>https://dev.to/crud5th-273-/exploiting-graphql-introspection-mapping-the-api-like-an-insider-2i0k</u>](https://dev.to/crud5th-273-/exploiting-graphql-introspection-mapping-the-api-like-an-insider-2i0k)

7.  Introspection - GraphQL, 檢索日期：1月 18, 2026， [<u>https://graphql.org/learn/introspection/</u>](https://graphql.org/learn/introspection/)

8.  Disable Introspection - Strawberry GraphQL, 檢索日期：1月 18, 2026， [<u>https://strawberry.rocks/docs/extensions/disable-introspection</u>](https://strawberry.rocks/docs/extensions/disable-introspection)

9.  A Comprehensive Guide to GraphQL Introspection - Escape DAST, 檢索日期：1月 18, 2026， [<u>https://escape.tech/blog/should-i-disable-introspection-in-graphql/</u>](https://escape.tech/blog/should-i-disable-introspection-in-graphql/)

10. Disable field suggestions when introspection disabled · Issue \#454 · webonyx/graphql-php, 檢索日期：1月 18, 2026， [<u>https://github.com/webonyx/graphql-php/issues/454</u>](https://github.com/webonyx/graphql-php/issues/454)

11. What is mass assignment? \| Tutorial & examples - Snyk Learn, 檢索日期：1月 18, 2026， [<u>https://learn.snyk.io/lesson/mass-assignment/</u>](https://learn.snyk.io/lesson/mass-assignment/)

12. Mass Assignment Vulnerabilities in APIs — Explained for Bug Bounty Hunters - Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/@merida-/mass-assignment-vulnerabilities-in-apis-explained-for-bug-bounty-hunters-1c84c9b06204</u>](https://medium.com/@merida-/mass-assignment-vulnerabilities-in-apis-explained-for-bug-bounty-hunters-1c84c9b06204)

13. Mass assignment \| APIs and the OWASP Top 10 guide - My F5, 檢索日期：1月 18, 2026， [<u>https://my.f5.com/manage/s/article/K51142652</u>](https://my.f5.com/manage/s/article/K51142652)

14. Testing for Mass Assignment - WSTG - Latest \| OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/20-Testing_for_Mass_Assignment</u>](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/20-Testing_for_Mass_Assignment)

15. Understanding and Protecting Against API1: Broken Object Level Authorization - StackHawk, 檢索日期：1月 18, 2026， [<u>https://www.stackhawk.com/blog/understanding-and-protecting-against-api1-broken-object-level-authorization/</u>](https://www.stackhawk.com/blog/understanding-and-protecting-against-api1-broken-object-level-authorization/)

16. API Broken Object Level Authorization - WSTG - Latest \| OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/12-API_Testing/02-API_Broken_Object_Level_Authorization</u>](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/12-API_Testing/02-API_Broken_Object_Level_Authorization)

17. How to Prevent and Fix Broken Object Level Authorization (BOLA) in APIs - Astra Security, 檢索日期：1月 18, 2026， [<u>https://www.getastra.com/blog/api-security/broken-object-level-authorization-bola/</u>](https://www.getastra.com/blog/api-security/broken-object-level-authorization-bola/)

18. Kill BOLAs Before They Escape: Prevent API Authorization Vulnerabilities - Aptori, 檢索日期：1月 18, 2026， [<u>https://www.aptori.com/guide/kill-bolas-prevent-api-authorization-vulnerabilities</u>](https://www.aptori.com/guide/kill-bolas-prevent-api-authorization-vulnerabilities)

19. Common JWT Vulnerabilities \| CodeSignal Learn, 檢索日期：1月 18, 2026， [<u>https://codesignal.com/learn/courses/jwt-security-attacks-defenses-1/lessons/common-jwt-vulnerabilities</u>](https://codesignal.com/learn/courses/jwt-security-attacks-defenses-1/lessons/common-jwt-vulnerabilities)

20. The Ultimate Guide to JWT Vulnerabilities and Attacks (with ..., 檢索日期：1月 18, 2026， [<u>https://pentesterlab.com/blog/jwt-vulnerabilities-attacks-guide</u>](https://pentesterlab.com/blog/jwt-vulnerabilities-attacks-guide)

21. JWT none algorithm supported - PortSwigger, 檢索日期：1月 18, 2026， [<u>https://portswigger.net/kb/issues/00200901_jwt-none-algorithm-supported</u>](https://portswigger.net/kb/issues/00200901_jwt-none-algorithm-supported)

22. Testing JSON Web Tokens - WSTG - Latest \| OWASP Foundation, 檢索日期：1月 18, 2026， [<u>https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/06-Session_Management_Testing/10-Testing_JSON_Web_Tokens</u>](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/06-Session_Management_Testing/10-Testing_JSON_Web_Tokens)

23. Exploiting JWT Vulnerabilities: Advanced Exploitation Guide - Intigriti, 檢索日期：1月 18, 2026， [<u>https://www.intigriti.com/researchers/blog/hacking-tools/exploiting-jwt-vulnerabilities</u>](https://www.intigriti.com/researchers/blog/hacking-tools/exploiting-jwt-vulnerabilities)

24. Signature Bypass None Algorithm Key confusion - NashTech Blog, 檢索日期：1月 18, 2026， [<u>https://blog.nashtechglobal.com/abusing-jwts-signature-bypass-none-algorithm-key-confusion/</u>](https://blog.nashtechglobal.com/abusing-jwts-signature-bypass-none-algorithm-key-confusion/)

25. Security issues in JWT Tokens - Medium, 檢索日期：1月 18, 2026， [<u>https://medium.com/@monethic/security-issues-in-jwt-tokens-d98b1afca423</u>](https://medium.com/@monethic/security-issues-in-jwt-tokens-d98b1afca423)

26. Cross-Site Websocket Hijacking (CSWSH) - Praetorian, 檢索日期：1月 18, 2026， [<u>https://www.praetorian.com/blog/cross-site-websocket-hijacking-cswsh/</u>](https://www.praetorian.com/blog/cross-site-websocket-hijacking-cswsh/)

27. Cross-site WebSocket hijacking \| Web Security Academy, 檢索日期：1月 18, 2026， [<u>https://portswigger.net/web-security/websockets/cross-site-websocket-hijacking</u>](https://portswigger.net/web-security/websockets/cross-site-websocket-hijacking)

28. Cross-Site WebSocket Hijacking (CSWSH), 檢索日期：1月 18, 2026， [<u>https://christian-schneider.net/blog/cross-site-websocket-hijacking/</u>](https://christian-schneider.net/blog/cross-site-websocket-hijacking/)

29. WebSocket connection failed: Error during WebSocket handshake: Unexpected response code: 400 - Stack Overflow, 檢索日期：1月 18, 2026， [<u>https://stackoverflow.com/questions/41381444/websocket-connection-failed-error-during-websocket-handshake-unexpected-respon</u>](https://stackoverflow.com/questions/41381444/websocket-connection-failed-error-during-websocket-handshake-unexpected-respon)

30. WebSocket upgrade failure: 403 or 404 and routes fail to load in PingGateway, 檢索日期：1月 18, 2026， [<u>https://support.pingidentity.com/s/article/WebSocket-upgrade-failure-403-or-404-and-routes-fail-to-load-in-PingGateway</u>](https://support.pingidentity.com/s/article/WebSocket-upgrade-failure-403-or-404-and-routes-fail-to-load-in-PingGateway)

31. Error during WebSocket handshake: Unexpected response code: 403 - Stack Overflow, 檢索日期：1月 18, 2026， [<u>https://stackoverflow.com/questions/39627017/error-during-websocket-handshake-unexpected-response-code-403</u>](https://stackoverflow.com/questions/39627017/error-during-websocket-handshake-unexpected-response-code-403)

32. 檢索日期：1月 18, 2026， [<u>https://kapeli.com/cheat_sheets/WebSocket_Status_Codes.docset/Contents/Resources/Documents/index#:~:text=1008%20indicates%20that%20an%20endpoint,specific%20details%20about%20the%20policy.</u>](https://kapeli.com/cheat_sheets/WebSocket_Status_Codes.docset/Contents/Resources/Documents/index#:~:text=1008%20indicates%20that%20an%20endpoint,specific%20details%20about%20the%20policy.)

33. CloseStatus (Spring Framework 7.0.2 API), 檢索日期：1月 18, 2026， [<u>https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/web/socket/CloseStatus.html</u>](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/web/socket/CloseStatus.html)

34. WebSocket messages and status codes - AWS IoT Wireless, 檢索日期：1月 18, 2026， [<u>https://docs.aws.amazon.com/iot-wireless/latest/developerguide/network-analyzer-messages-status.html</u>](https://docs.aws.amazon.com/iot-wireless/latest/developerguide/network-analyzer-messages-status.html)
