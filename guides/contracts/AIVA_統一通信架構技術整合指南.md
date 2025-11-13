# AIVA 統一通信架構技術整合指南

本指南詳細說明如何在 AIVA 平台中構建並導入**統一的通信架構**，將現有的
CLI 工具調用、JSON 消息隊列（MQ）和 gRPC
通信三種風格整合為一致的體系。透過單一資料合約來源驅動，結合統一的訊息封裝和契約代碼生成工具，可實現跨語言的低耦合協作通信。下文將依序介紹架構整合目標、方案設計、通道整合方式、實作工具鏈、優缺點分析、預期效益，以及分階段的推動建議。

## 架構整合目標：統一 CLI/JSON-MQ/gRPC 協定風格

**目標：**消弭 AIVA
系統中現有多種通信風格的割裂情形，統一為以**資料合約為核心**的通信架構。當前
AIVA 平台存在三種主要通信方式：

- **CLI 工具輸出：**某些安全掃描功能透過調用獨立的 CLI
  工具並解析其輸出（通常為 JSON）來獲取結果。
- **JSON-MQ 非同步消息：**核心服務之間利用消息隊列（如 RabbitMQ）傳遞
  JSON 格式的任務請求與結果，在 Python、Go、Rust
  等模組間進行跨語言調用[\[1\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L2-L5)。
- **gRPC 同步接口：**部分服務採用 gRPC
  定義嚴格的接口，以同步遠程調用或串流方式交換資料[\[1\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L2-L5)。

由於上述渠道各自為政，導致資料模型重複定義、轉換邏輯複雜，增加了維護成本和錯誤風險。**統一通信架構的目標**在於建立單一資料契約來源，讓
CLI、MQ、gRPC
**共享相同的資料模型和協定**，透過標準信封封裝和一致的Topic路由，實現各模組在不同通訊方式下的**行為一致性**。最終，開發人員無論使用哪種通道，都可遵循同一套資料結構與約定，降低跨語言整合難度[\[1\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L2-L5)。

## 整合方案設計：單一 SoT 產生資料模型與協定

為達成上述目標，AIVA
採取**契約先行**（Contract-First）的設計方案：使用**單一事實來源（SoT）**的架構描述來自動生成各語言的資料模型和通訊契約，以確保一致性。

- **單一資料合約來源 --
  core_schema_sot.yaml：**建立一個集中定義所有跨語言共享結構的 YAML 檔案
  **core_schema_sot.yaml**[\[2\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L44-L52)。此檔案成為唯一可信來源，記載各種資料模型（如任務請求、漏洞資訊、結果結構）以及枚舉類型、訊息信封格式等。未來所有語言的資料結構變更都在此維護，再由工具同步到各語言實現。這落實了
  *Single Source of Truth*
  原則[\[2\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L44-L52)。

- **自動合約代碼生成：**透過 **schema_codegen_tool.py** 工具，從
  core_schema_sot.yaml
  自動產生各語言的資料模型代碼[\[3\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L11-L19)。目前支援
  Python (Pydantic v2 模型)、TypeScript (介面定義)、Go (struct
  結構)、Rust (Serde
  結構)，確保四種語言的結構完全對齊[\[3\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L11-L19)。未來亦可擴充產生
  Protocol Buffers 定義，用於 gRPC 介面契約。每當 SoT
  定義更新，只需重新運行生成腳本，所有語言的模型即同步更新，杜絕人工同步誤差[\[4\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L40-L48)。例如，工具會根據
  YAML 定義自動生成 Python 的 Pydantic 模型檔案、Go
  的結構體檔案等並放入專案內對應路徑[\[5\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L50-L54)。

- **標準訊息信封 (Envelope)：**設計一種統一的消息封裝格式
  **AivaMessage**，用於封裝所有透過消息隊列傳遞的資料[\[6\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L15-L23)。AivaMessage
  包含標準訊息頭部（例如 message_id、trace_id、來源模組、時間戳等）以及
  Payload 載荷和 Topic
  主題等欄位[\[7\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L18-L21)。所有透過
  MQ
  傳遞的請求與結果均放入此信封中，確保訊息結構一致且附帶必要的上下文資訊[\[6\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L15-L23)。這樣，不同服務透過
  MQ 溝通時，都以 AivaMessage
  作為基本單位進行序列化和反序列化，大幅降低解析不一致的風險。同時
  AivaMessage 中的 `schema_version`
  欄位可標記資料結構版本，方便版本演進管理[\[7\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L18-L21)。

- **Topic 命名規則統一：**制定 **統一的 Topic
  命名慣例**並以枚舉列舉方式管理[\[8\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L250-L259)。例如，約定將任務請求類Topic以
  `tasks.<模組類別>.<動作>` 命名，結果類Topic以
  `results.<模組類別>.<結果>` 命名，事件通知以 `events.<領域>.<事件>`
  命名，命令請求以 `commands.<領域>.<動作>`
  命名等[\[9\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L260)。實際實例如下：掃描任務開始請求為
  **tasks.scan.start**，掃描完成結果為
  **results.scan.completed**，功能模組（如XSS檢測）任務為
  **tasks.function.xss**，其結果為 **results.function.completed**
  等[\[9\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L260)。所有這些Topic字串在
  `aiva_common.enums.Topic`
  枚舉中定義，程式碼中只能使用該枚舉值，避免手寫錯誤[\[8\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L250-L259)。統一的命名規則使得訊息路由清晰，發布/訂閱雙方對Topic含義有一致認知。

- **標準枚舉使用策略：**將所有常用常數類型（狀態、嚴重性級別、任務類型等）提取為枚舉定義，並在合約中引用。比如漏洞嚴重度Severity、任務狀態TaskStatus等在
  aiva_common 的 enums
  套件集中定義[\[10\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/common.py#L8-L16)[\[11\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/common.py#L22-L30)。開發人員在各語言中都使用由
  SoT
  生成的枚舉類，而非魔術字串，確保語意一致[\[12\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L57)。此外，資料驗證上嚴格限定只能使用這些枚舉值，違反則在模型層面拋錯，提高數據品質[\[12\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L57)。例如，上述
  Topic
  就是用枚舉類型，開發者只能從Topic枚舉選擇有效值[\[13\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L11-L19)。這種策略讓**「約定優於配置」**落實在代碼層，所有模組遵循統一的枚舉定義，可避免因不同人理解不同造成的參數不匹配。

整體而言，透過單一合約來源 + 自動代碼生成 + 統一信封與Topic規範 +
標準枚舉的組合方案，AIVA
在架構上建立了一套**跨語言一致的通信契約**。每個模組無論經由 CLI、MQ
還是 gRPC 通信，承載的資料都符合同一份結構定義，真正實現
*"一處定義，各處通用"*[\[2\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L44-L52)[\[14\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L99-L103)。

## 三大通道整合方式

針對 CLI 工具、MQ 非同步通信、gRPC 服務這三種主要通道，AIVA
規劃了相應的整合方式，使其都納入統一架構下運作：

- **MQ + Envelope（非同步任務與結果主幹）：**消息隊列將繼續擔當系統
  **非同步任務調度與結果傳遞的主幹**。整合後，所有進入 MQ
  的消息都採用前述 **AivaMessage**
  信封封裝[\[6\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L15-L23)。例如，Python
  核心發起一個 Go 編寫的掃描模組任務時，會將任務參數打包為
  AivaMessage，Topic 設置為對應的任務Topic（如
  `tasks.scan.start`），並發布到
  MQ[\[9\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L260)。Go
  微服務訂閱該Topic後取出消息，經由生成的結構解析出 Payload
  執行任務，然後將結果再封裝回 AivaMessage，Topic 改為結果Topic（如
  `results.scan.completed`），發回 MQ 供 Python
  核心接收處理[\[15\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L254-L262)。這種模式下，**MQ
  成為不同語言模組間可靠的事件總線**，以**JSON 格式**承載標準信封（由
  Pydantic/Serde
  等驗證），實現異步任務的解耦處理和結果聚合。由於所有MQ消息都帶有
  header 的 trace_id、source_module
  等資訊，系統可以追蹤任務全鏈路，進行調試和監控[\[7\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L18-L21)。整合方案中
  MQ 重點在於替換原有各模組自定義的JSON結構為統一的 AivaMessage，並統一
  Routing Key (Topic) 命名，以達到 **"格式統一、路由明確"** 的目的。

- **gRPC（同步通道與串流）：**在統一架構下，引入或強化 **gRPC**
  作為系統的**同步通信通道**。對於需要即時交互或雙向串流的場景（例如前端調用後端服務、長時任務的進度串流等），gRPC
  能提供高效的二進制傳輸和嚴格的接口約定。透過 SoT
  定義的資料模型，可以自動產生 .proto 檔案，再經由 `protoc`
  工具生成各語言的 gRPC
  接口代碼[\[16\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L994-L1002)。例如，根據
  core_schema_sot.yaml 中定義的 AivaMessage 和相關 Payload，生成
  `aiva_services.proto` 內含 message 和 service 定義，然後編譯出
  Python、Go、TypeScript、Rust
  的存根代碼[\[17\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L996-L1004)。Python
  核心可啟動 gRPC 伺服，同時 Go、TS 等也可各自實現 gRPC
  客戶端或伺服端，以需要時互相直連。**同步通道用途**包括：提供前端
  (TypeScript) 即時調用後端 (Python)
  的介面、在微服務間執行需要立即返回結果的請求，以及串流傳輸如掃描日誌、進度更新等[\[18\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L3-L5)。例如，前端可透過
  gRPC
  呼叫後端的`RunTool`方法，同步獲取執行結果；或訂閱一個`StreamProgress`串流來持續獲得進度消息。由於
  gRPC 基於**IDL**（接口定義語言）生成，多語言的一致性由工具保障，再配合
  SoT 的模型來源，可確保 **"編譯期即發現不兼容"**。需要注意的是，引入
  gRPC 也帶來部署上的複雜度，如服務註冊發現、負載均衡等，但透過 AIVA
  內建的服務發現機制可緩解此問題[\[19\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L148-L152)。總之，gRPC
  整合賦予 AIVA **高效同步與串流通信能力**，彌補 MQ 在即時性方面的不足。

- **Tool-Host（CLI 工具封裝為 gRPC 介面）：**為了統一 CLI
  工具的使用方式，AIVA 架構引入 **Tool-Host 微服務**
  的概念，即將原先以子進程CLI方式運行的安全工具，封裝在獨立服務中並提供
  gRPC API。每個重要的外部工具（如 Nmap、sqlmap 等）可以各自有對應的
  Tool-Host
  服務，或由單一服務統一調度。封裝時，**將工具原本的命令行參數和輸出映射到標準的資料模型**：例如，把
  sqlmap 所需參數定義為結構化的 SQLiTaskPayload，結果輸出映射為
  FindingPayload 列表等，皆寫入 SoT 模型。Tool-Host 服務實現 gRPC
  介面，如
  `RunSqlmap(TaskPayload) returns (FindingResult)`，在內部啟動對應 CLI
  程序執行，收集輸出並轉換為標準結果物件返回。這種方式下，其他模組不再直接呼叫系統shell或解析CLI輸出，而是透過
  gRPC 調用
  Tool-Host，就像調用普通微服務一樣[\[20\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L76-L84)。**優點**是將
  CLI
  工具納入統一通信體系：所有工具變成受管的服務，享受標準合約及錯誤處理機制（例如超時、重試策略），並可隨系統擴展自動部署。對開發者而言，使用封裝後的工具如同調用
  AIVA 內建功能，不必關心底層是透過 CLI
  實現。這不僅統一了介面，也方便後續替換或升級工具實現（例如日後用原生程式碼重寫某功能，只要保持介面相同即可無縫替換）。需要注意封裝帶來的性能開銷：每次調用需啟動進程，可能較直接內嵌調用略慢。但可透過長駐服務或預熱機制優化。此外，Tool-Host
  服務部署需考慮工具的環境相依性（例如安裝路徑、權限），可以使用容器封裝來隔離。總體而言，Tool-Host
  的整合使各種 CLI 安全工具成為**標準化服務**，徹底消除了 AIVA
  內"工具各自為政"的狀況，所有能力皆可透過統一協定訪問。

以上三種通道的整合，確保無論是**任務異步調度**（MQ）還是**同步查詢**（gRPC）、或**工具使用**（CLI），都遵循統一的資料結構和協議約定。在這三條路徑背後，實際承載的都是相同的契約：MQ與Tool-Host返回的皆是
FindingPayload 等標準模型，gRPC 接口傳輸的也是與 MQ
信封相同的內容。開發人員可以在需要時靈活選擇通道而無需轉換資料格式，極大提升了跨模組協作效率。

## 實作工具鏈

要順利落地上述統一架構，需要一套完善的實作工具鏈來支撐合約的生成、驗證與持續整合。AIVA
已經或計畫引入以下關鍵工具和流程：

- **Schema Codegen
  工具（schema_codegen_tool.py）：**此為核心的代碼自動生成工具[\[21\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L10-L18)。它讀取
  core_schema_sot.yaml，按照其中定義的結構和生成規則，輸出各語言對應的資料模型代碼[\[22\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L14-L19)。例如，可一鍵生成
  Python 的 Pydantic 模型類檔案到
  `services/aiva_common/schemas/generated/`[\[23\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L142-L150)；Go
  的資料結構檔案到
  `.../aiva_common_go/schemas/generated/`[\[24\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L150-L158)；Rust
  的模型檔到對應 crate；TypeScript 介面到 npm 套件等。透過 Jinja2
  模板機制，確保生成代碼符合各語言習慣（如 Go struct 自動加上 JSON
  tag）。目前該工具已封裝成 AIVA Converters Plugin
  的一部分，並支援命令列參數控制生成語言或輸出路徑[\[25\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L92-L100)。這使得**維護者無需手動撰寫重覆的模型代碼**，只關注
  YAML
  定義即可。[\[3\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L11-L19)

- **多語言自動產生與綁定：**工具鏈支援**多語言同步**：當開發者修改 SoT
  中的某個模型（如新增欄位），可立刻重新生成，使 Python、Go、Rust、TS
  的對應類/結構出現新欄位。一致的結構令跨語言調用變得自然------例如
  Python 將 Pydantic 對象序列化為 JSON，Go 用生成的 struct
  反序列化即可[\[26\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L28-L32)[\[14\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L99-L103)。除此之外，SoT
  也可用來生成**Protocol Buffers (.proto)** 合約，以便於 gRPC
  框架使用[\[16\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L994-L1002)。這通常透過在
  YAML 中加入對應 proto
  模板來實現。未來甚至可以擴展產生更多格式的契約，如 OpenAPI
  規範文件，從而自動生成 REST API 文檔和客戶端
  SDK。**跨語言驗證器**（CrossLanguageValidator）也在工具鏈中，用於比對不同語言生成結果的一致性，以防範生成模板問題[\[27\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L38-L46)。

- **CI 合約自動檢核：**將合約生成與驗證步驟融入 CI 流程中。一旦偵測到
  core_schema_sot.yaml 或相關模型代碼有變更，CI Pipeline 會自動執行
  schema_codegen_tool.py
  生成最新代碼[\[28\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L182-L188)。接著運行**合約一致性測試**，例如確定
  Python 模型和 Go 模型的關鍵欄位都存在且類型匹配，檢查所有枚舉值與 SoT
  定義一致等[\[29\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L59)。只有當生成和測試全部通過時，才允許合併變更，從流程上杜絕了"有人忘了同步某語言模型"的情況[\[28\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L182-L188)。此外，在
  CI
  中還加入**契約覆蓋率檢查**，即統計各模組使用標準契約的程度，避免出現繞過合約私下傳遞原始JSON的做法[\[30\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L20-L28)[\[31\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L33-L38)。比如近期的一份合約健康報告顯示核心合約導入正常、序列化/反序列化100%完整[\[32\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L13-L21)[\[33\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L34-L38)，這也將作為
  PR Gate 的審查基準之一。

- **Contract
  測試與驗證：**在單元測試和整合測試層面，也加強對資料合約的驗證。**契約測試**確保不同語言間序列化結果等價。例如，用
  Python 生成 JSON，再用 Go
  的結構解析，看能否得到相同物件。又或將不合法的輸入（違反枚舉、欄位約束等）餵給
  Pydantic 模型，確認其會拋出
  ValidationError，從而保證驗證規則生效[\[34\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L22-L31)。這些測試用例會伴隨每次契約修改而更新，成為迭代的安全網。此外，也定期執行
  **Contract Coverage Health Check**，如 11/01
  的檢查報告顯示所有核心合約載入正常，字段驗證嚴格，雙模式序列化完全正常[\[35\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L11-L19)[\[31\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L33-L38)。透過這類報告，團隊可以掌握合約實際執行情況，及時擴大覆蓋面（報告建議將合約使用覆蓋率從15.9%提升到20%再到25%等[\[36\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L61-L70)）。

- **Codemod 自動遷移工具：**為減輕舊代碼遷移到新架構的工作量，引入
  **Codemod（代碼批次改寫）**
  工具。透過靜態分析和AST重寫腳本，半自動地將原有直接使用JSON字典或硬編碼Topic字串的地方替換為使用新模型和枚舉。比如，把
  `data["finding_id"]` 改為 `finding.payload.finding_id` 屬性訪問，或把
  `"tasks.sast.start"` 改為
  `Topic.TASK_SCAN_START`[\[37\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L259)。也可以批量插入信封封裝/解封邏輯。這類腳本可在遷移PR中運行並附帶diff供審查，確保改動正確。Codemod
  的使用大幅縮短了改造時間，同時降低人工出錯機率。

- **PR Gate 檢查：**在 Pull Request 的 Gate
  環節，除了一般測試，特別增加了**契約相關的檢查**：包括確認
  core_schema_sot.yaml
  的版本有無提升（必要時要求更新版本號和變更日誌）、生成的代碼是否已更新（避免提交者忘記執行代碼生成工具）、所有契約測試是否綠燈等。只有滿足這些條件
  PR
  才可被合併。這相當於在流程上強制執行合約規範，**使契約管理內建到日常開發流程中**。例如，每次Schema更改都會自動觸發生成並在版本控制中同步更新[\[28\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L182-L188)；若未更新，PR
  會被標記為不通過。這種 Gate
  機制確保了統一架構的執行力，防止因人為疏忽破壞一致性。

綜上，透過**代碼生成**、**持續整合
(CI)**、**測試驗證**、**自動遷移**和**嚴格把關**等一系列工具和流程的配合，AIVA
的統一通信架構得以可靠落地並維持。這套工具鏈為開發團隊提供了高層次的自動化支持，使得契約驅動的開發模式成為日常習慣，大幅降低導入新架構的摩擦。

## 優缺點分析

導入統一通信架構雖然能帶來多方面好處，但也相應地引入一些成本和挑戰。以下對主要的優勢和可能的缺點進行分析：

- **優點 --
  統一合約管理：**所有模組共享單一資料合約，使跨語言結構保持一致[\[4\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L40-L48)。這消除了以往因手動同步模型導致的不一致問題，實踐表明可**消除90%以上的跨語言資料同步錯誤**[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)。統一合約還降低了溝通成本，開發人員對資料格式的理解不再各說各話，同一欄位在各語言含義相同。未來需求變更時，只需修改一處合約並生成代碼，各處自動適應，**避免重覆修改**。整體架構更加清晰可預期。

- **缺點 --
  學習曲線與流程改造：**開發團隊需要適應新的開發流程和工具。第一次使用
  schema SOT 和代碼生成，需學習 YAML schema
  語法、生成工具用法，這對不熟悉契約驅動開發的成員有一定學習曲線。此外，引入嚴格的合約驗證會在開發初期暴露更多格式錯誤，團隊需要時間習慣在模型層面修正資料，而非在執行期調試。流程上，CI
  Gate
  新增的強約束可能延長部分開發迭代時間。然而這些都是"磨合成本"，隨著使用漸趨熟練，收益將遠大於初始投入。

- **優點 -- gRPC 強類型通信：**相較於傳統 REST 或直傳 JSON，gRPC
  有明確的接口定義和強類型保證，**減少介面誤用**。透過自動生成的存根，開發者可以像調用本地函數般使用遠端服務，IDE
  下即可發現參數類型錯誤。gRPC
  的雙向串流特性亦為實現即時反饋提供了便利（如掃描進度每秒推送）。性能方面，gRPC
  基於 HTTP/2 和
  Protobuf，資料序列化緊湊、傳輸高效，適合需要高吞吐或低延遲的場景。綜合而言，引入
  gRPC **提高了通信效率和可靠性**，也讓 AIVA 對外提供更專業的 API
  介面（便於未來對接其他系統）。

- **缺點 -- gRPC 部署與運維複雜度：**採用 gRPC
  意味著需要在現有架構中增設 gRPC
  伺服器/客戶端，各服務需監聽額外埠口，並在微服務協調層增加**服務發現**和**負載均衡**機制[\[19\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L148-L152)。對
  DevOps 而言，這是一個新挑戰：例如需要設定 gRPC
  的健康檢查、處理版本升級的向後相容（proto
  需要維護版本號）等。另外，gRPC 的調試相對REST為難，需要專門工具（如
  grpcurl）或Client代碼。雖然 AIVA
  已內建部分服務發現與治理功能[\[39\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L98-L106)[\[40\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L124-L132)，但團隊仍需投入精力來確保
  gRPC 服務穩定運行並與現有MQ機制協同。換言之，**gRPC
  引入了架構的複雜性**，需要權衡其帶來的效益和維運成本。

- **優點 -- 封裝 CLI
  工具：**將眾多外部安全工具納入統一架構，帶來顯著的**功能覆蓋和重用**收益。AIVA
  可以利用成熟工具的能力（如 sqlmap 的 SQLi 檢測、Nmap
  的掃描），無需從頭實現相同功能，大幅縮短研發週期。同時透過標準介面調用，使
  AI
  策略決策層可以**動態選擇最佳工具組合**來執行任務[\[41\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L120-L129)[\[42\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L134-L139)。封裝也提高了穩定性------可以為這些工具執行加入超時、錯誤緩解機制，不會因單個CLI卡死而影響整體。藉由統一封裝，**去除了系統中的"黑盒"**：以前某模組裡偷偷執行一個系統命令可能隱患多多，現在一切皆明確為服務調用，可監控可管理。

- **缺點 -- 性能開銷與複雜度：**透過 Tool-Host 調用 CLI
  工具，較直接在程式碼中調用函式**效率略低**。啟動外部進程和進行IO讀寫增加了延遲，若大量高頻使用可能成為瓶頸。此外，一些互動式或長時間運行的工具管理起來也更複雜，需要在服務中實現非同步處理和狀態跟蹤（例如某些工具運行時間很長，要及時提供心跳或取消功能）。開發方面，為每個CLI編寫封裝服務也有工作量，尤其工具輸出格式千差萬別，解析映射需要仔細測試。最後，將多個外部工具納管後，系統部署體系更複雜（容器映像增多，需要資源隔離）。不過，這些問題可以透過優化來緩解，例如對高頻工具進行原生改寫或內嵌，或使用長駐進程來避免反覆啟動開銷。總體而言，封裝
  CLI
  工具有一定成本，但相對其帶來的功能整合價值，這些成本在可接受範圍內。

## 預期效益

導入統一通信架構後，AIVA
平台在開發、生產力和系統品質等方面將獲得明顯的提升：

- **消除重複與錯誤，提升品質：**由於所有模組共用單一定義的資料結構，**重複定義**將大幅減少甚至消除，避免因不一致導致的邏輯漏洞。此前需要在
  Python/Go
  各寫一遍的模型，現在一份定義即可自動生成，多處一致。[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)顯示，Schema自動化預計可**消除90%**的跨語言同步錯誤，維護成本降低
  **80%**
  以上[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)。數據格式錯誤減少，帶來系統**可靠性**提升，生產環境中因契約不匹配引發的故障將幾乎杜絕[\[43\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L82-L86)。

- **降低耦合度，模組演進更獨立：**統一合約相當於在模組間建立清晰的邊界，各模組透過標準介面交互，不再相互依賴內部實現。這種**低耦合**使得單個模組可以在不影響其他部分的前提下自由升級或替換，只要契約保持向後相容[\[44\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L184-L188)。例如，可以替換某個Rust高性能組件為新版本，只要輸入輸出結構一致，上層Python仍可照常調用。各語言團隊也能**並行開發**，因為契約先行定好了接口，不會你改了模型我這邊才發現。版本演進上，透過
  schema_version 等機制，可以讓新舊契約兼容存在，平滑過渡。

- **可治理的跨語言協作：**有了單一來源的資料約定，跨語言協作從以前的人對人溝通轉變為"以契約為依據"。每個開發者都能參考
  core_schema_sot.yaml
  或由此生成的文件來理解資料格式，不再需要翻閱多處文檔。契約的變更有明確流程（通過PR審核、CI驗證），確保所有人知曉變更內容並同步更新代碼[\[44\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L184-L188)。這使團隊協作更加有序可控。如發現某功能模組需要額外欄位，大家協商後在合約中添加，相關代碼一鍵生成，各語言實現者再各自使用，**協同成本和出錯機率都大幅降低**。

- **版本管理與演進流程明確：**統一架構下，資料合約本身成為一等公民，有版本號與變更日誌。在合約調整時，可明確標註**API
  版本**（例如引入新欄位但舊欄位依然保留deprecated狀態）[\[45\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/guides/CROSS_LANGUAGE_BEST_PRACTICES.md#L18-L23)[\[46\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/guides/CROSS_LANGUAGE_BEST_PRACTICES.md#L55-L62)。再配合契約測試，舊版本契約的覆蓋情況一目了然。如果沒有統一管理，版本演進往往混亂且不同語言步調不一；現在透過集中治理，可實現**按計畫漸進升級**而不中斷服務[\[44\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L184-L188)。這對於需要長期維護的平臺尤為重要。

- **開發效率提升：**統一合約節省了大量重覆編碼和對齊工作的時間。根據評估，引入
  Schema
  自動化和跨語言生成，可使**新功能開發速度提升3-5倍**[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)。開發者專注於業務邏輯，資料結構由工具處理，不再為了給不同語言寫結構體耗費精力。同時，因介面清晰標準，開發者在介接其他模組時不必試錯，多數情況下"對就對，錯則編譯不過"，減少了調試溝通成本。**Bug
  修復效率**也提高了：當發現某資料字段缺失，只需在 SOT
  添加並生成，相關模組立即獲取更新，不用逐個修改。綜合以上，可預期整體研發效率有顯著提升，團隊能將節省的時間投入更有價值的創新功能上。

- **系統整體性能與擴展性：**雖然引入封裝和gRPC有些開銷，但從全局看，統一架構有助於**性能優化**和**水平擴展**。比如
  gRPC 使用二進制協議，相比過去某些 REST JSON
  調用降低了帶寬和解析開銷。在需要高併發時，可方便地對某服務進行獨立擴容（因為低耦合），而消息隊列確保了高峰緩衝和平衡負載。統一協定還讓**監控更有效**：因為所有消息都有trace_id和標準格式，易於集中收集日誌和計算指標
  (如 cross_language_communication_latency
  等)[\[47\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L134-L141)。這將帶來更好的運維洞察，進而優化性能表現。

## 推動建議：分階段遷移策略 (PR1--PR5)

導入統一通信架構涉及範圍廣，建議採用**漸進式分階段**的方法來降低風險。以下規劃了五個階段（對應
5 個重要的 Pull Request），循序漸進完成整合：

- **PR1：奠定基礎的 Schema SoT 與代碼生成引入。**首先創建
  `services/aiva_common/core_schema_sot.yaml`，梳理匯總現有所有跨服務資料結構定義[\[2\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L44-L52)。實現
  `schema_codegen_tool.py` 的初版整合，能根據 SoT 成功產出
  Python/Go/Rust/TS
  的模型代碼[\[5\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L50-L54)。在此
  PR
  中，**不改變**現有模組通信方式，只是並行引入新結構定義與生成結果。一旦生成代碼文件到位，更新各語言的模組讓它們引用新的模型類（可以暫時做別名映射保持向下相容[\[44\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L184-L188)）。配置
  CI：新增步驟自動檢查 SoT
  和生成代碼是否同步[\[28\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L182-L188)。此階段完成後，團隊就擁有了統一的資料定義來源和自動生成管道，為後續階段提供基礎保障。

- **PR2：整合 MQ 通道，啟用統一 Envelope 與
  Topic。**在確保新模型穩定後，第二階段重點改造
  **消息佇列通信**。修改所有生產/消費 MQ
  訊息的模組，將原本直接傳遞JSON字典改為構造 **AivaMessage**
  物件[\[6\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L15-L23)，其中
  header 根據上下文填入（如來源模組名、自產 UUID）、payload
  放入原訊息內容、topic 使用 enums.Topic
  枚舉值[\[7\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L18-L21)[\[15\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L254-L262)。相應地，消費端用生成的模型類來解析
  AivaMessage.payload，而非手動解析
  JSON。為避免一次性切換風險，可先在生產端**同時發送**舊格式和新格式消息（雙軌）以測試兼容。在Topic命名上，將原先使用的各種routing
  key字串全部替換為Topic枚舉，例如 `"scan.start"` 改為
  `Topic.TASK_SCAN_START`[\[9\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L260)。確保訂閱端訂閱的新Topic，同時保留對舊Topic的兼容監聽直到切換完成。這個
  PR
  完成後，**系統內主要的非同步任務流皆已經使用統一信封和Topic**，實現消息格式標準化。測試重點是驗證不同語言服務通過MQ收發AivaMessage是否正常（例如Python發、Go收是否成功），並觀察延遲開銷變化。完成切換且穩定後，可移除舊格式的支持。

- **PR3：引入 gRPC
  服務介面層。**在確保MQ路徑穩定運行的同時，開啟同步通道的建設。基於 SoT
  中的模型，撰寫或生成 gRPC 的 `.proto` 定義檔，例如定義服務
  `AivaCoreService` 內有方法 `RunTask(TaskRequest) returns (TaskResult)`
  等等，資料欄位類型與 Pydantic
  模型對應[\[48\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L961-L970)[\[49\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L972-L980)。使用
  proto
  編譯器生成各語言的介面代碼[\[17\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L996-L1004)。在
  Python 核心服務中實作 gRPC
  伺服器端，將請求轉換調用現有邏輯並返回結果。在需要同步調用的客戶端（例如前端
  TypeScript
  或其他服務）中，使用生成的存根進行調用[\[50\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L63-L66)。初期可選擇幾個最常用的接口進行封裝，比如「啟動掃描任務」、「查詢掃描狀態」、「獲取報告結果」等。確保
  gRPC 服務與現有 REST API
  等不衝突，可併行提供。測試時特別關注串流場景，如需要的話在 proto
  中加入 `stream`
  方法並測試其穩定性。**階段目標**是在內部建立起一套可靠的 gRPC
  通道，供後續逐漸遷移部分同步交互使用，並為外部集成（如未來提供官方API給第三方）打下基礎。部署上，此
  PR 也涉及配置 gRPC 埠、TLS 證書等運維事項，一併在此階段解決。

- **PR4：開發 Tool-Host 微服務並封裝 CLI 工具。**選取系統中幾個關鍵的
  CLI 安全工具整合為獨立服務。例如：Nmap 掃描器、sqlmap
  注入測試工具、ffuf 暴力破解工具等等。為每個工具建立對應的 gRPC
  介面定義和實現。以 sqlmap 爲例，建立 `SqlmapService`，定義
  `RunSqlmap(SqlmapRequest) returns (SqlmapResult)`，其中請求結構包含目標URL、參數等，結果結構包含發現的漏洞列表等（這些結構在
  SoT 中定義）。`SqlmapService` 實現中，接收到 gRPC
  請求後，組裝命令行參數執行 sqlmap 程序，實時獲取工具輸出，解析為標準
  FindingPayload 等模型然後通過 gRPC
  返回[\[20\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L76-L84)。為降低延遲，可考慮讓服務預先啓動工具進程（如透過管道方式與服務通信），或將工具容器長時間運行。這些封裝服務可以用
  Go 或 Python 編寫，視具體工具介面易用性決定。完成一個工具封裝後，修改
  AIVA 核心邏輯，使其在需要該工具時，透過 gRPC 呼叫 Tool-Host 而非原先的
  CLI
  調用。逐一封裝並替換，順序上可先從易封裝、高價值的工具著手。例如先封裝
  Nmap（易解析輸出）再封裝複雜度更高的 Burp Suite API 等。這個 PR
  可能拆分為多個子 PR
  來封裝不同工具，但在指南中作為統一階段說明。完成此階段後，AIVA
  的**工具層全面納入統一通信體系**：所有安全檢測能力不是由內建代碼直接提供就是由
  Tool-Host 提供，杜絕了不受控的外部命令呼叫。

- **PR5：統一架構全面切換與遺留清理。**最後階段，在經過前面步驟的鋪墊後，進行全面的架構遷移完成和收尾優化。主要工作包括：移除或棄用舊的通信代碼（如原有解析MQ
  JSON的代碼、舊的CLI調用代碼），確保所有路徑都使用新機制。更新**開發文檔**，宣告未來只接受遵循統一合約的模組代碼變更。針對團隊進行培訓，分享新架構下開發調試的方法，例如如何添加新合約定義、如何模擬
  gRPC
  調用等。再者，監控新架構在生產環境的表現，特別是性能和資源開銷方面，做最後的調優。比如調整
  gRPC thread pool 大小、MQ
  消費者預取數量等參數，以達到與舊架構相當或更佳的效率。**漸進式開關**也是此階段策略的一部分：可在一段時間內同時運行舊新機制，比對結果，最終關閉舊路徑。待一切穩定，發佈版本公告，版本號提升以標誌架構升級。此時，統一通信架構算是真正落地完成，後續開發將遵循新模式進行。

透過以上分階段的遷移策略，團隊可以循序漸進地推動統一架構上線。在每個階段完成後都有一個**觀察驗證期**，確保沒有明顯回歸和問題再進入下一階段。這種穩健推進方式將把風險降到最低。同時，每階段的成果也讓團隊成員逐步體驗到新架構的好處，增加對其信心和熟練度，為最終全面切換做好準備。

**總結：**統一通信架構的整合是一項系統工程，但收益也是巨大的：消除技術債、提升效率、增強系統穩定性和可擴展性。透過嚴謹的方案設計和循序漸進的落實，AIVA
將實現 CLI、MQ、gRPC **三合一**
的通信體系，為跨語言的網路安全智能平臺奠定堅實基礎。在未來，AIVA
開發者可以更加專注於安全功能本身，而無須憂心底層通信細節，這正是架構演進所期望達到的終極目標。[\[26\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L28-L32)[\[43\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L82-L86)

**參考資料：**

- AIVA Phase 0/I 實施計畫 -- Schema
  自動化與跨語言同步[\[4\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L40-L48)[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)
- AIVA 多語言架構策略 -- 統一資料合約與 RabbitMQ
  訊息格式[\[48\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L961-L970)[\[51\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L1014-L1022)
- AIVA Common 模組架構圖 --
  單一事實來源與多語言支持[\[26\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L28-L32)[\[14\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L99-L103)
- 合約健康檢查報告 --
  契約導入完整性與驗證嚴格性[\[32\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L13-L21)[\[29\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L59)
- Scan Integration 路線圖 -- 跨語言整合 (MQ 調用 Go/Rust, TS 經
  HTTP/gRPC)[\[1\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L2-L5)
- HackingTool 整合分析 -- CLI
  工具標準化封裝建議[\[20\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L76-L84)[\[41\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L120-L129)

------------------------------------------------------------------------

[\[1\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L2-L5)
[\[18\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L3-L5)
[\[47\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L134-L141)
[\[50\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md#L63-L66)
SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/SCAN_INTEGRATION_IMPLEMENTATION_ROADMAP.md>

[\[2\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L44-L52)
[\[4\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L40-L48)
[\[5\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L50-L54)
[\[28\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L182-L188)
[\[38\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L142-L149)
[\[44\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md#L184-L188)
PHASE_0_I_IMPLEMENTATION_PLAN.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/plans/PHASE_0_I_IMPLEMENTATION_PLAN.md>

[\[3\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L11-L19)
[\[21\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L10-L18)
[\[22\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L14-L19)
[\[23\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L142-L150)
[\[24\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L150-L158)
[\[25\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L92-L100)
[\[27\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md#L38-L46)
README.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/plugins/aiva_converters/README.md>

[\[6\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L15-L23)
[\[7\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L18-L21)
[\[13\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py#L11-L19)
messaging.py

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/schemas/_base/messaging.py>

[\[8\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L250-L259)
[\[9\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L260)
[\[15\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L254-L262)
[\[37\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py#L252-L259)
modules.py

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/modules.py>

[\[10\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/common.py#L8-L16)
[\[11\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/common.py#L22-L30)
common.py

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/services/aiva_common/enums/common.py>

[\[12\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L57)
[\[29\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L52-L59)
[\[30\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L20-L28)
[\[31\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L33-L38)
[\[32\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L13-L21)
[\[33\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L34-L38)
[\[34\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L22-L31)
[\[35\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L11-L19)
[\[36\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L61-L70)
[\[43\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md#L82-L86)
contract_coverage_health_analysis_20251101.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/reports/contract_coverage_health_analysis_20251101.md>

[\[14\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L99-L103)
[\[26\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt#L28-L32)
aiva_common_architecture_20251025_200417.txt

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/architecture_diagrams/aiva_common_architecture_20251025_200417.txt>

[\[16\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L994-L1002)
[\[17\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L996-L1004)
[\[48\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L961-L970)
[\[49\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L972-L980)
[\[51\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md#L1014-L1022)
MULTILANG_STRATEGY.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/ARCHITECTURE/MULTILANG_STRATEGY.md>

[\[19\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L148-L152)
[\[39\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L98-L106)
[\[40\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md#L124-L132)
CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/_out/CROSS_LANGUAGE_ARCHITECTURE_TODO_COMPLETION_REPORT.md>

[\[20\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L76-L84)
[\[41\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L120-L129)
[\[42\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md#L134-L139)
HACKINGTOOL_INTEGRATION_ANALYSIS.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/HACKINGTOOL_INTEGRATION_ANALYSIS.md>

[\[45\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/guides/CROSS_LANGUAGE_BEST_PRACTICES.md#L18-L23)
[\[46\]](https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/guides/CROSS_LANGUAGE_BEST_PRACTICES.md#L55-L62)
CROSS_LANGUAGE_BEST_PRACTICES.md

<https://github.com/kyle0527/AIVA/blob/2f60eec0524d1f36be1e13c5eb56e8cb126dfa3b/docs/guides/CROSS_LANGUAGE_BEST_PRACTICES.md>
