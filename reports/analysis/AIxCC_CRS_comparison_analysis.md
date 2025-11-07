# AIVA 與 AIxCC 決賽七隊 CRS 實作比較報告

## 簡介

DARPA **AI Cyber Challenge (AIxCC)**
要求參賽隊伍構建全自動的網路推理系統（Cyber Reasoning System,
CRS），能在無人工干預下對開源軟體進行漏洞挖掘和自動修補[\[1\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L12-L15)。2025
年決賽共有七支隊伍入圍並公開其 CRS
程式碼，各隊的技術實現路線多元：有的在傳統程式分析技術上融合 AI
模型，有的則採用「AI
優先」的方法[\[1\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L12-L15)。本文將比較這七支決賽團隊的實作，從**技術棧、架構設計、功能模組和可重用元件**等方面分析哪些與
AIVA 平台（一套使用者開發的**AI
智能安全評估系統**）最相近或值得參考。AIVA（代號
*Warprecon-D*）目前是一個針對網頁應用的漏洞偵察與評估平台，採用**微服務架構**，包含掃描、核心、漏洞檢測、整合報告四大模組[\[2\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L10-L18)[\[3\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L22-L30)。AIVA
技術棧以 **Python 3.13+** 為主，後端框架採用 FastAPI，透過 **RabbitMQ**
等實現模組間異步協作，並可用 Docker Compose
部署[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)。AIVA核心特色在於利用
AI
輔助分析與風險評估，支援大規模併發掃描、多類型漏洞檢測（XSS、SQLi、SSRF
等）以及完整的掃描流程自動化[\[5\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L50-L58)[\[6\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L60-L68)。

下面將分別介紹各決賽隊伍的 CRS
系統，針對**技術選型**、**架構模組**、**功能機制**和**元件工具**進行說明，並指出其與
AIVA
的重疊或互補之處。同時會列出每支隊伍特別值得參考的實作細節（例如關鍵模組、檔案路徑或技術特性），以及建議
AIVA 可納入的開源工具或元件，供工程團隊在擴充架構時參考。

## Team Atlanta -- **"Atlantis"** (亞特蘭大團隊)

**技術棧與工具：**Atlantis 是冠軍隊伍 Atlanta 的
CRS，採用**多語言、多引擎的組合架構**[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)。它針對
C、C++、Java
等不同語言的目標，整合了多種分析技術（**模糊測試**、符號執行、靜態分析）平行運行，以提高漏洞覆蓋率[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)。團隊大量使用成熟的底層工具，例如**LibAFL**/**AFL++**
作為核心模糊測試引擎，以及擴充版的 **SymCC**
進行符號執行[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)。在
AI 部分，Atlantis
引入大型語言模型（LLM）輔助漏洞分析：他們微調了**Llama2
7B**模型來專門分析 C 語言程式碼，並設定 LLM
三種工作模式------*增強模式*（協助傳統分析）、*顧問模式*（提供提示）、*自主模式*（代理自動探索）[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)。此外，LLM
還用於生成測試輸入變異器（如語法字典）以及自動化撰寫攻擊腳本PoV[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)。整體技術棧偏向
C/C++ 原生工具加 Python 腳本協調：例如 Kubernetes
上部署多個容器節點，同時跑傳統模糊器和 LLM
分析服務[\[9\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L4)。

**架構設計：**Atlantis
採用**"N版本並行"**的架構理念[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)。不同分析模組（針對不同語言和漏洞類型的引擎）以平行管線方式執行，最終彙總結果。整個系統在
**Kubernetes**
上協調，具備良好的水平擴充能力[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)。這種架構與
AIVA 有相似之處：兩者都強調**模組化並行處理**，如 AIVA
中掃描、分析、各類漏洞檢測模組可以同步運行[\[3\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L22-L30)。不同的是，Atlantis
專注於程式層面的漏洞挖掘與修補，需同時跑多種分析引擎；而 AIVA 著重於 web
應用的資訊蒐集與測試，模組劃分是按功能類型（XSS、SQLi
等）而非分析方法。儘管領域不同，Atlantis 展現的**多引擎協同架構**值得
AIVA 借鑒：未來若 AIVA
擴充至檔案解析、原始碼檢查等領域，也可採取類似的並行策略，用不同子系統同步分析，提高覆蓋率。

**功能機制：**在漏洞挖掘方面，Atlantis
同時運行**多組模糊測試**實例（包括 AFL 系列和基於 libFuzzer
的定制模糊器）以及**符號執行分析**，以期最大程度探索程式狀態空間[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)。靜態分析則輔助發現可疑模式，例如使用
CodeQL 查詢或抽象語法樹分析（推測自其對 Java/C
不同實現的描述[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)）。LLM
模型被整合進整個流程中：例如在模糊測試卡住時，LLM
代理可建議新的測試策略（增強模式）；對模糊測試發現的 crash，LLM
進一步分析漏洞成因並提供修補思路（顧問模式）；甚至讓 LLM
自主代理對某模組進行完整漏洞搜索（自主模式）[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)。Atlantis
的**補丁策略**較為保守，決賽中禁用了沒有對應 PoV
的自動修補，以避免誤修補帶來扣分[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)。這點與
AIVA 的工作流程類似：AIVA
作為掃描器也應避免誤報或不精確的結論。在未來，若 AIVA
引入自動防禦措施（如 Web 應用防火牆規則自動產生），可參考 Atlantis
的做法，**僅在高可信度的前提下執行修補**，確保精確性。

**值得參考的實作細節：**Team Atlanta 公開的 Atlantis
程式碼中，值得關注的是其**多分析引擎整合**與**容器編排**方式。例如，Atlantis
如何在 Kubernetes 中部署多容器服務（模糊測試服務、符號執行服務、LLM
服務等）並行執行，是一個關鍵實現點[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)。這種基於容器的佈署方案可供
AIVA 團隊參考，用於提升現有 Docker Compose 部署的擴展性（必要時可升級至
Kubernetes 集群以支撐更大量並發掃描）。另一個細節是 **SymCC**
符號執行的擴充用法[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)：Atlantis
對 SymCC 作了定制以更好處理 C/Java 程式，AIVA
團隊若計畫加入**符號執行**以探索複雜邏輯（例如驗證碼繞過或深層業務邏輯漏洞），可研究
Atlantis 對符號工具的改造方式。此外，Atlantis 對 **LLM
的專項訓練**也很突出：他們微調了小型模型以適應程式分析[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)，表明針對安全領域進行模型精調能提升效果。AIVA
若有足夠程式碼數據，也可考慮訓練類似的小型專用模型，提高 AI
分析的專業性。

**與 AIVA 的重疊與互補：**Atlantis 與 AIVA
在架構理念上有**重疊**：雙方都強調組合多種檢測手段、平行處理提升效率。不同在於
Atlantis 面對二進位/原始碼漏洞（如緩衝區溢出、記憶體錯誤等），AIVA 面對
Web 應用漏洞（XSS、SQLi 等）。因此，Atlantis
的**模糊測試與符號執行模組**對 AIVA 屬於**互補**性質------AIVA
目前或許未大量應用此類低階分析，但未來擴充至協助審計後端原始碼或第三方組件漏洞時，這些技術將非常有用。相反地，AIVA
在 **Web
爬蟲、認證繞過**等面向上經驗豐富[\[2\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L10-L18)（如如何登入受保護頁面、提取動態內容），這方面並非
Atlantis 關注焦點，因此屬於 AIVA 的獨有強項。兩者結合看，Atlantis
提供了**高度自動化漏洞挖掘**框架範本，而 AIVA
提供**完整滲透測試流程**（從偵察到報告）的視角，兩者理念相通且各有側重，AIVA
可從 Atlantis 借鑒後端漏洞挖掘與 AI 結合的實踐。

**可納入的開源工具/元件：**根據 Atlantis 的實作，AIVA
團隊可考慮引入以下工具： - **AFL++
模糊測試引擎**[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)：業界領先的
fuzzing 引擎，已被 Atlantis 等系統廣泛驗證效能。AIVA 可將 AFL++
用於對某些輸入介面進行更深入的測試（例如 fuzz Web
應用的上傳檔案解析功能或 API 端點），提高涵蓋率。 - **LibFuzzer
平台**[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)：LibFuzzer
尤其適合對字串解析、檔案格式等進行變異測試。若 AIVA 在掃描中需要測試例如
PDF 解析、影像檔解析漏洞，LibFuzzer 可快速集成生成測試。 - **SymCC
符號執行框架**[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)：SymCC
是將符號執行融入編譯器的工具，Atlantis 用其找出深層邏輯漏洞。AIVA
未來如需對目標程式執行路徑探索（例如計算某輸入條件是否可觸發敏感路徑），可結合符號執行輔助。SymCC
開源且與 AFL++ 可組合（稱為 concolic execution），值得研究。 -
**Kubernetes 編排經驗**：Atlantis 利用 K8s 實現彈性伸縮和資源隔離。若
AIVA 部署規模擴大（如同時掃描上百目標），可考慮從單機 Docker Compose
過渡到 Kubernetes，以支撐**分散式掃描**。團隊可參照 Atlantis
部署腳本了解如何在 K8s 下管理多容器CRS系統。

## Trail of Bits -- **"Buttercup"** (Trail of Bits 公司隊伍)

**技術棧與工具：**Buttercup 是 Trail of Bits 團隊研發的
CRS，在決賽獲得第二名。它採用**Python 為主的協調邏輯**（推測，因 Trail
of Bits 常用 Python
開發安全工具）並結合多種安全測試工具形成**混合式管線**[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)。技術上突出的特點是將**模糊測試**與**靜態分析**緊密結合，並大量使用
LLM
改善測試用例生成與修補漏洞[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。Buttercup
集成了 **libFuzzer**（針對 C/C++ 程式）和 **Jazzer**（針對 JVM/Java
應用）的**覆蓋導向模糊測試**引擎，用以探索程式輸入空間[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。為了彌補傳統fuzzer在複雜輸入上的不足，系統透過
LLM 產生特殊測試案例（如結構化的 SQL 查詢或路徑字串），輔助 libFuzzer
等發現像
SQLi、路徑遍歷這類**高階語意漏洞**[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)。在靜態分析方面，Buttercup
使用 **Tree-sitter**
作為語法解析器並編寫了程式碼查詢規則，搜尋常見漏洞模式[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。例如，它能基於語法樹快速定位危險函式調用、未受信輸入的操作等[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。整體技術棧還注重**成本效益**：該隊刻意選擇「廉價」的模型（如開源或小參數量
LLM）並大量並行調用，以降低開銷（據報導，他們的 LLM 查詢次數超
100k，費用卻控制在較低水平）。這種"多而廉"的策略對需要大規模掃描的系統很有啟發。

**架構設計：**Buttercup 架構可概括為**"流水線 +
多代理"**。首先是流水線：對每個目標程式，先經過靜態分析產生潛在漏洞點，接著進入模糊測試與動態分析階段驗證，最後由修補模組自動生成並測試補丁[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。在此流程中，不同階段由不同子系統或代理負責。例如，它有專門的**輸入生成代理**（LLM
驅動）為fuzzer提供高價值測試案例[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)，以及**補丁代理**針對發現的漏洞產生修復程式碼[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。Trail
of Bits
強調系統的**高精確度**和**穩健性**，在決賽中他們達到超過90%的漏洞檢出準確率[\[14\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L7-L10)。為此，Buttercup
採取了**保守提交策略**：只有當某漏洞有對應的PoV（攻擊驗證)時才提交補丁[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。這點和
Team Atlanta、Shellphish
等類似，屬於"寧缺毋濫"風格[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。Buttercup
架構中的**靜態-動態結合**理念，對 AIVA 頗具參考價值。AIVA
目前以動態掃描為主，如果能引入靜態分析模組預先篩選高風險區域，再由動態攻擊驗證，將提升掃描效率與精準度。Buttercup
將兩者有機結合，並通過**中控協調程式**串起LLM代理、fuzzer、補丁模組，這種設計與
AIVA 的核心協調引擎作用類似（AIVA Core
Module負責任務生成、策略調整等[\[15\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L16-L24)）。不同之處在於
Buttercup 直接面向原始碼/二進位漏洞挖掘和修補，而 AIVA 面向
Web交互測試，不涉及自動修補。因此在架構上，Buttercup
多了**補丁生成/驗證**這一環節及其代理角色，AIVA
暫無對應部分。但未來若擴充例如自動產生 Web
應用防護規則或修復建議，Buttercup 的**多代理協作**框架值得借鏡。

**功能機制：**Buttercup 的漏洞挖掘流程從**靜態分析**開始：透過
Tree-sitter 對目標程式構建
AST，運行一系列**代碼查詢**規則來識別潛在漏洞（例如緩衝區未檢查、SQL查詢字串拼接等）[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。這些初步發現會轉換為**SARIF**格式的靜態報告。接著，系統對每個可疑點啟動**模糊測試**驗證：使用
libFuzzer（C/C++）或
Jazzer（Java）生成隨機輸入，同時計算覆蓋率，看能否觸發異常[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。為了攻克傳統fuzzer不易覆蓋的路徑，Buttercup
引入了一個**LLM測試生成器**，針對例如SQL
Injection、目錄遍歷這類需特定格式輸入的漏洞，讓 LLM
提供智能猜測的payload[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)。實踐中，他們可能給
LLM 一些程式上下文，讓其產生符合程式語義的輸入（如根據 SQL
查詢上下文生成各種惡意字串）。一旦模糊測試或LLM生成輸入觸發了漏洞（Crash或錯誤訊息），Buttercup
會記錄下具體輸入與現象。接著進入**自動修補階段**：另一個 LLM
代理基於漏洞位置的程式碼上下文，提出補丁（可能使用 prompt
要求模型輸出修正後的代碼片段）[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。生成的補丁代碼由系統自動套用並重新編譯程式，再跑測試驗證：不僅用PoV測試修補是否奏效，還跑回歸測試確保未引入新問題[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)。只有當補丁通過所有驗證且對應漏洞有PoV支援，系統才提交該補丁。[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。值得一提的是，Buttercup
以**成本控制**為目標，廣泛使用「非推理型」LLM（即相對簡單或便宜的模型）批量處理任務，如靜態報告分類、基礎代碼摘要等，而把昂貴的深度推理（如補丁生成）任務次數減到最少。這種機制保證了在嚴格資源限制下依然能高效運行。對
AIVA 而言，這提示我們可以**分層使用
AI**：大量低成本模型/規則做初篩和資料處理，少量高成本AI做關鍵決策，以達到效率與效益的平衡。

**值得參考的實作細節：**Buttercup 程式碼中一些關鍵模組對 AIVA
有啟發。例如，他們的**程式碼查詢模組**（如 `codequery.py`）實現了基於
Tree-sitter
的靜態規則檢索[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。AIVA
若計畫增加原始碼檢查能力，可參考該模組如何定義漏洞模式規則（如正則或AST模式）並高效搜索整個程式碼庫。又如
**LLM 測試生成**部分，Buttercup 可能有一套代理（也許在代碼中名為
*TestGenerator* 或類似）專門與 OpenAI API
互動。該代理如何將程式上下文編排進 prompt、如何解析 LLM
輸出轉化為測試輸入，都是值得學習的細節。特別地，Buttercup
很可能使用了**字典/文法檔**結合LLM，例如提供一組已知惡意字串讓 LLM
擴展，或者讓 LLM 產生類似 fuzz
字典的建議[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)。這對
AIVA 處理例如 XSS 攻擊字串、繞過 WAF payload
等很有幫助------團隊可以研究 Buttercup
涉及**Grammar字典**或**payload生成**的程式碼部分。最後，Buttercup
的**補丁驗證流程**也值得關注：AIVA
雖暫不自動修補，但可以借鑒其中**驗證迴圈**的理念，用於漏洞發現的確認。例如
Buttercup
只有在存在能觸發漏洞的PoC時才認定漏洞並修補[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)，AIVA
未來也可實現「**自動二次驗證**」------當掃描發現某疑似漏洞，例如SQLi，系統自動嘗試產生一個能證明資料庫錯誤的有效payload，再判定該漏洞是否真正存在，從而減少誤報。

**與 AIVA 的重疊與互補：**Buttercup 與 AIVA
有明顯的**重疊領域**在於對**Web應用漏洞**的關注。AIxCC 比賽要求覆蓋多種
CWE
漏洞，包括資料庫注入、路徑遍歷等[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)------這些類型正是
AIVA 功能模組（SQLi、檔案包含等）的核心檢測目標。Buttercup 通過 LLM
強化對此類漏洞的測試[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)，這與
AIVA 的目標高度一致。因此在漏洞偵測策略上，Buttercup 的經驗對 AIVA
屬**互補增強**：AIVA 可引入 Buttercup 的**Grammar Fuzz**思路，利用 AI
豐富惡意輸入的多樣性，彌補傳統字典的不足。同時兩者在**架構理念**上也有共同點，都採用多階段pipeline並強調**結果可信度**（Buttercup和AIVA都傾向在高度確信時才輸出結論/補丁[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)）。差異在於
Buttercup 比 AIVA **更智能自動化**：尤其補丁生成與應用完全自動，而 AIVA
仍主要把人作為最終決策（產生報告給人看）。因此 Buttercup 可以視為 AIVA
潛在的發展方向之一，即**從"發現問題"進一步走向"自動解決問題"**。總體而言，兩者在Web漏洞檢測上目標一致，Buttercup
的 AI 輔助技術為 AIVA 提供了寶貴的實踐經驗。

**可納入的開源工具/元件：**從 Buttercup 的實作出發，以下工具和技術可納入
AIVA 考量： - **Tree-sitter
語法解析與查詢**[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)：一個支援多語言的輕量級解析器，可方便地對源碼建立
AST。配合自定義查詢，能極大提升靜態分析效率。AIVA
若希望在不依賴大型商用SAST工具下增加程式審查能力，Tree-sitter
是理想選擇。 - **Jazzer
fuzzer**[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)：Google
開源的 Jazzer 可對 Java/JVM 應用進行模糊測試。AIVA 已包含對 Web
後端（可能有 Java服務）的測試需求，引入 Jazzer 能自動化對比如 Spring
應用、Android API 等的漏洞探測。 - **開源 LLM 平台**：Trail of Bits
強調控制成本，暗示他們可能使用開源模型（如 LLaMA、Falcon
等）在本地推理。AIVA 團隊可考慮整合一個開源
LLM（7B\~13B參數規模），微調於常見 Web
漏洞描述，供系統離線地生成payload或分析結果，減少對雲端API的依賴與成本。 -
**補丁建議生成**：Buttercup可自動產生補丁，這對 AIVA
而言可轉化為**修復建議**功能。可考慮利用開源的編碼輔助模型（如 CodeT5,
Polyglot Coding
等）來實現：輸入漏洞程式碼片段，生成建議的修復代碼或配置。這將使 AIVA
報告更具行動性，幫助開發者快速修復發現的漏洞。 -
**多代理協調框架**：Buttercup將不同AI代理嵌入管線，可能使用了類似
**LangChain** 之類的多步驟推理框架。AIVA 如果引入多階段 AI
分析，可使用此類框架構建 agent
chain。例如先由一個代理生成測試，再由另一代理判斷結果。這方面開源工具（LangChain、Haystack-Agent）成熟度不斷提高，可以加速開發。

## Theori -- **"RoboDuck"** (Theori 資安公司隊伍)

**技術棧與工具：**RoboDuck 是 Theori 團隊的
CRS，強調**"LLM優先"**的自動化漏洞挖掘，是決賽季軍[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。技術上，RoboDuck
幾乎將所有傳統分析環節都用**大型語言模型代理**替代或輔助：從閱讀程式碼理解、到想出攻擊方法、再到產生補丁，皆由
AI
代理執行[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。因此，其主要技術棧包含強大的
LLM 系統和輕量的輔助工具。RoboDuck
**最小化了傳統模糊測試與符號執行的使用**[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)------與其他隊伍動輒部署數百
fuzzer 相比，RoboDuck
僅在必要時調用模糊測試作為備用方案[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。他們利用
Meta 開源的 **Infer 靜態分析器**
來快速掃描程式，找出潛在的錯誤位置（如空指標解引用、整數溢位等）[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。Infer
生成的報告（例如潛在 bug 的函式路徑）會被 LLM
進一步審閱，以篩選值得深入分析的點[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。在必要時，RoboDuck
也會重用社群已有的 fuzz 測試 **harness**：如對那些已經有 OSS-Fuzz
測試的專案，直接利用其 libFuzzer 驅動，作為對 AI
發現的漏洞的驗證[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)。整體而言，RoboDuck
的技術棧核心在於**多代理 LLM 系統**（可能由 Python 實現代理邏輯，串接
OpenAI API 或自架模型）和一些**輔助分析腳本**（Infer
靜態分析、程式執行監控等）。這種"AI 主導 + 工具輔助"的組合為 CRS
提供了一種新範式。

**架構設計：**RoboDuck 採用了**代理 (Agent) 驅動的架構**。Theori
在賽後部落格中將它比作「讓 AI
代理按照逆向工程師的劇本去執行任務」[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。具體而言，RoboDuck
由多個專職 AI
代理組成：一部分代理負責**閱讀理解程式碼**（例如一個代理按函式逐個審查程式，另一代理通覽整體架構）[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)；另一些代理負責**執行具體任務**，如發現漏洞後，一個代理專攻**生成
exploit**，另有代理專攻**編寫修補程式**[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。這些代理透過一個中心調度（可能也是一個
LLM 或規則引擎）協同工作，類似多人團隊各司其職。為確保 AI
代理不偏離正軌，系統還設計了**行為約束**：比如預定一系列「逆向工程步驟」讓代理遵循，每步都有明確目標[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。在分析過程中，RoboDuck
還引入**推理鏈機制**：對每個發現的潛在漏洞，AI
會推導出漏洞成因，並嘗試產生可行的攻擊步驟（PoV）。值得注意的是，RoboDuck
能**生成傳統方法難以產生的
PoV**，包括特定格式輸入（URL、二進位通訊協議等），這都歸功於 LLM
的語意理解能力[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。在補丁決策方面，RoboDuck
採取了**進取型策略**：每當確認一個漏洞點（有PoV），即允許 AI
**提交多個試探性補丁**（最多兩個無PoV佐證的修補嘗試）來搶分[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)。這是很激進的做法，但在比賽計分上若命中真的漏洞即可加分（然而也冒著誤修補扣分風險）。整體而言，RoboDuck
架構的**高度 AI 自治**特性鮮明，這在傳統工具主導的安全測試中是革命性的。

對比 AIVA，RoboDuck 的架構幾乎走到了另一極端：AIVA
目前以工具和人工規則為核心（如定義XSS
payload模式，人工設定爬取策略），AI 主要輔助分析；而 RoboDuck
幾乎將**所有環節決策都交給
AI**。這兩者在架構上**互補**意義明顯------AIVA 可以從 RoboDuck 汲取 AI
自治的思想，在某些子任務上嘗試讓 AI 更自主。例如，AIVA
針對一個複雜業務邏輯漏洞，或許可以模仿 RoboDuck，啟動一個 "代理" LLM
去自動推理漏洞成因及可能的攻擊步驟，而不局限於預定的掃描字典。當然，AIVA
必須權衡這種自治性帶來的不確定性，可能需要類似 RoboDuck
的**監督機制**（如限制 AI 行為在某些範圍）。在架構可擴展性上，RoboDuck
的代理是很靈活的：增加或改變代理職責只需調整
prompt/腳本，系統就能應對新的漏洞類型。AIVA
若引入類似代理框架，長期看可提高面對新威脅的適應性。

**功能機制：**RoboDuck 的核心流程以 AI 驅動：首先**代碼理解階段**，AI
代理讀取整份程式碼庫，產生對程式功能和潛在問題的假設。Infer
靜態分析在此提供輔助輸入，它快速列出明顯的錯誤候選（如可疑的空指標操作）[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。AI
會審查這些靜態報告，結合自己的代碼理解，確定關注的漏洞點（可能透過
ranking 或篩選 false
positives）[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)。接下來**漏洞挖掘階段**，AI
代理針對關注點展開行動：例如對某函式，代理可能嘗試構造輸入誘發異常。如果需要執行程式才能驗證猜想，系統才會調用傳統模糊測試或執行測試（例如運行帶輸入的程式，查看是否崩潰）[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)。值得一提的是，RoboDuck
擁有在**無傳統fuzz支援下產生漏洞利用**的能力[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)：例如它能根據程式語意猜測出某種輸入格式直接導致漏洞，而非透過試爆無數payload才找出。這對那些高度結構化的輸入（如通訊協議封包）特別有用，AI
可以"讀懂"協議並找出漏洞利用方式。當發現漏洞後，RoboDuck 立即進入**PoV
生成**和**補丁**階段：一個 AI 代理編寫 exploit
程式（PoV），驗證漏洞可被利用[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)；另一代理嘗試修改程式碼修補漏洞。由於比賽中允許的話，RoboDuck
甚至對未完全確證的漏洞也提交補丁嘗試，以期先發制人[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)。每次補丁後系統再由
AI或腳本執行回gress測試，確保沒有破壞原功能（不過從結果看，RoboDuck
採取激進策略導致補丁準確度相對較低，這是其弱點）。整體流程中，AI
代理的行為被設計成接近人類分析師：按部就班地**閱讀 -\> 假設漏洞 -\> 驗證
-\> 攻擊/修補**，只是速度極快且不知疲倦。

**值得參考的實作細節：**RoboDuck 作為「AI滲透測試員」，其程式碼中與 AI
交互的部分對 AIVA 很有價值。比如，**代理提示的設計**：Theori
可能編寫了一系列 prompt 模板，引導 LLM
扮演不同角色（審計員、攻擊者、修補者）。AIVA 若要實現類似 AI
驅動分析，可參考其 prompt 編排方式。如對於「閱讀代碼找漏洞」，可能的
prompt
是"這份程式碼是否有安全漏洞？回答具體漏洞點"；對於「生成攻擊」，prompt
可能包含程式片段與目標效果。如能取得 Theori 開源的 prompt
腳本或代理代碼，將直接有助於複現這種行為。另一細節是 **Infer 與 AI
的結合**：Infer 輸出 JSON 或文本報告，而 AI
怎麼取用其中資訊？可能有轉換腳本將 Infer 的重點結果串進
prompt，或AI代理直接解析報告。這部分處理值得學習：AIVA
也可以用現成靜態分析（如 CodeQL、SpotBugs 等）的結果作為 AI
輔助輸入。因此可以參考 RoboDuck 程式碼中是如何load靜態工具結果並與 AI
交互的。第三，RoboDuck 的**PoV
生成**代理可能運用了特定模板。例如，對一個HTTP服務漏洞，它或許讓AI輸出一段可以在終端運行的
`curl` 命令作為PoC。這類模板 AIVA 也可考慮，未來可以自動產生PoC（例如
XSS 的惡意
URL）。最後，**多代理協調**的實現方式是重點：可能採用了類似狀態機或任務隊列，每個
AI 完成任務後將結果交給下一角色。AIVA
若想引入AI代理，需要設計這種任務分發機制，RoboDuck
的實現是極佳範例（如隊伍是否使用了現有 Agent 框架或者自行開發）。

**與 AIVA 的重疊與互補：**RoboDuck 與 AIVA
**重疊較少**，因為前者重點在**原始碼級漏洞挖掘與攻防自動化**，而 AIVA
目前是**黑箱的 Web
應用掃描**。然而雙方目標一致：提升漏洞發現的自動化和智能化程度。因此
RoboDuck 的 AI Agent 模型對 AIVA 而言是非常強的**互補**借鑒。尤其在 AIVA
想擴充"**智慧分析**"能力時，RoboDuck 提供了一個範本：讓 AI
像資安專家一樣去思考。比如，AIVA 在掃描一個 Web 應用時，可以嘗試增加一個
RoboDuck 式的步驟：讓 AI
閱讀應用文件或常見路由，推測哪裡可能有邏輯漏洞，然後指導掃描器去測試。這將突破傳統掃描預定流程的限制。當然，RoboDuck
激進的漏洞修補策略在企業環境未必適用，但其**快速 PoC 驗證**思路可融入
AIVA，用於降低誤報。總體看，RoboDuck 代表最前沿的"AI滲透測試員"，AIVA
的強項是可靠的掃描框架和實戰經驗，兩者結合可望產生強大的**AI
賦能安檢**系統：即保留 AIVA 穩健的模組式流程，同時注入 RoboDuck 類的 AI
智囊，提高對未知威脅的洞察。

**可納入的開源工具/元件：**從 RoboDuck 經驗，AIVA 可考慮： - **Infer
靜態分析器**[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)：Facebook
開源的靜態分析工具，擅長發現 Null Pointer、資源洩漏等 bug。Infer
自動化程度高，適合集成到流水線。AIVA
若掃描對象包含後端原始碼（Java、C等），可用 Infer
快速篩查潛在問題，再交給動態測試驗證，提高效率。 - **CodeQL
查詢庫**：類似 Infer，CodeQL 可自定義安全漏洞查詢。Theori
博文提及他們有解析 SARIF 報告整合到 AI
判斷中[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)。AIVA
團隊可以使用 CodeQL 分析 Web
應用程式碼（若拿得到），並將結果與掃描發現交叉印證。 -
**多模特化AI協同**：RoboDuck 大量使用 LLM，但未明說是否單一模型。AIVA
可學習 Lacrosse/Shellphish
等隊伍的**多模型共識**策略（見後述），引入AIVA現有的不同特化AI模型對掃描結果進行交叉驗證。例如EnhancedDecisionAgent、BioNeuronMasterController，同時分析某可疑事件，只有雙方都認定高風險才報告，藉此降低AI判斷誤差。 -
**Agent 管理框架**：如前述，若要實現 AI 代理，多步驟推理框架如
**LangChain**、**GPT-Engineer** 等可以幫助串聯複數提示和行動。Theori
沒明說用何框架，但其流程本質屬於 "思維鏈+工具調用"。AIVA 可試用
LangChain 的 Chains/Agents來模擬簡化的
RoboDuck流程，例如："讓AI閱讀漏洞描述-\>讓AI呼叫掃描模組驗證-\>AI判定結果"。這類框架能減少自行實現代理管理的工作量。

## All You Need is a Fuzzing Brain -- **"Fuzzing�Brain"** (AllYouNeed隊，中譯「萬事俱備，只欠模糊」)

**技術棧與工具：**"Fuzzing Brain" 是一支特立獨行的隊伍（名取自 "All You
Need is a Fuzzing Brain"），其 CRS 幾乎完全由 AI
所打造與驅動，堪稱**AIxCC
中最「AI化」的系統**[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)。據賽後報告，團隊90%以上的代碼都是利用
AI
助手生成的[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)。技術棧方面，可以推測他們廣泛使用
Python 來串接各種服務，並在其中嵌入大量 AI
模型調用。這套系統部署了**成千上萬個並行的 LLM
代理**[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)來分析程式、推理漏洞、決策補丁，真正實現大規模
AI
併發。傳統技術在他們系統中處於輔助地位，只是作為驗證或備份：例如有一組普通的模糊測試管線在
AI
找到漏洞後再去驗證，避免假陽性[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)。由於這隊對計算資源的大量消耗（推測使用了雲端或高效能叢集），其在比賽成績上**發現漏洞速度最快**，曾達成5分鐘內找到漏洞的紀錄，並在靜態檢測（SARIF報告準確率）上名列前茅[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)。整體技術選型上，Fuzzing Brain
可能使用了**高度平行計算框架**（如 Ray、Dask 或雲函式）來調度上千 AI
任務，以及**向量資料庫**或快取來讓 AI
共享知識。這方面的細節外界不得而知，但可以肯定的是，他們以**極端堆砌 AI
agent**的方式實現了前所未有的自動漏洞挖掘。

**架構設計：**Fuzzing Brain 的架構可形容為**"AI
雲腦"**：把問題分拆成許多子任務，丟給大批 AI
並行處理，最後彙總結果。具體架構可能包括：一個**集中任務分發器**，將程式碼按模組/檔案/函式分片，交給眾多
AI worker
分析[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)；每個
AI worker
對自己的片段嘗試回答「有無漏洞、可能的攻擊、如何修補」等，再把見解回報中央；中央彙總後，再指派更多
AI
去驗證關鍵點，或者產生PoC/補丁[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)。如此多輪反覆，直到系統對哪些是漏洞達成一定信心，再提交結果。在這過程中，該隊設計了**AI三重驗證**來保障品質：他們聲稱透過
AI
分層篩選，讓靜態報告的80%以上錯誤分類準確[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)。這暗示他們可能讓多個模型對同一件事投票表決（例如三個模型各自判斷一個SARIF警告是否真漏洞，取多數意見），或採用模型間互批改的策略。整體上，Fuzzing Brain
更像一個概念驗證，展示了**純 AI
方法**的潛力。然而其缺點在於**資源利用效率偏低**：大量 agent
平行可能有重覆工作，加之模型推理昂貴，在比賽限制下雖有好成績但性價比低，據統計其每分得一分所耗資源遠高於前幾名[\[20\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)。

對於 AIVA 而言，Fuzzing Brain 提供了一個極端參考樣本------**如果完全依賴
AI，我們能做到什麼程度？** 當前 AIVA
較注重系統化和決策可控性，不可能走如此極端路線。但其中一些**架構理念**可以借鑒。首先是**高度平行的Agent調度**：AIVA
目前雖能多執行緒掃描多目標，但尚未充分利用AI並行。參考該隊，AIVA
未來可嘗試**並行啟動多個AI實例**關注不同問題：例如一個模型專找XSS、一個專找SQLi，再匯總結果，這有點類似人手分工，提高效率。其次，Fuzzing Brain
展現了**AI分級篩選**的效果------讓AI先濾掉大部分明顯無關的事件，再精查剩餘部分。AIVA
可以在報告生成前加一個AI復核步驟：把掃描結果交給模型，請它標記最可能的真漏洞，從而為使用者重點提示高風險項或降低誤報。第三，他們**快速戰果**的能力啟發我們：也許可以增加一個"快速模式"，在掃描初期就用AI猜測幾個最可能漏洞去嘗試，短時間內給出初步結果------這對某些時效要求高的場景很有價值。當然，全盤的
AI
云腦在工程上目前不可行，但其中蘊含的**可伸縮性**與**AI智力榨取**理念確實值得思考。

**功能機制：**由於 Fuzzing Brain 主要由 AI
決策，其具體功能流程與人類直覺差異較大。我們推測其機制如下：系統上線後，首先**初始化上千個
LLM
agent**等待任務。接著遇到一個待測程式，系統會將其拆成許多**小單元**（如函式級別），分發給
AI
代理。**第一輪**，每個代理閱讀自己的程式碼段，產生**初步漏洞假說**：例如
"函式 X 可能有緩衝區溢出"、"函式 Y
SQL查詢未轉義"等。這些假說彙總後進入**第二輪**驗證：可以想像再啟動一些代理，針對每條假說嘗試**推導攻擊方法**或**尋找對應證據**。若有假說在這輪被證明有效（如
AI
給出了具體攻擊步驟），則標記為確認的漏洞。對仍存疑的，可能進入**第三輪**由更多AI討論或透過傳統fuzzer檢驗[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)。該隊似乎也使用了
fuzzing，但僅僅是為了驗證 AI
的發現，而非主要探索手段[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)。隨著輪數增加，系統收斂出一批確認漏洞及其PoC，再交由另一組AI嘗試編寫**補丁**。補丁出來後，再有AI或腳本進行**編譯測試**，以及**效能檢查**（確保補丁不影響功能）。比賽中，該隊提交了不少補丁，但只有部分是有效的，顯示其**補丁驗證**可能不夠嚴謹，或許由於時間關係未對每個補丁進行充分測試。不過，他們在**靜態分析正確率**方面拿下"Czar
of the
SARIF"獎，意味著他們能較準確地分析並分類靜態結果[\[21\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L43)。這可能是因為
AI
群體智慧濾除了大部分誤報，只留下真問題------一種**AI驅動漏洞三角驗證**的成功案例。

**值得參考的實作細節：**Fuzzing Brain
的代碼若公開，其中關鍵部分在於**如何管理海量 AI 任務**。AIVA
工程師可重點關注：1）**任務切分策略**：他們如何把大問題拆解，比如是否有
`task_manager.py`
之類模組在負責切分檔案、函式，並為每段生成適當的提示任務給AI。2）**並行計算框架**：可能出現如
`parallel_executor`、`agent_pool` 等模組，實現同時調用數千AI。這類實現對
AIVA
具參考意義，未來如需掃描大型目標或海量站點，如何擴展併發是一大挑戰，他們的代碼或設計思路可供借鑒。3）**AI結果彙總與評斷**：代碼中也許有將多個AI輸出整合的步驟，如統計某漏洞被幾個AI報告、信心評分等。如果能找到類似
`aggregate_findings()` 的函式，就能學到 AI
投票或共識的形成方法。4）**錯誤容忍與重試**：在大規模AI調用下，必然有部分任務失敗或AI產生無用輸出。他們或許實現了檢測無效輸出的機制，並會重新分配任務。AIVA
如增加AI單元，亦需考慮穩定性，可以參照其錯誤處理實現。總之，AllYouNeed
隊的代碼可能相當複雜，但哪怕抽取部分並行框架和AI協同的精華，也能讓 AIVA
如虎添翼。

**與 AIVA 的重疊與互補：**Fuzzing Brain 和 AIVA
幾乎沒有直接重疊------前者是超前實驗性系統，後者是實用導向的平台。然而在理念上，雙方有共同追求即**全自動化漏洞挖掘**。Fuzzing Brain
走的是極端 AI 化路線，這對 AIVA 是一種**互補激勵**：提醒我們不能忽視 AI
的潛力。AIVA 可以借鑒其成功之處，例如 AI
高效分類漏洞，提高掃描結果準確度，也要引以為戒其不足，例如過度依賴AI可能導致誤補漏洞。在架構上，AIVA
偏重穩定的模組調度，而 Fuzzing Brain 偏重動態的 agent
湧現。兩者結合或可取長補短------AIVA
保留核心模組架構以確保流程可靠，同時局部引入 agent
提升智能。比如保留當前掃描器負責遍歷頁面和基本測試，再加一層AI
agent對輸出結果或目標進行深度推理。這樣不會大幅改動架構，又能享受AI紅利。總體而言，Fuzzing Brain
給 AIVA 帶來的是「**未來圖景**」的映照：證明即使沒有人為干預，AI
也能在一定程度上完成我們以前由人/工具結合才能完成的任務。AIVA
作為實戰系統，可以挑選其中可行的部分逐步實現，循序漸進往高智能化發展。

**可納入的開源工具/元件：**從 Fuzzing Brain 的理念出發，AIVA 可引入： -
**分散式任務調度框架**：如 **Ray**、**Celery**
等，用於調度大量並發任務。Ray 特別適合 Python
任務的高併發分發，可讓AIVA在單機或叢集上方便地啟動上百上千個任務(worker)。配合AI，可以開啟多worker並行分析不同頁面或參數，提升速度。 -
**向量資料庫與檢索**：大規模AI代理往往需要共享上下文。引入向量數據庫（如
Faiss、Milvus），可讓AI將中間分析結果embedding存儲，再供其他AI檢索。AIVA
可用它來讓不同模組/AI共享發現。例如一個模組發現某參數有特殊行為，可存入DB，另一AI代理在分析相關頁面時檢索到這資訊，提高一致性。 -
**Prompt 質量控制**：大量AI並行容易產生噪訊結果，可考慮引入 **OpenAI
Evals** 或自己設計的 prompt
測試集，評估每個提示模板的有效性。AllYouNeed隊或許也經歷了調適 prompt
的過程，有無自動化工具未知，但我們可主動使用現有評估框架優化AIVA的AI提示，確保投入計算力得到有意義輸出。 -
**快速模式選項**：如前述構想，提供"fast
mode"掃描，可先讓一輪AI粗略檢查應用結構和常見漏洞，再決定詳細掃描策略。這需要一些腳本支持，例如使用現有的
**Wappalyzer API** 或 **AI-driven crawler**
來快速摸底目標，再把結果交給AI判斷哪裡最可能有洞。這種前置模組可以在開源社群尋找雛形，例如
OWASP 有一些專案做應用指紋識別，可與AI結合達到快速偵察效果。

## Shellphish -- **"Artiphishell"** (Shellphish 戰隊)

**技術棧與工具：**Artiphishell 是由著名學術駭客團隊 Shellphish 打造的
CRS。他們曾是 DARPA CGC 2016
決賽隊之一，具有豐富的自主漏洞挖掘系統經驗。Artiphishell
延續了該團隊傳統的強項（如符號執行、動態分析）並結合了當代的 AI
技術，形成一套**大規模多智能體**的系統[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。技術棧上，Shellphish
偏好使用 **Python** 作為膠水語言（其著名的 angr 框架即 Python
編寫），此次他們的系統也很可能主要以 Python 寫成，並透過 **Docker
容器**與 **Kubernetes** 佈署在 Azure
雲端叢集上[\[23\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L40-L48)[\[24\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L66-L74)。Artiphishell
包含超過**60個 AI
Agent**，協同執行程式分析、模糊測試、漏洞利用、補丁等任務[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。這些
Agent 部分依賴外部大型模型 API，例如 OpenAI GPT-4、Anthropic
Claude，以及據稱還接入了 Google 的 Gemini
模型[\[25\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L36)。藉助多種LLM，系統可以取長補短，避免單一模型的偏誤。除
AI 模型外，Artiphishell 繼承了 Shellphish 過去工具鏈：如使用
**angr/Driller**
等符號執行和測試生成框架（推測自官方描述）[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。另外，他們研發了一個名為**"Grammar
Guy"**的 AI
子系統，用來自動學習程式輸入的文法，以提升模糊測試對複雜輸入格式的效率[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。整體工具組合反映了
Shellphish 團隊**人機結合**的風格：以先進 AI
力量增強既有安全分析技術，使系統具備深度和廣度兼顧的能力。

**架構設計：**Artiphishell
採用**分散協作式多模組架構**。從架構圖上看（團隊可能有內部架構圖，但從描述推測），它由一系列微服務或容器組成，各自承擔特定職能，由中央協調組件（類似**任務管理器**）控制部署與通信[\[24\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L66-L74)。各模組大致包括：**解析代理**（分析原始碼/二進位，提取函式、控制流程等資訊）、**模糊測試代理**（運行多種fuzzer，並收集覆蓋）、**Grammar
Guy代理**（根據fuzzer反饋調整輸入文法）、**漏洞利用代理**（生成PoV）、**補丁代理**（給出修補程式碼建議）等等[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)[\[27\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L37)。這些代理之間透過**遙測與日誌**系統交換訊息：值得一提是
Shellphish
特別強調了**遙測（Telemetry）**，即詳細地紀錄每個AI決策、每個模組狀態，方便賽後分析系統運作[\[28\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L37)。他們的遙測系統甚至獲得"最佳遙測"獎，表明其在系統可觀測性上做得非常好[\[28\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L37)。在決策策略上，Artiphishell
屬於**保守派**：例如在修補漏洞方面，他們和 Trail of Bits 一樣**嚴格要求
PoV
驗證**，絕不提交未經PoV證實的補丁[\[27\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L37)[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)，以確保接近100%的補丁正確率（實際達到95%以上）[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。這種謹慎降低了得分上限（因為有些漏洞可能來不及補就放棄），但提升了系統可靠度。相較
AIVA，Artiphishell 在架構上非常類似------都是**微服務模組群 +
中央協調**。AIVA
的14個微服務模組[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)對應
Shellphish
的眾多agent，一個偏安全測試域、一個偏Web掃描域，但都是把系統拆成細粒度元件，各司其職。兩者不同之處在於
Artiphishell 幾乎完全自治運行，不需要人介入；而 AIVA
目前仍預設人員審查報告和調整策略。Artiphishell
展現的**高自治多Agent協作**，正是 AIVA 未來升級的方向之一。

**功能機制：**Artiphishell 的漏洞挖掘流程融合了**傳統程序分析**與**AI
驅動**技術。一開始，系統會利用類似 angr
的符號執行去探索程式路徑，找出可能的漏洞位置（例如可達的危險函式調用）[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。並行地，**多組模糊測試**在運行，包括
AFL++、QEMU 模式fuzzer
等，試圖觸發崩潰[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。在這些過程中，AI
代理插入來提升效率：**Grammar Guy** 代理分析 fuzzer
輸入與覆蓋的關係，動態調整輸入格式推斷，例如如果發現某輸入欄位似乎是
JSON，Agent 會生成針對 JSON 結構的payload，讓fuzzer
引入結構化變異[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。這極大增強了fuzzer對複雜語法（SQL查詢、URL等）的探索能力[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。同時，其他
AI
代理監控每個LLM的輸出，扮演"**保姆**"：如發現某代理偏離任務，會糾正其提示或重啟該任務[\[27\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L37)。Shellphish
提到他們使用**角色提示**和多模型交叉，確保AI保持在正確軌道[\[27\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L37)。當傳統分析（fuzz或符號執行）捕獲
crash 時，系統生成報告交給 AI
代理解析crash原因並分類漏洞（例如區分獲得EIP控制vs僅僅null deref）。如果
crash 看起來是真漏洞，則進一步由**PoV代理**利用 AI 腦力來構造
exploit（特別是對1MB以上巨型輸入，他們也成功搞定並獲獎[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)）。在**補丁階段**，Artiphishell
延續其保守策略：不確定就不補。AI
代理在此更多是輔助確認漏洞/補丁有效性，而非主導產生補丁（可能覺得AI寫補丁不可靠，所以乾脆不濫用AI在這步驟）。結果是
Artiphishell
在決賽中補丁正確率最高，但補丁數相對較少[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。綜上，Artiphishell
最大特色在於**AI助力的傳統強項**：把 Shellphish 過去在 CGC
場上淬煉的符號執行、fuzz、動態分析，插上AI翅膀，飛得更高。同時保留審慎作風，寧可少而精，體現了學術團隊嚴謹風格。

**值得參考的實作細節：**對 AIVA 而言，Artiphishell
程式碼中的幾個部分值得深入研究。首先是**Grammar
Guy**的實現[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)。這可能涉及監聽fuzzer產生的輸入並計算覆蓋變化，然後AI基於這些資料建議新的輸入形態。如果
AIVA 能引入類似機制，對付 Web中複雜輸入（如 JSON
API、JWT、富文本）將更有把握。可關注 artiphishell 專案中是否有關於
grammar inference 的模組（例如包含 "grammar"
關鍵字的程式碼檔），學習如何對接AI與fuzzer。其次，Artiphishell
使用**多個LLM
API**[\[25\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L36)（OpenAI、Anthropic等）。程式碼中如何抽象出一層接口來調用不同服務商模型也是重點。AIVA
也許未來會需要同時用多種模型（如結合 ChatGPT 與本地模型），那麼
Shellphish
的實踐可以作參考，看他們是否實現了統一的LLM請求模組，可配置切換 API
金鑰等[\[29\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L24-L32)。第三，**遙測日誌系統**：AIVA
現有日誌可能只記錄掃描步驟，而 Shellphish
詳細記錄了AI的思考歷程[\[28\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L37)。若他們開源了
telemetry 相關代碼，AIVA
團隊可學習如何有效地紀錄大量agent的狀態和決策------這對日後除錯及向客戶解釋AI決策非常有用。最後，Shellphish
在**補丁決策**上的保守邏輯也可參考：程式碼中應該有對每個潛在補丁打標記是否有PoV支持的流程，AIVA
將來如做自動修復，也應堅持先驗證再執行的原則，Shellphish
的實現能提供現成模板。

**與 AIVA 的重疊與互補：**Artiphishell 與 AIVA
是最具**相似性**的一對：兩者都是模組化架構、強調多元漏洞檢測方法、追求高精確性。可以說，Artiphishell
是專注於二進位/程式碼漏洞的"AI賦能CRS"，而 AIVA 是面向
Web的"AI賦能掃描器"。雙方很多設計思想可互相印證。例如，AIVA
已實現的**微服務架構**[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)在
Artiphishell 得到了類似運用，只不過後者微服務的行為者是 AI agent。這說明
AIVA
的架構具有前瞻性，完全可以容納日後增加AI代理等先進功能。又如雙方都非常注重**結果準確**，Shellphish
堅持補丁必須有PoV驗證[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)，AIVA
也可在報告階段對重要漏洞嘗試PoC（現有部分模組可能已有PoC功能，如 XSS
在DOM中執行檢查），這種理念是一致的。另一方面，Artiphishell 有一些 AIVA
目前缺乏的能力，是**互補**價值所在：例如**符號執行**，Shellphish
可能使用 angr
等在二進位上找到了深層漏洞[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)；AIVA
或許未用符號執行，但將來可用 angr
對JS代碼或重要演算法路徑分析，找出隱蔽漏洞。再如**LLM
Grammar學習**，AIVA 還沒有類似的智慧模塊，這正是可以借鑒 Shellphish
的地方，使掃描器更"聰明"地測試特殊輸入。總體而言，Artiphishell 對 AIVA
來說更像是一個同道先進者------它證明了成熟安全分析技術與AI結合後威力大增，AIVA
完全可以朝這方向演進。兩者領域不同，但有許多共通方法論，可以預見 AIVA
參考 Artiphishell 經驗後，能在 Web 安全自動化上取得類似的突破。

**可納入的開源工具/元件：**從 Artiphishell 汲取靈感，AIVA 可考慮： -
**Angr 與
Driller**[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)：Shellphish
開發的開源符號執行與模糊測試框架（CGC期間問世），對分析二進位程式極為有用。AIVA
在掃描時，如需要針對特定執行檔（如上傳檔案後端處理）進行深入挖掘，可結合
angr 來探索程式路徑，找出邏輯漏洞。 - **Grammar Fuzz
工具**：如果不從零開始實作 GrammarGuy，可考慮現有一些grammar
fuzz輔助工具。如 **Antlr4** 提供語法定義後可生成測試，或者 **Skywalker**
這類從樣本學習輸入格式的研究原型。將這類工具與 fuzzer 集成，可以部分達到
GrammarGuy 的效果。在缺乏工具時，甚至簡單地用 AI
幫忙生成字典也有幫助（Shellphish靠AI學文法，我們可以請AI看目標規格，輸出一些字典項）。 -
**多雲模型支持**：Shellphish能同時調用多家LLM服務[\[25\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L36)。AIVA
若計畫廣泛使用外部AI，可封裝出一套**多API支持庫**，按配置輪詢使用不同模型，避免對單一服務依賴。同時可用多模型
cross-check 結果。開源工具如 **LangChain**
已支援多提供商接口，可直接利用。 -
**雲端遙測與日誌分析**：為提升大型系統可觀測性，建議引入像 **Elastic
Stack (ELK)** 或 **Prometheus + Grafana** 這樣的方案。Shellphish強調
telemetry，AIVA
可在部署架構中加入集中式日誌、指標收集。比如記錄每次AI調用耗時、每個模組檢測到多少可疑點等，方便日後優化。同時也為客戶報告提供依據（類似
Shellphish 在AIxCC提供詳細系統報告）。如果 Shellphish
開源了他們的遙測配置，直接採用會更省力。 - **PoC 自動生成**：Shellphish
的PoV代理和超大PoC產生展示了AI在這方面的能力[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)。AIVA
將來可引入**PoC生成模組**，利用 AI
將掃描出的漏洞自動構造成一個可重現的攻擊腳本。例如掃描發現SQLi，模組自動輸出一條攻擊URL證明資料庫洩露。開源的
**AttackFlow**、**Metasploit** 都有部分自動化 exploit
生成思路，可調研後融入AIVA，使報告從"發現漏洞"走向"證明漏洞"。

## Team 42 & Team B3yond & Team 6ug -- **"Bug Buster"** (聯合隊伍)

**技術棧與工具：**"Bug Buster" 是一個由 **42**、**b3yond**、**6ug**
三支隊伍聯合組成的聯隊（名稱中的42-b3yond-6ug即來源於此），他們的 CRS
著重於**高並發模糊測試平臺**的打造[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)。技術棧上，以
**Go 語言** 為核心開發，其關鍵組件 **BandFuzz** 就是使用 Go
編寫的RL調度系統[\[31\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L28-L37)。BandFuzz
能動態協調成百上千個 fuzzer 節點，使用**強化學習（Reinforcement
Learning）**來決定資源分配和測試策略[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。整個系統可同時跑多達約
2000
個模糊測試實例，進行大規模平行模糊[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)。為管理這樣的龐大分散環境，Bug
Buster 團隊引入了企業級的訊息與資料組件，如 **PostgreSQL**
數據庫保存測試結果、**RabbitMQ** 消息列隊分發任務、**Redis**
快取共享狀態[\[33\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L17)。這些與
AIVA 在微服務架構中使用的技術非常相似（AIVA 也使用
RabbitMQ、資料庫等來協調多模組[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)）。此外，團隊在靜態分析方面也有所涉獵，嘗試了**程式切片（Program
Slicing）**技術，試圖透過靜態分析剪除不相關代碼，使模糊測試重點聚焦於易出問題的區域[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)。他們還整合
**SARIF** 標準的靜態掃描報告，使用 AI 進一步分析比對，以過濾重覆的 crash
或確認漏洞類型[\[34\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)。在補丁部分，Bug
Buster 引入了一個有趣概念：**"超級補丁"**（Super
Patch），期望用一個補丁修復同一類型的多個漏洞[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)。實現上可能是讓
AI 分析多個 crash
共通的根本原因，生成一次性修復，但比賽結果看，這招有風險且成功率不高[\[35\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L41-L43)。綜合而言，Bug
Buster 的技術棧是**傳統工業級架構 +
創新AI調度**。他們將分散式系統、資料庫、訊息中間件等成熟方案用在CRS中，保證了系統穩定和可擴展，同時用AI優化模糊測試效率，兩相結合。

**架構設計：**Bug Buster
的架構可以比喻為**"雲端模糊測試實驗室"**。其核心是 **BandFuzz
調度器**，像大腦一樣控制著數千模糊測試節點[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。系統在雲端部署多個容器或VM，每個運行一個fuzzer實例，所有節點向
BandFuzz
匯報狀態（覆蓋率、Crash等），由BandFuzz決定下一步該把更多資源投入哪個target[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。這裡用到了**強化學習**：BandFuzz
根據歷史表現（reward可能設為找到crash數或覆蓋增益）動態調整對不同模糊策略的調度[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。例如，如果某個target或某種變異策略在前一時段收效很好，調度器會增加對其的執行比重。另一方面，若某些作業長久無斬獲，則減少資源浪費[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。資料庫（PostgreSQL）則用來匯總所有節點結果，進行
Crash
去重和統計分析[\[35\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L41-L43)。為了自動識別重覆漏洞，團隊實現了**Crash
Deduplication**邏輯，透過比較 Crash
堆疊或錯誤簽名將相同根源的歸類[\[35\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L41-L43)。AI
元件在架構中主要輔助兩部分：一是**強化學習代理**，這本質也是種AI（可能用PyTorch實現一個policy
network），嵌在BandFuzz裡學習何時分配資源[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)；二是**LLM
分析助手**，用於分析程式、生成輸入、概括補丁等[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)。例如，他們訓練了一個
LLM
來生成有趣的測試種子[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)（或從既有fuzz輸入中學習模式後產生新種子），以提高fuzzer初始效率。還有讓
LLM 分析 Crash
日誌產生**"漏洞物件"**的做法[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)：將Crash資訊（堆疊、輸入）摘要為漏洞描述，便於補丁模組使用。整體架構非常偏重**規模和效能**，這與AIVA的企業應用場景不謀而合：兩者都採用消息隊列、資料庫、分散節點等確保可伸縮和穩定。可以說
Bug Buster 的系統架構與 AIVA
最接近"一般後端服務架構"，因此AIVA團隊容易理解其設計並加以借鑒。

**功能機制：**Bug Buster 的核心功能在於**智能化的大規模模糊測試**。其
BandFuzz 系統工作如下：首先，在對某個目標軟體進行模糊前，BandFuzz
會將任務（Target及fuzz設定）發布到**工作隊列**，一批fuzzer worker
從隊列取任務開始跑[\[38\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L44-L52)。每個fuzzer
worker都被容器化，可以是不同fuzzer組合（AFL++、honggfuzz等）以增加多樣性。一段時間後，worker
回報當前**覆蓋率、發現Crash數**等指標。BandFuzz
收集所有回報，以這些量作為**狀態**，通過其內建的 RL agent
計算下一輪的**動作**：如增減某個target的fuzzer實例數，或改變某類變異策略（例如多嘗試字典插入或改為結構感知模式）[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。然後調度器發出新的任務分配，worker據此調整，進入下一迭代。如此循環，在24小時的賽事時間內動態尋優。這種
RL
調度可看作自適應模糊：能在易出漏洞的點投入更多火力，在困難點避免浪費時間[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。除了模糊本身，Bug
Buster
也輔以**靜態輔助**：如前述，他們試圖用程式切片減少無效代碼區塊的模糊，但工具相容性不佳未充分奏效[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)。另外，對一些
Crash，系統會把程式切片或執行路徑給 LLM看，讓AI嘗試概括出造成 Crash
的根因，甚至建議patch方向[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)。然後由開發者基於這些"漏洞物件"撰寫補丁（或AI嘗試寫超級補丁）。然而，比賽成績顯示他們找到了
**41** 個漏洞但只補上 **3**
個[\[39\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L46)，主要是補丁系統出現Bug或時間不足[\[40\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L43)。不過在漏洞檢出上，他們PoV分數是第二高，且靜態誤報過濾做得最好[\[39\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L46)。這說明其**核心模糊測試能力極強**，AI在此功不可沒：無論
RL 調度還是
LLM輔助種子，都提高了效率。同時也暴露了**補丁模組不穩**的問題------這也許是AI過度泛化或缺乏驗證導致的教訓。

**值得參考的實作細節：**Bug Buster 程式碼中，最值得AIVA關注的是
**BandFuzz**
調度框架[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)[\[41\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L21)。BandFuzz
本身已開源，包含詳細的README說明其功能[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。AIVA
團隊可研讀其 **Go
實現**，看看如何建立一個**強化學習環路**來管理模糊測試資源。例如，他們如何定義狀態和獎勵、用哪種
RL 演算法（可能 Policy Gradient 或 Q-learning）以及如何探索/利用。儘管
AIVA
不是純fuzzer平台，但類似思想可用於掃描任務調度：比如學習在大範圍IP掃描時，應將更多精力放在有反應的host上，減少無回應host。另一細節是
**RabbitMQ 任務佇列** 的結構：程式碼裡可能定義了幾種隊列（如"pending
targets"、"running targets"），以及 worker 如何從隊列取任務。AIVA
目前也用RabbitMQ，但調度邏輯較固定；參考BandFuzz，可改進AIVA的任務分配，使其更動態高效。再有，**Crash
Deduplication** 和 **SARIF整合** 在代碼中應有體現，如一個
`crash_handler`
模組，先對Crash進行簽名（可能用崩潰位址+堆疊hash），然後對SARIF靜態結果進行匹配，以確認那些Crash已知成因。AIVA
如日後掃描源碼，產生静态報告，就可仿效這種**交叉印證**：將動態發現與靜態結果對比，減少重覆報告。最後，他們的**超級補丁**機制或許在代碼中也有雛形，例如對多個漏洞物件共用一段patch。如果能了解這是如何描述/套用的，可為AIVA未來做批量修復建議提供思路（如對多處相似XSS輸入點，給出統一修補建議）。縱觀Bug
Buster程式碼，可以學到如何構建**企業級可靠模糊基礎設施**，這對AIVA工程化發展很有裨益。

**與 AIVA 的重疊與互補：**Bug Buster 與 AIVA
在**系統工程層面重疊最多**。兩者都使用微服務和消息機制來實現大規模任務協調，都強調模組獨立可擴展，並關注降低誤報（Bug
Buster拿了SARIF準確獎[\[39\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L46)、AIVA強調零誤報品質[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)）。可以說，AIVA
在 Web 掃描領域實踐的很多工程技巧，在 Bug Buster CRS 上有對應：例如
RabbitMQ+Redis 的用法、容器佈署、Crash 去重類似於 AIVA
對掃描結果相同Payload只報一次等邏輯。這種共通點意味著 AIVA **容易借鑒**
Bug Buster 成果並平移到自身系統。另一方面，雙方在技術重心上又互補：Bug
Buster極強在**底層二進位漏洞模糊**，AIVA 則專精**高階Web邏輯檢測**。AIVA
可以引入Bug
Buster的**RL調度**理念來優化掃描資源使用。例如大範圍網段掃描時，學習自動關注那些回應異常的主機並深挖。同時，Bug
Buster 大量使用的**傳統模糊技術**（AFL++ 等）也可補強 AIVA
對特殊輸入場景的測試能力。反之，AIVA 的
Web知識（如身份驗證繞過、Session處理）這支聯隊未必擅長，可算是各有千秋。總的來看，Bug
Buster 更像是一個**通用自動漏洞挖掘平臺**雛形，而 AIVA
是一個**專業領域安全掃描器**。兩者在架構設計上英雄所見略同，而具體技術上可相互補足，AIVA
尤其可以輸入更多 "智能" 基因，比如來自Bug Buster的 RL與AI輔助測試創新。

**可納入的開源工具/元件：**基於 Bug Buster 經驗，AIVA 可考慮： -
**BandFuzz
強化學習調度框架**[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)[\[41\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L21)：這個工具已在學術會議發表[\[42\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L78-L86)並開源，是協調大規模fuzzer的現成方案。AIVA
或可對其改造，用於協調大量 Web 掃描 agent。例如大企業同時掃很多站點，用
BandFuzz 類似思路學習哪類站點需要更深入掃描，優化全局掃描計畫。 -
**程式切片/依賴分析**：Bug Buster
雖未成功用程式切片，但該想法值得探究。開源的 **CodeSurfer**、**Frama-C**
等工具能對代碼進行依賴分析，找到跟輸入相關的代碼區域。AIVA
若掃描有原始碼的Web應用，可在掃描前用這類工具剔除明顯無關區域，加快分析。這和
AIVA
**範圍定義**階段類似[\[6\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L60-L68)，但更細緻到代碼級。 -
**Crash 去重演算法**：當檢測大量漏洞時，歸納同類問題很關鍵。可考慮引入
**Stack hashing** 或 **Exploitability判斷**（像Microsoft
!exploitable工具）來自動分類掃描結果。這可用於AIVA報告整理：例如偵測到100處XSS，其實成因類似，可自動合併一條。Bug
Buster 在Crash dedup上或有現成腳本可參考。 - **開源規則/字典集**：Bug
Buster 透過LLM生成了一些 fuzz
種子[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)，AIVA
也可利用社群現有字典和透過AI擴充。比如 OWASP 零日字典、fuzzdb
等都可納入AIVA字典庫，再用AI根據目標特性（參數名稱等）做定製。這種結合傳統資料+AI創新的做法，是
Bug Buster 取得良好效果的原因之一。 - **CodeQL/Joern**：Bug Buster
能成為SARIF沙皇，可能他們整合了CodeQL查出問題再由AI過濾[\[34\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)。AIVA
若獲取到目標程式碼，可跑
CodeQL查已知漏洞模式，再和動態結果交叉驗證，提高置信度。Linux基金會開源的
**Joern** 也類似，可圖譜分析代碼，對發現邏輯漏洞有幫助。

## SIFT -- **"Lacrosse"** (SIFT 公司隊伍)

**技術棧與工具：**Lacrosse 是由軟體工程公司 SIFT 所開發的
CRS，在決賽排名第七[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這支隊伍的特殊之處在於，其系統基於一個「**具有10年歷史的遺產平臺**」演進而來[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。換言之，他們將多年來積累的一套自動化漏洞挖掘框架加以現代化升級，並融入
AI 模組形成
Lacrosse[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。技術棧上，因為是舊系統改造，可能包含較多
C++、Java 等舊語言程式碼，同時為適應AI也引入 Python 腳本或 API
介接。Lacrosse 著重於**經典模糊測試技術**，他們部署了 **300-500
個平行模糊節點**，規模與 Trail of Bits
不相上下[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這些節點由一個中央代理
"**Optimus Zero**"
統籌調度[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)（名稱戲仿變形金剛中的首領），以確保持續穩定地對目標進行模糊測試。而
AI 在 Lacrosse
中的角色較有限：主要用於**高層推理與分析**。例如，每當發現一個
crash，會有一個 LLM
代理生成詳細的「**漏洞物件**」描述，包含漏洞類型分類、成因分析和潛在修補方案[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。另外在補丁決策上，他們採用了
**多模型共識**機制：必須經過多個 AI
模型都認可的補丁方案，才會付諸實施[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這類似引入
AI 的**審議投票**，避免單一模型出錯導致錯誤補丁。傳統工具方面，Lacrosse
大量使用現成的**靜態分析**和**模糊測試**組件：如程式提及 "standard
static tools" 來篩選
crash[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)，可能用到了
Coverity、CodeSonar 這類商用SAST，或開源工具（未明示）。總之，Lacrosse
的技術策略是"**穩健第一，創新其次**"：相信經時間驗證的方法（大規模fuzz、綜合利用多工具），對新技術則謹慎評估後採用，以確保系統不出大錯[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。

**架構設計：**Lacrosse 架構可視作**"傳統CRS +
AI模組點綴"**。核心結構沿襲他們多年系統：主控端 + 多模糊測試worker +
資料庫 + 任務腳本。Optimus Zero
這個中央代理負責將目標程式分配給眾多模糊節點運行，並監控各節點進度[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。與
BandFuzz 不同，Optimus Zero
未必採用學習調度，而更像基於啟發式或固定策略分配，但考慮到他們聲稱部署規模很大，也許內有一些負載均衡演算法。Lacrosse
架構的亮點在於**AI共識機制**：在漏洞補丁這類關鍵決策上，他們引入了多個
AI
模型參與審核[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。具體做法可能是在補丁生成後，用幾個不同的大型模型各自審查補丁是否合理，如果意見統一（例如都認為補丁不會破壞功能且能解決漏洞），才交付執行[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這相當於給
AI 再加了一道保險。Lacrosse
還有一個特色是**非常保守的漏洞挖掘策略**：他們幾乎完全依賴**傳統模糊**找
PoV，對花哨的 AI
探索並不熱衷[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。雖然這使得進度較慢，但優點是較少誤報干擾。架構中應該有一個**全域狀態管理**模組，維護所有
fuzz節點的共同狀態，例如哪些發現已提交、哪些漏洞在修補中等[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這有助於那麼多並行任務協同，防止重覆工作或衝突。總的來看，Lacrosse
架構層級分明、強調**穩定與正確**，在比賽中它確實幾乎沒犯錯，但也因過於謹慎只提交了極少成果（找到1個漏洞並補上1個）[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。對
AIVA 而言，Lacrosse 彷彿是一面鏡子：AIVA
也是極力追求零誤報、高可靠的掃描輸出[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)。Lacrosse
告訴我們，**謹慎有餘則攻擊性不足**，如何平衡值得思考。

**功能機制：**Lacrosse
的漏洞偵測大多依賴**模糊測試**。據描述，他們執行了數百個fuzzer平行，且這些fuzzer以**傳統方式**找
PoV[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。意思是，他們不太使用
LLM
直接推理漏洞，而是讓fuzzer自己撞，因而PoV雖晚但確實可靠[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。在每個
crash 發生後，系統啟動**LLM分析**：該 LLM 看 crash dump
產生「漏洞物件」，其中記載漏洞類型（基於stack
pattern分類，例如UAF、Stack
Overflow等）、具體成因推測、影響範圍，甚至建議潛在修補思路[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。這些資訊一方面給團隊成員賽後分析用，一方面也餵給**補丁模組**作參考。補丁模組可能採用傳統靜態分析搭配AI評估：如先用符號執行驗證漏洞點，再AI協助判斷如何修改程式碼才能修復且不影響其他功能。為避免AI的一己之見出錯，他們要求**多個模型達成共識**[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。可能具體做法是給GPT-4、Claude等同樣的補丁與上下文，問"這補丁是否安全有效？"，只有都回答肯定才算通過，否則退回人工。此機制反映出
Lacrosse
把AI當作諮詢而非決策者，這和AIVA目前的AI定位相似（AIVA也主要輔助分析，而最終報告策略仍人工制定）。在
fuzz過程中，Lacrosse 並未大書特書AI作用，但猜測也有一些 AI
介入：例如**語意指導fuzzer**[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)，LLM或許讀代碼幫忙告訴fuzzer重點函式或輸入格式，使fuzzer更有效。或者**安全閥**：Lacrosse
重視不犯錯，興許用AI寫了一個"信心演算法"對每個發現打分，低信心的不提交[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。總體而言，Lacrosse
功能是**穩扎穩打**，沒什麼花俏，但在可靠性上達到了賽事頂尖（沒亂打補丁扣分，且發現了一個別隊漏掉的真漏洞[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)）。這種務實風格對產品化系統如
AIVA 很有啟發。

**值得參考的實作細節：**Lacrosse
儘管表現平平，但其**共識補丁決策**實作值得AIVA深入研究[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。代碼中應有一個協調模組，比如
`patch_consensus.py` 或類似，負責收集不同AI對補丁的評價然後決策。AIVA
未來若實現AI自動建議修復，也可以用多模型交叉驗證的方式。此處或許可借助
openai API的 logit 或簡單 majority vote
來實現。另一個細節是**全局狀態管理**，Optimus Zero
類似的中央控制如何記錄所有fuzzer進展？AIVA
也有Core模組管全局狀態[\[15\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L16-L24)，可以對比學習。如果SIFT有開源部分腳本如
`lax-run-optimus0`等，可以看看指令流程，加深對如此多fuzzer協調的理解。此外，他們**避免重覆提交**和**避免誤報**的方法也可挖掘。可能在程式碼中體現為"每發現一個漏洞，要判斷是否別隊已提（避免重覆）"和"沒有一定信心不發"。前者在AIVA不直接相關，後者可以結合
AI 做------AIVA 報告可標註信心值，只顯示高信心條目。Lacrosse
用多AI共識計算信心，AIVA 可以考慮 simpler
consensus，比如模型+規則結合。總之，Lacrosse
代碼雖可能未如其他隊前沿，但其**工程嚴謹性**和**質量控制**部分極具價值，值得AIVA研習。

**與 AIVA 的重疊與互補：**Lacrosse 和 AIVA
的**理念高度契合**：都奉行**穩定可靠優先**。AIVA
強調零誤報、高精度[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)；Lacrosse
乾脆慢慢找，寧可少拿分也不出錯[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。在這方面兩者有共鳴。這意味
AIVA 可以安心地在某些地方仿效 Lacrosse
的保守策略（如前述共識決策、信心評分），因為那符合AIVA服務客戶的需求。但也要避免重蹈Lacrosse覆轍------**過於保守導致遺漏**。畢竟
Lacrosse 只找到1個漏洞，說明有不少漏洞它們沒趕上找到。AIVA
作為掃描器若漏報太多就失職。因此互補關係在於：AIVA 可借鑒 Lacrosse
的**質量保障措施**，同時要結合其他隊的方法提高**挖掘廣度**。具體而言，AIVA
可以在**結果輸出**階段學習
Lacrosse，多重驗證後再報告，以保持可靠；但在**掃描探索**階段要學習如
Shellphish、Trail of Bits
那樣大膽運用AI和新技術，別錯失漏洞。總體來說，Lacrosse 給 AIVA
的啟示是：**穩健與創新需平衡**。AIVA
本就做得不錯，在新的AI元素引入後，更要保持這種平衡，不偏不倚，成為即有創新又讓人放心的掃描系統。

**可納入的開源工具/元件：**從 Lacrosse 的做法出發，AIVA 可導入： -
**多模型協同機制**：如之前所述，可考慮結合多個 AI
模型對重要決策投票。技術上，可利用 **HuggingFace**
提供的多模型介面，同時調用開源大模型和遠端API，彙總輸出。或者使用
**Ensemble Learning**
思想，把多個模型看法作為特徵輸入一個最終分類器。這會增大計算量，但在關鍵環節（如判斷一個高風險漏洞是否真的成立）值得投入。 -
**信心指標計算**：可以引入一些統計學方法給每個發現評分。Lacrosse或許用了簡單的
heuristics，比如按覆蓋率增量給漏洞打分或結合AI評語長度/確定詞等。AIVA
也可以研發一套信心度算法，比如根據payload命中率、返回異常特徵明顯程度等給分，高於閾值才報告。 -
**穩定性測試**：SIFT的系統經年打磨，可能加入許多檢查。例如在CRS提交補丁前，他們會用幾組完全無關輸入測試patch穩定。AIVA
若考慮自動熱修補（如臨時封鎖某攻擊），也應加這種檢測防止誤傷正常功能。可借助CI工具或快照回滾技術實現出錯自動恢復。 -
**內部文檔與模式庫**：老牌系統往往有豐富規則庫。AIVA
團隊可積累自己的漏洞模式庫並用AI維護。SIFT
10年平台想必有一套漏洞模式知識（不然AI也難生成漏洞物件）。我們可用
**Yaml/JSON** 定義常見漏洞模式，AI 每發現一件事就比對模式提建議，類似
Lacrosse
用靜態工具篩Crash。模式庫可部分來自開源（CWE資料庫、OWASP範本）。 -
**容錯和超時**：為了穩定，對AI回應超時或無效要有預案。SIFT系統可能設置了LLM最大等待時間，或fallback方案（模型A不行換模型B）。AIVA
引入AI後，也需在代碼中加入這些容錯。例如OpenAI
API超時就重試，連續失敗就跳過該AI步驟，不能整個流程卡死。

## 總結與建議

經過上述逐隊分析，可以發現 AIxCC 七支決賽隊伍的 CRS
系統各有側重，但都蘊含著可供 AIVA 借鑒的寶貴經驗。**表1**
概括了各隊值得參考的重點以及與 AIVA 的關聯性：

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  團隊（系統）         值得參考的實作與特色                                                                                                                                                                                                                                                                                                          與 AIVA 的關係
  -------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Team Atlanta**     \- **多引擎並行架構**：K8s容器化多種分析引擎並行[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)\<br/\>- **LLM 融合傳統分析**：微調 Llama2 模型輔助                                                                                   **重疊**：模組化並行理念類似，追求高覆蓋率\<br/\>**互補**：符號執行等低階技術增強AIVA程式分析深度
  （Atlantis）         fuzz/符號執行[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)\<br/\>- **強化模糊+符號**：LibAFL/AFL++ 結合 SymCC                                                                                                                        
                       提升覆蓋[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)\<br/\>-                                                                                                                                                                       
                       **保守補丁策略**：禁用無PoV補丁避免誤修[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)                                                                                                                                                

  **Trail of Bits**    \- **靜態+動態結合**：Tree-sitter 靜態查詢定位漏洞，再 fuzz                                                                                                                                                                                                                                                                   **重疊**：聚焦Web常見漏洞（SQLi等），重視結果正確\<br/\>**互補**：AI 生成惡意測試用例、靜態預檢，提高AIVA漏洞發現率和精度
  （Buttercup）        驗證[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)\<br/\>- **LLM                             
                       測試生成**：模型產生SQLi等複雜payload輔助fuzz[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)\<br/\>-                                                                                                                                  
                       **多Agent**：分離漏洞分析代理和補丁代理並行工作[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)\<br/\>- **高精度策略**：PoV                                                                                                            
                       驗證後才提交補丁，準確率90%+[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)                                                                                                                                                          

  **Theori**           \- **全AI代理框架**：多個LLM代理分工找漏洞、寫exploit、出補丁[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)\<br/\>-                                                                                                                **重疊**：目標都是全自動漏洞挖掘，使用靜態分析輔助\<br/\>**互補**：AI代理自治能力極強，可為AIVA引入智能推理、邏輯漏洞發現的新模式
  （RoboDuck）         **最小化傳統Fuzz**：主要靠AI理解程式推理漏洞[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)\<br/\>- **Infer 靜態輔助**：用 Facebook                                                                                                 
                       Infer找潛在bug，供AI深入分析[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)\<br/\>-                                                                                                                                                 
                       **積極補丁策略**：允許無PoV補丁嘗試，加快修補節奏[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)                                                                                                                                    

  **AllYouNeed**       \- **超大規模AI併發**：平行啟動上千LLM代理分析，同時嘗試多路徑[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)\<br/\>-                                                                                                               **重疊**：都是應用AI於安全測試，自動化程度高\<br/\>**互補**：AI平行計算、AI過濾誤報思路可用於提升AIVA效率與結果品質，但需節制資源，找平衡
  （Fuzzing Brain）    **AI三角驗證**：AI濾除80%誤報，SARIF精度全場最佳[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)\<br/\>-                                                                                                                             
                       **敏捷發現**："Pizza速度"5分鐘內出漏洞，得首報獎[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)\<br/\>-                                                                                                                             
                       **AI代碼產出**：90%代碼由AI輔助生成，驗證AI編程潛力[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)                                                                                                                                  

  **Shellphish**       \- **多AI多模塊**：60+代理分工合作，模糊、符號、exploit全部門聯動[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)\<br/\>- **Grammar 学習**：Grammar Guy                                                                              **重疊**：微服務/agent架構類似，模糊測試+符號執行+AI與AIVA思路相近\<br/\>**互補**：符號執行強項補足AIVA弱點；引入Grammar推理、自適應fuzz讓AIVA更聰明；多模型冗餘提高AIVA
  （Artiphishell）     AI從反饋歸納輸入格式，提升fuzz效果[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)\<br/\>-                                                                                                                                           AI決策可靠性
                       **多雲LLM**：結合OpenAI/Anthropic/Gemini優勢，避免單點偏差[\[25\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L36)\<br/\>-                                                                                                                   
                       **頂尖遙測**：詳細記錄AI決策流程，方便除錯和解釋[\[28\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L37)\<br/\>-                                                                                                                             
                       **補丁零誤差**：堅持無PoV不補丁，補丁成功率95%+[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)                                                                                                                                      

  **42/b3yond/6ug**    \- **BandFuzz RL調度**：強化學習分配2000+ fuzz                                                                                                                                                                                                                                                                                **重疊**：架構類似企業級系統，AIVA可直接借鑒其消息、資料庫協調方案\<br/\>**互補**：RL自適應調度可優化AIVA掃描任務分配；AI輔助輸入、靜態結合動態結果分析提升AIVA效率和低誤報
  （Bug Buster）       worker，最優利用資源[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)[\[43\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L20)\<br/\>-   
                       **全佈署架構**：RabbitMQ+PostgreSQL+Redis組合，支撐大規模並發[\[33\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L17)\<br/\>-                                                                                                         
                       **LLM輔助種子**：訓練模型生成有價值fuzz初始輸入，提速探索[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)\<br/\>-                                                                                                                    
                       **Crash去重**：自動聚類Crash根因，SARIF靜態驗證，拿SARIF準確獎[\[39\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L46)\<br/\>-                                                                                                               
                       **超級補丁**：嘗試單一補丁修多漏洞，概念超前但實用性待驗[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)                                                                                                                             

  **SIFT**             \- **舊平臺新生**：10年CRS經驗+AI升級，穩定可靠[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)\<br/\>- **Optimus Zero**：中央管理500                                                                                                **重疊**：強調穩定性和高精確，AIVA的零誤報目標與其一致\<br/\>**互補**：AI多模共識、信心評分機制可增強AIVA結果可信度；AIVA可從其經驗汲取穩健風控措施，同時避免過度保守導致漏報
  （Lacrosse）         fuzz節點，穩定輸出但速度較慢[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)\<br/\>-                                                                                                                                                 
                       **AI漏洞物件**：LLM將Crash描述成漏洞對象，含分類和修復建議[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)\<br/\>-                                                                                                                   
                       **AI共識補丁**：多模型審核補丁方案，無共識不提交，極低誤修率[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)\<br/\>-                                                                                                                 
                       **傳統優先**：主要靠經典fuzz+工具找洞，AI作輔助決策，誤報極少[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)                                                                                                                        
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

綜上所述，針對 AIVA 平臺的架構擴充，建議重點關注以下幾方面：

1.  **引入 AI 輔助測試生成與分析**：利用 Trail of Bits、Shellphish
    等隊經驗，讓AIVA現有的特化AI系統參與漏洞掃描的關鍵環節。例如，開發基於 AIVA特化AI
    的 Payload
    Generator，針對發現的線索自動產生高涵蓋率的測試輸入（如特殊字串、結構化資料），提升對複雜漏洞（SQLi、RCE）的探測能力[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)。同時，部署
    AI
    驅動的結果分析代理，對掃描可能產生的大量警告/異常進行歸類和篩選，過濾明顯誤報，僅保留高可信度結果（類似
    AllYouNeed 利用 AI 濾除 \>80%
    靜態誤報[\[44\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L16-L19)）。這將大幅減輕人工審核負擔，使
    AIVA 的報告更精簡精準。

2.  **融合靜態分析與動態掃描**：借鑒 Buttercup、RoboDuck 等的做法，在
    AIVA
    現有動態掃描流程前後加入靜態分析模組。一方面，可在掃描前用靜態工具（如
    CodeQL、Infer[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)）快速標記潛在漏洞熱點，指導
    AIVA
    重點測試高風險區域；另一方面，對掃描產生的漏洞候選，進一步用靜態分析驗證背後的漏洞模式是否存在，提升置信度。例如，若動態掃描懷疑某API有SQLi，可用靜態分析檢查該API實現中是否直接拼接SQL字串[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)。靜態與動態結合，將提高覆蓋率並降低誤報，實現1+1\>2的效果。

3.  **應用大規模分散式調度優化**：參考 Bug Buster 的 BandFuzz
    平臺[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)和
    Team Atlanta Kubernetes
    佈署經驗[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)，將
    AIVA
    的任務調度升級為智能化、可伸縮的雲端系統。一方面，引入強化學習或自適應演算法，根據掃描過程中的反饋動態調整爬虫深度、測試強度分配------例如，某些子域頻繁出現漏洞，則自動增加對該子域的測試；反之則減少資源浪費。另一方面，充分利用容器編排（Kubernetes）在多節點上平行執行掃描模組，提高掃描速度和併發能力。RabbitMQ、Redis
    等中介件已在 AIVA 中使用，可擴充其規模佈署並結合 RL
    調度，使系統既能處理單一大型目標的深度掃描，也能同時覆蓋海量目標的廣度掃描，而不浪費資源。

4.  **增強輸入智能處理與Grammar推理**：藉鑒 Shellphish "Grammar Guy"
    模組[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)，使
    AIVA
    對特殊輸入格式具備學習與適應能力。實現路徑包括：利用已掃描的HTTP請求/回應數據構建語法樹模型，讓AI分析參數值的結構（JSON、JWT、XML等），然後自動構造符合該結構的惡意payload進行測試。也可以讓AI觀察多輪fuzz測試沒覆蓋到的路徑，推測需要哪種類型的輸入才能觸發（例如根據程式錯誤訊息，AI
    推測某參數需要特定格式才進入漏洞點）。將這些功能作為模組加入 AIVA
    的"核心分析引擎"中[\[15\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L16-L24)，可顯著提升對複雜場景（例如二階段驗證、多步驟交易流程）的掃描效果。

5.  **建立多層次驗證與共識機制**：受 Lacrosse 啟發，對 AIVA
    報告輸出的漏洞採用**多層驗證**。對每個高危漏洞，由系統嘗試自動生成PoC並再次測試確認；同時，引入**多模組/多模型共識**：例如讓不同偵測引擎（XSS和Crawler）或不同AI模型各自對該漏洞給出判斷，只有一致認定為真漏洞才列入報告[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)。舉例而言，若OpenAI模型和本地Fine-tune模型都判斷某輸出為SQLi，則誤報機率大降。這種共識機制也可應用在自動防禦決策上，如未來實裝自動封堵攻擊IP，需多重條件滿足才執行，避免誤封。總之，透過**投票表決、信心評分**等方式，提升
    AIVA 輸出的可靠度，維持產品可信賴形象。

6.  **強化元件復用與社群合作**：競賽催生了許多實用工具，AIVA
    應大膽引入這些**開源元件**：例如 DARPA 官方開源的 **SHERPA** 測試
    harness
    自動生成工具[\[45\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L16)。SHERPA
    可智慧選擇軟體中未被測試的外部輸入點，利用 LLM 自動生成相應fuzz
    harness[\[45\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L16)，對
    AIVA
    來說，這意味未來若要對某Web應用進行源碼級fuzz，可快速產生測試框架，無需工程師手寫樣板代碼。又如
    **BandFuzz** (北西大團隊開源)
    可作為子系統融入，用於對特定模組（比如上傳檔案解析引擎）進行強化學習優化模糊測試[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)。再看
    Shellphish 開源的 **angr** 符號執行框架，可集成進 AIVA
    對某些深度路徑進行符號分析，找出一般fuzz不到的邏輯漏洞。利用這些開源利器，能省卻大量開發時間，並站在巨人肩膀上完善
    AIVA 功能模組。建議團隊持續關注 AIxCC
    及相關社群動態，積極採納成熟工具。例如，在確保授權兼容的前提下，把
    BandFuzz、SHERPA 作為 AIVA 的插件，提供進階掃描模式；或者將
    Shellphish 的 Grammar學習邏輯融入 AIVA 的模糊測試模組等。

最後，AIVA
團隊在擴充架構時應當綜合考量**工程實用**與**前沿探索**的平衡。AIxCC
七強系統提供了寶貴的多樣化路線：有的側重AI創新（如
RoboDuck、Fuzzing Brain），有的堅守穩健傳統（如
Lacrosse），還有的融合二者達成佳績（如 Buttercup、Artiphishell）。對
AIVA
而言，最佳策略可能是**漸進融合**：先引入可靠度高、易實現的改進（如靜態分析輔助、共識機制、開源工具整合），在確保現有功能品質不受影響前提下提升效率與精度；同時，在研發路線上投入對
AI
代理、自適應學習等前沿方向的預研，小範圍試點應用（比如新增一兩個AI代理輔助特定漏洞類型掃描），逐步驗證效果。如此一來，AIVA
平臺將能穩步演進，既借鑒競賽頂尖成果，又保持產品級穩定性，最終實現**更智能、高效、全面**的安全評估能力，在自動化漏洞掃描領域保持領先。[\[5\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L50-L58)[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)

**參考資料：**[\[1\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L12-L15)[\[2\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L10-L18)[\[9\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L4)[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)

------------------------------------------------------------------------

[\[1\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L12-L15)
[\[7\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L19-L23)
[\[8\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L3)
[\[9\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L1-L4)
[\[10\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L2-L4)
[\[11\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L4-L6)
[\[12\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L5-L9)
[\[13\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L10)
[\[14\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L7-L10)
[\[16\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L28-L30)
[\[17\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L29-L30)
[\[18\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L35)
[\[19\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)
[\[20\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L32-L34)
[\[21\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L43)
[\[22\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L39)
[\[25\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L36)
[\[26\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L39)
[\[27\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L36-L37)
[\[28\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L35-L37)
[\[30\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L39-L41)
[\[34\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)
[\[35\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L41-L43)
[\[36\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L40-L41)
[\[37\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L44-L46)
[\[39\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L46)
[\[40\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L42-L43)
[\[44\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L16-L19)
[\[45\]](https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md#L8-L16)
aixcc.md

<https://github.com/CyberSecAI/CyberSecAI.github.io/blob/6ff998b8b63473d54717f78eead297bf2650064e/docs/software/aixcc.md>

[\[2\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L10-L18)
[\[3\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L22-L30)
[\[4\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L52-L58)
[\[5\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L50-L58)
[\[6\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L60-L68)
[\[15\]](https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md#L16-L24)
README.md

<https://github.com/kyle0527/azuredev-0c43/blob/dc68a59963585171d521871af16b210aa6f6bfa9/D/Warprecon-D/README.md>

[\[23\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L40-L48)
[\[24\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L66-L74)
[\[29\]](https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md#L24-L32)
README.md

<https://github.com/shellphish/artiphishell/blob/951db005027caccb279aeb20291e7da495d43781/README.md>

[\[31\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L28-L37)
[\[32\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L6-L14)
[\[33\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L17)
[\[38\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L44-L52)
[\[41\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L21)
[\[42\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L78-L86)
[\[43\]](https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md#L13-L20)
README.md

<https://github.com/42-b3yond-6ug/42-b3yond-6ug-crs/blob/1f249de171df528ec7427787701b67a0eaeb5840/components/bandfuzz/README.md>
