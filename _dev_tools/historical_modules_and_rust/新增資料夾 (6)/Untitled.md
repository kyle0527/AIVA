# Go Engine SSRF 掃描器修復與深度驗證指南

本指南旨在提供一份詳盡的技術說明與操作手冊，協助開發者、安全工程師及測試人員應用關鍵的代碼修復補丁。主要目標是解決
Go Engine 在針對需要身份驗證的複雜靶場（如 WebGoat、DVWA
或企業內部應用）進行 Server-Side Request Forgery (SSRF)
掃描時所遇到的核心問題。此外，本文件還包含了完整的編譯環境建置流程、詳細的分步驗證步驟，以及針對真實網路環境的故障排除策略與進階調試技巧。

## **0. 背景與問題分析**

在現代動態應用程式安全測試 (DAST)
中，許多高風險的漏洞往往隱藏在受保護的端點（Protected
Endpoints）之後，僅對已登錄的用戶開放。傳統的掃描器若缺乏處理會話狀態（Session
State）的能力，所有的探測請求（Probes）都會被應用程式的權限過濾器攔截，並重定向至登錄頁面（通常是
HTTP 302/301 跳轉）。

這會導致兩個嚴重的後果：

1.  **漏報 (False
    Negatives)**：掃描器無法觸及真正的業務邏輯，導致隱藏在深處的 SSRF
    漏洞未被發現。

2.  **誤報 (False
    Positives)**：掃描器可能錯誤地將登錄頁面的某些特徵解析為漏洞指標，或者因為無法正確解析重定向後的頁面而產生錯誤判斷。

本次修復的核心目標，即是賦予 SSRF 引擎處理自定義 HTTP Headers（包括
Cookie、Authorization Bearer Token、X-API-Key
等）的能力，使其能夠模擬合法用戶的行為，穿透認證防線進行深層掃描。

## **1. 修復內容技術詳解**

我們對 internal/ssrf/detector/ssrf.go
核心檢測邏輯進行了架構層面的升級。以下是本次修復的技術細節、代碼變更原理及其影響範圍：

### **A. 增強 JSON 輸入處理架構 (Data Unmarshalling)**

原本的架構設計中，雖然定義了數據傳輸對象 (DTO) 結構，且 JSON
輸入中包含了 headers
欄位，但在數據流傳遞至底層掃描函數的過程中，這個欄位被意外忽略了。這導致
Go Engine 與 Python Coordinator 之間的通訊協議存在斷層。

- 修復前狀態：\
  掃描函數 Scan 僅接收 targets 列表作為參數。當 Python 端傳遞包含
  Session ID 的 JSON 時，Go 端雖然能解碼
  JSON，但在呼叫業務邏輯時丟棄了這些關鍵資訊。

- 修復後狀態：\
  我們重構了 Scan 與 scanSingleTarget 方法的函式簽名 (Function
  Signature)，使其能夠接收並傳遞 map\[string\]string 類型的 headers
  參數。這使得引擎不僅能處理標準的 Cookie 驗證，還能靈活適應各種現代化的
  API 驗證機制（如 OAuth2 Token 或自定義的安全標頭）。

### **B. HTTP 請求注入與客戶端行為控制 (Request Injection)**

除了數據傳遞，我們還在底層的 http.NewRequest 構造流程中加入了 Header
注入邏輯，並微調了 HTTP Client 的行為。

- 動態請求構建：\
  在建立每一個 SSRF 測試 Payload 的 HTTP
  請求時，系統現在會自動遍歷配置中的 Headers map，並使用
  req.Header.Set(key, value)
  方法將其逐一注入。這確保了每一個探測包（Probe）------無論是用於檢測
  AWS Metadata
  的請求，還是測試本地文件讀取的請求------都帶有有效的身份憑證。

- 業務邏輯必要性：\
  以 WebGoat 為例，這是一個基於 Java Spring Boot
  的應用。若請求中缺少有效的 JSESSIONID Cookie，Spring Security
  過濾器會立即攔截請求並返回 HTTP 302 Redirect，將使用者導向
  /WebGoat/login。

  - **未修復時**：掃描器收到
    302，自動跟隨跳轉至登錄頁，檢測到登錄表單特徵，判定為「非漏洞」，導致掃描無效。

  - **修復後**：掃描器攜帶 Session ID，服務器返回 HTTP 200
    及業務頁面內容，掃描器成功注入 Payload 並觸發 SSRF
    行為（如延時或特定內容回顯），成功識別漏洞。

## **2. 完整應用修復與編譯流程**

為了確保修復生效，必須重新編譯 Go 二進制文件。請依照以下詳細步驟，在
PowerShell 環境中操作。確保您的開發環境中已安裝 Go 1.20
或更高版本，並且環境變數 GOPATH 與 GOROOT 設置正確。

\# 1. 切換工作目錄\
\# 請確保您位於 Go Engine 的根目錄，這是所有模組依賴解析的基準點 (Module
Root)\
cd C:\\D\\fold7\\AIVA-git\\services\\scan\\engines\\go_engine\
\
\# 2. 環境檢查與清理 (Best Practice)\
\# 檢查 Go 版本以避免語法兼容性問題\
go version\
\# 刪除舊檔案是為了避免 Windows 檔案鎖定 (File Locking)
問題，並確保我們運行的是全新編譯的版本\
Write-Host \"正在清理舊的執行檔\...\" -ForegroundColor Yellow\
if (Test-Path bin/ssrf-scanner.exe) {\
Remove-Item bin\\ssrf-scanner.exe -Force -ErrorAction SilentlyContinue\
}\
\
\# 3. 整理依賴模組 (Dependency Management)\
\# 如果您修改了 import 路徑或 go.mod 文件，此步驟至關重要。\
\# 它會自動下載缺少的包並移除不再使用的依賴，更新 go.sum 校驗和。\
Write-Host \"正在整理 Go 模組依賴\...\" -ForegroundColor Cyan\
go mod tidy\
\
\# 4. 編譯新的掃描器 (Build Process)\
\# -o 指定輸出路徑\
\# -v 顯示編譯過程中的包名稱 (Verbose)\
\# cmd/ssrf-scanner/main.go 是程式的入口點 (Main Package)\
Write-Host \"正在編譯 SSRF Scanner\...\" -ForegroundColor Cyan\
go build -v -o bin/ssrf-scanner.exe cmd/ssrf-scanner/main.go\
\
\# 5. 驗證編譯結果 (Verification)\
if (Test-Path bin/ssrf-scanner.exe) {\
\$item = Get-Item bin\\ssrf-scanner.exe\
Write-Host \"\`n✅ 編譯成功！\" -ForegroundColor Green\
Write-Host \" 檔案位置: \$(\$item.FullName)\"\
Write-Host \" 檔案大小: \$(\[math\]::Round(\$item.Length / 1MB, 2))
MB\"\
Write-Host \" 修改時間: \$(\$item.LastWriteTime)\"\
\
\# 簡單運行一次以確保沒有 Runtime Panic (如 DLL 缺失)\
.\\bin\\ssrf-scanner.exe \--help 2\>\$null\
} else {\
Write-Error \"❌ 編譯失敗，請檢查上方的編譯器錯誤訊息 (Compiler
Errors)。\"\
exit 1\
}

## **3. 深度驗證步驟操作手冊**

驗證過程分為三個階段：前期的憑證獲取
(Reconnaissance)、針對單一漏洞點的深度掃描
(Exploitation)、以及針對多目標的混合壓力測試 (Regression Testing)。

### 步驟 A: 獲取 WebGoat Session ID **(偵察階段)**

WebGoat 使用 JSESSIONID cookie 來追蹤用戶登錄狀態。由於此 Session
具有時效性（通常閒置 30
分鐘後過期），每次開始新的測試回合前，建議重新獲取。

1.  啟動與訪問：\
    確保您的 WebGoat Docker 容器已啟動且運行正常。使用瀏覽器訪問服務地址
    (通常是 http://localhost:8080/WebGoat)。

2.  登錄操作：\
    輸入您的使用者名稱與密碼進行登錄。如果您是初次使用該靶場，可能需要先註冊一個新帳號。登錄成功後，確保能看到主儀表板。

3.  開啟開發者工具：\
    在瀏覽器頁面按 F12 或右鍵點擊頁面選擇「檢查 (Inspect)」以打開
    DevTools。

4.  **定位 Cookie 存儲**：

    - **Chrome/Edge**: 點擊上方標籤列的 **Application**，在左側選單展開
      **Storage** -\> **Cookies**，點擊 WebGoat 的網址。

    - **Firefox**: 點擊上方標籤列的 **Storage**，在左側選單展開
      **Cookies**。

5.  提取憑證：\
    在右側列表中找到名稱為 JSESSIONID
    的條目。雙擊其「Value」欄位並複製該字串（格式通常類似
    C2B134\...）。注意：不要複製到分號或空格。

### **步驟 B: 執行帶權限的深度掃描 (漏洞驗證階段)**

使用我們專門編寫的 test_webgoat_auth.ps1 腳本進行測試。此腳本模擬了
Python Adapter 的行為，構造標準的 JSON 結構並將其注入到掃描器的標準輸入
(stdin) 中。

.\\test_webgoat_auth.ps1

操作提示：

當腳本執行後，終端機將暫停並提示 請輸入 JSESSIONID。請直接貼上剛才複製的
ID 字串並按 Enter 鍵。

**預期結果與日誌分析**:

- 執行狀態 (Status)：\
  JSON 輸出中的 status 欄位必須顯示為 success。若顯示 partial 或
  failed，請檢查下方的 errors 陣列以獲取詳細錯誤原因。

- 資產發現 (Assets Count)：\
  assets 陣列的長度應大於 0。WebGoat 的 SSRF 練習題 (Task 1)
  通常設計為可被利用，因此掃描器應能發現至少一個漏洞。

- 日誌行為監控：\
  觀察 fix_test.log 或控制台輸出。修復後的版本不應再出現大量的「Redirect
  to login」或 HTTP 302 狀態碼。正常的 Payload 測試請求應返回 HTTP
  200，或者在觸發服務端錯誤時返回 HTTP 500。

- **漏洞詳情解讀**：

  - **Severity (嚴重性)**：根據檢測到的行為，應標示為 High (如成功讀取
    Metadata) 或 Medium (如掃描到內網端口)。

  - **Confidence (置信度)**：如果是基於精確特徵（例如 Payload 請求了 AWS
    Metadata 且響應中包含 ami-id），置信度應為
    High。如果僅是狀態碼異常，則為 Low。

### **步驟 C: 多目標與連通性壓力測試 (回歸測試階段)**

為了驗證修復代碼是否引入了副作用（例如破壞了對無須認證目標的掃描能力），以及驗證掃描器在並發環境下的穩定性，請執行多目標測試腳本：

.\\test_multi_targets.ps1

測試場景設計說明：

此腳本同時對三個不同性質的目標發起掃描，測試引擎的調度能力：

1.  httpbin.org：公網 Echo 服務。用於驗證掃描器對外網的連通性、DNS
    解析能力以及對標準 HTTP 協議的支援。

2.  example.com：簡單的靜態頁面。作為「陰性對照組」，預期不應發現任何
    SSRF 漏洞。若發現漏洞，則可能存在誤報邏輯。

3.  localhost:8080：WebGoat 內網服務。測試對 localhost loopback
    地址的解析，以及在請求被拒絕 (Connection Refused) 時的錯誤處理機制。

**預期結果**:

- 掃描器應能成功解析並連接所有網路可達的目標。

- httpbin.org 的掃描應快速完成並返回結果。

- **無崩潰 (Panic
  Free)**：掃描過程應平穩結束，不會因為某個目標連接超時或重置而導致整個進程崩潰。

## **4. 常見問題與故障排除 (FAQ)**

在實際操作中，您可能會遇到以下問題。請參考對應的解決方案：

Q1: 為什麼掃描結果顯示 \"0 assets found\"，但我確定 Session ID
是正確的？

A: 這種情況通常由網路隔離或路徑錯誤引起：

1.  **Docker 網路隔離**：如果您在 Docker 容器內運行
    WebGoat，而掃描器在宿主機 (Windows) 上運行，localhost
    對宿主機而言是通的。但如果兩者都在**不同的** Docker
    容器中，它們之間是隔離的。您可能需要使用 host.docker.internal
    或容器的真實 IP 地址。

2.  **URL 路徑不精確**：WebGoat 的漏洞通常位於特定的練習頁面（如
    /WebGoat/SSRF/task1），而非首頁。掃描器只會對輸入的 URL 及其參數進行
    Fuzzing，不會自動爬取全站。請確認 targets 列表中的 URL
    是漏洞所在的具體入口點。

3.  **Session 過早過期**：WebGoat 的 Session
    預設超時時間較短。如果掃描時間設置過長（例如 timeout \>
    300s），Session 可能在掃描中途失效。請嘗試重新登錄獲取新的
    ID，並適當減少並發數或超時時間。

Q2: 執行時出現 \"exec: \...: file does not exist\" 或 \"The system
cannot find the path specified\" 錯誤？

A: 這表示編譯未成功或路徑引用錯誤。

1.  確認 bin/ssrf-scanner.exe 檔案是否確實存在於目錄中。

2.  PowerShell 對路徑解析較為嚴格，確認您是以相對路徑
    .\\bin\\ssrf-scanner.exe 呼叫，而不是直接輸入文件名。

Q3: 如何查看更詳細的 HTTP 請求與響應內容？

A: 目前的 main.go 初始化時使用了 zap.NewDevelopment() Logger，它會在
DEBUG 層級記錄詳細資訊。

您可以在 JSON 輸入中添加 \"enable_verbose_log\": true
(如果代碼支援該參數)，或者直接觀察 stderr 的輸出。在 fix_test.log
中可以看到每個 Payload
的注入情況、目標響應的狀態碼以及部分響應內容，這對於分析為何某些 Payload
未能觸發漏洞非常有幫助。

Q4: 掃描器是否支援 HTTPS 目標？遇到自簽名證書怎麼辦？

A: 是的，掃描器完全支援 HTTPS。然而，如果靶場使用自簽名證書 (Self-signed
Certificate) 或過期的證書，標準的 Go HTTP Client 預設會拒絕連接並報錯
x509: certificate signed by unknown authority。

目前的實作中，NewSSRFDetector 內的 http.Client 若未配置
InsecureSkipVerify:
true，則會遭遇此問題。若您需要掃描此類目標，請在源代碼中修改
TLSClientConfig 設定。
