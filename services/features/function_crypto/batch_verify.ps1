# AIVA Crypto Scanner - 批量驗證腳本 (WebGoat 路徑修正版)
# 自動偵測執行檔路徑並針對正確 URL 進行掃描

# 1. 自動偵測 crypto-scanner.exe 位置
$PossiblePaths = @(
    "C:\D\fold7\AIVA-git\function_crypto\rust_core\target\release\crypto-scanner.exe",
    "C:\D\fold7\AIVA-git\target\release\crypto-scanner.exe"
)

$ExePath = ""
foreach ($path in $PossiblePaths) {
    if (Test-Path $path) {
        $ExePath = $path
        Write-Host "[*] Found scanner at: $ExePath" -ForegroundColor Gray
        break
    }
}

if (-not $ExePath) {
    Write-Host "[!] 找不到 crypto-scanner.exe，請確認是否已編譯 (cargo build --release)" -ForegroundColor Red
    exit 1
}

# 2. 定義目標 (修正 WebGoat/WebWolf 的路徑)
$Targets = @(
    @{ Name="JuiceShop-Live"; Url="http://localhost:3000"; Port=3000 },
    @{ Name="JuiceShop-Dev";  Url="http://localhost:3001"; Port=3001 },
    @{ Name="JuiceShop-Test"; Url="http://localhost:3003"; Port=3003 },
    @{ Name="WebGoat-App";    Url="http://localhost:8080/WebGoat"; Port=8080 }, # 修正：加上 /WebGoat
    @{ Name="WebWolf";        Url="http://localhost:9090/WebWolf"; Port=9090 }  # 修正：加上 /WebWolf (通常 9090 是 WebWolf)
)

Write-Host "========== AIVA Crypto Scanner 驗證修正版 ==========" -ForegroundColor Cyan
$totalFindings = 0

foreach ($t in $Targets) {
    Write-Host "`n[+] 正在掃描: $($t.Name) ($($t.Url))" -ForegroundColor Yellow
    
    try {
        # 3. 取得 HTTP 回應
        # WebGoat 可能會轉址到登入頁，Session 預設會保留
        $r = Invoke-WebRequest -Uri $t.Url -UseBasicParsing -SessionVariable "sess" -ErrorAction Stop
        Write-Host "    ✓ HTTP 連接成功 (狀態碼: $($r.StatusCode))" -ForegroundColor Green
        
        # --- 分析 Headers ---
        $h = @{}
        $r.Headers.GetEnumerator() | % { $h[$_.Key] = $_.Value -join ', ' }
        $jsonHeaders = $h | ConvertTo-Json -Compress
        
        Write-Host "    [*] 分析安全頭..." -NoNewline
        $headerResult = & $ExePath analyze-headers --url $t.Url --headers-json $jsonHeaders | ConvertFrom-Json
        if ($headerResult.summary.total_findings -gt 0) {
            Write-Host " 發現問題!" -ForegroundColor Red
            $headerResult.findings | ForEach-Object { Write-Host "        - [$($_.severity)] $($_.title)" -ForegroundColor Yellow }
            $totalFindings += $headerResult.summary.total_findings
        } else {
            Write-Host " OK" -ForegroundColor Green
        }

        # --- 分析 Cookies ---
        $cookies = @()
        
        # 嘗試從回應或 Session 中抓取 Cookie
        if ($r.Headers['Set-Cookie']) {
            if ($r.Headers['Set-Cookie'] -is [array]) { $cookies += $r.Headers['Set-Cookie'] } 
            else { $cookies += @($r.Headers['Set-Cookie']) }
        }
        # 如果有 Session Cookie (如 JSESSIONID)
        if ($sess -and $sess.Cookies) {
            $sess.Cookies.GetCookies($t.Url) | ForEach-Object { 
                # 簡單格式化為 Set-Cookie 風格字串以便分析
                $cStr = "$($_.Name)=$($_.Value); Path=$($_.Path)"
                if ($_.Secure) { $cStr += "; Secure" }
                if ($_.HttpOnly) { $cStr += "; HttpOnly" }
                $cookies += $cStr
            }
        }
        
        # 去重
        $cookies = $cookies | Select-Object -Unique

        if ($cookies.Count -gt 0) {
            Write-Host "    [*] 分析 Cookies ($($cookies.Count))..." -NoNewline
            $cookieJson = $cookies | ConvertTo-Json -Compress
            $cookieResult = & $ExePath analyze-cookies --url $t.Url --cookies-json $cookieJson | ConvertFrom-Json
            
            if ($cookieResult.summary.total_findings -gt 0) {
                Write-Host " 發現問題!" -ForegroundColor Red
                $cookieResult.findings | ForEach-Object { Write-Host "        - [$($_.severity)] $($_.title)" -ForegroundColor Yellow }
                $totalFindings += $cookieResult.summary.total_findings
            } else {
                Write-Host " OK" -ForegroundColor Green
            }
        } else {
            Write-Host "    [*] 無 Cookie 可分析" -ForegroundColor Gray
        }

        # --- TLS 分析 (使用 Port) ---
        Write-Host "    [*] 分析 TLS 配置..." -NoNewline
        $tlsResult = & $ExePath analyze-tls --target localhost --port $t.Port | ConvertFrom-Json
        
        if ($tlsResult.summary.total_findings -gt 0) {
            Write-Host " 結果: $($tlsResult.findings[0].title)" -ForegroundColor Gray
            $totalFindings += $tlsResult.summary.total_findings
        } else {
            Write-Host " OK" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "    [!] 掃描失敗: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n========== 驗證結束 (總計: $totalFindings 個問題) ==========" -ForegroundColor Cyan