# =====================================================================
# Docker 靶場多目標驗證腳本
# =====================================================================
# 用途: 對截圖中的 Docker 容器進行多種類型的安全掃描
# 目標:
#   - juice-shop-live (3000:3000)
#   - ecstatic_ritchie (3001:3000)
#   - vigilant_shockle (3003:3000)
#   - laughing_jang (8080:8080, webgoat/webgoat)
# =====================================================================

param(
    [switch]$Verbose
)

Write-Host "`n🎯 ============================================" -ForegroundColor Cyan
Write-Host "   AIVA Go Engine - 多目標靶場驗證" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# 定義目標
$targets = @(
    @{
        Name = "juice-shop-live"
        URL = "http://localhost:3000"
        Port = 3000
        Type = "OWASP Juice Shop"
        Tests = @("SSRF", "XSS", "SQL Injection")
    },
    @{
        Name = "ecstatic_ritchie"
        URL = "http://localhost:3001"
        Port = 3001
        Type = "Unknown Service"
        Tests = @("SSRF", "Port Scan")
    },
    @{
        Name = "vigilant_shockle"
        URL = "http://localhost:3003"
        Port = 3003
        Type = "Unknown Service"
        Tests = @("SSRF", "Port Scan")
    },
    @{
        Name = "laughing_jang"
        URL = "http://localhost:8080"
        Port = 8080
        Type = "WebGoat"
        Tests = @("SSRF", "Path Traversal")
    }
)

# SSRF Payloads
$ssrfPayloads = @(
    @{
        Name = "AWS IMDS"
        URL = "http://169.254.169.254/latest/meta-data/"
        Risk = "HIGH"
    },
    @{
        Name = "GCP Metadata"
        URL = "http://metadata.google.internal/computeMetadata/v1/"
        Risk = "HIGH"
    },
    @{
        Name = "Localhost Admin"
        URL = "http://127.0.0.1:80/admin"
        Risk = "MEDIUM"
    },
    @{
        Name = "Localhost Alt Port"
        URL = "http://127.0.0.1:8080/"
        Risk = "MEDIUM"
    },
    @{
        Name = "IPv6 Localhost"
        URL = "http://[::1]/"
        Risk = "MEDIUM"
    },
    @{
        Name = "Wildcard IP"
        URL = "http://0.0.0.0/"
        Risk = "MEDIUM"
    },
    @{
        Name = "Private IP"
        URL = "http://192.168.1.1/"
        Risk = "LOW"
    }
)

# 結果統計
$global:totalTests = 0
$global:vulnerableTests = 0
$global:blockedTests = 0
$global:errorTests = 0

# =====================================================================
# 測試函數
# =====================================================================

function Test-ContainerAvailability {
    param([hashtable]$Target)
    
    Write-Host "  [檢查] 容器可用性..." -ForegroundColor Gray
    
    try {
        $response = Invoke-WebRequest -Uri $Target.URL -Method Head -TimeoutSec 3 -ErrorAction Stop
        Write-Host "  ✓ 容器在線 (Status: $($response.StatusCode))" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ✗ 容器離線或無響應" -ForegroundColor Red
        return $false
    }
}

function Test-SSRFVulnerability {
    param(
        [hashtable]$Target,
        [hashtable]$Payload
    )
    
    $global:totalTests++
    
    # 構造測試 URL (常見 SSRF 參數名)
    $testParams = @("url", "target", "redirect", "link", "src", "dest", "path")
    
    foreach ($param in $testParams) {
        $testUrl = "$($Target.URL)?$param=$([System.Uri]::EscapeDataString($Payload.URL))"
        
        if ($Verbose) {
            Write-Host "    [測試] $($Payload.Name) via `$$param" -ForegroundColor DarkGray
        }
        
        try {
            $response = Invoke-WebRequest -Uri $testUrl -TimeoutSec 5 -MaximumRedirection 0 -ErrorAction Stop
            
            $bodyPreview = $response.Content.Substring(0, [Math]::Min(200, $response.Content.Length))
            
            # 檢查敏感關鍵字
            $sensitiveKeywords = @("ami-id", "instance-id", "iam/security-credentials", "computeMetadata", 
                                   "config", "password", "secret", "token", "api_key", "AWS", "credentials")
            
            $foundKeywords = $sensitiveKeywords | Where-Object { $bodyPreview -match $_ }
            
            if ($response.StatusCode -eq 200 -and $foundKeywords.Count -gt 0) {
                Write-Host "    🚨 VULNERABLE: $($Payload.Name) [$($Payload.Risk)]" -ForegroundColor Red
                Write-Host "       參數: $param" -ForegroundColor Yellow
                Write-Host "       關鍵字: $($foundKeywords -join ', ')" -ForegroundColor Yellow
                Write-Host "       響應預覽: $($bodyPreview.Substring(0, [Math]::Min(80, $bodyPreview.Length)))..." -ForegroundColor DarkYellow
                $global:vulnerableTests++
                return $true
            } elseif ($response.StatusCode -eq 200) {
                if ($Verbose) {
                    Write-Host "    ⚠️  可疑響應 (200 但無敏感資訊)" -ForegroundColor Yellow
                }
            }
            
        } catch [System.Net.WebException] {
            $statusCode = [int]$_.Exception.Response.StatusCode
            if ($statusCode -in @(403, 404, 500)) {
                if ($Verbose) {
                    Write-Host "    ✓ 已阻擋或不存在 (Status: $statusCode)" -ForegroundColor Green
                }
                $global:blockedTests++
            } else {
                if ($Verbose) {
                    Write-Host "    ⚠️  錯誤: $statusCode" -ForegroundColor Yellow
                }
                $global:errorTests++
            }
        } catch {
            if ($Verbose) {
                Write-Host "    ✓ 連接失敗 (已阻擋)" -ForegroundColor Green
            }
            $global:blockedTests++
        }
    }
    
    return $false
}

function Test-ContainerInternals {
    param([hashtable]$Target)
    
    Write-Host "`n  [深度檢測] 容器內部掃描" -ForegroundColor Cyan
    
    # 檢查容器是否存在
    $containerExists = docker ps --format "{{.Names}}" | Select-String -Pattern "^$($Target.Name)$" -Quiet
    
    if (-not $containerExists) {
        Write-Host "    ⚠️  容器 $($Target.Name) 未運行,跳過內部掃描" -ForegroundColor Yellow
        return
    }
    
    # 1. 檢查開放端口
    Write-Host "    [端口掃描]" -ForegroundColor Gray
    try {
        $ports = docker exec $Target.Name netstat -tuln 2>&1
        if ($ports -match "LISTEN") {
            $listenPorts = $ports | Select-String -Pattern ":(\d+).*LISTEN" -AllMatches | 
                           ForEach-Object { $_.Matches.Groups[1].Value } | Sort-Object -Unique
            Write-Host "      開放端口: $($listenPorts -join ', ')" -ForegroundColor White
        }
    } catch {
        Write-Host "      ⚠️  無法執行端口掃描 (netstat 不可用)" -ForegroundColor Yellow
    }
    
    # 2. 檢查環境變數
    Write-Host "    [環境變數檢查]" -ForegroundColor Gray
    try {
        $envVars = docker exec $Target.Name env 2>&1
        $sensitiveEnvs = $envVars | Select-String -Pattern "(PASSWORD|SECRET|KEY|TOKEN|API)" -AllMatches
        if ($sensitiveEnvs) {
            Write-Host "      ⚠️  發現敏感環境變數:" -ForegroundColor Yellow
            $sensitiveEnvs | ForEach-Object {
                Write-Host "        - $($_.Line.Split('=')[0])" -ForegroundColor DarkYellow
            }
        } else {
            Write-Host "      ✓ 未發現明顯敏感環境變數" -ForegroundColor Green
        }
    } catch {
        Write-Host "      ⚠️  無法讀取環境變數" -ForegroundColor Yellow
    }
    
    # 3. 檢查文件系統
    Write-Host "    [文件系統檢查]" -ForegroundColor Gray
    try {
        $sensitiveFiles = docker exec $Target.Name find / -maxdepth 3 -name "*.config" -o -name "*.env" -o -name "*.key" 2>$null
        if ($sensitiveFiles) {
            Write-Host "      發現配置文件:" -ForegroundColor White
            $sensitiveFiles | Select-Object -First 5 | ForEach-Object {
                Write-Host "        - $_" -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "      ⚠️  無法掃描文件系統" -ForegroundColor Yellow
    }
}

function Get-DockerNetworkInfo {
    Write-Host "`n📡 Docker 網絡拓撲分析" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    try {
        $networks = docker network ls --format "{{.Name}}" | Where-Object { $_ -ne "none" -and $_ -ne "host" }
        
        foreach ($network in $networks) {
            Write-Host "`n  網絡: $network" -ForegroundColor Yellow
            $inspect = docker network inspect $network | ConvertFrom-Json
            
            if ($inspect.Containers) {
                Write-Host "  連接的容器:" -ForegroundColor Gray
                $inspect.Containers.PSObject.Properties | ForEach-Object {
                    $container = $_.Value
                    Write-Host "    - $($container.Name) ($($container.IPv4Address))" -ForegroundColor White
                }
            }
        }
    } catch {
        Write-Host "  ⚠️  無法獲取網絡資訊" -ForegroundColor Yellow
    }
}

# =====================================================================
# 主要執行邏輯
# =====================================================================

Write-Host "📋 目標列表:" -ForegroundColor Green
$targets | ForEach-Object {
    Write-Host "  • $($_.Name) - $($_.URL) [$($_.Type)]" -ForegroundColor White
}

Write-Host "`n🧪 測試清單:" -ForegroundColor Green
Write-Host "  • SSRF 漏洞檢測 ($($ssrfPayloads.Count) 種 Payload)" -ForegroundColor White
Write-Host "  • 容器內部掃描 (端口/環境變數/文件系統)" -ForegroundColor White
Write-Host "  • Docker 網絡拓撲分析" -ForegroundColor White

Write-Host "`n⏳ 開始掃描..." -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Gray

# 遍歷每個目標
foreach ($target in $targets) {
    Write-Host "🎯 目標: $($target.Name) [$($target.Type)]" -ForegroundColor Cyan
    Write-Host "   URL: $($target.URL)" -ForegroundColor Gray
    
    # 檢查容器可用性
    $isAvailable = Test-ContainerAvailability -Target $target
    
    if (-not $isAvailable) {
        Write-Host "  ⚠️  跳過此目標`n" -ForegroundColor Yellow
        continue
    }
    
    # SSRF 測試
    Write-Host "`n  [SSRF 漏洞掃描]" -ForegroundColor Cyan
    $vulnerabilityFound = $false
    
    foreach ($payload in $ssrfPayloads) {
        Write-Host "    測試: $($payload.Name) [$($payload.Risk)]" -ForegroundColor Gray
        $result = Test-SSRFVulnerability -Target $target -Payload $payload
        if ($result) {
            $vulnerabilityFound = $true
        }
    }
    
    if (-not $vulnerabilityFound) {
        Write-Host "    ✓ 未發現 SSRF 漏洞" -ForegroundColor Green
    }
    
    # 容器內部掃描
    Test-ContainerInternals -Target $target
    
    Write-Host "`n" + ("─" * 60) + "`n" -ForegroundColor DarkGray
}

# Docker 網絡分析
Get-DockerNetworkInfo

# =====================================================================
# 結果統計
# =====================================================================

Write-Host "`n📊 掃描統計" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  總測試數: $global:totalTests" -ForegroundColor White
Write-Host "  🚨 發現漏洞: $global:vulnerableTests" -ForegroundColor Red
Write-Host "  ✓ 已阻擋: $global:blockedTests" -ForegroundColor Green
Write-Host "  ⚠️  錯誤/未知: $global:errorTests" -ForegroundColor Yellow

$riskLevel = if ($global:vulnerableTests -gt 0) { 
    "HIGH" 
} elseif ($global:errorTests -gt 5) { 
    "MEDIUM" 
} else { 
    "LOW" 
}

Write-Host "`n🎯 整體風險等級: $riskLevel" -ForegroundColor $(
    if ($riskLevel -eq "HIGH") { "Red" } 
    elseif ($riskLevel -eq "MEDIUM") { "Yellow" } 
    else { "Green" }
)

Write-Host "`n✅ 掃描完成!`n" -ForegroundColor Green
