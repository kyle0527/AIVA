# AIVA 編碼檢測工具腳本集

## 1. 主要編碼檢測函數

```powershell
function Test-FileEncoding {
    <#
    .SYNOPSIS
    檢測檔案的字符編碼格式
    
    .DESCRIPTION
    使用多層檢測策略確定檔案的編碼格式：
    1. BOM (Byte Order Mark) 檢測
    2. UTF-8 嚴格模式驗證  
    3. 傳統中文編碼測試
    
    .PARAMETER FilePath
    要檢測的檔案完整路徑
    
    .EXAMPLE
    Test-FileEncoding -FilePath "C:\example\file.txt"
    
    .NOTES
    基於 Microsoft 官方文件和 Stack Overflow 最佳實踐開發
    #>
    
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath
    )
    
    # 檔案存在性檢查
    if (-not (Test-Path $FilePath)) {
        return "檔案不存在"
    }
    
    # 讀取檔案位元組
    try {
        $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    } catch {
        return "檔案讀取失敗: $($_.Exception.Message)"
    }
    
    # 空檔案處理
    if ($bytes.Length -eq 0) {
        return "空檔案"
    }
    
    # === 第一層：BOM 檢測 ===
    # UTF-8 BOM: EF BB BF
    if ($bytes.Length -ge 3 -and 
        $bytes[0] -eq 0xEF -and 
        $bytes[1] -eq 0xBB -and 
        $bytes[2] -eq 0xBF) {
        return "UTF-8 BOM"
    }
    
    # UTF-16 Little Endian: FF FE
    if ($bytes.Length -ge 2 -and 
        $bytes[0] -eq 0xFF -and 
        $bytes[1] -eq 0xFE) {
        return "UTF-16 LE"
    }
    
    # UTF-16 Big Endian: FE FF
    if ($bytes.Length -ge 2 -and 
        $bytes[0] -eq 0xFE -and 
        $bytes[1] -eq 0xFF) {
        return "UTF-16 BE"
    }
    
    # UTF-32 Little Endian: FF FE 00 00
    if ($bytes.Length -ge 4 -and 
        $bytes[0] -eq 0xFF -and 
        $bytes[1] -eq 0xFE -and 
        $bytes[2] -eq 0x00 -and 
        $bytes[3] -eq 0x00) {
        return "UTF-32 LE"
    }
    
    # UTF-32 Big Endian: 00 00 FE FF
    if ($bytes.Length -ge 4 -and 
        $bytes[0] -eq 0x00 -and 
        $bytes[1] -eq 0x00 -and 
        $bytes[2] -eq 0xFE -and 
        $bytes[3] -eq 0xFF) {
        return "UTF-32 BE"
    }
    
    # === 第二層：UTF-8 嚴格模式驗證 ===
    try {
        # 創建嚴格的 UTF-8 編碼器（會對無效序列拋出異常）
        $utf8Strict = [System.Text.UTF8Encoding]::new($false, $true)
        $utf8Content = $utf8Strict.GetString($bytes)
        
        # 檢查是否包含中文字符
        if ($utf8Content -match '[\u4e00-\u9fff]') {
            return "UTF-8 (含中文)"
        } elseif ($utf8Content -match '[^\x00-\x7F]') {
            return "UTF-8 (含非ASCII字符)"
        } else {
            return "UTF-8 或 ASCII"
        }
    } catch [System.Text.DecoderFallbackException] {
        # UTF-8 嚴格解碼失敗，繼續測試其他編碼
        Write-Verbose "UTF-8 嚴格解碼失敗，測試其他編碼"
    } catch {
        Write-Verbose "UTF-8 解碼異常: $($_.Exception.Message)"
    }
    
    # === 第三層：傳統中文編碼測試 ===
    
    # CP950 (Big5 繁體中文)
    try {
        $cp950Encoder = [System.Text.Encoding]::GetEncoding(950, 
            [System.Text.EncoderFallback]::ExceptionFallback, 
            [System.Text.DecoderFallback]::ExceptionFallback)
        $cp950Content = $cp950Encoder.GetString($bytes)
        
        if ($cp950Content -match '[\u4e00-\u9fff]') {
            return "CP950 (Big5 繁體中文)"
        }
    } catch {
        Write-Verbose "CP950 解碼失敗"
    }
    
    # GBK/CP936 (簡體中文)
    try {
        $gbkEncoder = [System.Text.Encoding]::GetEncoding(936,
            [System.Text.EncoderFallback]::ExceptionFallback, 
            [System.Text.DecoderFallback]::ExceptionFallback)
        $gbkContent = $gbkEncoder.GetString($bytes)
        
        if ($gbkContent -match '[\u4e00-\u9fff]') {
            return "GBK (CP936 簡體中文)"
        }
    } catch {
        Write-Verbose "GBK 解碼失敗"
    }
    
    # === 第四層：其他常見編碼 ===
    
    # ISO-8859-1 (Latin-1)
    try {
        $latin1Content = [System.Text.Encoding]::GetEncoding("ISO-8859-1").GetString($bytes)
        # 檢查是否包含 Latin-1 特有字符
        if ($latin1Content -match '[\u00A0-\u00FF]') {
            return "ISO-8859-1 (Latin-1)"
        }
    } catch {
        Write-Verbose "ISO-8859-1 解碼失敗"
    }
    
    # 最後回傳
    return "無法確定編碼 (可能是二進位檔案或不支援的編碼)"
}
```

## 2. 批次檢測函數

```powershell
function Scan-ProjectEncoding {
    <#
    .SYNOPSIS
    掃描整個專案的檔案編碼
    
    .PARAMETER ProjectPath
    專案根目錄路徑
    
    .PARAMETER FileExtensions
    要掃描的檔案副檔名陣列
    
    .PARAMETER OutputReport
    是否輸出詳細報告
    #>
    
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectPath,
        
        [string[]]$FileExtensions = @('.py', '.js', '.ts', '.go', '.rs', '.md', 
                                      '.json', '.yaml', '.yml', '.ps1', '.sh', '.txt'),
        
        [switch]$OutputReport
    )
    
    Write-Host "開始掃描專案編碼..." -ForegroundColor Green
    Write-Host "專案路徑: $ProjectPath" -ForegroundColor Cyan
    Write-Host "檔案類型: $($FileExtensions -join ', ')" -ForegroundColor Cyan
    
    # 獲取所有目標檔案
    $allFiles = Get-ChildItem -Path $ProjectPath -Recurse -File | 
                Where-Object { $_.Extension -in $FileExtensions }
    
    Write-Host "找到 $($allFiles.Count) 個檔案待檢測" -ForegroundColor Yellow
    
    # 編碼統計
    $encodingStats = @{}
    $results = @()
    
    # 進度追蹤
    $progress = 0
    $total = $allFiles.Count
    
    foreach ($file in $allFiles) {
        $progress++
        $percentComplete = [math]::Round(($progress / $total) * 100, 1)
        
        Write-Progress -Activity "檢測檔案編碼" -Status "處理中... ($progress/$total)" -PercentComplete $percentComplete
        
        $encoding = Test-FileEncoding -FilePath $file.FullName
        
        # 統計編碼類型
        if ($encodingStats.ContainsKey($encoding)) {
            $encodingStats[$encoding]++
        } else {
            $encodingStats[$encoding] = 1
        }
        
        # 記錄結果
        $result = [PSCustomObject]@{
            FilePath = $file.FullName
            FileName = $file.Name
            Extension = $file.Extension
            Encoding = $encoding
            Size = $file.Length
        }
        $results += $result
        
        # 即時輸出可疑檔案
        if ($encoding -notlike "UTF-8*" -and $encoding -ne "UTF-8 或 ASCII") {
            Write-Host "發現非UTF-8檔案: $($file.FullName) -> $encoding" -ForegroundColor Red
        }
    }
    
    Write-Progress -Activity "檢測檔案編碼" -Completed
    
    # 輸出統計結果
    Write-Host "`n=== 編碼統計結果 ===" -ForegroundColor Green
    $encodingStats.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
        $percentage = [math]::Round(($_.Value / $total) * 100, 2)
        Write-Host "$($_.Key): $($_.Value) 個檔案 ($percentage%)" -ForegroundColor Cyan
    }
    
    # 輸出詳細報告
    if ($OutputReport) {
        $reportPath = Join-Path $ProjectPath "reports\encoding_scan_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
        $results | Export-Csv -Path $reportPath -Encoding UTF8 -NoTypeInformation
        Write-Host "`n詳細報告已儲存至: $reportPath" -ForegroundColor Magenta
    }
    
    return $results
}
```

## 3. BOM 檢測專用函數

```powershell
function Find-BOMFiles {
    <#
    .SYNOPSIS
    專門檢測包含 BOM 的檔案
    #>
    
    param(
        [string]$Path = ".",
        [string[]]$Extensions = @('.py', '.js', '.ts', '.go', '.rs', '.md', '.json', '.yaml', '.yml', '.ps1', '.sh', '.txt')
    )
    
    $bomFiles = @()
    
    Get-ChildItem -Path $Path -Recurse -File | 
        Where-Object { $_.Extension -in $Extensions } | 
        ForEach-Object {
            $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
            
            $bomType = $null
            if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
                $bomType = "UTF-8 BOM"
            } elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
                $bomType = "UTF-16 LE"
            } elseif ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
                $bomType = "UTF-16 BE"
            } elseif ($bytes.Length -ge 4 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE -and $bytes[2] -eq 0x00 -and $bytes[3] -eq 0x00) {
                $bomType = "UTF-32 LE"
            } elseif ($bytes.Length -ge 4 -and $bytes[0] -eq 0x00 -and $bytes[1] -eq 0x00 -and $bytes[2] -eq 0xFE -and $bytes[3] -eq 0xFF) {
                $bomType = "UTF-32 BE"
            }
            
            if ($bomType) {
                $bomFiles += [PSCustomObject]@{
                    FilePath = $_.FullName
                    BOMType = $bomType
                    FirstBytes = ($bytes[0..7] | ForEach-Object { $_.ToString("X2") }) -join " "
                }
            }
        }
    
    return $bomFiles
}
```

## 4. 使用範例

```powershell
# 基本使用
$encoding = Test-FileEncoding -FilePath "C:\example\file.txt"
Write-Host "檔案編碼: $encoding"

# 批次掃描專案
$results = Scan-ProjectEncoding -ProjectPath "C:\D\fold7\AIVA-git" -OutputReport
Write-Host "掃描完成，共處理 $($results.Count) 個檔案"

# 查找 BOM 檔案
$bomFiles = Find-BOMFiles -Path "C:\D\fold7\AIVA-git"
if ($bomFiles.Count -gt 0) {
    Write-Host "發現 $($bomFiles.Count) 個包含 BOM 的檔案:"
    $bomFiles | Format-Table -AutoSize
} else {
    Write-Host "沒有發現包含 BOM 的檔案"
}

# 驗證特定檔案
$testFiles = @(
    "C:\D\fold7\AIVA-git\README.md",
    "C:\D\fold7\AIVA-git\services\features\docs\FILE_ORGANIZATION.md"
)

Write-Host "=== 檔案編碼驗證 ===" -ForegroundColor Green
foreach ($file in $testFiles) {
    if (Test-Path $file) {
        $encoding = Test-FileEncoding -FilePath $file
        Write-Host "$($file): $encoding" -ForegroundColor Cyan
    } else {
        Write-Host "$($file): 檔案不存在" -ForegroundColor Red
    }
}
```

## 5. 除錯與驗證函數

```powershell
function Debug-FileBytes {
    <#
    .SYNOPSIS
    顯示檔案的位元組內容，用於除錯編碼問題
    #>
    
    param(
        [string]$FilePath,
        [int]$ByteCount = 32
    )
    
    if (-not (Test-Path $FilePath)) {
        Write-Error "檔案不存在: $FilePath"
        return
    }
    
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    $displayBytes = $bytes[0..([Math]::Min($ByteCount - 1, $bytes.Length - 1))]
    
    Write-Host "檔案: $FilePath" -ForegroundColor Cyan
    Write-Host "總大小: $($bytes.Length) 位元組" -ForegroundColor Yellow
    Write-Host "前 $($displayBytes.Length) 個位元組:" -ForegroundColor Green
    
    # 十六進位顯示
    $hexDisplay = ($displayBytes | ForEach-Object { $_.ToString("X2") }) -join " "
    Write-Host "HEX: $hexDisplay" -ForegroundColor White
    
    # ASCII 顯示
    $asciiDisplay = $displayBytes | ForEach-Object { 
        if ($_ -ge 32 -and $_ -le 126) { 
            [char]$_ 
        } else { 
            "." 
        } 
    }
    Write-Host "ASCII: $($asciiDisplay -join '')" -ForegroundColor Gray
    
    # 編碼檢測
    $encoding = Test-FileEncoding -FilePath $FilePath
    Write-Host "檢測編碼: $encoding" -ForegroundColor Magenta
}
```

<#
這個完整的腳本集提供了：
1. 嚴格的編碼檢測邏輯
2. 批次處理能力
3. 詳細的報告功能
4. 除錯和驗證工具
5. 完整的錯誤處理

所有函數都基於我們從網路研究中學到的最佳實踐，確保檢測結果的準確性。
#>