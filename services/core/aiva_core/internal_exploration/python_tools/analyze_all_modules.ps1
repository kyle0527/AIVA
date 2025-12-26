# AIVA 自動化分析腳本
# 用途: 批次分析所有模組並生成統一的分析結果
# 作者: AIVA Team
# 日期: 2025-12-15

param(
    [string]$ProjectRoot = "C:\D\fold7\AIVA-git",
    [string[]]$Modules = @("core", "features", "scan", "integration")
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "AIVA 模組自動化分析工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Join-Path $ProjectRoot "services\core\aiva_core\internal_exploration\python_tools\aiva_exploration_pipeline.py"

if (-not (Test-Path $scriptPath)) {
    Write-Host "✗ 找不到分析腳本: $scriptPath" -ForegroundColor Red
    exit 1
}

$results = @()

foreach ($module in $Modules) {
    Write-Host "[$module] 開始分析..." -ForegroundColor Yellow
    $startTime = Get-Date
    
    try {
        # 執行分析
        Push-Location $ProjectRoot
        python $scriptPath --target $module --module $module
        Pop-Location
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = ($endTime - $startTime).TotalSeconds
            
            Write-Host "[$module] ✓ 分析完成 (耗時: ${duration}s)" -ForegroundColor Green
            
            # 驗證輸出文件
            $capFile = Join-Path $ProjectRoot "services\integration\analysis_data\$module\capabilities\latest_${module}_capabilities.json"
            
            if (Test-Path $capFile) {
                $content = Get-Content $capFile -Raw | ConvertFrom-Json
                $flowCount = $content.metadata.total_flows
                $size = [math]::Round((Get-Item $capFile).Length / 1KB, 2)
                
                Write-Host "[$module]   流程數: $flowCount" -ForegroundColor Gray
                Write-Host "[$module]   文件大小: ${size} KB" -ForegroundColor Gray
                
                $results += @{
                    Module = $module
                    Status = "Success"
                    FlowCount = $flowCount
                    Duration = $duration
                    FileSize = $size
                }
            } else {
                Write-Host "[$module] ⚠ 輸出文件未生成" -ForegroundColor Yellow
                $results += @{
                    Module = $module
                    Status = "Warning"
                    FlowCount = 0
                    Duration = $duration
                }
            }
        } else {
            throw "分析進程返回錯誤代碼: $LASTEXITCODE"
        }
    }
    catch {
        Write-Host "[$module] ✗ 分析失敗: $_" -ForegroundColor Red
        $results += @{
            Module = $module
            Status = "Failed"
            Error = $_.Exception.Message
        }
    }
    
    Write-Host ""
}

# 輸出摘要
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "分析摘要" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$successCount = ($results | Where-Object { $_.Status -eq "Success" }).Count
$failedCount = ($results | Where-Object { $_.Status -eq "Failed" }).Count
$totalFlows = ($results | Where-Object { $_.Status -eq "Success" } | Measure-Object -Property FlowCount -Sum).Sum

Write-Host "成功: $successCount / $($Modules.Count)" -ForegroundColor Green
Write-Host "失敗: $failedCount / $($Modules.Count)" -ForegroundColor $(if ($failedCount -gt 0) { "Red" } else { "Gray" })
Write-Host "總流程數: $totalFlows" -ForegroundColor White
Write-Host ""

# 詳細列表
$results | Format-Table -Property Module, Status, FlowCount, @{Name="Duration(s)"; Expression={$_.Duration}}, @{Name="Size(KB)"; Expression={$_.FileSize}} -AutoSize

Write-Host ""
Write-Host "輸出位置: $ProjectRoot\services\integration\analysis_data\" -ForegroundColor Cyan
Write-Host "歷史版本: $ProjectRoot\services\integration\data\internal_exploration\analysis_history\" -ForegroundColor Cyan
