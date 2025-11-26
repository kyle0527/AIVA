# AIVA AI 持續運作服務 - PowerShell 啟動腳本

Write-Host "🚀 AIVA AI 服務啟動器" -ForegroundColor Cyan
Write-Host "=" * 60

# 設置項目路徑
$ProjectRoot = "C:\D\fold7\AIVA-git"
$env:PYTHONPATH = $ProjectRoot

# 顯示選單
function Show-Menu {
    Write-Host ""
    Write-Host "請選擇運作模式:" -ForegroundColor Yellow
    Write-Host "1. API 服務模式 (推薦) - REST API 持續監聽"
    Write-Host "2. 後台監控模式 - 自動定期掃描"
    Write-Host "3. 交互式模式 - 命令行交互"
    Write-Host "4. 守護進程模式 - API + 監控雙重模式"
    Write-Host "5. Windows 服務模式 (後台運行)"
    Write-Host "Q. 退出"
    Write-Host ""
}

# 啟動 API 模式
function Start-APIMode {
    Write-Host ""
    Write-Host "🌐 啟動 API 服務模式..." -ForegroundColor Green
    Write-Host "   訪問: http://localhost:8000"
    Write-Host "   文檔: http://localhost:8000/docs"
    Write-Host ""
    Write-Host "按 Ctrl+C 停止服務" -ForegroundColor Yellow
    Write-Host ""
    
    python "$ProjectRoot\start_ai_service.py" --mode api --port 8000
}

# 啟動監控模式
function Start-MonitorMode {
    Write-Host ""
    $targets = Read-Host "請輸入監控目標 URL (多個用空格分隔，回車使用預設 localhost:3000)"
    if ([string]::IsNullOrWhiteSpace($targets)) {
        $targets = "http://localhost:3000"
    }
    
    $interval = Read-Host "請輸入掃描間隔（秒）(回車使用預設 3600 = 1小時)"
    if ([string]::IsNullOrWhiteSpace($interval)) {
        $interval = 3600
    }
    
    Write-Host ""
    Write-Host "🔍 啟動監控模式..." -ForegroundColor Green
    Write-Host "   目標: $targets"
    Write-Host "   間隔: $interval 秒 ($([int]$interval/60) 分鐘)"
    Write-Host ""
    Write-Host "按 Ctrl+C 停止服務" -ForegroundColor Yellow
    Write-Host ""
    
    $targetArray = $targets -split '\s+'
    python "$ProjectRoot\start_ai_service.py" --mode monitor --targets $targetArray --interval $interval
}

# 啟動交互模式
function Start-InteractiveMode {
    Write-Host ""
    Write-Host "💬 啟動交互式模式..." -ForegroundColor Green
    Write-Host "   輸入 'help' 查看可用命令"
    Write-Host "   輸入 'quit' 退出"
    Write-Host ""
    
    python "$ProjectRoot\start_ai_service.py" --mode interactive
}

# 啟動守護進程模式
function Start-DaemonMode {
    Write-Host ""
    Write-Host "🔄 啟動守護進程模式 (API + 監控)..." -ForegroundColor Green
    Write-Host "   API: http://localhost:8000"
    Write-Host "   監控: 每小時掃描一次"
    Write-Host ""
    Write-Host "按 Ctrl+C 停止服務" -ForegroundColor Yellow
    Write-Host ""
    
    python "$ProjectRoot\start_ai_service.py" --mode daemon --port 8000 --targets "http://localhost:3000" --interval 3600
}

# 啟動 Windows 服務模式
function Start-WindowsServiceMode {
    Write-Host ""
    Write-Host "⚙️  配置 Windows 服務模式..." -ForegroundColor Green
    
    # 檢查是否以管理員身份運行
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Host "❌ 需要管理員權限來安裝 Windows 服務" -ForegroundColor Red
        Write-Host "請以管理員身份重新運行此腳本" -ForegroundColor Yellow
        return
    }
    
    # 使用 NSSM (Non-Sucking Service Manager) 或創建計劃任務
    Write-Host ""
    Write-Host "選擇後台運行方式:" -ForegroundColor Yellow
    Write-Host "1. 計劃任務 (推薦) - 開機自動啟動"
    Write-Host "2. 當前會話後台運行"
    Write-Host ""
    
    $choice = Read-Host "請選擇 (1/2)"
    
    switch ($choice) {
        "1" {
            # 創建計劃任務
            $taskName = "AIVA_AI_Service"
            $scriptPath = "$ProjectRoot\start_ai_service.py"
            $pythonPath = (Get-Command python).Source
            
            $action = New-ScheduledTaskAction -Execute $pythonPath -Argument "$scriptPath --mode daemon --port 8000" -WorkingDirectory $ProjectRoot
            $trigger = New-ScheduledTaskTrigger -AtStartup
            $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
            $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
            
            try {
                Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
                Write-Host ""
                Write-Host "✅ Windows 計劃任務已創建: $taskName" -ForegroundColor Green
                Write-Host "   任務將在系統啟動時自動運行" -ForegroundColor Green
                Write-Host ""
                Write-Host "管理命令:" -ForegroundColor Yellow
                Write-Host "   啟動: Start-ScheduledTask -TaskName '$taskName'"
                Write-Host "   停止: Stop-ScheduledTask -TaskName '$taskName'"
                Write-Host "   刪除: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
                Write-Host ""
                
                $startNow = Read-Host "是否立即啟動服務? (Y/N)"
                if ($startNow -eq "Y" -or $startNow -eq "y") {
                    Start-ScheduledTask -TaskName $taskName
                    Write-Host "✅ 服務已啟動" -ForegroundColor Green
                }
            }
            catch {
                Write-Host "❌ 創建計劃任務失敗: $_" -ForegroundColor Red
            }
        }
        "2" {
            # 當前會話後台運行
            Write-Host ""
            Write-Host "🔄 在新窗口啟動後台服務..." -ForegroundColor Green
            
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot'; `$env:PYTHONPATH='$ProjectRoot'; python start_ai_service.py --mode daemon --port 8000"
            
            Write-Host "✅ 服務已在新窗口啟動" -ForegroundColor Green
            Write-Host "   關閉該窗口即可停止服務" -ForegroundColor Yellow
        }
    }
}

# 主循環
while ($true) {
    Show-Menu
    $choice = Read-Host "請選擇 (1-5 或 Q)"
    
    switch ($choice) {
        "1" { Start-APIMode }
        "2" { Start-MonitorMode }
        "3" { Start-InteractiveMode }
        "4" { Start-DaemonMode }
        "5" { Start-WindowsServiceMode }
        "Q" { 
            Write-Host ""
            Write-Host "👋 再見！" -ForegroundColor Cyan
            exit 
        }
        "q" { 
            Write-Host ""
            Write-Host "👋 再見！" -ForegroundColor Cyan
            exit 
        }
        default { 
            Write-Host "❌ 無效選擇，請重新輸入" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "按任意鍵返回主選單..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
