#Requires -Version 5.1
#Requires -PSEdition Desktop,Core

# RabbitMQ 死信隊列配置腳本 (PowerShell 版本)
# 為 AIVA 平台配置統一的死信隊列策略，防止 poison pill 消息問題

<#
.SYNOPSIS
    為 AIVA 平台配置 RabbitMQ 死信隊列策略

.DESCRIPTION
    此腳本自動配置 RabbitMQ 死信隊列策略，包括：
    - 為 tasks.*, findings.*, *results 隊列設置死信策略
    - 創建死信交換機 (aiva.dead_letter)
    - 創建死信隊列 (aiva.dead_letter.failed)
    - 配置隊列綁定和策略
    
    防止 poison pill 消息問題，確保系統穩定性。

.PARAMETER RabbitMQHost
    RabbitMQ 服務器主機名或IP地址 (預設: localhost)

.PARAMETER RabbitMQUser
    RabbitMQ 用戶名 (預設: guest)

.PARAMETER RabbitMQPassword
    RabbitMQ 密碼，使用安全字符串格式

.PARAMETER VHost
    RabbitMQ 虛擬主機 (預設: /)

.PARAMETER Verbose
    啟用詳細輸出

.EXAMPLE
    .\setup_dead_letter_queues.ps1
    使用預設參數配置死信隊列

.EXAMPLE
    .\setup_dead_letter_queues.ps1 -RabbitMQHost "rabbitmq.example.com" -RabbitMQUser "admin"
    指定主機和用戶名配置死信隊列

.EXAMPLE
    $securePassword = ConvertTo-SecureString "mypassword" -AsPlainText -Force
    .\setup_dead_letter_queues.ps1 -RabbitMQPassword $securePassword
    使用安全字符串密碼配置死信隊列

.NOTES
    文件名: setup_dead_letter_queues.ps1
    作者: AIVA 開發團隊
    版本: 1.1.0
    最後修改: 2025-10-27
    
    要求:
    - PowerShell 5.1 或更高版本
    - RabbitMQ Management 插件已啟用
    - 適當的 RabbitMQ 用戶權限

.LINK
    https://github.com/kyle0527/aiva
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [Parameter(Mandatory = $false, HelpMessage = "RabbitMQ server hostname or IP address")]
    [ValidateNotNullOrEmpty()]
    [ValidatePattern('^([a-zA-Z0-9-]+\.)*[a-zA-Z0-9-]+$|^([0-9]{1,3}\.){3}[0-9]{1,3}$|^localhost$')]
    [string]$RabbitMQHost = $(if ($env:AIVA_RABBITMQ_HOST) { $env:AIVA_RABBITMQ_HOST } else { "localhost" }),
    
    [Parameter(Mandatory = $false, HelpMessage = "RabbitMQ username")]
    [ValidateNotNullOrEmpty()]
    [ValidateLength(1, 255)]
    [string]$RabbitMQUser = $(if ($env:AIVA_RABBITMQ_USER) { $env:AIVA_RABBITMQ_USER } else { "guest" }),
    
    [Parameter(Mandatory = $false, HelpMessage = "RabbitMQ password as SecureString")]
    [SecureString]$RabbitMQPassword,
    
    [Parameter(Mandatory = $false, HelpMessage = "RabbitMQ virtual host")]
    [ValidateNotNullOrEmpty()]
    [ValidateLength(1, 255)]
    [string]$VHost = $(if ($env:AIVA_RABBITMQ_VHOST) { $env:AIVA_RABBITMQ_VHOST } else { "/" }),
    
    [Parameter(Mandatory = $false, HelpMessage = "Enable verbose output")]
    [switch]$Verbose
)

# 安全地處理密碼參數 - 遵循 SecureString 最佳實踐
if (-not $RabbitMQPassword) {
    $passwordPlain = $env:AIVA_RABBITMQ_PASSWORD
    if (-not $passwordPlain) {
        if ($PSCmdlet.ShouldProcess("密碼輸入", "提示輸入 RabbitMQ 密碼")) {
            # 安全地提示輸入密碼，避免使用預設值
            Write-Host "請輸入 RabbitMQ 密碼（按 Enter 使用預設 'guest'）: " -NoNewline
            $RabbitMQPassword = Read-Host -AsSecureString
            
            # 如果用戶未輸入任何內容，使用預設密碼
            if ($RabbitMQPassword.Length -eq 0) {
                Write-Warning "使用預設密碼 'guest'，建議在生產環境中使用強密碼"
                $passwordPlain = "guest"
            }
        }
        else {
            # WhatIf 模式下使用預設值
            Write-Warning "WhatIf 模式：將使用預設密碼 'guest'"
            $passwordPlain = "guest"
        }
    }
    
    # 只有在有明文密碼時才轉換
    if ($passwordPlain) {
        try {
            $RabbitMQPassword = ConvertTo-SecureString -String $passwordPlain -AsPlainText -Force
            # 立即清除明文密碼變數以提高安全性
            $passwordPlain = $null
            Remove-Variable -Name passwordPlain -ErrorAction SilentlyContinue
        }
        catch {
            Write-Error "無法轉換密碼為安全字符串: $($_.Exception.Message)"
            exit 1
        }
    }
}

# 函數定義

# 驗證輸入參數的安全性和有效性
function Test-InputParameters {
    [CmdletBinding()]
    param(
        [string]$HostName,
        [string]$User,
        [string]$VirtualHost
    )
    
    # 驗證主機名不包含危險字符
    if ($HostName -match '[<>"`|&;]') {
        throw "主機名包含不安全的字符: $HostName"
    }
    
    # 驗證用戶名不包含危險字符
    if ($User -match '[<>"`|&;]') {
        throw "用戶名包含不安全的字符: $User"
    }
    
    # 驗證虛擬主機路徑
    if ($VirtualHost -match '[<>"`|&;]' -and $VirtualHost -ne '/') {
        throw "虛擬主機路徑包含不安全的字符: $VirtualHost"
    }
    
    Write-Verbose "輸入參數驗證通過"
    return $true
}

function Test-RabbitMQConnection {
    [CmdletBinding()]
    param(
        [string]$HostName,
        [string]$User,
        [SecureString]$Password,
        [string]$VirtualHost
    )
    
    try {
        $null = & rabbitmqctl status 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Write-LogMessage {
    [CmdletBinding()]
    param(
        [string]$Message,
        [ValidateSet('Info', 'Warning', 'Error', 'Success')]
        [string]$Level = 'Info'
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        'Info' { 'White' }
        'Warning' { 'Yellow' }
        'Error' { 'Red' }
        'Success' { 'Green' }
    }
    
    Write-Host "[$timestamp] $Message" -ForegroundColor $color
}

# 主要執行邏輯
Write-LogMessage "配置 AIVA 平台的 RabbitMQ 死信隊列策略..." -Level Success

# 驗證輸入參數的安全性
try {
    Test-InputParameters -HostName $RabbitMQHost -User $RabbitMQUser -VirtualHost $VHost
}
catch {
    Write-LogMessage "參數驗證失敗: $($_.Exception.Message)" -Level Error
    exit 1
}

# 驗證 RabbitMQ 連接
Write-LogMessage "檢查 RabbitMQ 服務狀態..." -Level Info
if (-not (Test-RabbitMQConnection -HostName $RabbitMQHost -User $RabbitMQUser -Password $RabbitMQPassword -VirtualHost $VHost)) {
    Write-LogMessage "警告: 無法連接到 RabbitMQ 服務，繼續執行配置..." -Level Warning
}

try {
    # 定義死信隊列策略配置
    $policies = @(
        @{
            Name = "aiva-tasks-dlx-policy"
            Pattern = "^tasks\."
            Definition = '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 10000}'
            Description = "任務隊列死信策略"
        },
        @{
            Name = "aiva-findings-dlx-policy"
            Pattern = "^findings\."
            Definition = '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 50000}'
            Description = "發現隊列死信策略"
        },
        @{
            Name = "aiva-results-dlx-policy"
            Pattern = "^.*results$"
            Definition = '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 50000}'
            Description = "結果隊列死信策略"
        }
    )

    # 設置隊列策略
    foreach ($policy in $policies) {
        if ($PSCmdlet.ShouldProcess("RabbitMQ VHost '$VHost'", "設置 $($policy.Description)")) {
            Write-LogMessage "設置 $($policy.Description)..." -Level Info
            
            $policyArgs = @(
                "set_policy"
                "--vhost", "`"$VHost`""
                "`"$($policy.Name)`""
                "`"$($policy.Pattern)`""
                "`'$($policy.Definition)`'"
                "--priority", "7"
                "--apply-to", "queues"
            )
            
            & rabbitmqctl @policyArgs
            
            if ($LASTEXITCODE -ne 0) {
                throw "$($policy.Description)設置失敗 (Exit Code: $LASTEXITCODE)"
            }
            
            Write-LogMessage "$($policy.Description)設置成功" -Level Success
        }
        else {
            Write-LogMessage "跳過 $($policy.Description) (WhatIf 模式)" -Level Info
        }
    }



    # 創建死信基礎設施
    $exchangeScript = "rabbit_exchange:declare(resource(<<`"$VHost`">>, exchange, <<`"aiva.dead_letter`">>), topic, true, false, false, [])."
    $queueScript = "rabbit_amqqueue:declare(resource(<<`"$VHost`">>, queue, <<`"aiva.dead_letter.failed`">>), true, false, [], none, <<`"admin`">>)."
    $bindingScript = "rabbit_binding:add(binding(resource(<<`"$VHost`">>, exchange, <<`"aiva.dead_letter`">>), <<`"failed`">>, resource(<<`"$VHost`">>, queue, <<`"aiva.dead_letter.failed`">>), []))."
    
    $infrastructureComponents = @(
        @{
            Name = "死信交換機"
            Script = $exchangeScript
        },
        @{
            Name = "死信隊列"  
            Script = $queueScript
        },
        @{
            Name = "死信隊列綁定"
            Script = $bindingScript
        }
    )

    foreach ($component in $infrastructureComponents) {
        if ($PSCmdlet.ShouldProcess("RabbitMQ VHost '$VHost'", "創建 $($component.Name)")) {
            Write-LogMessage "創建 $($component.Name)..." -Level Info
            
            try {
                & rabbitmqctl eval $component.Script
                
                if ($LASTEXITCODE -ne 0) {
                    throw "$($component.Name)創建失敗 (Exit Code: $LASTEXITCODE)"
                }
                
                Write-LogMessage "$($component.Name)創建成功" -Level Success
            }
            catch {
                Write-LogMessage "$($component.Name)創建失敗: $($_.Exception.Message)" -Level Error
                throw
            }
        }
        else {
            Write-LogMessage "跳過 $($component.Name) (WhatIf 模式)" -Level Info
        }
    }

    Write-LogMessage "✅ RabbitMQ 死信隊列配置完成！" -Level Success
    Write-Host ""
    
    # 配置摘要
    Write-Host "配置摘要：" -ForegroundColor Cyan
    Write-Host "- 死信交換機: aiva.dead_letter" -ForegroundColor White
    Write-Host "- 死信隊列: aiva.dead_letter.failed" -ForegroundColor White
    Write-Host "- 消息 TTL: 24 小時" -ForegroundColor White
    Write-Host "- 適用範圍: tasks.*, findings.*, *results 隊列" -ForegroundColor White
    Write-Host "- 虛擬主機: $VHost" -ForegroundColor White
    Write-Host ""
    
    # 驗證命令
    Write-Host "驗證配置命令：" -ForegroundColor Cyan
    $verificationCommands = @(
        "rabbitmqctl list_policies --vhost `"$VHost`"",
        "rabbitmqctl list_exchanges --vhost `"$VHost`" | Select-String dead_letter",
        "rabbitmqctl list_queues --vhost `"$VHost`" | Select-String dead_letter"
    )
    
    foreach ($cmd in $verificationCommands) {
        Write-Host "  $cmd" -ForegroundColor White
    }
    
    Write-Host ""
    Write-LogMessage "腳本執行完成，請使用上述命令驗證配置" -Level Info

} catch {
    Write-LogMessage "❌ 配置失敗: $($_.Exception.Message)" -Level Error
    Write-LogMessage "錯誤詳情: $($_.Exception.StackTrace)" -Level Error
    
    # 提供故障排除建議
    Write-Host ""
    Write-Host "故障排除建議：" -ForegroundColor Yellow
    Write-Host "1. 確保 RabbitMQ 服務正在運行" -ForegroundColor White
    Write-Host "2. 驗證用戶權限和虛擬主機存在" -ForegroundColor White
    Write-Host "3. 檢查防火牆和網路連接" -ForegroundColor White
    Write-Host "4. 查看 RabbitMQ 日誌: rabbitmqctl log_tail" -ForegroundColor White
    
    exit 1
}