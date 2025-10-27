# RabbitMQ 死信隊列配置腳本 (PowerShell 版本)
# 為 AIVA 平台配置統一的死信隊列策略，防止 poison pill 消息問題

param(
    [string]$RabbitMQHost = $env:AIVA_RABBITMQ_HOST ?? "localhost",
    [string]$RabbitMQUser = $env:AIVA_RABBITMQ_USER ?? "guest", 
    [SecureString]$RabbitMQPassword,
    [string]$VHost = $env:AIVA_RABBITMQ_VHOST ?? "/"
)

# 如果沒有提供密碼參數，從環境變數讀取
if (-not $RabbitMQPassword) {
    $passwordPlain = $env:AIVA_RABBITMQ_PASSWORD ?? "guest"
    $RabbitMQPassword = ConvertTo-SecureString -String $passwordPlain -AsPlainText -Force
}

Write-Host "配置 AIVA 平台的 RabbitMQ 死信隊列策略..." -ForegroundColor Green

try {
    # 設置所有任務隊列的死信交換機和重試策略
    Write-Host "設置任務隊列死信策略..." -ForegroundColor Yellow
    & rabbitmqctl set_policy `
        --vhost "$VHost" `
        "aiva-tasks-dlx-policy" `
        "^tasks\." `
        '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 10000}' `
        --priority 7 `
        --apply-to queues

    if ($LASTEXITCODE -ne 0) { throw "任務隊列死信策略設置失敗" }

    # 設置發現隊列的死信策略  
    Write-Host "設置發現隊列死信策略..." -ForegroundColor Yellow
    & rabbitmqctl set_policy `
        --vhost "$VHost" `
        "aiva-findings-dlx-policy" `
        "^findings\." `
        '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 50000}' `
        --priority 7 `
        --apply-to queues

    if ($LASTEXITCODE -ne 0) { throw "發現隊列死信策略設置失敗" }

    # 設置結果隊列的死信策略
    Write-Host "設置結果隊列死信策略..." -ForegroundColor Yellow  
    & rabbitmqctl set_policy `
        --vhost "$VHost" `
        "aiva-results-dlx-policy" `
        "^.*results$" `
        '{"dead-letter-exchange": "aiva.dead_letter", "dead-letter-routing-key": "failed", "message-ttl": 86400000, "max-length": 50000}' `
        --priority 7 `
        --apply-to queues

    if ($LASTEXITCODE -ne 0) { throw "結果隊列死信策略設置失敗" }

    # 創建死信交換機
    Write-Host "創建死信交換機..." -ForegroundColor Yellow
    $createExchangeScript = @"
rabbit_exchange:declare(
    resource(<<"$VHost">>, exchange, <<"aiva.dead_letter">>),
    topic,
    true,
    false,
    false,
    []
).
"@
    & rabbitmqctl eval $createExchangeScript

    if ($LASTEXITCODE -ne 0) { throw "死信交換機創建失敗" }

    # 創建死信隊列
    Write-Host "創建死信隊列..." -ForegroundColor Yellow
    $createQueueScript = @"
rabbit_amqqueue:declare(
    resource(<<"$VHost">>, queue, <<"aiva.dead_letter.failed">>),
    true,
    false,
    [],
    none,
    <<"admin">>
).
"@
    & rabbitmqctl eval $createQueueScript

    if ($LASTEXITCODE -ne 0) { throw "死信隊列創建失敗" }

    # 綁定死信隊列到死信交換機
    Write-Host "綁定死信隊列..." -ForegroundColor Yellow
    $bindQueueScript = @"
rabbit_binding:add(
    binding(
        resource(<<"$VHost">>, exchange, <<"aiva.dead_letter">>),
        <<"failed">>,
        resource(<<"$VHost">>, queue, <<"aiva.dead_letter.failed">>),
        []
    )
).
"@
    & rabbitmqctl eval $bindQueueScript

    if ($LASTEXITCODE -ne 0) { throw "死信隊列綁定失敗" }

    Write-Host "✅ RabbitMQ 死信隊列配置完成！" -ForegroundColor Green
    Write-Host ""
    Write-Host "配置摘要：" -ForegroundColor Cyan
    Write-Host "- 死信交換機: aiva.dead_letter"
    Write-Host "- 死信隊列: aiva.dead_letter.failed"  
    Write-Host "- 消息 TTL: 24 小時"
    Write-Host "- 適用於所有 tasks.*, findings.*, *results 隊列"
    Write-Host ""
    Write-Host "查看配置：" -ForegroundColor Cyan
    Write-Host "  rabbitmqctl list_policies --vhost $VHost"
    Write-Host "  rabbitmqctl list_exchanges --vhost $VHost | Select-String dead_letter"
    Write-Host "  rabbitmqctl list_queues --vhost $VHost | Select-String dead_letter"

} catch {
    Write-Host "❌ 配置失敗: $_" -ForegroundColor Red
    exit 1
}