<#
.SYNOPSIS
    自動識別並組合跨模組流程圖
    
.DESCRIPTION
    1. 掃描所有 mermaid 圖，識別包含 Topic/MessageBroker 的圖
    2. 根據 Topic 自動匹配發送端和接收端
    3. 生成完整的跨模組流程組合圖
    4. 起點可以是任何模組，不限於 Core
    
.EXAMPLE
    .\auto_combine_cross_module_flows.ps1 -SourceDir "_out1101016\mermaid_details" -OutputDir "_out\cross_module_flows"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$SourceDir = "_out1101016\mermaid_details",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "_out\cross_module_flows"
)

Write-Host "`n🔍 AIVA 跨模組流程自動組合工具" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan

# ============================================================================
# 1. 定義模組識別規則
# ============================================================================

$ModulePatterns = @{
    'core' = @('core_aiva_core', 'core_messaging')
    'scan' = @('scan_aiva_scan', 'scan_discovery')
    'function' = @('function_sqli', 'function_xss', 'function_ssrf', 'function_idor')
    'integration' = @('integration_')
    'common' = @('aiva_common')
}

function Get-ModuleFromPath {
    param([string]$FilePath)
    
    $relativePath = $FilePath -replace [regex]::Escape($SourceDir), ''
    
    foreach ($module in $ModulePatterns.Keys) {
        foreach ($pattern in $ModulePatterns[$module]) {
            if ($relativePath -match $pattern) {
                return $module
            }
        }
    }
    return 'unknown'
}

# ============================================================================
# 2. 掃描並分類所有流程圖
# ============================================================================

Write-Host "`n📂 掃描流程圖檔案..." -ForegroundColor Yellow

$allFiles = Get-ChildItem -Path $SourceDir -Filter "*.mmd" -Recurse

Write-Host "   找到 $($allFiles.Count) 個 mermaid 檔案" -ForegroundColor Gray

# ============================================================================
# 3. 識別包含跨模組通訊的圖
# ============================================================================

Write-Host "`n🔎 識別跨模組通訊點..." -ForegroundColor Yellow

$communicationPoints = @()

foreach ($file in $allFiles) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    
    # 檢查是否包含跨模組通訊關鍵字（考慮 HTML 編碼）
    $hasTopic = $content -match 'Topic\.|Topic\.|\&amp\;&#35;39;Topic'
    $hasMessageBroker = $content -match 'MessageBroker|message_broker|MessageBroker'
    $hasPublish = $content -match 'publish|send|emit|\.publish|\.send'
    $hasSubscribe = $content -match 'subscribe|consume|on_message|callback|register_handler'
    $hasTaskPrefix = $content -match 'TASK_|RESULTS_|EVENT_|COMMAND_|TASKS_'
    
    if ($hasTopic -or $hasMessageBroker -or $hasTaskPrefix) {
        $module = Get-ModuleFromPath -FilePath $file.FullName
        
        # 提取所有 Topic 引用（包括 HTML 編碼格式）
        $topics = @()
        
        # 正常格式: Topic.TASK_SCAN
        $topicMatches = [regex]::Matches($content, 'Topic\.([A-Z_]+)')
        foreach ($match in $topicMatches) {
            $topics += $match.Groups[1].Value
        }
        
        # HTML 編碼格式: function_sqli&amp;&#35;39;
        $encodedMatches = [regex]::Matches($content, '(function_sqli|scan_discovery|function_xss|function_ssrf|function_idor|integration_analysis)')
        foreach ($match in $encodedMatches) {
            # 將模組名轉換為 Topic
            $moduleName = $match.Value
            switch ($moduleName) {
                'function_sqli' { $topics += 'TASK_FUNCTION_SQLI'; $topics += 'RESULTS_FUNCTION_SQLI' }
                'scan_discovery' { $topics += 'TASK_SCAN_START'; $topics += 'RESULTS_SCAN_COMPLETED' }
                'function_xss' { $topics += 'TASK_FUNCTION_XSS'; $topics += 'RESULTS_FUNCTION_XSS' }
                'function_ssrf' { $topics += 'TASK_FUNCTION_SSRF'; $topics += 'RESULTS_FUNCTION_SSRF' }
                'function_idor' { $topics += 'TASK_FUNCTION_IDOR'; $topics += 'RESULTS_FUNCTION_IDOR' }
                'integration_analysis' { $topics += 'TASK_INTEGRATION_ANALYSIS'; $topics += 'RESULTS_INTEGRATION_ANALYSIS' }
            }
        }
        
        # 直接字串: TASK_SCAN_START
        $stringMatches = [regex]::Matches($content, '(TASKS?_|RESULTS_|EVENT_|COMMAND_)[A-Z_]+')
        foreach ($match in $stringMatches) {
            $topics += $match.Value
        }
        
        $topics = $topics | Select-Object -Unique
        
        if ($topics.Count -gt 0) {
            $communicationPoints += [PSCustomObject]@{
                File = $file.Name
                FullPath = $file.FullName
                Module = $module
                Topics = $topics
                IsSender = $hasPublish
                IsReceiver = $hasSubscribe
                Content = $content
            }
        }
    }
}

Write-Host "   識別到 $($communicationPoints.Count) 個通訊點" -ForegroundColor Green

# Debug: 顯示前 10 個通訊點
Write-Host "`n📋 通訊點範例 (前 10 個):" -ForegroundColor Cyan
$communicationPoints | Select-Object -First 10 | ForEach-Object {
    Write-Host "   • [$($_.Module)] $($_.File)" -ForegroundColor Gray
    Write-Host "     Topics: $($_.Topics -join ', ')" -ForegroundColor DarkGray
}

# ============================================================================
# 4. 按 Topic 分組並匹配發送端和接收端
# ============================================================================

Write-Host "`n🔗 匹配跨模組通訊流程..." -ForegroundColor Yellow

$flowGroups = @{}

foreach ($point in $communicationPoints) {
    foreach ($topic in $point.Topics) {
        if (-not $flowGroups.ContainsKey($topic)) {
            $flowGroups[$topic] = @{
                Topic = $topic
                Senders = @()
                Receivers = @()
            }
        }
        
        if ($point.IsSender) {
            $flowGroups[$topic].Senders += $point
        }
        if ($point.IsReceiver) {
            $flowGroups[$topic].Receivers += $point
        }
    }
}

# 過濾出真正的跨模組流程（發送端和接收端在不同模組）
$crossModuleFlows = @()

foreach ($topic in $flowGroups.Keys) {
    $group = $flowGroups[$topic]
    
    foreach ($senderNode in $group.Senders) {
        foreach ($receiverNode in $group.Receivers) {
            if ($senderNode.Module -ne $receiverNode.Module -and 
                $senderNode.Module -ne 'unknown' -and 
                $receiverNode.Module -ne 'unknown') {
                
                $crossModuleFlows += [PSCustomObject]@{
                    Topic = $topic
                    SenderModule = $senderNode.Module
                    ReceiverModule = $receiverNode.Module
                    SenderFile = $senderNode.File
                    ReceiverFile = $receiverNode.File
                    SenderPath = $senderNode.FullPath
                    ReceiverPath = $receiverNode.FullPath
                    SenderContent = $senderNode.Content
                    ReceiverContent = $receiverNode.Content
                }
            }
        }
    }
}

Write-Host "   找到 $($crossModuleFlows.Count) 個跨模組流程" -ForegroundColor Green

# ============================================================================
# 5. 按流程類型分組
# ============================================================================

$flowsByPattern = $crossModuleFlows | Group-Object -Property {
    "$($_.SenderModule) → $($_.ReceiverModule)"
}

Write-Host "`n📊 流程模式統計:" -ForegroundColor Cyan
foreach ($group in $flowsByPattern) {
    Write-Host "   • $($group.Name): $($group.Count) 個流程" -ForegroundColor Yellow
}

# ============================================================================
# 6. 生成組合流程圖
# ============================================================================

Write-Host "`n🎨 生成組合流程圖..." -ForegroundColor Yellow

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$generatedCount = 0

foreach ($flow in $crossModuleFlows) {
    $flowName = "$($flow.SenderModule)_to_$($flow.ReceiverModule)_$($flow.Topic)"
    $outputFile = Join-Path $OutputDir "$flowName.md"
    
    # 生成組合圖
    $combinedContent = @"
# 跨模組流程: $($flow.SenderModule) → $($flow.ReceiverModule)

**Topic**: ``$($flow.Topic)``

## 📤 發送端: $($flow.SenderModule)

**檔案**: ``$($flow.SenderFile)``

``````mermaid
$($flow.SenderContent -replace '```mermaid.radar', '' -replace '```', '')
``````

---

## 📥 接收端: $($flow.ReceiverModule)

**檔案**: ``$($flow.ReceiverFile)``

``````mermaid
$($flow.ReceiverContent -replace '```mermaid.radar', '' -replace '```', '')
``````

---

## 🔄 完整流程組合

``````mermaid
sequenceDiagram
    participant Sender as $($flow.SenderModule.ToUpper())<br/>📤 $($flow.SenderFile -replace '_Function.*|_Module.*', '')
    participant MQ as 📨 Message Queue<br/>Topic: $($flow.Topic)
    participant Receiver as $($flow.ReceiverModule.ToUpper())<br/>📥 $($flow.ReceiverFile -replace '_Function.*|_Module.*', '')
    
    Sender->>MQ: Publish<br/>$($flow.Topic)
    MQ->>Receiver: Deliver Message
    Receiver->>Receiver: Process
    Note over Receiver: 執行業務邏輯
    Receiver->>MQ: Publish Result
    MQ->>Sender: Result Delivered
``````

---

**生成時間**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  
**工具**: AIVA 跨模組流程自動組合工具
"@

    $combinedContent | Out-File -FilePath $outputFile -Encoding UTF8
    $generatedCount++
}

Write-Host "   ✅ 已生成 $generatedCount 個組合流程圖" -ForegroundColor Green

# ============================================================================
# 7. 生成索引文件
# ============================================================================

Write-Host "`n📄 生成索引文件..." -ForegroundColor Yellow

$indexContent = @"
# AIVA 跨模組流程索引

**生成時間**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  
**總流程數**: $($crossModuleFlows.Count)

---

## 📊 流程模式統計

| 流程模式 | 數量 |
|---------|------|
"@

foreach ($group in $flowsByPattern) {
    $indexContent += "`n| $($group.Name) | $($group.Count) |"
}

$indexContent += @"


---

## 📋 所有跨模組流程

| Topic | 發送端 | 接收端 | 檔案 |
|-------|--------|--------|------|
"@

foreach ($flow in $crossModuleFlows | Sort-Object Topic) {
    $fileName = "$($flow.SenderModule)_to_$($flow.ReceiverModule)_$($flow.Topic).md"
    $indexContent += "`n| ``$($flow.Topic)`` | $($flow.SenderModule) | $($flow.ReceiverModule) | [$fileName]($fileName) |"
}

$indexContent += @"


---

## 🎯 使用方式

1. **瀏覽流程**: 點擊上表中的檔案連結查看完整流程
2. **分析通訊**: 每個流程圖包含發送端、接收端和完整序列圖
3. **生成 CLI**: 基於這些流程可以生成對應的 CLI 命令

---

## 📝 流程圖說明

每個流程圖包含三個部分：

1. **發送端流程圖**: 顯示消息發送邏輯
2. **接收端流程圖**: 顯示消息處理邏輯
3. **完整序列圖**: 顯示端到端的通訊流程

---

**維護**: 執行 ``.\scripts\auto_combine_cross_module_flows.ps1`` 重新生成
"@

$indexFile = Join-Path $OutputDir "INDEX.md"
$indexContent | Out-File -FilePath $indexFile -Encoding UTF8

Write-Host "   ✅ 已生成索引: $indexFile" -ForegroundColor Green

# ============================================================================
# 8. 生成 Topic 對照表
# ============================================================================

Write-Host "`n📚 生成 Topic 對照表..." -ForegroundColor Yellow

$topicTableContent = @"
# AIVA Topic 對照表

**生成時間**: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

---

## 📨 所有 Topics

| Topic | 類型 | 發送模組 | 接收模組 | 流程數 |
|-------|------|---------|---------|--------|
"@

$topicStats = $crossModuleFlows | Group-Object Topic | ForEach-Object {
    $topic = $_.Name
    $flows = $_.Group
    $type = if ($topic -match '^TASK_') { 'Task' }
            elseif ($topic -match '^RESULTS_') { 'Result' }
            elseif ($topic -match '^EVENT_') { 'Event' }
            elseif ($topic -match '^COMMAND_') { 'Command' }
            else { 'Other' }
    
    $senders = ($flows.SenderModule | Select-Object -Unique) -join ', '
    $receivers = ($flows.ReceiverModule | Select-Object -Unique) -join ', '
    
    [PSCustomObject]@{
        Topic = $topic
        Type = $type
        Senders = $senders
        Receivers = $receivers
        Count = $flows.Count
    }
} | Sort-Object Type, Topic

foreach ($stat in $topicStats) {
    $topicTableContent += "`n| ``$($stat.Topic)`` | $($stat.Type) | $($stat.Senders) | $($stat.Receivers) | $($stat.Count) |"
}

$topicTableFile = Join-Path $OutputDir "TOPIC_TABLE.md"
$topicTableContent | Out-File -FilePath $topicTableFile -Encoding UTF8

Write-Host "   ✅ 已生成 Topic 對照表: $topicTableFile" -ForegroundColor Green

# ============================================================================
# 完成
# ============================================================================

Write-Host "`n" -ForegroundColor White
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✅ 完成！" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "`n📁 輸出目錄: $OutputDir" -ForegroundColor Cyan
Write-Host "📊 生成統計:" -ForegroundColor Cyan
Write-Host "   • 跨模組流程圖: $generatedCount 個" -ForegroundColor Yellow
Write-Host "   • 通訊點: $($communicationPoints.Count) 個" -ForegroundColor Yellow
Write-Host "   • Topics: $($topicStats.Count) 個" -ForegroundColor Yellow
Write-Host "`n💡 下一步:" -ForegroundColor Yellow
Write-Host "   1. 查看索引: $indexFile" -ForegroundColor White
Write-Host "   2. 查看 Topic 對照表: $topicTableFile" -ForegroundColor White
Write-Host "   3. 基於流程生成 CLI 命令" -ForegroundColor White
Write-Host "`n" -ForegroundColor White
