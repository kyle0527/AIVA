<#
.SYNOPSIS
    基於跨模組流程圖生成 CLI 命令
    
.DESCRIPTION
    分析架構圖中的跨模組流程，自動生成對應的 CLI 命令實現
    
.EXAMPLE
    .\generate_cli_from_flows.ps1 -ArchDiagramsDir "_out1101016\architecture_diagrams" -OutputDir "services\cli\generated"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ArchDiagramsDir = "_out1101016\architecture_diagrams",
    
    [Parameter(Mandatory=$false)]
    [string]$OutputDir = "services\cli\generated"
)

# ============================================================================
# 跨模組流程定義（從架構圖分析得出）
# ============================================================================

$CrossModuleFlows = @{
    "scan_flow" = @{
        Title = "掃描流程"
        Description = "User → API → Core → Scan → MQ"
        Modules = @("Core", "Scan", "MQ")
        Commands = @(
            @{
                Name = "scan start"
                Args = "<url>"
                Flags = @("--max-depth <int>", "--max-pages <int>", "--wait", "--format json")
                Topics = @("TASK_SCAN_START", "RESULTS_SCAN_PROGRESS", "RESULTS_SCAN_COMPLETED")
            },
            @{
                Name = "scan status"
                Args = "<scan_id>"
                Flags = @("--format json")
                Topics = @("QUERY_SCAN_STATUS", "RESULTS_SCAN_STATUS")
            },
            @{
                Name = "scan assets"
                Args = "<scan_id>"
                Flags = @("--type <web|api|file>", "--format json")
                Topics = @("QUERY_SCAN_ASSETS", "RESULTS_SCAN_ASSETS")
            }
        )
    }
    
    "sqli_detection_flow" = @{
        Title = "SQL 注入檢測流程"
        Description = "Core → Function(SQLi) → MQ → Core"
        Modules = @("Core", "Function", "MQ")
        Commands = @(
            @{
                Name = "detect sqli"
                Args = "<url>"
                Flags = @("--param <name>", "--method <GET|POST>", "--engines <list>", "--wait", "--format json")
                Topics = @("TASK_FUNCTION_SQLI", "RESULTS_FUNCTION_PROGRESS", "EVENT_FUNCTION_VULN_FOUND", "RESULTS_FUNCTION_COMPLETED")
            }
        )
    }
    
    "xss_detection_flow" = @{
        Title = "XSS 檢測流程"
        Description = "Core → Function(XSS) → MQ → Core"
        Modules = @("Core", "Function", "MQ")
        Commands = @(
            @{
                Name = "detect xss"
                Args = "<url>"
                Flags = @("--param <name>", "--type <reflected|stored|dom>", "--wait", "--format json")
                Topics = @("TASK_FUNCTION_XSS", "RESULTS_FUNCTION_COMPLETED")
            }
        )
    }
    
    "ssrf_detection_flow" = @{
        Title = "SSRF 檢測流程"
        Description = "Core → Function(SSRF) → MQ → Core"
        Modules = @("Core", "Function", "MQ")
        Commands = @(
            @{
                Name = "detect ssrf"
                Args = "<url>"
                Flags = @("--param <name>", "--callback-url <url>", "--wait", "--format json")
                Topics = @("TASK_FUNCTION_SSRF", "RESULTS_FUNCTION_COMPLETED")
            }
        )
    }
    
    "idor_detection_flow" = @{
        Title = "IDOR 檢測流程"
        Description = "Core → Function(IDOR) → MQ → Core"
        Modules = @("Core", "Function", "MQ")
        Commands = @(
            @{
                Name = "detect idor"
                Args = "<url>"
                Flags = @("--param <name>", "--user-context <json>", "--wait", "--format json")
                Topics = @("FUNCTION_IDOR_TASK", "RESULTS_FUNCTION_COMPLETED")
            }
        )
    }
    
    "integration_flow" = @{
        Title = "整合分析流程"
        Description = "Core → Integration → MQ → Core"
        Modules = @("Core", "Integration", "MQ")
        Commands = @(
            @{
                Name = "report generate"
                Args = "<scan_id>"
                Flags = @("--format <pdf|html|json>", "--output <file>", "--no-findings", "--include-remediation")
                Topics = @("TASK_INTEGRATION_ANALYSIS_START", "RESULTS_INTEGRATION_ANALYSIS_COMPLETED", "COMMAND_INTEGRATION_REPORT_GENERATE", "EVENT_INTEGRATION_REPORT_GENERATED")
            },
            @{
                Name = "report status"
                Args = "<analysis_id>"
                Flags = @("--format json")
                Topics = @("QUERY_INTEGRATION_STATUS", "RESULTS_INTEGRATION_STATUS")
            },
            @{
                Name = "report export"
                Args = "<scan_id>"
                Flags = @("--format <csv|excel|json>", "--output <file>")
                Topics = @("COMMAND_INTEGRATION_EXPORT")
            }
        )
    }
    
    "ai_training_flow" = @{
        Title = "AI 訓練流程"
        Description = "Core(AI) → MQ → Knowledge Base"
        Modules = @("Core", "MQ")
        Commands = @(
            @{
                Name = "ai train"
                Args = ""
                Flags = @("--mode <realtime|replay|simulation>", "--epochs <int>", "--scenarios <int>", "--storage-path <path>", "--format json")
                Topics = @("TASK_AI_TRAINING_START", "TASK_AI_TRAINING_EPISODE", "RESULTS_AI_TRAINING_PROGRESS", "EVENT_AI_EXPERIENCE_CREATED", "EVENT_AI_MODEL_UPDATED", "RESULTS_AI_TRAINING_COMPLETED")
            },
            @{
                Name = "ai status"
                Args = ""
                Flags = @("--storage-path <path>", "--format json")
                Topics = @("QUERY_AI_STATUS", "RESULTS_AI_STATUS")
            },
            @{
                Name = "ai stop"
                Args = "<training_id>"
                Flags = @("--format json")
                Topics = @("TASK_AI_TRAINING_STOP")
            },
            @{
                Name = "ai deploy"
                Args = "<model_id>"
                Flags = @("--environment <dev|staging|prod>", "--format json")
                Topics = @("COMMAND_AI_MODEL_DEPLOY")
            }
        )
    }
    
    "threat_intel_flow" = @{
        Title = "威脅情報流程"
        Description = "Core → Threat Intel → External APIs → Core"
        Modules = @("Core", "ThreatIntel", "MQ")
        Commands = @(
            @{
                Name = "threat lookup"
                Args = "<ioc>"
                Flags = @("--type <ip|domain|hash|url>", "--enrich", "--mitre", "--format json")
                Topics = @("TASK_THREAT_INTEL_LOOKUP", "TASK_IOC_ENRICHMENT", "TASK_MITRE_MAPPING", "RESULTS_THREAT_INTEL")
            },
            @{
                Name = "threat batch"
                Args = "<file>"
                Flags = @("--format json")
                Topics = @("TASK_THREAT_INTEL_BATCH")
            },
            @{
                Name = "threat mitre"
                Args = "<technique_id>"
                Flags = @("--format json")
                Topics = @("TASK_MITRE_MAPPING", "RESULTS_THREAT_INTEL")
            }
        )
    }
    
    "authz_flow" = @{
        Title = "權限檢測流程"
        Description = "Core → AuthZ → MQ → Core"
        Modules = @("Core", "AuthZ", "MQ")
        Commands = @(
            @{
                Name = "authz check"
                Args = "<url>"
                Flags = @("--user-context <json>", "--test-escalation", "--format json")
                Topics = @("TASK_AUTHZ_CHECK", "RESULTS_AUTHZ")
            },
            @{
                Name = "authz analyze"
                Args = "<scan_id>"
                Flags = @("--format json")
                Topics = @("TASK_AUTHZ_ANALYZE", "RESULTS_AUTHZ")
            }
        )
    }
    
    "remediation_flow" = @{
        Title = "修復建議流程"
        Description = "Core → Remediation → MQ → Core"
        Modules = @("Core", "Remediation", "MQ")
        Commands = @(
            @{
                Name = "remedy generate"
                Args = "<vuln_id>"
                Flags = @("--language <python|java|go|...>", "--framework <flask|django|...>", "--format json")
                Topics = @("TASK_REMEDIATION_GENERATE", "RESULTS_REMEDIATION")
            },
            @{
                Name = "remedy batch"
                Args = "<scan_id>"
                Flags = @("--format json")
                Topics = @("TASK_REMEDIATION_BATCH", "RESULTS_REMEDIATION")
            }
        )
    }
}

# ============================================================================
# 生成函數
# ============================================================================

function New-CLICommandImplementation {
    param(
        [string]$FlowName,
        [hashtable]$FlowData
    )
    
    $commandsCode = ""
    
    foreach ($cmd in $FlowData.Commands) {
        $funcName = "cmd_" + ($cmd.Name -replace ' ', '_')
        $argsList = if ($cmd.Args) { ", args: list[str]" } else { "" }
        
        $commandsCode += @"

def $funcName(${argsList}):
    '''$($FlowData.Title) - $($cmd.Name)
    
    Description: $($FlowData.Description)
    Modules: $($FlowData.Modules -join ' → ')
    
    Topics:
$($cmd.Topics | ForEach-Object { "    - $_`n" })
    
    Args:
        $($cmd.Args)
    
    Flags:
$($cmd.Flags | ForEach-Object { "        $_`n" })
    '''
    # TODO: 實現跨模組通訊邏輯
    # 1. 解析參數
    # 2. 發送到對應 Topic
    # 3. 等待結果
    # 4. 格式化輸出
    pass

"@
    }
    
    return $commandsCode
}

function New-CLIFlowDocument {
    param(
        [hashtable]$AllFlows
    )
    
    $doc = @"
# AIVA 跨模組流程與 CLI 命令對應

生成時間: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## 📊 流程總覽

| 流程名稱 | 涉及模組 | CLI 命令數 | Topic 數 |
|---------|---------|-----------|---------|
"@

    foreach ($flowName in $AllFlows.Keys) {
        $flow = $AllFlows[$flowName]
        $cmdCount = $flow.Commands.Count
        $topicCount = ($flow.Commands | ForEach-Object { $_.Topics } | Select-Object -Unique).Count
        
        $doc += "`n| $($flow.Title) | $($flow.Modules -join ', ') | $cmdCount | $topicCount |"
    }
    
    $doc += @"

## 📋 詳細命令列表

"@

    foreach ($flowName in $AllFlows.Keys) {
        $flow = $AllFlows[$flowName]
        
        $doc += @"

### $($flow.Title)

**描述**: $($flow.Description)  
**涉及模組**: $($flow.Modules -join ' → ')

#### 命令

"@
        
        foreach ($cmd in $flow.Commands) {
            $doc += @"

##### ``aiva $($cmd.Name) $($cmd.Args)``

**選項**:
"@
            foreach ($flag in $cmd.Flags) {
                $doc += "`n- ``$flag``"
            }
            
            $doc += @"


**對應 Topics**:
"@
            foreach ($topic in $cmd.Topics) {
                $doc += "`n- ``$topic``"
            }
            
            $doc += "`n"
        }
    }
    
    return $doc
}

function New-MermaidFlowDiagram {
    param(
        [string]$FlowName,
        [hashtable]$FlowData
    )
    
    $mermaid = @"
# $($FlowData.Title)

``````mermaid
sequenceDiagram
    participant CLI as 🖥️ CLI
    participant Core as 🤖 Core
"@

    # 添加模組參與者
    foreach ($module in $FlowData.Modules) {
        if ($module -eq "Scan") {
            $mermaid += "`n    participant Scan as 🔍 Scan"
        } elseif ($module -eq "Function") {
            $mermaid += "`n    participant Func as ⚡ Function"
        } elseif ($module -eq "Integration") {
            $mermaid += "`n    participant Intg as 🔗 Integration"
        } elseif ($module -eq "MQ") {
            $mermaid += "`n    participant MQ as 📨 MQ"
        }
    }
    
    $mermaid += "`n`n    CLI->>Core: $($FlowData.Commands[0].Name)"
    
    # 簡化的流程
    foreach ($module in $FlowData.Modules) {
        if ($module -ne "Core" -and $module -ne "MQ") {
            $mermaid += "`n    Core->>MQ: Send Task"
            $mermaid += "`n    MQ->>${module}: Process"
            $mermaid += "`n    ${module}->>MQ: Result"
            $mermaid += "`n    MQ->>Core: Complete"
            break
        }
    }
    
    $mermaid += "`n    Core->>CLI: Return Result"
    $mermaid += "`n```````n`n"
    
    return $mermaid
}

# ============================================================================
# 主程式
# ============================================================================

Write-Host "`n🚀 AIVA 跨模組流程 CLI 生成器" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 建立輸出目錄
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "📁 已建立輸出目錄: $OutputDir" -ForegroundColor Yellow
}

# 生成 Python CLI 實現
Write-Host "`n📝 生成 CLI 命令實現..." -ForegroundColor Cyan
$pythonCode = @"
'''
AIVA 跨模組流程 CLI 命令實現
自動生成於: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
'''

from typing import Optional
import asyncio
from services.aiva_common.enums.modules import Topic, ModuleName


"@

foreach ($flowName in $CrossModuleFlows.Keys) {
    $pythonCode += New-CLICommandImplementation -FlowName $flowName -FlowData $CrossModuleFlows[$flowName]
}

$pythonFilePath = Join-Path $OutputDir "cross_module_commands.py"
$pythonCode | Out-File -FilePath $pythonFilePath -Encoding UTF8
Write-Host "✅ 已生成: $pythonFilePath" -ForegroundColor Green

# 生成文檔
Write-Host "`n📄 生成文檔..." -ForegroundColor Cyan
$docContent = New-CLIFlowDocument -AllFlows $CrossModuleFlows
$docFilePath = Join-Path $OutputDir "CLI_FLOWS.md"
$docContent | Out-File -FilePath $docFilePath -Encoding UTF8
Write-Host "✅ 已生成: $docFilePath" -ForegroundColor Green

# 生成 Mermaid 流程圖
Write-Host "`n🎨 生成流程圖..." -ForegroundColor Cyan
$mermaidDir = Join-Path $OutputDir "flow_diagrams"
if (-not (Test-Path $mermaidDir)) {
    New-Item -ItemType Directory -Path $mermaidDir -Force | Out-Null
}

foreach ($flowName in $CrossModuleFlows.Keys) {
    $mermaidContent = New-MermaidFlowDiagram -FlowName $flowName -FlowData $CrossModuleFlows[$flowName]
    $mermaidFilePath = Join-Path $mermaidDir "$flowName.md"
    $mermaidContent | Out-File -FilePath $mermaidFilePath -Encoding UTF8
    Write-Host "  ✓ $flowName.md" -ForegroundColor Gray
}

# 統計
Write-Host "`n📊 統計資料:" -ForegroundColor Cyan
$totalFlows = $CrossModuleFlows.Count
$totalCommands = ($CrossModuleFlows.Values | ForEach-Object { $_.Commands.Count } | Measure-Object -Sum).Sum
$totalTopics = ($CrossModuleFlows.Values | ForEach-Object { $_.Commands | ForEach-Object { $_.Topics } } | Select-Object -Unique).Count

Write-Host "  • 跨模組流程: $totalFlows 個" -ForegroundColor Yellow
Write-Host "  • CLI 命令: $totalCommands 個" -ForegroundColor Yellow
Write-Host "  • 涉及 Topics: $totalTopics 個" -ForegroundColor Yellow

# 完成
Write-Host "`n" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "✅ 完成！" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "`n📁 輸出目錄: $OutputDir" -ForegroundColor Cyan
Write-Host "`n💡 下一步:" -ForegroundColor Yellow
Write-Host "  1. 查看生成的文檔: $docFilePath" -ForegroundColor White
Write-Host "  2. 查看 Python 實現: $pythonFilePath" -ForegroundColor White
Write-Host "  3. 查看流程圖: $mermaidDir" -ForegroundColor White
Write-Host "  4. 整合到主 CLI: services\cli\aiva_cli.py" -ForegroundColor White
Write-Host "`n" -ForegroundColor White
