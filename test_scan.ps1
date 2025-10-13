# AIVA 測試掃描腳本
# 日期: 2025-10-13
# 用途: 發送測試掃描任務

param(
    [string]$TargetUrl = "https://testphp.vulnweb.com",
    [int]$MaxDepth = 2,
    [int]$MaxPages = 10
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🧪 AIVA 測試掃描" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "目標 URL:  $TargetUrl" -ForegroundColor Yellow
Write-Host "最大深度:  $MaxDepth" -ForegroundColor Yellow
Write-Host "最大頁數:  $MaxPages" -ForegroundColor Yellow
Write-Host ""

# 啟動虛擬環境
& .\.venv\Scripts\Activate.ps1

# 建立臨時 Python 腳本
$pythonScript = @"
import asyncio
import json
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, ScanStartPayload, MessageHeader
from services.aiva_common.enums import Topic, ModuleName
from services.aiva_common.utils import new_id

async def send_scan_task():
    print('📡 連線到 RabbitMQ...')
    broker = await get_broker()
    
    scan_id = new_id('scan')
    payload = ScanStartPayload(
        scan_id=scan_id,
        target_url='$TargetUrl',
        max_depth=$MaxDepth,
        max_pages=$MaxPages
    )
    
    msg = AivaMessage(
        header=MessageHeader(
            message_id=new_id('msg'),
            trace_id=new_id('trace'),
            correlation_id=scan_id,
            source_module=ModuleName.CORE
        ),
        topic=Topic.TASK_SCAN_START,
        payload=payload.model_dump()
    )
    
    print(f'📤 發送掃描任務: {scan_id}')
    await broker.publish(
        Topic.TASK_SCAN_START,
        json.dumps(msg.model_dump()).encode('utf-8')
    )
    
    print(f'✅ 掃描任務已發送!')
    print(f'📋 掃描 ID: {scan_id}')
    print(f'🔍 請查看 Scan Worker 視窗的輸出')
    
    # 監聽結果 (等待 30 秒)
    print(f'\n📡 監聽掃描結果 (最多等待 30 秒)...')
    import asyncio
    try:
        async with asyncio.timeout(30):
            async for result_msg in broker.subscribe(Topic.RESULTS_SCAN_COMPLETED):
                result = AivaMessage.model_validate_json(result_msg.body)
                if result.header.correlation_id == scan_id:
                    print(f'\n✅ 收到掃描結果!')
                    result_data = result.payload
                    print(f'   • 發現資產: {len(result_data.get("assets", []))}')
                    print(f'   • 發現漏洞: {len(result_data.get("vulnerabilities", []))}')
                    break
    except asyncio.TimeoutError:
        print(f'\n⏱️  等待逾時 (30秒)')
        print(f'   提示: 掃描可能仍在進行中,請查看 Worker 日誌')

asyncio.run(send_scan_task())
"@

# 執行 Python 腳本
$pythonScript | python -

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "測試完成" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
