#!/usr/bin/env python3
"""實際目標掃描腳本 - 掃描 Docker 容器"""

import asyncio
from services.core.aiva_core.core_capabilities.dialog.assistant import AIVADialogAssistant

async def scan_real_targets():
    assistant = AIVADialogAssistant()
    
    # Docker 容器目標
    targets = [
        ('http://localhost:8080', 'WebGoat - OWASP 漏洞教學平台'),
        ('http://localhost:3000', 'Juice Shop - OWASP 靶機'),
        ('http://localhost:3001', 'bkimminich容器1'),
        ('http://localhost:3003', 'bkimminich容器2'),
    ]
    
    results = []
    
    for target_url, description in targets:
        print(f'\n{"="*70}')
        print(f'🎯 掃描目標: {target_url}')
        print(f'📝 描述: {description}')
        print(f'{"="*70}')
        
        result = await assistant.process_user_input(f'掃描 {target_url}')
        
        print(f'\n📊 結果:')
        print(f'  意圖: {result["intent"]}')
        print(f'  可執行: {result["executable"]}')
        print(f'\n💬 AI 回應:')
        print(result["message"])
        
        if result.get("data"):
            results.append({
                'target': target_url,
                'description': description,
                'scan_id': result["data"].get("scan_id"),
                'status': result["data"].get("status"),
                'assets': result["data"].get("assets", [])
            })
    
    # 總結報告
    print(f'\n\n{"="*70}')
    print('📈 掃描總結報告')
    print(f'{"="*70}')
    print(f'總共掃描目標: {len(targets)}')
    print(f'成功執行: {len(results)}')
    print(f'\n詳細結果:')
    for r in results:
        print(f'\n  目標: {r["target"]}')
        print(f'  描述: {r["description"]}')
        print(f'  掃描ID: {r["scan_id"]}')
        print(f'  狀態: {r["status"]}')
        print(f'  發現資產: {len(r["assets"])} 個')
        if r["assets"]:
            for asset in r["assets"][:3]:
                print(f'    - {asset["type"]}: {asset["value"]}')

if __name__ == "__main__":
    asyncio.run(scan_real_targets())
