#!/usr/bin/env python3
"""
Real BizLogic Scan - 對實際靶場執行業務邏輯掃描

目標靶場:
- Juice Shop (localhost:3000) - 電商靶場
- WebGoat (localhost:8080) - 多功能靶場
- 其他靶場 (localhost:3001, 3003)

掃描類型:
- 競態條件 (Race Condition)
- 價格操控 (Price Manipulation)  
- 工作流程繞過 (Workflow Bypass)
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# 添加項目根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.aiva_common.schemas.commands import AICommand, CommandType, CommandPriority
from services.features.features_ready.function_bizlogic.command_handler import BizLogicCommandHandler


# 端點常量
ENDPOINT_CHECKOUT = "/checkout"
ENDPOINT_BASKET = "/basket"
ENDPOINT_COUPON = "/coupon"

# 靶場配置
TARGETS = {
    "juice_shop": {
        "name": "OWASP Juice Shop",
        "base_url": "http://localhost:3000",
        "endpoints": {
            "checkout": "/rest/basket/checkout",
            "basket": "/rest/basket",
            "coupon": "/rest/coupon",
            "products": "/rest/products"
        },
        "tests": ["race_condition", "price_manipulation", "workflow_bypass"]
    },
    "webgoat": {
        "name": "WebGoat",
        "base_url": "http://localhost:8080",
        "endpoints": {
            "registration": "/WebGoat/register",
            "login": "/WebGoat/login",
            "lessons": "/WebGoat/lessons"
        },
        "tests": ["workflow_bypass"]
    },
    "target_3001": {
        "name": "Custom Target (3001)",
        "base_url": "http://localhost:3001",
        "endpoints": {
            "api": "/api"
        },
        "tests": ["race_condition", "price_manipulation"]
    }
}


def _build_scan_options(target_config: dict) -> dict:
    """構建掃描選項（降低認知複雜度）"""
    options = {
        "test_race_condition": "race_condition" in target_config['tests'],
        "test_price_manipulation": "price_manipulation" in target_config['tests'],
        "test_workflow_bypass": "workflow_bypass" in target_config['tests'],
        "concurrent_count": 20,
        "race_test_endpoints": [],
        "workflow_steps": []
    }
    
    # 添加競態條件測試端點
    if "race_condition" in target_config['tests']:
        endpoints = target_config['endpoints']
        options["race_test_endpoints"] = [
            endpoints.get("checkout", ENDPOINT_CHECKOUT),
            endpoints.get("coupon", ENDPOINT_COUPON),
            endpoints.get("basket", ENDPOINT_BASKET)
        ]
    
    # 添加工作流程步驟
    if "workflow_bypass" in target_config['tests']:
        endpoints = target_config['endpoints']
        options["workflow_steps"] = [
            endpoints.get("basket", ENDPOINT_BASKET),
            endpoints.get("checkout", ENDPOINT_CHECKOUT)
        ]
        options["checkout_endpoint"] = endpoints.get("checkout", ENDPOINT_CHECKOUT)
    
    return options


def _print_scan_header(target_config: dict):
    """打印掃描頭部信息"""
    print("\n" + "="*80)
    print(f"🎯 掃描目標: {target_config['name']}")
    print(f"🌐 URL: {target_config['base_url']}")
    print(f"📋 測試類型: {', '.join(target_config['tests'])}")
    print("="*80)


def _print_vulnerability_details(scan_result: dict):
    """打印漏洞詳細信息（降低認知複雜度）"""
    if scan_result.get('total_vulnerabilities', 0) == 0:
        print("\n✨ 未發現明顯的業務邏輯漏洞")
        return
    
    print("\n🔍 發現的漏洞:")
    
    # 競態條件
    for i, vuln in enumerate(scan_result.get('race_conditions', []), 1):
        print(f"\n   [{i}] 競態條件漏洞")
        print(f"       類型: {vuln.get('type')}")
        print(f"       嚴重性: {vuln.get('severity')}")
        print(f"       證據: {vuln.get('evidence')}")
    
    # 價格操控
    for i, vuln in enumerate(scan_result.get('price_manipulations', []), 1):
        print(f"\n   [{i}] 價格操控漏洞")
        print(f"       類型: {vuln.get('type')}")
        print(f"       嚴重性: {vuln.get('severity')}")
        print(f"       證據: {vuln.get('evidence')}")
    
    # 工作流程繞過
    for i, vuln in enumerate(scan_result.get('workflow_bypasses', []), 1):
        print(f"\n   [{i}] 工作流程繞過漏洞")
        print(f"       類型: {vuln.get('type')}")
        print(f"       嚴重性: {vuln.get('severity')}")
        print(f"       證據: {vuln.get('evidence')}")


async def scan_target(handler: BizLogicCommandHandler, target_key: str, target_config: dict):
    """掃描單個靶場"""
    _print_scan_header(target_config)
    
    # 構建掃描選項
    options = _build_scan_options(target_config)
    
    # 創建掃描命令
    command = AICommand(
        command_id=f"scan_{target_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        command_type=CommandType.FEATURE_BIZLOGIC_TEST,
        target_module="features.bizlogic",
        payload={
            "target_url": target_config['base_url'],
            "options": options,
            "authorization_token": "test_scan_token_" + "x"*32,  # 模擬 token
            "user_role": "customer"
        },
        trace_id=f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        session_id=f"session_{target_key}",
        parent_command_id=None,
        callback_url=None
    )
    
    # 執行掃描
    print(f"\n⏳ 開始掃描 {target_config['name']}...")
    start_time = datetime.now()
    
    try:
        result = await handler.handle_command(command)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ 掃描完成 (耗時: {duration:.2f}秒)")
        print(f"   - 狀態: {result.status.value}")
        print(f"   - 成功: {'是' if result.success else '否'}")
        
        if result.success and result.result:
            scan_result = result.result
            
            # 顯示漏洞統計
            print("\n📊 漏洞統計:")
            print(f"   ├─ 總漏洞數: {scan_result.get('total_vulnerabilities', 0)}")
            print(f"   ├─ 競態條件: {len(scan_result.get('race_conditions', []))} 個")
            print(f"   ├─ 價格操控: {len(scan_result.get('price_manipulations', []))} 個")
            print(f"   └─ 工作流程繞過: {len(scan_result.get('workflow_bypasses', []))} 個")
            
            # 顯示詳細漏洞（使用輔助函數）
            _print_vulnerability_details(scan_result)
            
            # 掃描信息
            scan_info = scan_result.get('scan_info', {})
            print("\n📋 掃描信息:")
            print(f"   ├─ 執行測試: {', '.join(scan_info.get('tests_performed', []))}")
            print(f"   ├─ 掃描時長: {scan_info.get('duration', 0):.2f}秒")
            print(f"   └─ 開始時間: {scan_info.get('start_time', 'N/A')}")
            
            return {
                "target": target_key,
                "success": True,
                "vulnerabilities": scan_result.get('total_vulnerabilities', 0),
                "duration": duration,
                "details": scan_result
            }
        
        elif result.error:
            print(f"\n❌ 掃描失敗: {result.error}")
            print(f"   錯誤代碼: {result.error_code}")
            
            return {
                "target": target_key,
                "success": False,
                "error": result.error,
                "duration": duration
            }
        
        else:
            print("\n⚠️  掃描完成但無結果")
            return {
                "target": target_key,
                "success": False,
                "error": "No result returned",
                "duration": duration
            }
    
    except Exception as e:
        print(f"\n❌ 掃描異常: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "target": target_key,
            "success": False,
            "error": str(e),
            "duration": 0
        }


async def _check_target_availability(config: dict) -> bool:
    """檢查單個靶場是否可用"""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(config['base_url'])
            if response.status_code < 500:
                print(f"   ✅ {config['name']} ({config['base_url']}) - 可用")
                return True
            print(f"   ⚠️  {config['name']} ({config['base_url']}) - 服務器錯誤")
            return False
    except Exception as e:
        print(f"   ❌ {config['name']} ({config['base_url']}) - 不可用 ({e})")
        return False


async def _detect_available_targets() -> dict:
    """檢測可用的靶場（降低認知複雜度）"""
    print("\n📌 檢測靶場可用性...")
    available_targets = {}
    
    for key, config in TARGETS.items():
        if await _check_target_availability(config):
            available_targets[key] = config
    
    return available_targets


def _save_scan_report(results: list, total_targets: int, successful_scans: int, 
                     total_vulnerabilities: int, total_duration: float):
    """保存掃描報告
    
    Note: 使用同步文件操作是可以接受的，因為：
    1. 報告文件較小，不會阻塞太久
    2. 這是程序結束前的最後一步
    3. 避免引入額外的 aiofiles 依賴
    """
    report_file = project_root / "data" / "logs" / f"bizlogic_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "scan_time": datetime.now().isoformat(),
        "targets_scanned": total_targets,
        "successful_scans": successful_scans,
        "total_vulnerabilities": total_vulnerabilities,
        "total_duration": total_duration,
        "results": results
    }
    
    # 同步文件操作在此處是可接受的（見函數文檔說明）
    with open(report_file, 'w', encoding='utf-8') as f:  # noqa: S7493
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 詳細報告已保存: {report_file}")


async def main():
    """主掃描流程"""
    
    print("\n" + "🚀"*40)
    print("AIVA BizLogic Real Scan - 實際靶場掃描")
    print("🚀"*40)
    
    # 初始化處理器
    print("\n📌 初始化 BizLogic Command Handler...")
    try:
        handler = BizLogicCommandHandler()
        print("✅ Handler 初始化成功")
    except Exception as e:
        print(f"❌ Handler 初始化失敗: {e}")
        return 1
    
    # 檢測可用靶場
    available_targets = await _detect_available_targets()
    
    if not available_targets:
        print("\n❌ 沒有可用的靶場，請確認 Docker 容器已啟動")
        print("   提示: docker ps 查看運行中的容器")
        return 1
    
    print(f"\n✅ 發現 {len(available_targets)} 個可用靶場")
    
    # 執行掃描
    print("\n" + "="*80)
    print("開始實際掃描")
    print("="*80)
    
    results = []
    for target_key, target_config in available_targets.items():
        result = await scan_target(handler, target_key, target_config)
        results.append(result)
    
    # 總結報告
    print("\n" + "="*80)
    print("📊 掃描總結報告")
    print("="*80)
    
    total_targets = len(results)
    successful_scans = sum(1 for r in results if r['success'])
    total_vulnerabilities = sum(r.get('vulnerabilities', 0) for r in results if r['success'])
    total_duration = sum(r.get('duration', 0) for r in results)
    
    print("\n統計數據:")
    print(f"   ├─ 掃描靶場: {total_targets} 個")
    print(f"   ├─ 成功掃描: {successful_scans} 個")
    print(f"   ├─ 發現漏洞: {total_vulnerabilities} 個")
    print(f"   └─ 總耗時: {total_duration:.2f} 秒")
    
    print("\n詳細結果:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        vuln_count = result.get('vulnerabilities', 0) if result['success'] else 'N/A'
        print(f"   {status} {result['target']}: {vuln_count} 個漏洞 (耗時: {result.get('duration', 0):.2f}s)")
        if not result['success']:
            print(f"      錯誤: {result.get('error', 'Unknown')}")
    
    # 保存報告
    _save_scan_report(results, total_targets, successful_scans, total_vulnerabilities, total_duration)
    
    print("\n" + "="*80)
    if total_vulnerabilities > 0:
        print(f"⚠️  發現 {total_vulnerabilities} 個業務邏輯漏洞!")
        print("="*80 + "\n")
        return 0
    else:
        print("✅ 未發現明顯的業務邏輯漏洞")
        print("="*80 + "\n")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
