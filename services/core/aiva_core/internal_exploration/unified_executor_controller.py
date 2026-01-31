"""
統一執行器控制層 - AI 可直接調用的執行接口

整合兩個執行器：
1. aiva_internal_executor.py - 內部探索能力 (286 flows)
2. aiva_external_executor.py - 外部攻擊能力 (多語言)

參考現有 .bat 文件的執行方式，提供統一的 Python 接口

使用方式:
    # 方式 1: Python 調用
    from aiva_core.internal_exploration.unified_executor_controller import UnifiedExecutorController
    
    controller = UnifiedExecutorController()
    result = controller.execute_capability(
        capability="sqli",
        target="http://test.com",
        executor_type="auto"  # auto/internal/external
    )
    
    # 方式 2: 命令列
    python unified_executor_controller.py --capability sqli --target "http://test.com"
    
    # 方式 3: 互動式選單
    python unified_executor_controller.py --menu

實現日期: 2026-01-20
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

# 添加專案路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "services" / "core"))
sys.path.insert(0, str(PROJECT_ROOT / "services"))


@dataclass
class ExecutionResult:
    """執行結果"""
    success: bool
    executor: str  # "internal" or "external"
    capability: str
    flow_id: Optional[int] = None
    output: str = ""
    error: str = ""


class UnifiedExecutorController:
    """統一執行器控制器
    
    讓 AI 可以通過簡單接口控制內部和外部執行器
    """
    
    def __init__(self):
        """初始化控制器"""
        self.internal_executor = None
        self.external_executor = None
        self._initialized = False
        
        # 能力映射到執行器類型
        self.capability_executor_map = {
            # 外部攻擊能力 → external executor
            "sqli": "external",
            "xss": "external", 
            "csrf": "external",
            "ssrf": "external",
            "rce": "external",
            "lfi": "external",
            "rfi": "external",
            "xxe": "external",
            "deserial": "external",
            "auth_bypass": "external",
            "unified_attack": "external",
            "port_scan": "external",
            "web_crawl": "external",
            "fuzzing": "external",
            "business_logic": "external",
            
            # 內部探索能力 → internal executor
            "system_explore": "internal",
            "capability_scan": "internal",
            "health_check": "internal",
            "dependency_analysis": "internal",
            "code_analysis": "internal",
        }
        
        # 能力到 Flow ID 的映射 (從 latest_classification.json 動態載入)
        self.capability_flow_map = {}
    
    def initialize(self):
        """初始化執行器實例"""
        if self._initialized:
            return
        
        print("初始化統一執行器控制層...")
        
        # 1. 導入並初始化內部執行器
        try:
            from aiva_core.internal_exploration.aiva_internal_executor import FlowExecutor
            self.internal_executor = FlowExecutor()
            print("   - 內部執行器 (aiva_internal_executor) 已載入")
        except Exception as e:
            print(f"   - 內部執行器載入失敗: {e}")
        
        # 2. 導入並初始化外部執行器
        try:
            from aiva_core.internal_exploration.aiva_external_executor import MultiLangExecutor
            self.external_executor = MultiLangExecutor()
            print("   - 外部執行器 (aiva_external_executor) 已載入")
        except Exception as e:
            print(f"   - 外部執行器載入失敗: {e}")
        
        # 3. 載入能力映射
        self._load_capability_mappings()
        
        self._initialized = True
        print("統一執行器控制層初始化完成\n")
    
    def _load_capability_mappings(self):
        """從兩個數據源載入能力映射：
        1. latest_classification.json (內部探索能力 - 286 flows)
        2. classification_data.json (外部攻擊模組 - 210 flows)
        """
        
        # === 1. 載入內部探索能力 (latest_classification.json) ===
        internal_file = PROJECT_ROOT / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
        internal_count = 0
        
        if internal_file.exists():
            try:
                with open(internal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                flows = data.get('flows', [])
                
                # 內部能力關鍵字（匹配 primary_module 和 path）
                internal_keywords = {
                    'system_explore': ['internal_exploration', 'self_exploration', 'capability_discovery'],
                    'capability_scan': ['capability_scan', 'capability_registry', 'core_capabilities'],
                    'health_check': ['health_check', 'system_health', 'service_backbone'],
                    'dependency_analysis': ['dependency', 'dependency_analysis', 'learning_system'],
                    'code_analysis': ['code_analysis', 'static_analysis', 'analysis_engine'],
                }
                
                for flow in flows:
                    flow_id = flow.get('id')
                    primary_module = flow.get('primary_module', '').lower()
                    path = flow.get('path', [])
                    start = flow.get('start', '').lower()
                    end = flow.get('end', '').lower()
                    
                    # 合併所有可搜尋的文本
                    path_str = ' '.join(path).lower() if path else ''
                    searchable_text = f"{primary_module} {path_str} {start} {end}"
                    
                    for capability, keywords in internal_keywords.items():
                        if any(kw in searchable_text for kw in keywords):
                            self.capability_flow_map.setdefault(capability, []).append(flow_id)
                            internal_count += 1
                            break
                
                print(f"   - 內部能力: 載入 {internal_count} 個映射")
                
            except Exception as e:
                print(f"   - 載入內部能力映射失敗: {e}")
        else:
            print(f"   - 找不到內部探索數據: {internal_file}")
        
        # === 2. 載入外部攻擊模組 (classification_data.json) ===
        external_file = PROJECT_ROOT / "features_classification" / "classification_data.json"
        external_count = 0
        
        if external_file.exists():
            try:
                with open(external_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                flows = data.get('flows', [])
                
                # 外部攻擊能力關鍵字（匹配 module 名稱）
                external_keywords = {
                    'sqli': ['function_sqli', 'sql_injection'],
                    'xss': ['function_xss', 'cross_site_scripting'],
                    'ssrf': ['function_ssrf', 'server_side_request'],
                    'csrf': ['csrf', 'cross_site_request'],
                    'rce': ['rce', 'remote_code_execution', 'command_injection'],
                    'lfi': ['lfi', 'local_file_inclusion'],
                    'rfi': ['rfi', 'remote_file_inclusion'],
                    'xxe': ['xxe', 'xml_external_entity'],
                    'deserial': ['deserial', 'deserialization', 'unserialize'],
                    'auth_bypass': ['function_authn', 'authentication', 'auth_bypass'],
                    'idor': ['function_idor', 'access_control'],
                    'business_logic': ['function_bizlogic', 'business_logic'],
                    'port_scan': ['port_scan', 'nmap'],
                    'web_crawl': ['web_crawl', 'crawler', 'spider'],
                    'fuzzing': ['fuzzing', 'fuzzer', 'fuzz'],
                    'unified_attack': ['unified_attack', 'attack_coordinator'],
                }
                
                for flow in flows:
                    flow_id = flow.get('id')
                    module = flow.get('module', '').lower()
                    flow_type = flow.get('type', '').lower()
                    language = flow.get('language', 'Python').lower()
                    
                    searchable = f"{module} {flow_type}"
                    
                    for capability, keywords in external_keywords.items():
                        if any(kw in searchable for kw in keywords):
                            # 外部模組的映射需要包含語言資訊
                            self.capability_flow_map.setdefault(capability, []).append({
                                'id': flow_id,
                                'language': language,
                                'module': module
                            })
                            external_count += 1
                            break
                
                print(f"   - 外部模組: 載入 {external_count} 個映射")
                
            except Exception as e:
                print(f"   - 載入外部模組映射失敗: {e}")
        else:
            print(f"   - 找不到外部模組數據: {external_file}")
        
        print(f"   - 總計: {len(self.capability_flow_map)} 種能力，{internal_count + external_count} 個映射")
    
    def execute_capability(
        self,
        capability: str,
        target: Optional[str] = None,
        executor_type: Literal["auto", "internal", "external"] = "auto",
        flow_id: Optional[int] = None,
        dry_run: bool = False,
        **kwargs
    ) -> ExecutionResult:
        """執行指定能力
        
        Args:
            capability: 能力名稱，如 "sqli", "xss"
            target: 目標 URL (外部攻擊需要)
            executor_type: 執行器類型 - "auto"(自動選擇) / "internal" / "external"
            flow_id: 指定 Flow ID (可選，不指定則自動查找)
            dry_run: 是否 Dry Run 模式
            **kwargs: 其他參數
            
        Returns:
            ExecutionResult
        """
        if not self._initialized:
            self.initialize()
        
        # 1. 決定使用哪個執行器
        if executor_type == "auto":
            executor_type = self.capability_executor_map.get(capability, "external")
        
        print(f"執行能力: {capability}")
        print(f"   執行器: {executor_type}")
        print(f"   目標: {target or 'N/A'}")
        print(f"   Dry Run: {dry_run}")
        
        # 2. 查找 Flow ID (如果未指定)
        flow_info = None
        if flow_id is None and capability in self.capability_flow_map:
            flow_list = self.capability_flow_map[capability]
            if flow_list:
                first_item = flow_list[0]
                
                # 檢查是 dict (外部模組) 還是 int (內部能力)
                if isinstance(first_item, dict):
                    flow_info = first_item
                    flow_id = first_item['id']
                    print(f"   自動選擇外部 Flow ID: {flow_id} ({first_item['language']}, {first_item['module']})")
                else:
                    flow_id = first_item
                    print(f"   自動選擇內部 Flow ID: {flow_id}")
        
        # 3. 執行
        try:
            if executor_type == "internal":
                return self._execute_internal(capability, flow_id, dry_run, **kwargs)
            else:
                return self._execute_external(capability, flow_id, flow_info, target, dry_run, **kwargs)
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                executor=executor_type,
                capability=capability,
                flow_id=flow_id,
                error=str(e)
            )
    
    def _execute_internal(
        self,
        capability: str,
        flow_id: Optional[int],
        dry_run: bool,
        **kwargs
    ) -> ExecutionResult:
        """使用內部執行器執行"""
        if not self.internal_executor:
            raise RuntimeError("內部執行器未初始化")
        
        if flow_id is None:
            raise ValueError(f"能力 '{capability}' 沒有對應的 Flow ID")
        
        print(f"\n   → 調用 aiva_internal_executor.py")
        print(f"   → Flow ID: {flow_id}")
        
        try:
            # 調用內部執行器
            self.internal_executor.execute_flow(
                flow_id=flow_id,
                context_data=kwargs,
                dry_run=dry_run
            )
            
            return ExecutionResult(
                success=True,
                executor="internal",
                capability=capability,
                flow_id=flow_id,
                output=f"Flow {flow_id} 執行完成"
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                executor="internal",
                capability=capability,
                flow_id=flow_id,
                error=str(e)
            )
    
    def _execute_external(
        self,
        capability: str,
        flow_id: Optional[int],
        flow_info: Optional[dict],
        target: Optional[str],
        dry_run: bool,
        **kwargs
    ) -> ExecutionResult:
        """使用外部執行器執行
        
        Args:
            capability: 能力名稱
            flow_id: Flow ID
            flow_info: Flow 詳細資訊 (包含 language, module 等)
            target: 目標 URL
            dry_run: 是否 Dry Run
        """
        if not self.external_executor:
            raise RuntimeError("外部執行器未初始化")
        
        print(f"\n   → 調用 aiva_external_executor.py")
        
        try:
            # 判斷語言類型
            language = flow_info.get('language', 'python') if flow_info else 'python'
            
            if flow_id:
                print(f"   → Flow ID: {flow_id} ({language})")
                
                # 根據語言調用對應的執行方法
                if language == 'python':
                    result = self.external_executor.execute_python(
                        flow_id=flow_id,
                        dry_run=dry_run,
                        **kwargs
                    )
                elif language == 'go':
                    result = self.external_executor.execute_go(
                        flow_id=flow_id,
                        dry_run=dry_run,
                        **kwargs
                    )
                elif language == 'typescript':
                    result = self.external_executor.execute_typescript(
                        flow_id=flow_id,
                        dry_run=dry_run,
                        **kwargs
                    )
                elif language == 'rust':
                    result = self.external_executor.execute_rust(
                        func_name=flow_info.get('module', capability) if flow_info else capability,
                        dry_run=dry_run,
                        **kwargs
                    )
                else:
                    raise ValueError(f"不支援的語言: {language}")
            else:
                # 沒有 Flow ID，使用能力名稱執行（默認 Python）
                print(f"   → 能力: {capability} (無 Flow ID，使用 Python)")
                result = self.external_executor.execute_python(
                    func_name=capability,
                    dry_run=dry_run,
                    target=target,
                    **kwargs
                )
            
            return ExecutionResult(
                success=True,
                executor="external",
                capability=capability,
                flow_id=flow_id,
                output=str(result) if result else "執行完成"
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                executor="external",
                capability=capability,
                flow_id=flow_id,
                error=str(e)
            )
    
    def list_capabilities(self, executor_type: Optional[str] = None) -> dict:
        """列出可用能力
        
        Args:
            executor_type: 過濾執行器類型 - None(全部) / "internal" / "external"
            
        Returns:
            {
                "internal": ["system_explore", ...],
                "external": ["sqli", "xss", ...]
            }
        """
        if not self._initialized:
            self.initialize()
        
        capabilities = {"internal": [], "external": []}
        
        for cap, exec_type in self.capability_executor_map.items():
            if executor_type is None or executor_type == exec_type:
                capabilities[exec_type].append(cap)
        
        return capabilities
    
    def show_menu(self):
        """顯示互動式選單 (參考 .bat 文件方式)"""
        if not self._initialized:
            self.initialize()
        
        while True:
            print("\n" + "=" * 60)
            print("AIVA 統一執行器控制台".center(60))
            print("=" * 60)
            
            print("\n【1】執行外部攻擊能力")
            print("【2】執行內部探索能力")
            print("【3】列出所有能力")
            print("【4】查看能力詳情")
            print("【0】退出")
            
            choice = input("\n請選擇 (0-4): ").strip()
            
            if choice == "0":
                print("\n再見！")
                break
            
            elif choice == "1":
                self._menu_external_capabilities()
            
            elif choice == "2":
                self._menu_internal_capabilities()
            
            elif choice == "3":
                self._menu_list_capabilities()
            
            elif choice == "4":
                self._menu_capability_details()
            
            else:
                print("- 無效選擇")
    
    def _menu_external_capabilities(self):
        """外部攻擊能力選單"""
        print("\n" + "-" * 60)
        print("外部攻擊能力".center(60))
        print("-" * 60)
        
        caps = self.list_capabilities("external")["external"]
        
        for i, cap in enumerate(caps, 1):
            print(f"  {i}. {cap}")
        
        print(f"  0. 返回")
        
        choice = input("\n請選擇能力 (0-{}): ".format(len(caps))).strip()
        
        if choice == "0":
            return
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(caps):
                capability = caps[idx]
                target = input("請輸入目標 URL: ").strip()
                
                if target:
                    result = self.execute_capability(
                        capability=capability,
                        target=target,
                        executor_type="external",
                        dry_run=True
                    )
                    
                    print("\n執行結果:")
                    print(f"  成功: {result.success}")
                    if result.success:
                        print(f"  輸出: {result.output}")
                    else:
                        print(f"  錯誤: {result.error}")
        
        except ValueError:
            print("- 無效輸入")
    
    def _menu_internal_capabilities(self):
        """內部探索能力選單"""
        print("\n" + "-" * 60)
        print("內部探索能力".center(60))
        print("-" * 60)
        
        caps = self.list_capabilities("internal")["internal"]
        
        if not caps:
            print("  目前沒有已識別的內部探索能力")
            print("  提示: 可以使用 Flow ID 直接執行")
            
            flow_id = input("\n請輸入 Flow ID (或按 Enter 返回): ").strip()
            if flow_id:
                try:
                    result = self.execute_capability(
                        capability="custom",
                        executor_type="internal",
                        flow_id=int(flow_id),
                        dry_run=True
                    )
                    print(f"\n執行結果: {result.success}")
                except ValueError:
                    print("- 無效的 Flow ID")
            return
        
        for i, cap in enumerate(caps, 1):
            print(f"  {i}. {cap}")
        
        print(f"  0. 返回")
        
        choice = input("\n請選擇能力: ").strip()
        
        if choice != "0":
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(caps):
                    capability = caps[idx]
                    result = self.execute_capability(
                        capability=capability,
                        executor_type="internal",
                        dry_run=True
                    )
                    print(f"\n執行結果: {result.success}")
            except ValueError:
                print("- 無效輸入")
    
    def _menu_list_capabilities(self):
        """列出所有能力"""
        print("\n" + "-" * 60)
        print("所有可用能力".center(60))
        print("-" * 60)
        
        all_caps = self.list_capabilities()
        
        print("\n- 外部攻擊能力:")
        for cap in all_caps["external"]:
            flow_ids = self.capability_flow_map.get(cap, [])
            flow_info = f" (Flow: {flow_ids[0]})" if flow_ids else ""
            print(f"  - {cap}{flow_info}")
        
        print("\n- 內部探索能力:")
        for cap in all_caps["internal"]:
            flow_ids = self.capability_flow_map.get(cap, [])
            flow_info = f" (Flow: {flow_ids[0]})" if flow_ids else ""
            print(f"  - {cap}{flow_info}")
        
        input("\n按 Enter 繼續...")
    
    def _menu_capability_details(self):
        """查看能力詳情"""
        capability = input("\n請輸入能力名稱 (如 sqli, xss): ").strip().lower()
        
        if not capability:
            return
        
        print(f"\n能力: {capability}")
        print("-" * 40)
        
        executor = self.capability_executor_map.get(capability, "未知")
        print(f"執行器類型: {executor}")
        
        flow_ids = self.capability_flow_map.get(capability, [])
        if flow_ids:
            print(f"對應 Flow IDs: {flow_ids}")
        else:
            print("對應 Flow IDs: (未找到)")
        
        input("\n按 Enter 繼續...")


def main():
    """主函數 - 命令列接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AIVA 統一執行器控制台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 互動式選單
  python unified_executor_controller.py --menu
  
  # 執行 SQLi 測試
  python unified_executor_controller.py --capability sqli --target "http://test.com"
  
  # 執行指定 Flow ID
  python unified_executor_controller.py --flow 11 --executor internal --dry-run
  
  # 列出所有能力
  python unified_executor_controller.py --list
        """
    )
    
    parser.add_argument('--menu', action='store_true', help='啟動互動式選單')
    parser.add_argument('--capability', type=str, help='能力名稱 (如 sqli, xss)')
    parser.add_argument('--target', type=str, help='目標 URL')
    parser.add_argument('--flow', type=int, help='Flow ID')
    parser.add_argument('--executor', choices=['auto', 'internal', 'external'], 
                       default='auto', help='執行器類型')
    parser.add_argument('--dry-run', action='store_true', help='Dry Run 模式')
    parser.add_argument('--list', action='store_true', help='列出所有能力')
    
    args = parser.parse_args()
    
    controller = UnifiedExecutorController()
    controller.initialize()
    
    if args.menu:
        controller.show_menu()
    
    elif args.list:
        caps = controller.list_capabilities()
        print("\n外部攻擊能力:")
        for cap in caps["external"]:
            print(f"  - {cap}")
        print("\n內部探索能力:")
        for cap in caps["internal"]:
            print(f"  - {cap}")
    
    elif args.capability or args.flow:
        result = controller.execute_capability(
            capability=args.capability or "custom",
            target=args.target,
            executor_type=args.executor,
            flow_id=args.flow,
            dry_run=args.dry_run
        )
        
        print("\n" + "=" * 60)
        print("執行結果".center(60))
        print("=" * 60)
        print(f"成功: {result.success}")
        print(f"執行器: {result.executor}")
        print(f"能力: {result.capability}")
        if result.flow_id:
            print(f"Flow ID: {result.flow_id}")
        if result.output:
            print(f"輸出: {result.output}")
        if result.error:
            print(f"錯誤: {result.error}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
