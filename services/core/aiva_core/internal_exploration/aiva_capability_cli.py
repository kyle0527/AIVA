#!/usr/bin/env python3
"""
AIVA 能力查詢與執行工具 (Enhanced CLI)

核心功能：
1. **能力搜尋** (--search): 讓 AI 在執行前找到合適的能力
2. **能力資訊** (--info): 查看詳細的能力描述
3. **能力執行** (--flow): 執行指定的 Flow
4. **列表模式** (--list): 瀏覽所有可用能力

使用範例：
    # AI 搜尋 XSS 相關能力
    python aiva_capability_cli.py --search xss
    
    # 查看 Flow 313 的詳細資訊
    python aiva_capability_cli.py --info 313
    
    # 執行 Flow 313
    python aiva_capability_cli.py --flow 313
    
    # 按模組搜尋
    python aiva_capability_cli.py --search cognitive --search-by module
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 設定專案根目錄
CURRENT_DIR = Path(__file__).resolve().parent
# python_tools → internal_exploration → aiva_core → core → services → PROJECT_ROOT
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AIVACapabilityManager:
    """AIVA 能力管理器 - AI 友好版"""
    
    def __init__(self):
        self.data_path = self._find_data_file()
        self.data = self._load_data()
        
    def _find_data_file(self) -> Path:
        """尋找數據文件（使用統一路徑配置，無降級）"""
        # 直接載入 config 模組避免 __init__.py 的相對導入問題
        import importlib.util
        services_root = CURRENT_DIR.parent.parent.parent.parent
        config_path = services_root / "integration" / "aiva_integration" / "config.py"
        
        if not config_path.exists():
            raise FileNotFoundError(f"❌ 找不到配置文件: {config_path}")
        
        spec = importlib.util.spec_from_file_location("aiva_integration_config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"❌ 無法載入配置模組: {config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        LATEST_CLASSIFICATION_JSON = config_module.LATEST_CLASSIFICATION_JSON
        INTERNAL_EXPLORATION_DIR = config_module.INTERNAL_EXPLORATION_DIR
        
        # 優先使用 enriched 版本
        enriched_path = INTERNAL_EXPLORATION_DIR / "enriched_classification.json"
        if enriched_path.exists():
            has_capability = self._check_has_capability(enriched_path)
            print(f"[OK] 使用增強版數據: {enriched_path.name} {'(含能力資訊)' if has_capability else '(舊版)'}")
            return enriched_path
        
        if LATEST_CLASSIFICATION_JSON.exists():
            has_capability = self._check_has_capability(LATEST_CLASSIFICATION_JSON)
            print(f"[OK] 使用 Integration 配置: {LATEST_CLASSIFICATION_JSON.name} {'(含能力資訊)' if has_capability else '(舊版)'}")
            return LATEST_CLASSIFICATION_JSON
        
        raise FileNotFoundError(
            f"[ERROR] 找不到數據文件\n"
            f"   檢查路徑: {enriched_path}\n"
            f"   檢查路徑: {LATEST_CLASSIFICATION_JSON}\n"
            f"   請先執行: python aiva_exploration_pipeline.py --target core"
        )
    
    def _check_has_capability(self, path: Path) -> bool:
        """檢查數據文件是否包含能力資訊"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                flows = data.get("flows", [])
                return len(flows) > 0 and "capability" in flows[0]
        except:
            return False
    
    def _load_data(self) -> Dict[str, Any]:
        """載入數據"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 讀取數據失敗: {e}")
            return {"flows": []}
    
    def _get_flow_module(self, flow: Dict) -> str:
        """獲取 Flow 的模組（兼容新舊格式）"""
        # 新版格式（有 capability 物件）
        cap = flow.get("capability")
        if cap:
            return cap.get("module", "unknown")
        # 舊版格式（直接使用 primary_module）
        return flow.get("primary_module", "unknown")
    
    def _get_flow_tags(self, flow: Dict) -> List[str]:
        """獲取 Flow 的標籤（兼容新舊格式）"""
        cap = flow.get("capability")
        if cap:
            return cap.get("tags", [])
        # 舊版使用 structured_tags
        return flow.get("structured_tags", [])
    
    def _get_flow_name(self, flow: Dict) -> str:
        """獲取 Flow 名稱（兼容新舊格式）"""
        cap = flow.get("capability")
        if cap:
            return cap.get("name", "Unknown")
        # 舊版：從 func_names 組合或使用 path
        func_names = flow.get("func_names", [])
        if func_names:
            return " -> ".join(func_names[:2])
        path = flow.get("path", [])
        return " -> ".join(path[:2]) if path else "Unknown"
    
    def _get_flow_description(self, flow: Dict) -> str:
        """獲取 Flow 描述（兼容新舊格式）"""
        cap = flow.get("capability")
        if cap:
            return cap.get("description", "")
        # 舊版：從 classifications 取第一個 description
        classifications = flow.get("classifications", [])
        if classifications:
            return classifications[0].get("description", "")
        return ""
    
    def search(self, query: str, by: str = "all", limit: int = 50) -> List[Dict]:
        """
        搜尋能力
        
        Args:
            query: 搜尋關鍵字
            by: 搜尋範圍 ("all", "name", "tag", "module", "description")
            limit: 最多返回結果數
            
        Returns:
            符合條件的 Flow 列表
        """
        results = []
        query_lower = query.lower()
        
        for flow in self.data.get("flows", []):
            # 跳過已合併的流程
            if 'merged_into' in flow:
                continue
                
            matched = False
            
            if by in ["all", "name"]:
                name = self._get_flow_name(flow)
                if query_lower in name.lower():
                    matched = True
                # 也搜尋 func_names
                for fn in flow.get("func_names", []):
                    if query_lower in fn.lower():
                        matched = True
                        break
            
            if by in ["all", "tag"]:
                tags = self._get_flow_tags(flow)
                if any(query_lower in tag.lower() for tag in tags):
                    matched = True
            
            if by in ["all", "module"]:
                module = self._get_flow_module(flow)
                if query_lower in module.lower():
                    matched = True
                # 也檢查 modules 列表（舊版）
                for mod in flow.get("modules", []):
                    if query_lower in mod.lower():
                        matched = True
                        break
            
            if by in ["all", "description"]:
                desc = self._get_flow_description(flow)
                if query_lower in desc.lower():
                    matched = True
            
            if matched:
                results.append(flow)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_flow(self, flow_id: int) -> Optional[Dict]:
        """獲取指定 Flow"""
        for flow in self.data.get("flows", []):
            if flow["id"] == flow_id:
                return flow
        return None
    
    def show_info(self, flow_id: int):
        """顯示 Flow 的詳細能力資訊"""
        flow = self.get_flow(flow_id)
        if not flow:
            print(f"[ERROR] Flow {flow_id} 不存在")
            return
        
        # 獲取資訊（兼容新舊格式）
        name = self._get_flow_name(flow)
        module = self._get_flow_module(flow)
        tags = self._get_flow_tags(flow)
        desc = self._get_flow_description(flow)
        
        # 顯示詳細資訊
        print(f"\n{'='*70}")
        print(f"🎯 Flow {flow_id}: {name}")
        print(f"{'='*70}")
        
        print(f"\n📋 能力描述:")
        print(f"   {desc if desc else '無描述資訊'}")
        
        # CLI 指令
        cli_cmd = flow.get("cli_command", f"python aiva_cli_implementation.py --flow {flow_id}")
        print(f"\n🔧 CLI 指令:")
        print(f"   {cli_cmd}")
        
        print(f"\n🏷️  標籤: {', '.join(tags) if tags else '無標籤'}")
        
        module_names = {
            "cognitive_core": "認知核心",
            "internal_exploration": "內探模組",
            "task_planning": "任務規劃",
            "external_learning": "外學模組",
            "core_capabilities": "核心能力",
            "service_backbone": "服務骨幹",
            "learning_system": "學習系統"
        }
        module_zh = module_names.get(module, module)
        print(f"📦 模組: {module_zh} ({module})")
        
        # 複雜度（從 classifications 中獲取或計算）
        component_type = flow.get("primary_component_type", "Unknown")
        print(f"📊 組件類型: {component_type}")
        print(f"🔀 流程長度: {flow.get('length', len(flow.get('path', [])))} 步")
        
        # 是否為 AI 能力
        is_ai = flow.get("is_ai_capability", False)
        if is_ai:
            print(f"🤖 AI 能力: ✅")
        
        print(f"\n📍 完整路徑:")
        path = flow.get("path", [])
        full_path = flow.get("full_path", [])
        func_names = flow.get("func_names", [])
        
        for i, step in enumerate(path):
            arrow = "   ↓" if i < len(path) - 1 else ""
            func = func_names[i] if i < len(func_names) else ""
            file_path = full_path[i] if i < len(full_path) else ""
            print(f"   {i+1}. {step}")
            if func:
                print(f"      函數: {func}")
            if file_path:
                # 只顯示相對路徑
                short_path = file_path.replace("C:\\D\\fold7\\AIVA-git\\", "")
                print(f"      檔案: {short_path}")
            if arrow:
                print(arrow)
        
        print(f"\n💡 執行方式:")
        print(f"   # 預覽執行（dry-run）")
        print(f"   python aiva_cli_implementation.py --flow {flow_id} --dry-run")
        print(f"   ")
        print(f"   # 實際執行")
        print(f"   python aiva_cli_implementation.py --flow {flow_id}")
        
        print(f"\n{'='*70}\n")
    
    def show_search_results(self, results: List[Dict], query: str):
        """顯示搜尋結果"""
        if not results:
            print(f"\n❌ 找不到符合 '{query}' 的能力\n")
            return
        
        print(f"\n🔍 搜尋結果: 找到 {len(results)} 個能力\n")
        print(f"{'Flow ID':<10} {'能力名稱':<35} {'模組':<20} {'標籤'}")
        print("-" * 100)
        
        for flow in results:
            flow_id = flow['id']
            name = self._get_flow_name(flow)[:33]
            module = self._get_flow_module(flow)[:18]
            tags = ', '.join(self._get_flow_tags(flow)[:3])[:30]
            print(f"{flow_id:<10} {name:<35} {module:<20} {tags}")
        
        print(f"\n💡 提示:")
        print(f"   - 使用 --info <flow_id> 查看詳細資訊")
        print(f"   - 使用 --flow <flow_id> 執行能力")
        print()
    
    def list_all(self, group_by: str = "module"):
        """列出所有能力"""
        from collections import defaultdict
        
        all_flows = self.data.get("flows", [])
        # 過濾掉已合併的流程
        flows = [f for f in all_flows if 'merged_into' not in f]
        
        if group_by == "module":
            # 按模組分組，再按 (start, end) 分組統計能力種類
            module_flows = defaultdict(list)
            module_abilities = defaultdict(lambda: defaultdict(list))  # module -> (start,end) -> [flows]
            
            for flow in flows:
                module = self._get_flow_module(flow)
                module_flows[module].append(flow)
                key = (flow.get('start', ''), flow.get('end', ''))
                module_abilities[module][key].append(flow)
            
            # 計算總能力種類數和有多路徑的能力數
            total_abilities = sum(len(abilities) for abilities in module_abilities.values())
            multi_path_abilities = sum(1 for abilities in module_abilities.values() 
                                       for flows_list in abilities.values() if len(flows_list) > 1)
            
            module_names = {
                "cognitive_core": "認知核心",
                "internal_exploration": "內探模組",
                "task_planning": "任務規劃",
                "external_learning": "外學模組",
                "core_capabilities": "核心能力",
                "service_backbone": "服務骨幹",
                "learning_system": "學習系統"
            }
            
            # 統計表
            print(f"\n{'='*65}")
            print(f"📊 AIVA 能力統計摘要")
            print(f"{'='*65}\n")
            
            print(f"| {'模組':<20} | {'能力數':>8} | {'多路徑能力':>10} | {'數據流':>8} |")
            print(f"|{'-'*22}|{'-'*10}|{'-'*12}|{'-'*10}|")
            
            total_multi = 0
            for module in sorted(module_flows.keys()):
                name_zh = module_names.get(module, module)
                abilities = module_abilities[module]
                ability_count = len(abilities)
                flow_count = len(module_flows[module])
                multi_count = sum(1 for flows_list in abilities.values() if len(flows_list) > 1)
                total_multi += multi_count
                print(f"| {name_zh:<20} | {ability_count:>8} | {multi_count:>10} | {flow_count:>8} |")
            
            print(f"|{'-'*22}|{'-'*10}|{'-'*12}|{'-'*10}|")
            print(f"| {'**總計**':<20} | {total_abilities:>8} | {total_multi:>10} | {len(flows):>8} |")
            print()
            print(f"📌 說明：")
            print(f"   - 能力數：唯一的 (起點→終點) 組合")
            print(f"   - 多路徑能力：有 2 條以上不同執行路徑的能力")  
            print(f"   - 數據流：所有執行路徑總數")
            
            # 詳細列表
            print(f"\n{'='*65}")
            print(f"📋 各模組能力清單")
            print(f"{'='*65}")
            
            for module in sorted(module_flows.keys()):
                name_zh = module_names.get(module, module)
                abilities = module_abilities[module]
                
                # 分離單路徑和多路徑能力
                single_path = [(k, v) for k, v in abilities.items() if len(v) == 1]
                multi_path = [(k, v) for k, v in abilities.items() if len(v) > 1]
                
                print(f"\n### {name_zh} ({module})")
                print(f"    {len(abilities)} 種能力 | {len(multi_path)} 種有多路徑 | {len(module_flows[module])} 條數據流\n")
                
                # 先顯示多路徑能力
                if multi_path:
                    print(f"  **多路徑能力** ({len(multi_path)} 種):")
                    for (start, end), ability_flows in sorted(multi_path, key=lambda x: -len(x[1]))[:5]:
                        flow_ids = [f['id'] for f in ability_flows]
                        # 顯示路徑長度差異
                        lengths = [f.get('length', len(f.get('path', []))) for f in ability_flows]
                        len_info = f"長度: {min(lengths)}-{max(lengths)}步" if min(lengths) != max(lengths) else f"長度: {min(lengths)}步"
                        print(f"    • {start} → {end}")
                        print(f"      {len(flow_ids)} 條路徑 | {len_info} | ID: {','.join(map(str, flow_ids[:5]))}")
                    if len(multi_path) > 5:
                        print(f"    ... 還有 {len(multi_path) - 5} 種多路徑能力")
                
                # 再顯示單路徑能力摘要
                if single_path:
                    print(f"\n  **單路徑能力** ({len(single_path)} 種):")
                    for (start, end), ability_flows in list(single_path)[:5]:
                        flow_id = ability_flows[0]['id']
                        length = ability_flows[0].get('length', len(ability_flows[0].get('path', [])))
                        print(f"    • [{flow_id}] {start} → {end} ({length}步)")
                    if len(single_path) > 5:
                        print(f"    ... 還有 {len(single_path) - 5} 種單路徑能力")
            
            print(f"\n💡 使用 --search <關鍵字> 搜尋特定能力")
            print(f"💡 使用 --info <flow_id> 查看路徑詳情")


def main():
    """主程式"""
    parser = argparse.ArgumentParser(
        description="AIVA 能力查詢與執行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 搜尋能力
  %(prog)s --search xss
  %(prog)s --search 掃描 --search-by name
  %(prog)s --search cognitive --search-by module
  
  # 查看能力資訊
  %(prog)s --info 313
  
  # 列出所有能力
  %(prog)s --list
        """
    )
    
    parser.add_argument('--search', type=str,
                        help='搜尋能力（關鍵字）')
    parser.add_argument('--search-by', 
                        choices=['all', 'name', 'tag', 'module', 'description'],
                        default='all', 
                        help='搜尋範圍（預設: all）')
    parser.add_argument('--info', type=int,
                        help='顯示指定 Flow 的詳細能力資訊')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用能力')
    parser.add_argument('--limit', type=int, default=50,
                        help='搜尋結果最大數量（預設: 50）')
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = AIVACapabilityManager()
    
    # 搜尋模式
    if args.search:
        results = manager.search(args.search, args.search_by, args.limit)
        manager.show_search_results(results, args.search)
        return
    
    # 資訊模式
    if args.info is not None:
        manager.show_info(args.info)
        return
    
    # 列表模式
    if args.list:
        manager.list_all()
        return
    
    # 無參數時顯示幫助
    parser.print_help()


if __name__ == "__main__":
    main()
