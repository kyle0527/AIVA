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
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent.parent  # 到 AIVA-git

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AIVACapabilityManager:
    """AIVA 能力管理器 - AI 友好版"""
    
    def __init__(self):
        self.data_path = self._find_data_file()
        self.data = self._load_data()
        
    def _find_data_file(self) -> Path:
        """尋找數據文件（優先使用 enriched 版本）"""
        possible_paths = [
            Path("C:/Users/User/Downloads/data/internal_exploration/enriched_classification.json"),
            Path("C:/Users/User/Downloads/data/internal_exploration/latest_classification.json"),
            Path("C:/D/fold7/AIVA-git/data/internal_exploration/enriched_classification.json"),
            Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/enriched_classification.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                has_capability = self._check_has_capability(path)
                print(f"✅ 使用數據: {path.name} {'(含能力資訊)' if has_capability else '(舊版)'}")
                return path
        
        print("❌ 找不到數據文件")
        sys.exit(1)
    
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
            capability = flow.get("capability")
            if not capability:
                continue
            
            matched = False
            
            if by in ["all", "name"]:
                if query_lower in capability.get("name", "").lower():
                    matched = True
                if query_lower in capability.get("identifier", "").lower():
                    matched = True
            
            if by in ["all", "tag"]:
                tags = capability.get("tags", [])
                if any(query_lower in tag.lower() for tag in tags):
                    matched = True
            
            if by in ["all", "module"]:
                if query_lower in capability.get("module", "").lower():
                    matched = True
                if query_lower in capability.get("module_zh", "").lower():
                    matched = True
            
            if by in ["all", "description"]:
                if query_lower in capability.get("description", "").lower():
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
            print(f"❌ Flow {flow_id} 不存在")
            return
        
        capability = flow.get("capability")
        if not capability:
            print(f"⚠️ Flow {flow_id} 無能力資訊（使用舊版數據）")
            print(f"路徑: {' -> '.join(flow['path'])}")
            return
        
        # 顯示詳細資訊
        print(f"\n{'='*70}")
        print(f"🎯 Flow {flow_id}: {capability['name']}")
        print(f"{'='*70}")
        
        print(f"\n📋 能力描述:")
        print(f"   {capability['description']}")
        
        print(f"\n🔧 CLI 指令:")
        print(f"   {capability['command_template']}")
        
        print(f"\n🏷️  標籤: {', '.join(capability['tags'])}")
        print(f"📦 模組: {capability['module_zh']} ({capability['module']})")
        print(f"📊 複雜度: {capability['complexity']}")
        print(f"🔀 流程長度: {flow['length']} 步")
        
        print(f"\n📍 完整路徑:")
        for i, step in enumerate(flow['path'], 1):
            arrow = "   ↓" if i < len(flow['path']) else ""
            print(f"   {i}. {step}")
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
        print(f"{'Flow ID':<10} {'能力名稱':<25} {'模組':<15} {'標籤'}")
        print("-" * 85)
        
        for flow in results:
            cap = flow.get("capability", {})
            flow_id = flow['id']
            name = cap.get('name', 'Unknown')[:23]
            module = cap.get('module_zh', 'Unknown')[:13]
            tags = ', '.join(cap.get('tags', [])[:3])[:35]
            print(f"{flow_id:<10} {name:<25} {module:<15} {tags}")
        
        print(f"\n💡 提示:")
        print(f"   - 使用 --info <flow_id> 查看詳細資訊")
        print(f"   - 使用 --flow <flow_id> 執行能力")
        print()
    
    def list_all(self, group_by: str = "module"):
        """列出所有能力"""
        from collections import defaultdict
        
        flows = self.data.get("flows", [])
        
        if group_by == "module":
            grouped = defaultdict(list)
            for flow in flows:
                cap = flow.get("capability", {})
                module = cap.get("module", "unknown")
                grouped[module].append(flow)
            
            print(f"\n📊 所有能力（按模組分組）- 總計 {len(flows)} 個\n")
            
            module_names = {
                "cognitive_core": "認知核心",
                "internal_exploration": "內探模組",
                "task_planning": "任務規劃",
                "external_learning": "外學模組",
                "core_capabilities": "核心能力",
                "service_backbone": "服務骨幹"
            }
            
            for module in sorted(grouped.keys()):
                name_zh = module_names.get(module, module)
                count = len(grouped[module])
                print(f"\n{name_zh} ({module}): {count} 個能力")
                print("-" * 70)
                
                for flow in grouped[module][:10]:  # 每個模組顯示前 10 個
                    cap = flow.get("capability", {})
                    print(f"   Flow {flow['id']}: {cap.get('name', '?')}")
                
                if count > 10:
                    print(f"   ... 還有 {count - 10} 個")
            
            print(f"\n💡 使用 --search <關鍵字> 搜尋特定能力\n")


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
