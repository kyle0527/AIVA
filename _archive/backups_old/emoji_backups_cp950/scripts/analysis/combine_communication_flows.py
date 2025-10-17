#!/usr/bin/env python3
"""
組合通訊流程圖腳本
目的: 將分散的 Module/Function 流程圖組合成端到端流程
基於 1655 個 py2mermaid 生成的詳細流程圖
"""

from pathlib import Path
from typing import List, Dict, Tuple
import re
import json

class MermaidFlowCombiner:
    """Mermaid 流程圖組合器"""
    
    def __init__(self, diagram_dir: Path):
        self.diagram_dir = diagram_dir
        self.flows = self._load_all_flows()
        print(f"✅ 已載入 {len(self.flows)} 個流程圖")
    
    def _load_all_flows(self) -> Dict[str, str]:
        """載入所有 .mmd 檔案"""
        flows = {}
        for mmd_file in self.diagram_dir.glob("**/*.mmd"):
            try:
                flows[mmd_file.stem] = mmd_file.read_text(encoding='utf-8')
            except Exception as e:
                print(f"⚠️ 無法讀取 {mmd_file.name}: {e}")
        return flows
    
    def combine_task_dispatch_flow(self) -> str:
        """組合任務派發完整流程"""
        print("\n📊 組合任務派發流程...")
        components = [
            "core_aiva_core_messaging_task_dispatcher_Function___get_topic_for_tool",
            "core_aiva_core_messaging_task_dispatcher_Function___build_task_payload",
            "core_aiva_core_messaging_task_dispatcher_Function___build_message",
            "aiva_common_mq_Module",  # MQ 發送
        ]
        
        combined = self._create_combined_diagram(
            components,
            "任務派發完整流程 (Core → Worker)",
            "從用戶請求到消息發布的完整流程"
        )
        return combined
    
    def combine_sqli_detection_flow(self) -> str:
        """組合 SQLi 檢測完整流程"""
        print("\n📊 組合 SQLi 檢測流程...")
        components = [
            "function_function_sqli_aiva_func_sqli_worker_Module",
            "function_function_sqli_aiva_func_sqli_engines_error_detection_engine_Module",
            "function_function_sqli_aiva_func_sqli_engines_boolean_detection_engine_Module",
            "function_function_sqli_aiva_func_sqli_result_binder_publisher_Module",
        ]
        
        combined = self._create_combined_diagram(
            components,
            "SQLi 檢測完整流程 (Worker 執行)",
            "從接收任務到發布結果的完整流程"
        )
        return combined
    
    def combine_result_collection_flow(self) -> str:
        """組合結果收集完整流程"""
        print("\n📊 組合結果收集流程...")
        components = [
            "core_aiva_core_messaging_result_collector_Module",
            "core_aiva_core_messaging_result_collector_Function__register_handler",
            "core_aiva_core_messaging_result_collector_Function___set_pending_result",
        ]
        
        combined = self._create_combined_diagram(
            components,
            "結果收集完整流程 (Worker → Core)",
            "從訂閱結果 Topic 到處理結果的完整流程"
        )
        return combined
    
    def combine_scan_workflow(self) -> str:
        """組合掃描工作流程"""
        print("\n📊 組合掃描工作流程...")
        components = [
            "scan_aiva_scan_scan_orchestrator_Module",
            "scan_aiva_scan_core_crawling_engine_url_queue_manager_Module",
            "scan_aiva_scan_dynamic_engine_headless_browser_pool_Module",
            "scan_aiva_scan_fingerprint_manager_Module",
        ]
        
        combined = self._create_combined_diagram(
            components,
            "掃描工作流程 (Scan Worker)",
            "從接收掃描任務到發現 Asset 的完整流程"
        )
        return combined
    
    def _create_combined_diagram(self, components: List[str], title: str, description: str) -> str:
        """創建組合圖表（簡化版 - 按順序連接）"""
        sections = []
        
        for idx, comp_name in enumerate(components):
            if comp_name not in self.flows:
                print(f"  ⚠️ 未找到流程圖: {comp_name}")
                continue
            
            # 提取簡化的流程描述
            flow_content = self.flows[comp_name]
            summary = self._extract_flow_summary(flow_content, comp_name)
            sections.append(f"  subgraph S{idx} [\"{self._clean_name(comp_name)}\"]\n{summary}\n  end")
        
        # 按順序連接子圖
        connections = []
        for i in range(len(sections) - 1):
            connections.append(f"  S{i} -->|下一步| S{i+1}")
        
        diagram = f"""```mermaid
---
title: {title}
---
flowchart TB
  Start([開始: {description}]) --> S0
{chr(10).join(sections)}
{chr(10).join(connections)}
  S{len(sections)-1} --> End([結束])
  
  style Start fill:#90EE90
  style End fill:#FFB6C1
```"""
        
        print(f"  ✅ 已組合 {len([c for c in components if c in self.flows])} 個組件")
        return diagram
    
    def _extract_flow_summary(self, content: str, comp_name: str) -> str:
        """從流程圖中提取關鍵步驟摘要"""
        # 提取節點定義（簡化版）
        node_pattern = r'n\d+\[(.*?)\]'
        nodes = re.findall(node_pattern, content)
        
        # 清理並限制數量
        clean_nodes = []
        for node in nodes[:5]:  # 只取前 5 個關鍵節點
            clean_text = node.replace('&amp;', '&').replace('&#35;', '#').replace('&#40;', '(').replace('&#41;', ')')
            if len(clean_text) > 50:
                clean_text = clean_text[:47] + "..."
            clean_nodes.append(f"    {clean_text}")
        
        if len(nodes) > 5:
            clean_nodes.append(f"    ... 還有 {len(nodes) - 5} 個步驟")
        
        return "\n".join(clean_nodes)
    
    def _clean_name(self, name: str) -> str:
        """清理組件名稱"""
        # 移除前綴
        name = name.replace('core_aiva_core_', '').replace('function_function_', '').replace('scan_aiva_scan_', '')
        # 移除後綴
        name = name.replace('_Module', '').replace('_Function', '')
        # 替換底線為空格
        name = name.replace('_', ' ').title()
        return name
    
    def generate_all_combined_flows(self, output_dir: Path):
        """生成所有組合流程圖"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        flows = {
            "01_task_dispatch_complete.mmd": self.combine_task_dispatch_flow(),
            "02_sqli_detection_complete.mmd": self.combine_sqli_detection_flow(),
            "03_result_collection_complete.mmd": self.combine_result_collection_flow(),
            "04_scan_workflow_complete.mmd": self.combine_scan_workflow(),
        }
        
        print("\n💾 儲存組合流程圖...")
        for filename, content in flows.items():
            output_file = output_dir / filename
            output_file.write_text(content, encoding='utf-8')
            print(f"  ✅ {output_file.name}")
        
        # 生成 README
        self._generate_readme(output_dir, flows)
    
    def _generate_readme(self, output_dir: Path, flows: Dict[str, str]):
        """生成說明文件"""
        readme_content = """# 組合流程圖說明

本目錄包含從 1655 個詳細流程圖組合而成的端到端流程。

## 檔案列表

| 檔案 | 說明 | 組件數量 |
|------|------|---------|
| `01_task_dispatch_complete.mmd` | 任務派發完整流程 (Core → Worker) | 4 |
| `02_sqli_detection_complete.mmd` | SQLi 檢測完整流程 (Worker 執行) | 4 |
| `03_result_collection_complete.mmd` | 結果收集完整流程 (Worker → Core) | 3 |
| `04_scan_workflow_complete.mmd` | 掃描工作流程 (Scan Worker) | 4 |

## 使用方式

### 在 VS Code 中預覽

1. 安裝 Mermaid 預覽擴展
2. 開啟 `.mmd` 檔案
3. 使用預覽功能查看圖表

### 產生 PNG 圖片

```bash
# 使用 Mermaid CLI (需先安裝 @mermaid-js/mermaid-cli)
mmdc -i 01_task_dispatch_complete.mmd -o 01_task_dispatch.png
```

### 嵌入文檔

將 `.mmd` 檔案內容複製到 Markdown 文件中即可。

## 相關文檔

- [核心模組通訊流程分析](./../CORE_MODULE_COMMUNICATION_FLOW_ANALYSIS.md)
- [跨模組通訊 CLI 參考](./../CROSS_MODULE_COMMUNICATION_CLI_REFERENCE.md)

---

**生成時間**: 2025-10-16  
**來源**: 1655 個 py2mermaid 流程圖
"""
        
        readme_file = output_dir / "README.md"
        readme_file.write_text(readme_content, encoding='utf-8')
        print(f"  ✅ README.md")


def main():
    """主函式"""
    print("=" * 60)
    print("🔧 AIVA 通訊流程圖組合工具")
    print("=" * 60)
    
    # 設定路徑
    base_dir = Path(__file__).parent.parent.parent
    diagram_dir = base_dir / "_out1101016" / "mermaid_details" / "all_services"
    output_dir = base_dir / "_out1101016" / "combined_flows"
    
    if not diagram_dir.exists():
        print(f"❌ 錯誤: 找不到流程圖目錄 {diagram_dir}")
        return
    
    # 執行組合
    combiner = MermaidFlowCombiner(diagram_dir)
    combiner.generate_all_combined_flows(output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 組圖完成！")
    print(f"📂 輸出目錄: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
