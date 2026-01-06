#!/usr/bin/env python3
"""
AI 能力參考手冊生成器
從 enriched_classification.json 生成 AI 友好的能力查詢手冊

輸出格式：
1. Markdown 格式（人類閱讀）
2. JSON 格式（AI 快速檢索）
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class CapabilityReferenceGenerator:
    """能力參考手冊生成器"""
    
    def __init__(self, input_json: str):
        self.input_path = Path(input_json)
        self.output_dir = self.input_path.parent / "capability_references"
        self.output_dir.mkdir(exist_ok=True)
        
    def generate(self):
        """生成參考手冊"""
        
        print(f"📖 讀取數據: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flows = data.get("flows", [])
        print(f"📊 總共 {len(flows)} 個 Flows")
        
        # 按模組分類
        by_module = defaultdict(list)
        by_tags = defaultdict(list)
        by_complexity = defaultdict(list)
        
        for flow in flows:
            cap = flow.get("capability")
            if not cap:
                continue
            
            module = cap.get("module", "unknown")
            tags = cap.get("tags", [])
            complexity = cap.get("complexity", "unknown")
            
            by_module[module].append((flow['id'], cap))
            by_complexity[complexity].append((flow['id'], cap))
            
            for tag in tags:
                by_tags[tag].append((flow['id'], cap))
        
        # 生成 Markdown
        self._generate_markdown(flows, by_module, by_tags, by_complexity)
        
        # 生成 JSON 索引
        self._generate_json_index(flows, by_module, by_tags, by_complexity)
        
        print(f"\n✅ 完成！參考手冊已生成於: {self.output_dir}")
    
    def _generate_markdown(self, flows, by_module, by_tags, by_complexity):
        """生成 Markdown 格式手冊"""
        
        output_file = self.output_dir / "AI_Capability_Reference.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# AIVA AI 能力參考手冊\n\n")
            f.write(f"> 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"> 總能力數: {len(flows)} 個\n\n")
            
            f.write("## 📑 目錄\n\n")
            f.write("- [按模組分類](#按模組分類)\n")
            f.write("- [按複雜度分類](#按複雜度分類)\n")
            f.write("- [按標籤分類](#按標籤分類)\n")
            f.write("- [完整能力清單](#完整能力清單)\n\n")
            
            # 統計摘要
            f.write("## 📊 統計摘要\n\n")
            f.write("### 模組分佈\n\n")
            f.write("| 模組 | 能力數 |\n")
            f.write("|------|--------|\n")
            
            module_names = {
                "cognitive_core": "認知核心",
                "internal_exploration": "內探模組",
                "task_planning": "任務規劃",
                "external_learning": "外學模組",
                "core_capabilities": "核心能力",
                "service_backbone": "服務骨幹"
            }
            
            for module in sorted(by_module.keys()):
                name = module_names.get(module, module)
                count = len(by_module[module])
                f.write(f"| {name} ({module}) | {count} |\n")
            
            f.write("\n### 複雜度分佈\n\n")
            f.write("| 複雜度 | 能力數 | 說明 |\n")
            f.write("|--------|--------|------|\n")
            
            complexity_desc = {
                "simple": "簡單（1-2步）",
                "medium": "中等（3-4步）",
                "complex": "複雜（5步以上）"
            }
            
            for level in ["simple", "medium", "complex"]:
                count = len(by_complexity.get(level, []))
                desc = complexity_desc.get(level, level)
                f.write(f"| {level} | {count} | {desc} |\n")
            
            # 按模組詳細列表
            f.write("\n## 按模組分類\n\n")
            
            for module in sorted(by_module.keys()):
                name = module_names.get(module, module)
                capabilities = by_module[module]
                
                f.write(f"### {name} ({len(capabilities)} 個能力)\n\n")
                
                for flow_id, cap in sorted(capabilities, key=lambda x: x[0])[:20]:  # 每模組顯示前 20 個
                    f.write(f"#### Flow {flow_id}: {cap['name']}\n\n")
                    f.write(f"- **描述**: {cap['description']}\n")
                    f.write(f"- **CLI指令**: `{cap['command_template']}`\n")
                    f.write(f"- **標籤**: {', '.join(cap['tags'])}\n")
                    f.write(f"- **複雜度**: {cap['complexity']}\n\n")
                
                if len(capabilities) > 20:
                    f.write(f"*... 還有 {len(capabilities) - 20} 個能力，詳見完整清單*\n\n")
            
            # 完整清單（表格形式）
            f.write("\n## 完整能力清單\n\n")
            f.write("| Flow ID | 能力名稱 | 模組 | 複雜度 | 標籤 |\n")
            f.write("|---------|----------|------|--------|------|\n")
            
            for flow in flows:
                cap = flow.get("capability")
                if not cap:
                    continue
                
                flow_id = flow['id']
                name = cap['name']
                module = cap['module_zh']
                complexity = cap['complexity']
                tags = ', '.join(cap['tags'][:3])  # 只顯示前 3 個標籤
                
                f.write(f"| {flow_id} | {name} | {module} | {complexity} | {tags} |\n")
        
        print(f"  ✅ Markdown 手冊: {output_file}")
    
    def _generate_json_index(self, flows, by_module, by_tags, by_complexity):
        """生成 JSON 索引（供 AI 快速檢索）"""
        
        output_file = self.output_dir / "capability_index.json"
        
        # 構建索引結構
        index = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_capabilities": len(flows),
                "version": "1.0"
            },
            "capabilities": [],
            "indices": {
                "by_module": {},
                "by_tag": {},
                "by_complexity": {}
            }
        }
        
        # 完整能力列表
        for flow in flows:
            cap = flow.get("capability")
            if not cap:
                continue
            
            index["capabilities"].append({
                "flow_id": flow['id'],
                "name": cap['name'],
                "identifier": cap['identifier'],
                "description": cap['description'],
                "command_template": cap['command_template'],
                "tags": cap['tags'],
                "module": cap['module'],
                "module_zh": cap['module_zh'],
                "complexity": cap['complexity']
            })
        
        # 模組索引
        for module, items in by_module.items():
            index["indices"]["by_module"][module] = [item[0] for item in items]
        
        # 標籤索引
        for tag, items in by_tags.items():
            index["indices"]["by_tag"][tag] = [item[0] for item in items]
        
        # 複雜度索引
        for level, items in by_complexity.items():
            index["indices"]["by_complexity"][level] = [item[0] for item in items]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ JSON 索引: {output_file}")


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成 AI 能力參考手冊")
    parser.add_argument(
        '--input',
        default='C:/Users/User/Downloads/data/internal_exploration/enriched_classification.json',
        help='輸入 enriched JSON 檔案路徑'
    )
    
    args = parser.parse_args()
    
    generator = CapabilityReferenceGenerator(args.input)
    generator.generate()


if __name__ == "__main__":
    main()
