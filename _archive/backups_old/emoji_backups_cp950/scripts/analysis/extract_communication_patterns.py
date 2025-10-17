#!/usr/bin/env python3
"""
從流程圖中提取通訊模式
目的: 自動分析所有 broker.publish/subscribe 調用，建立通訊關係圖
基於 1655 個 py2mermaid 生成的詳細流程圖
"""

from pathlib import Path
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import json

class CommunicationPatternExtractor:
    """通訊模式提取器"""
    
    def __init__(self, diagram_dir: Path):
        self.diagram_dir = diagram_dir
        self.patterns = {
            'publishes': [],     # (source_module, topic, details)
            'subscribes': [],    # (source_module, topic, details)
            'topics': [],        # (source_module, topic_constant)
            'message_flows': [], # (from_module, to_module, topic)
        }
        self.stats = defaultdict(int)
    
    def extract_all_patterns(self):
        """提取所有通訊模式"""
        print("🔍 開始掃描流程圖...")
        
        mmd_files = list(self.diagram_dir.glob("**/*.mmd"))
        total = len(mmd_files)
        
        for idx, mmd_file in enumerate(mmd_files, 1):
            if idx % 100 == 0:
                print(f"  進度: {idx}/{total}")
            
            try:
                content = mmd_file.read_text(encoding='utf-8')
                module_name = self._extract_module_name(mmd_file.stem)
                
                # 提取各種模式
                self._find_publishes(content, module_name, mmd_file.stem)
                self._find_subscribes(content, module_name, mmd_file.stem)
                self._find_topics(content, module_name)
                self._find_message_handlers(content, module_name)
                
            except Exception as e:
                print(f"  ⚠️ 處理 {mmd_file.name} 時出錯: {e}")
        
        print(f"✅ 完成掃描 {total} 個流程圖\n")
        self._calculate_stats()
    
    def _extract_module_name(self, filename: str) -> str:
        """從檔案名提取模組名稱"""
        # 範例: core_aiva_core_messaging_task_dispatcher_Module → Core.TaskDispatcher
        parts = filename.split('_')
        
        if 'core' in parts[:2]:
            return f"Core.{parts[-2] if parts[-1] == 'Module' else parts[-1]}"
        elif 'function' in parts[:2]:
            func_type = parts[2] if len(parts) > 2 else 'unknown'
            return f"Function.{func_type.upper()}"
        elif 'scan' in parts[:2]:
            return "Scan"
        elif 'integration' in parts[:2]:
            return "Integration"
        else:
            return parts[0].capitalize()
    
    def _find_publishes(self, content: str, module: str, filename: str):
        """查找 publish 調用"""
        # 匹配模式: broker.publish(...) 或 publish(...)
        patterns = [
            r'broker\.publish\s*\((.*?)\)',
            r'self\.broker\.publish\s*\((.*?)\)',
            r'\.publish\s*\((.*?)\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # 嘗試提取 topic
                topic = self._extract_topic_from_call(match)
                if topic:
                    self.patterns['publishes'].append((module, topic, filename))
                    self.stats['total_publishes'] += 1
    
    def _find_subscribes(self, content: str, module: str, filename: str):
        """查找 subscribe 調用"""
        patterns = [
            r'broker\.subscribe\s*\((.*?)\)',
            r'self\.broker\.subscribe\s*\((.*?)\)',
            r'\.subscribe\s*\((.*?)\)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                topic = self._extract_topic_from_call(match)
                if topic:
                    self.patterns['subscribes'].append((module, topic, filename))
                    self.stats['total_subscribes'] += 1
    
    def _find_topics(self, content: str, module: str):
        """查找 Topic 常量使用"""
        # 匹配: Topic.TASKS_FUNCTION_SQLI 等
        pattern = r'Topic\.((?:TASKS|RESULTS|EVENTS|COMMANDS|STATUS|FEEDBACK|LOG)_[A-Z_]+)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            full_topic = f"Topic.{match}"
            self.patterns['topics'].append((module, full_topic))
            self.stats['unique_topics'] = len(set([t for _, t in self.patterns['topics']]))
    
    def _find_message_handlers(self, content: str, module: str):
        """查找消息處理函式"""
        # 匹配: async def _handle_message, def process_task 等
        patterns = [
            r'def\s+(_handle_\w+|process_\w+|_on_\w+)\s*\(',
            r'async\s+def\s+(_handle_\w+|process_\w+|_on_\w+)\s*\(',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            self.stats[f'{module}_handlers'] += len(matches)
    
    def _extract_topic_from_call(self, call_str: str) -> str:
        """從函式調用中提取 Topic"""
        # 嘗試匹配 Topic.XXX
        topic_match = re.search(r'Topic\.([A-Z_]+)', call_str)
        if topic_match:
            return f"Topic.{topic_match.group(1)}"
        
        # 嘗試匹配字串常量
        string_match = re.search(r'["\']([a-z_.]+)["\']', call_str)
        if string_match:
            return string_match.group(1)
        
        return None
    
    def _calculate_stats(self):
        """計算統計資料"""
        # 模組發布統計
        publish_by_module = defaultdict(int)
        for module, _, _ in self.patterns['publishes']:
            publish_by_module[module] += 1
        self.stats['publish_by_module'] = dict(publish_by_module)
        
        # 模組訂閱統計
        subscribe_by_module = defaultdict(int)
        for module, _, _ in self.patterns['subscribes']:
            subscribe_by_module[module] += 1
        self.stats['subscribe_by_module'] = dict(subscribe_by_module)
        
        # Topic 使用統計
        topic_usage = defaultdict(int)
        for _, topic in self.patterns['topics']:
            topic_usage[topic] += 1
        self.stats['topic_usage'] = dict(sorted(topic_usage.items(), key=lambda x: -x[1]))
        
        # 建立通訊流
        self._build_message_flows()
    
    def _build_message_flows(self):
        """建立消息流向關係"""
        # Publisher -> Topic 映射
        topic_publishers = defaultdict(set)
        for module, topic, _ in self.patterns['publishes']:
            topic_publishers[topic].add(module)
        
        # Topic -> Subscriber 映射
        topic_subscribers = defaultdict(set)
        for module, topic, _ in self.patterns['subscribes']:
            topic_subscribers[topic].add(module)
        
        # 建立 Publisher -> Subscriber 流
        for topic in set(topic_publishers.keys()) | set(topic_subscribers.keys()):
            publishers = topic_publishers.get(topic, set())
            subscribers = topic_subscribers.get(topic, set())
            
            for pub in publishers:
                for sub in subscribers:
                    if pub != sub:  # 避免自己發送給自己
                        self.patterns['message_flows'].append((pub, sub, topic))
    
    def generate_report(self) -> str:
        """生成詳細分析報告"""
        lines = [
            "# AIVA 通訊模式分析報告",
            "",
            f"**生成時間**: 2025-10-16",
            f"**分析檔案數**: {self.stats.get('total_publishes', 0) + self.stats.get('total_subscribes', 0)}",
            "",
            "---",
            "",
            "## 📊 整體統計",
            "",
            f"- **總發布次數**: {self.stats.get('total_publishes', 0)}",
            f"- **總訂閱次數**: {self.stats.get('total_subscribes', 0)}",
            f"- **唯一 Topic 數量**: {self.stats.get('unique_topics', 0)}",
            f"- **消息流向數量**: {len(self.patterns['message_flows'])}",
            "",
            "---",
            "",
        ]
        
        # Publisher 統計
        lines.extend([
            "## 📤 Publisher 統計 (按模組)",
            "",
            "| 模組 | 發布次數 |",
            "|------|---------|",
        ])
        
        for module, count in sorted(
            self.stats.get('publish_by_module', {}).items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"| {module} | {count} |")
        
        lines.extend(["", "---", ""])
        
        # Subscriber 統計
        lines.extend([
            "## 📥 Subscriber 統計 (按模組)",
            "",
            "| 模組 | 訂閱次數 |",
            "|------|---------|",
        ])
        
        for module, count in sorted(
            self.stats.get('subscribe_by_module', {}).items(),
            key=lambda x: -x[1]
        ):
            lines.append(f"| {module} | {count} |")
        
        lines.extend(["", "---", ""])
        
        # Topic 使用統計
        lines.extend([
            "## 🏷️ Topic 使用頻率 (Top 30)",
            "",
            "| Topic | 使用次數 | 類型 |",
            "|-------|---------|------|",
        ])
        
        for topic, count in list(self.stats.get('topic_usage', {}).items())[:30]:
            topic_type = self._classify_topic(topic)
            lines.append(f"| `{topic}` | {count} | {topic_type} |")
        
        lines.extend(["", "---", ""])
        
        # 消息流向
        lines.extend([
            "## 🔄 主要消息流向 (Top 20)",
            "",
            "| 發送方 | 接收方 | Topic |",
            "|-------|-------|-------|",
        ])
        
        # 計算流向頻率
        flow_counts = defaultdict(int)
        for pub, sub, topic in self.patterns['message_flows']:
            flow_counts[(pub, sub, topic)] += 1
        
        for (pub, sub, topic), count in sorted(
            flow_counts.items(),
            key=lambda x: -x[1]
        )[:20]:
            lines.append(f"| {pub} | {sub} | `{topic}` |")
        
        lines.extend(["", "---", ""])
        
        # 通訊模式分類
        lines.extend(self._generate_pattern_classification())
        
        return "\n".join(lines)
    
    def _classify_topic(self, topic: str) -> str:
        """分類 Topic 類型"""
        if 'TASKS_' in topic:
            return "📋 任務派發"
        elif 'RESULTS_' in topic:
            return "📊 結果回報"
        elif 'EVENTS_' in topic:
            return "🎯 事件廣播"
        elif 'COMMANDS_' in topic:
            return "🎮 指令控制"
        elif 'STATUS_' in topic:
            return "💡 狀態同步"
        elif 'FEEDBACK_' in topic:
            return "🔁 反饋循環"
        elif 'LOG_' in topic:
            return "📝 日誌聚合"
        else:
            return "❓ 其他"
    
    def _generate_pattern_classification(self) -> List[str]:
        """生成通訊模式分類"""
        lines = [
            "## 🎯 通訊模式分類",
            "",
        ]
        
        # 按 Topic 類型分組
        patterns_by_type = defaultdict(list)
        for pub, sub, topic in self.patterns['message_flows']:
            pattern_type = self._classify_topic(topic)
            patterns_by_type[pattern_type].append((pub, sub, topic))
        
        for pattern_type, flows in sorted(patterns_by_type.items()):
            lines.extend([
                f"### {pattern_type}",
                "",
                f"**流向數量**: {len(flows)}",
                "",
            ])
            
            # 列出前 5 個範例
            if flows:
                lines.append("**範例**:")
                lines.append("")
                for pub, sub, topic in flows[:5]:
                    lines.append(f"- {pub} → {sub} (`{topic}`)")
                if len(flows) > 5:
                    lines.append(f"- ... 還有 {len(flows) - 5} 個")
                lines.append("")
        
        return lines
    
    def export_graph(self, output_file: Path):
        """匯出通訊圖 (GraphViz DOT 格式)"""
        lines = [
            "digraph communication {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded];",
            "  ",
            "  // 模組節點",
        ]
        
        # 收集所有模組
        all_modules = set()
        for pub, sub, _ in self.patterns['message_flows']:
            all_modules.add(pub)
            all_modules.add(sub)
        
        # 添加模組節點（按類型著色）
        for module in sorted(all_modules):
            color = self._get_module_color(module)
            lines.append(f'  "{module}" [fillcolor="{color}", style="filled,rounded"];')
        
        lines.extend([
            "  ",
            "  // 消息流向",
        ])
        
        # 添加邊（按 Topic 類型分組）
        flow_groups = defaultdict(list)
        for pub, sub, topic in self.patterns['message_flows']:
            flow_groups[(pub, sub)].append(topic)
        
        for (pub, sub), topics in flow_groups.items():
            # 合併相同路徑的多個 Topic
            if len(topics) == 1:
                label = topics[0].replace('Topic.', '')
            else:
                label = f"{len(topics)} topics"
            
            lines.append(f'  "{pub}" -> "{sub}" [label="{label}"];')
        
        lines.append("}")
        
        output_file.write_text("\n".join(lines), encoding='utf-8')
        print(f"✅ 通訊圖已匯出: {output_file}")
    
    def _get_module_color(self, module: str) -> str:
        """取得模組顏色"""
        if module.startswith('Core'):
            return "lightblue"
        elif module.startswith('Function'):
            return "lightgreen"
        elif module.startswith('Scan'):
            return "lightyellow"
        elif module.startswith('Integration'):
            return "lightpink"
        else:
            return "lightgray"
    
    def export_json(self, output_file: Path):
        """匯出 JSON 格式資料"""
        data = {
            'stats': dict(self.stats),
            'patterns': {
                'publishes': [
                    {'module': m, 'topic': t, 'source': s}
                    for m, t, s in self.patterns['publishes']
                ],
                'subscribes': [
                    {'module': m, 'topic': t, 'source': s}
                    for m, t, s in self.patterns['subscribes']
                ],
                'message_flows': [
                    {'from': pub, 'to': sub, 'topic': topic}
                    for pub, sub, topic in self.patterns['message_flows']
                ],
            },
            'metadata': {
                'generated_at': '2025-10-16',
                'total_diagrams_scanned': len(list(self.diagram_dir.glob("**/*.mmd"))),
            }
        }
        
        output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"✅ JSON 資料已匯出: {output_file}")


def main():
    """主函式"""
    print("=" * 60)
    print("🔍 AIVA 通訊模式提取工具")
    print("=" * 60)
    print()
    
    # 設定路徑
    base_dir = Path(__file__).parent.parent.parent
    diagram_dir = base_dir / "_out1101016" / "mermaid_details" / "all_services"
    output_dir = base_dir / "_out1101016"
    
    if not diagram_dir.exists():
        print(f"❌ 錯誤: 找不到流程圖目錄 {diagram_dir}")
        return
    
    # 執行提取
    extractor = CommunicationPatternExtractor(diagram_dir)
    extractor.extract_all_patterns()
    
    # 生成報告
    print("📝 生成報告...")
    report = extractor.generate_report()
    report_file = output_dir / "COMMUNICATION_PATTERN_ANALYSIS.md"
    report_file.write_text(report, encoding='utf-8')
    print(f"  ✅ {report_file.name}")
    
    # 匯出通訊圖
    print("\n🎨 匯出通訊圖...")
    graph_file = output_dir / "communication_graph.dot"
    extractor.export_graph(graph_file)
    
    # 匯出 JSON
    print("\n💾 匯出 JSON 資料...")
    json_file = output_dir / "communication_patterns.json"
    extractor.export_json(json_file)
    
    print("\n" + "=" * 60)
    print("🎉 分析完成！")
    print(f"📂 輸出目錄: {output_dir}")
    print("\n📌 後續步驟:")
    print(f"  1. 查看分析報告: {report_file.name}")
    print(f"  2. 產生通訊圖 PNG:")
    print(f"     dot -Tpng {graph_file.name} -o communication_graph.png")
    print(f"  3. 查看 JSON 資料: {json_file.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
