#!/usr/bin/env python3
"""
性能基準測試 - BioNeuronCore vs 其他方法

對比:
1. BioNeuronCore + 簡單匹配器
2. 純關鍵字匹配
3. 純隨機（基準線）
"""
import sys
from pathlib import Path
import time
import random
from typing import List, Tuple
import json

sys.path.insert(0, str(Path(__file__).parent))

from services.core.aiva_core.ai_engine.bio_neuron_core import BioNeuronRAGAgent
from services.core.aiva_core.ai_engine.simple_matcher import SimpleTaskMatcher
from services.core.aiva_core.ai_engine.cli_tools import get_all_tools


class PerformanceBenchmark:
    """性能基準測試"""
    
    def __init__(self):
        """初始化測試"""
        self.test_cases = [
            ("掃描目標網站 example.com", "ScanTrigger"),
            ("開始安全掃描 https://test.com", "ScanTrigger"),
            ("檢測 SQL 注入漏洞在登入頁面", "SQLiDetector"),
            ("測試 SQL injection 在用戶表單", "SQLiDetector"),
            ("檢測 XSS 漏洞在搜索框", "XSSDetector"),
            ("測試跨站腳本攻擊", "XSSDetector"),
            ("分析 services/core 目錄代碼", "CodeAnalyzer"),
            ("檢查代碼質量和結構", "CodeAnalyzer"),
            ("讀取 pyproject.toml 配置", "CodeReader"),
            ("查看 README.md 文件內容", "CodeReader"),
            ("寫入配置到 config.json", "CodeWriter"),
            ("創建新的Python文件", "CodeWriter"),
            ("生成完整掃描報告", "ReportGenerator"),
            ("輸出漏洞檢測結果", "ReportGenerator"),
        ]
        
        # 初始化工具
        cli_tools = get_all_tools()
        self.tools = [{"name": n, "instance": o} for n, o in cli_tools.items()]
        self.tool_names = [t["name"] for t in self.tools]
    
    def test_random_baseline(self) -> dict:
        """測試隨機選擇（基準線）"""
        print("\n[方法 1] 純隨機選擇（基準線）")
        print("-" * 70)
        
        correct = 0
        times = []
        
        for task, expected in self.test_cases:
            start = time.perf_counter()
            predicted = random.choice(self.tool_names)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            if predicted == expected:
                correct += 1
        
        accuracy = correct / len(self.test_cases)
        avg_time = sum(times) / len(times) * 1000  # 轉換為毫秒
        
        print(f"  準確度: {correct}/{len(self.test_cases)} = {accuracy:.1%}")
        print(f"  平均延遲: {avg_time:.4f} ms")
        print(f"  吞吐量: {1000/avg_time:.0f} 推理/秒")
        
        return {
            "method": "Random Baseline",
            "accuracy": accuracy,
            "avg_latency_ms": avg_time,
            "throughput_per_sec": 1000 / avg_time,
            "correct": correct,
            "total": len(self.test_cases)
        }
    
    def test_simple_matcher(self) -> dict:
        """測試簡單匹配器"""
        print("\n[方法 2] 簡單匹配器（關鍵字）")
        print("-" * 70)
        
        matcher = SimpleTaskMatcher(self.tools)
        
        correct = 0
        times = []
        confidences = []
        
        for task, expected in self.test_cases:
            start = time.perf_counter()
            predicted, confidence = matcher.match(task)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            confidences.append(confidence)
            if predicted == expected:
                correct += 1
        
        accuracy = correct / len(self.test_cases)
        avg_time = sum(times) / len(times) * 1000
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"  準確度: {correct}/{len(self.test_cases)} = {accuracy:.1%}")
        print(f"  平均延遲: {avg_time:.4f} ms")
        print(f"  吞吐量: {1000/avg_time:.0f} 推理/秒")
        print(f"  平均信心度: {avg_confidence:.1%}")
        
        return {
            "method": "Simple Matcher",
            "accuracy": accuracy,
            "avg_latency_ms": avg_time,
            "throughput_per_sec": 1000 / avg_time,
            "avg_confidence": avg_confidence,
            "correct": correct,
            "total": len(self.test_cases)
        }
    
    def test_bioneuron_core(self) -> dict:
        """測試 BioNeuronCore AI"""
        print("\n[方法 3] BioNeuronCore AI")
        print("-" * 70)
        
        # 創建 AI 代理
        codebase = str(Path(__file__).parent)
        agent = BioNeuronRAGAgent(
            codebase_path=codebase,
            enable_planner=False,
            enable_tracer=False,
            enable_experience=False
        )
        
        # 替換工具
        agent.tools = self.tools
        agent.tool_map = {tool["name"]: tool for tool in self.tools}
        
        # 重新創建決策核心
        from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
        agent.decision_core = ScalableBioNet(agent.input_vector_size, len(self.tools))
        
        correct = 0
        times = []
        confidences = []
        
        for task, expected in self.test_cases:
            start = time.perf_counter()
            result = agent.invoke(task)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            predicted = result.get('tool_used')
            confidence = result.get('confidence', 0)
            confidences.append(confidence)
            
            if predicted == expected:
                correct += 1
        
        accuracy = correct / len(self.test_cases)
        avg_time = sum(times) / len(times) * 1000
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"  準確度: {correct}/{len(self.test_cases)} = {accuracy:.1%}")
        print(f"  平均延遲: {avg_time:.4f} ms")
        print(f"  吞吐量: {1000/avg_time:.0f} 推理/秒")
        print(f"  平均信心度: {avg_confidence:.1%}")
        print(f"  神經網路參數: {agent.decision_core.total_params:,}")
        
        return {
            "method": "BioNeuronCore AI",
            "accuracy": accuracy,
            "avg_latency_ms": avg_time,
            "throughput_per_sec": 1000 / avg_time,
            "avg_confidence": avg_confidence,
            "neural_params": agent.decision_core.total_params,
            "correct": correct,
            "total": len(self.test_cases)
        }
    
    def test_hybrid_approach(self) -> dict:
        """測試混合方法（匹配器 + 神經網路）"""
        print("\n[方法 4] 混合方法（推薦）")
        print("-" * 70)
        
        # 創建兩個組件
        matcher = SimpleTaskMatcher(self.tools)
        
        codebase = str(Path(__file__).parent)
        agent = BioNeuronRAGAgent(
            codebase_path=codebase,
            enable_planner=False,
            enable_tracer=False,
            enable_experience=False
        )
        agent.tools = self.tools
        agent.tool_map = {tool["name"]: tool for tool in self.tools}
        
        from services.core.aiva_core.ai_engine.bio_neuron_core import ScalableBioNet
        agent.decision_core = ScalableBioNet(agent.input_vector_size, len(self.tools))
        
        correct = 0
        times = []
        confidences = []
        matcher_used = 0
        neural_used = 0
        
        for task, expected in self.test_cases:
            start = time.perf_counter()
            
            # 1. 嘗試簡單匹配
            match_tool, match_conf = matcher.match(task)
            
            # 2. 如果信心度低，使用神經網路
            if match_conf >= 0.7:
                predicted = match_tool
                confidence = match_conf
                matcher_used += 1
            else:
                result = agent.invoke(task)
                predicted = result.get('tool_used')
                confidence = result.get('confidence', 0)
                neural_used += 1
            
            elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            confidences.append(confidence)
            
            if predicted == expected:
                correct += 1
        
        accuracy = correct / len(self.test_cases)
        avg_time = sum(times) / len(times) * 1000
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"  準確度: {correct}/{len(self.test_cases)} = {accuracy:.1%}")
        print(f"  平均延遲: {avg_time:.4f} ms")
        print(f"  吞吐量: {1000/avg_time:.0f} 推理/秒")
        print(f"  平均信心度: {avg_confidence:.1%}")
        print(f"  使用匹配器: {matcher_used} 次")
        print(f"  使用神經網路: {neural_used} 次")
        
        return {
            "method": "Hybrid (Matcher + Neural)",
            "accuracy": accuracy,
            "avg_latency_ms": avg_time,
            "throughput_per_sec": 1000 / avg_time,
            "avg_confidence": avg_confidence,
            "matcher_used": matcher_used,
            "neural_used": neural_used,
            "correct": correct,
            "total": len(self.test_cases)
        }
    
    def generate_comparison_report(self, results: List[dict]):
        """生成對比報告"""
        print("\n" + "="*70)
        print("性能對比總結")
        print("="*70)
        
        # 排序（按準確度）
        results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        print("\n[準確度排名]")
        for i, r in enumerate(results_sorted, 1):
            print(f"  {i}. {r['method']}: {r['accuracy']:.1%}")
        
        # 排序（按速度）
        results_by_speed = sorted(results, key=lambda x: x['avg_latency_ms'])
        
        print("\n[速度排名（延遲越低越好）]")
        for i, r in enumerate(results_by_speed, 1):
            print(f"  {i}. {r['method']}: {r['avg_latency_ms']:.4f} ms")
        
        # 綜合評分
        print("\n[綜合評分（準確度×速度權重）]")
        for r in results:
            # 綜合分數 = 準確度 * 0.7 + (1 - 標準化延遲) * 0.3
            max_latency = max(res['avg_latency_ms'] for res in results)
            normalized_latency = r['avg_latency_ms'] / max_latency
            score = r['accuracy'] * 0.7 + (1 - normalized_latency) * 0.3
            r['综合分數'] = score
        
        results_by_score = sorted(results, key=lambda x: x['综合分數'], reverse=True)
        for i, r in enumerate(results_by_score, 1):
            print(f"  {i}. {r['method']}: {r['综合分數']:.3f}")
        
        # 保存報告
        output_file = Path("data/ai_training/benchmark_report.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_cases": len(self.test_cases),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 完整報告已保存: {output_file}")
        
        # 勝出者
        winner = results_by_score[0]
        print(f"\n🏆 綜合冠軍: {winner['method']}")
        print(f"   準確度: {winner['accuracy']:.1%}")
        print(f"   延遲: {winner['avg_latency_ms']:.4f} ms")
        print(f"   綜合分數: {winner['综合分數']:.3f}")


def main():
    """主測試流程"""
    print("="*70)
    print("BioNeuronCore AI 性能基準測試")
    print("="*70)
    print(f"\n測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    benchmark = PerformanceBenchmark()
    print(f"測試案例數: {len(benchmark.test_cases)}")
    print(f"工具數量: {len(benchmark.tools)}")
    
    results = []
    
    # 測試 1: 隨機基準線
    results.append(benchmark.test_random_baseline())
    
    # 測試 2: 簡單匹配器
    results.append(benchmark.test_simple_matcher())
    
    # 測試 3: BioNeuronCore
    results.append(benchmark.test_bioneuron_core())
    
    # 測試 4: 混合方法
    results.append(benchmark.test_hybrid_approach())
    
    # 生成對比報告
    benchmark.generate_comparison_report(results)
    
    print("\n" + "="*70)
    print("測試完成！")
    print("="*70)


if __name__ == "__main__":
    main()
