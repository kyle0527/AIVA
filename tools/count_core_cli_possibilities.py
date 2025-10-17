"""
Core CLI Possibilities Counter
計算核心模組 CLI 使用可能性數量

這個工具會：
1. 掃描 services/core/** 下的所有 CLI 入口點
2. 分析每個入口點的參數空間
3. 計算可能的使用組合數量
4. 輸出機器可讀的統計報告
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent.parent
CORE_ROOT = PROJECT_ROOT / "services" / "core"


def factorial(n: int) -> int:
    """計算階乘."""
    return math.factorial(n)


def permutation(n: int, k: int) -> int:
    """計算排列數 P(n,k) = n! / (n-k)!"""
    if k > n:
        return 0
    return factorial(n) // factorial(n - k)


def count_ordered_sequences(candidate_set_size: int) -> int:
    """
    計算從大小為 N 的集合中選取的所有非空有序序列數量.
    
    公式: Σ(k=1 to N) P(N,k) = Σ(k=1 to N) N!/(N-k)!
    
    Args:
        candidate_set_size: 候選集合大小 N
        
    Returns:
        所有可能的非空有序序列總數
    """
    total = 0
    for k in range(1, candidate_set_size + 1):
        total += permutation(candidate_set_size, k)
    return total


class CLIAnalyzer:
    """CLI 參數分析器."""
    
    def __init__(self, config_path: Path | None = None):
        """
        初始化分析器.
        
        Args:
            config_path: 設定檔路徑（包含 host 和 port 候選集合）
        """
        self.config = self._load_config(config_path)
        self.results: dict[str, Any] = {}
    
    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """載入設定檔."""
        default_config = {
            "host_candidates": ["127.0.0.1"],
            "port_candidates": [3000, 8000, 8080, 8888, 9000],
            "scope": "services/core/**"
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def analyze_auto_server(self) -> dict[str, Any]:
        """
        分析 auto_server.py 的 CLI 參數空間.
        
        Returns:
            分析結果字典
        """
        # 參數定義（基於實際程式碼）
        mode_values = ["ui", "ai", "hybrid"]
        host_candidates = self.config["host_candidates"]
        port_candidates = self.config["port_candidates"]
        
        # 計算各維度的可能性
        mode_count = len(mode_values)
        host_count = len(host_candidates)
        
        # --ports 參數：有序序列，順序有意義
        N = len(port_candidates)
        port_sequences_count = count_ordered_sequences(N)
        
        # 計算總組合數
        # 注意：--ports 是可選的，所以要加上「不提供 --ports」的情況
        total_with_ports = mode_count * host_count * port_sequences_count
        total_without_ports = mode_count * host_count  # 使用預設端口自動選擇
        total = total_with_ports + total_without_ports
        
        # 下界（最小可區分的使用方式）
        lower_bound = mode_count  # 只變 mode，其他都用預設
        
        # 依 mode 分組統計
        by_mode = {}
        for mode in mode_values:
            by_mode[mode] = {
                "with_ports": host_count * port_sequences_count,
                "without_ports": host_count,
                "total": host_count * (port_sequences_count + 1)
            }
        
        return {
            "cli_entry": "services/core/aiva_core/ui_panel/auto_server.py",
            "command_style": "single-command",
            "parameters": {
                "mode": {
                    "type": "enum",
                    "values": mode_values,
                    "count": mode_count,
                    "evidence": "parser.add_argument('--mode', choices=['ui', 'ai', 'hybrid'])"
                },
                "host": {
                    "type": "string",
                    "default": "127.0.0.1",
                    "candidates": host_candidates,
                    "count": host_count,
                    "evidence": "parser.add_argument('--host', default='127.0.0.1')"
                },
                "ports": {
                    "type": "list[int]",
                    "nargs": "+",
                    "order_matters": True,
                    "optional": True,
                    "candidates": port_candidates,
                    "candidate_count": N,
                    "sequence_count": port_sequences_count,
                    "evidence": "parser.add_argument('--ports', nargs='+', type=int)"
                }
            },
            "counts": {
                "lower_bound": lower_bound,
                "with_ports_specified": total_with_ports,
                "without_ports_specified": total_without_ports,
                "total": total,
                "by_mode": by_mode
            },
            "formulas": {
                "port_sequences": f"Σ(k=1 to {N}) P({N},k) = {port_sequences_count}",
                "total": f"{mode_count} (modes) × {host_count} (hosts) × ({port_sequences_count} (port sequences) + 1 (auto)) = {total}"
            }
        }
    
    def analyze_all(self) -> dict[str, Any]:
        """
        分析核心模組中所有的 CLI 入口點.
        
        Returns:
            完整分析報告
        """
        # 目前只有 auto_server.py 一個 CLI 入口
        cli_entries = []
        
        # 分析 auto_server.py
        auto_server_analysis = self.analyze_auto_server()
        cli_entries.append(auto_server_analysis)
        
        # 彙總結果
        total_cli_count = len(cli_entries)
        total_possibilities = sum(
            entry["counts"]["total"] 
            for entry in cli_entries
        )
        
        self.results = {
            "scope": self.config["scope"],
            "scan_timestamp": None,  # 可以加入時間戳
            "configuration": {
                "host_candidates": self.config["host_candidates"],
                "port_candidates": self.config["port_candidates"]
            },
            "summary": {
                "cli_entry_count": total_cli_count,
                "total_usage_possibilities": total_possibilities,
                "note": "在核心模組樹內只找到一個 argparse CLI 入口點"
            },
            "cli_entries": cli_entries
        }
        
        return self.results
    
    def generate_top_k_examples(self, k: int = 10) -> list[dict[str, Any]]:
        """
        生成 Top-K 最常用的組合範例.
        
        Args:
            k: 要生成的範例數量
            
        Returns:
            範例列表
        """
        examples = []
        
        mode_values = ["ui", "ai", "hybrid"]
        host_candidates = self.config["host_candidates"]
        port_candidates = self.config["port_candidates"]
        
        # 1. 最簡單的組合（只用預設）
        for mode in mode_values:
            examples.append({
                "rank": len(examples) + 1,
                "category": "minimal",
                "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode}",
                "description": f"使用 {mode} 模式，所有其他參數使用預設值"
            })
            if len(examples) >= k:
                return examples[:k]
        
        # 2. 指定單一偏好端口
        for mode in mode_values[:min(2, len(mode_values))]:
            for port in port_candidates[:min(3, len(port_candidates))]:
                examples.append({
                    "rank": len(examples) + 1,
                    "category": "single-port",
                    "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --ports {port}",
                    "description": f"使用 {mode} 模式，偏好端口 {port}"
                })
                if len(examples) >= k:
                    return examples[:k]
        
        # 3. 指定多個偏好端口（有順序）
        for mode in mode_values[:1]:
            port_seq = port_candidates[:min(3, len(port_candidates))]
            examples.append({
                "rank": len(examples) + 1,
                "category": "multi-port",
                "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --ports {' '.join(map(str, port_seq))}",
                "description": f"使用 {mode} 模式，依序嘗試端口 {port_seq}"
            })
            if len(examples) >= k:
                return examples[:k]
        
        # 4. 指定主機
        if len(host_candidates) > 1:
            for mode in mode_values[:1]:
                for host in host_candidates[1:2]:
                    examples.append({
                        "rank": len(examples) + 1,
                        "category": "custom-host",
                        "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host}",
                        "description": f"使用 {mode} 模式，綁定到 {host}"
                    })
                    if len(examples) >= k:
                        return examples[:k]
        
        return examples[:k]
    
    def save_report(self, output_path: Path) -> None:
        """
        儲存分析報告.
        
        Args:
            output_path: 輸出檔案路徑
        """
        if not self.results:
            self.analyze_all()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 報告已儲存至: {output_path}")
    
    def print_summary(self) -> None:
        """印出摘要資訊."""
        if not self.results:
            self.analyze_all()
        
        summary = self.results["summary"]
        
        print("\n" + "=" * 70)
        print("  核心模組 CLI 使用可能性統計")
        print("=" * 70)
        print(f"範圍: {self.results['scope']}")
        print(f"CLI 入口點數量: {summary['cli_entry_count']}")
        print(f"總使用可能性: {summary['total_usage_possibilities']:,}")
        print("-" * 70)
        
        for entry in self.results["cli_entries"]:
            print(f"\n入口點: {entry['cli_entry']}")
            counts = entry["counts"]
            print(f"  下界（最小）: {counts['lower_bound']}")
            print(f"  有指定 --ports: {counts['with_ports_specified']:,}")
            print(f"  無指定 --ports: {counts['without_ports_specified']}")
            print(f"  總計: {counts['total']:,}")
            
            print("\n  依模式分組:")
            for mode, mode_counts in counts["by_mode"].items():
                print(f"    {mode}: {mode_counts['total']:,} 種可能")
        
        print("\n" + "=" * 70)
        print(f"註: {summary['note']}")
        print("=" * 70 + "\n")


def main():
    """主程式."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='計算核心模組 CLI 使用可能性數量'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='設定檔路徑（JSON 格式，包含 host_candidates 和 port_candidates）'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / '_out' / 'core_cli_possibilities.json',
        help='輸出報告路徑（預設: _out/core_cli_possibilities.json）'
    )
    parser.add_argument(
        '--examples',
        type=int,
        default=10,
        help='生成 Top-K 範例數量（預設: 10）'
    )
    parser.add_argument(
        '--examples-output',
        type=Path,
        help='範例輸出路徑（預設: 與主報告同目錄）'
    )
    
    args = parser.parse_args()
    
    # 建立分析器
    analyzer = CLIAnalyzer(config_path=args.config)
    
    # 執行分析
    analyzer.analyze_all()
    
    # 印出摘要
    analyzer.print_summary()
    
    # 確保輸出目錄存在
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 儲存主報告
    analyzer.save_report(args.output)
    
    # 生成並儲存範例
    if args.examples > 0:
        examples = analyzer.generate_top_k_examples(k=args.examples)
        
        examples_path = args.examples_output or (
            args.output.parent / f"{args.output.stem}_examples.json"
        )
        
        with open(examples_path, 'w', encoding='utf-8') as f:
            json.dump({
                "count": len(examples),
                "examples": examples
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 範例已儲存至: {examples_path}")
        
        # 印出前幾個範例
        print(f"\n前 {min(5, len(examples))} 個常用範例:")
        for ex in examples[:5]:
            print(f"\n{ex['rank']}. [{ex['category']}]")
            print(f"   {ex['command']}")
            print(f"   → {ex['description']}")


if __name__ == "__main__":
    main()
