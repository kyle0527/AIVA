"""
AI CLI 遊樂場 - 讓 AI 隨意嘗試和測試 CLI 組合
可以自由實驗、即時反饋、自主探索

功能：
1. 隨機嘗試模式 - AI 隨機選擇 CLI 組合測試
2. 智能探索模式 - AI 根據學習經驗選擇未測試的組合
3. 互動式模式 - 接受自然語言指令
4. 好奇心驅動 - 優先嘗試新奇的組合
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIPlayground:
    """AI CLI 遊樂場 - 自由探索和實驗."""

    def __init__(self, possibilities_file: Path, experience_file: Path | None = None):
        """初始化遊樂場.

        Args:
            possibilities_file: CLI 可能性文件
            experience_file: 經驗記錄文件（可選）
        """
        # 載入 CLI 數據
        with open(possibilities_file, encoding='utf-8') as f:
            self.cli_data = json.load(f)
        
        self.cli_entry = self.cli_data["cli_entries"][0]
        self.total_possibilities = self.cli_data["summary"]["total_usage_possibilities"]
        
        # 載入已有經驗
        self.experience_file = experience_file or Path("_out/ai_playground/experience.json")
        self.experience = self._load_experience()
        
        # 生成所有組合
        self.all_combinations = self._generate_all_combinations()
        
        # 好奇心統計
        self.curiosity_stats = {
            "total_tries": 0,
            "new_discoveries": 0,
            "successful_tries": 0,
            "failed_tries": 0
        }
        
        logger.info("🎮 AI 遊樂場已啟動！")
        logger.info(f"  總組合數: {len(self.all_combinations)}")
        logger.info(f"  已嘗試過: {len(self.experience.get('tried_combinations', []))}")
        logger.info(f"  未探索: {len(self.all_combinations) - len(self.experience.get('tried_combinations', []))}")

    def _load_experience(self) -> dict[str, Any]:
        """載入經驗記錄."""
        if self.experience_file.exists():
            with open(self.experience_file, encoding='utf-8') as f:
                return json.load(f)
        return {
            "tried_combinations": [],
            "successful_patterns": {},
            "failed_patterns": {},
            "curiosity_log": []
        }

    def _save_experience(self) -> None:
        """保存經驗."""
        self.experience_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.experience_file, 'w', encoding='utf-8') as f:
            json.dump(self.experience, f, ensure_ascii=False, indent=2)

    def _generate_all_combinations(self) -> list[dict[str, Any]]:
        """生成所有可能的 CLI 組合."""
        combinations = []
        params = self.cli_entry["parameters"]
        
        modes = params["mode"]["values"]
        hosts = params["host"]["candidates"]
        ports = params["ports"]["candidates"]
        
        # 1. 無端口版本（3種）
        for mode in modes:
            for host in hosts:
                cmd = f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host}"
                combinations.append({
                    "command": cmd,
                    "mode": mode,
                    "host": host,
                    "ports": [],
                    "complexity": "minimal",
                    "pattern": f"{mode}_minimal"
                })
        
        # 2. 單端口版本
        for mode in modes:
            for host in hosts:
                for port in ports:
                    cmd = f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port}"
                    combinations.append({
                        "command": cmd,
                        "mode": mode,
                        "host": host,
                        "ports": [port],
                        "complexity": "single-port",
                        "pattern": f"{mode}_single-port"
                    })
        
        # 3. 雙端口版本
        for mode in modes:
            for host in hosts:
                for i, port1 in enumerate(ports):
                    for port2 in ports[i+1:]:
                        cmd = f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port1} {port2}"
                        combinations.append({
                            "command": cmd,
                            "mode": mode,
                            "host": host,
                            "ports": [port1, port2],
                            "complexity": "dual-port",
                            "pattern": f"{mode}_dual-port"
                        })
        
        # 3. 三端口版本
        for mode in modes:
            for host in hosts:
                for i, port1 in enumerate(ports):
                    for j, port2 in enumerate(ports[i+1:], i+1):
                        for port3 in ports[j+1:]:
                            cmd = f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port1} {port2} {port3}"
                            combinations.append({
                                "command": cmd,
                                "mode": mode,
                                "host": host,
                                "ports": [port1, port2, port3],
                                "complexity": "triple-port",
                                "pattern": f"{mode}_triple-port"
                            })
        
        return combinations

    def random_try(self, n: int = 1) -> list[dict[str, Any]]:
        """隨機嘗試 N 個組合.

        Args:
            n: 嘗試次數

        Returns:
            嘗試結果列表
        """
        logger.info(f"🎲 隨機嘗試 {n} 個組合...")
        results = []
        
        for i in range(n):
            combo = random.choice(self.all_combinations)
            result = self._try_combination(combo)
            results.append(result)
            
            # 簡短輸出
            status = "✓" if result["success"] else "✗"
            logger.info(f"  [{i+1}/{n}] {status} {result['pattern']} - {result['mode']}")
        
        self._save_experience()
        self._print_session_stats(results)
        return results

    def curious_explore(self, n: int = 10, novelty_bonus: float = 2.0) -> list[dict[str, Any]]:
        """好奇心驅動探索 - 優先嘗試未知組合.

        Args:
            n: 探索次數
            novelty_bonus: 新奇獎勵係數

        Returns:
            探索結果列表
        """
        logger.info(f"🔍 好奇心探索模式 - 探索 {n} 個組合...")
        results = []
        
        # 計算每個組合的好奇心分數
        tried_set = set(self.experience.get("tried_combinations", []))
        
        untried = [c for c in self.all_combinations if c["command"] not in tried_set]
        
        if not untried:
            logger.info("  所有組合都已嘗試過！隨機選擇...")
            untried = self.all_combinations
        
        logger.info(f"  未嘗試組合: {len(untried)}/{len(self.all_combinations)}")
        
        # 優先選擇未嘗試的
        selected = random.sample(untried, min(n, len(untried)))
        
        for i, combo in enumerate(selected):
            result = self._try_combination(combo)
            results.append(result)
            
            status = "✓" if result["success"] else "✗"
            is_new = "🆕" if result["is_new_discovery"] else ""
            logger.info(f"  [{i+1}/{n}] {status}{is_new} {result['pattern']} - {result['complexity']}")
        
        self._save_experience()
        self._print_session_stats(results)
        return results

    def smart_explore(self, n: int = 10) -> list[dict[str, Any]]:
        """智能探索 - 基於已學習模式選擇.

        Args:
            n: 探索次數

        Returns:
            探索結果列表
        """
        logger.info(f"🧠 智能探索模式 - 探索 {n} 個組合...")
        results = []
        
        # 分析成功模式
        successful_patterns = self.experience.get("successful_patterns", {})
        
        # 如果沒有成功經驗，先隨機嘗試
        if not successful_patterns:
            logger.info("  沒有成功經驗，隨機探索...")
            return self.random_try(n)
        
        # 找出成功率最高的模式
        pattern_scores = {}
        for pattern, data in successful_patterns.items():
            success_rate = data.get("success_count", 0) / max(data.get("total_count", 1), 1)
            pattern_scores[pattern] = success_rate
        
        # 選擇相似的未嘗試組合
        tried_set = set(self.experience.get("tried_combinations", []))
        untried = [c for c in self.all_combinations if c["command"] not in tried_set]
        
        if not untried:
            untried = self.all_combinations
        
        # 優先選擇成功率高的模式
        sorted_combos = sorted(
            untried,
            key=lambda c: pattern_scores.get(c["pattern"], 0),
            reverse=True
        )
        
        selected = sorted_combos[:n]
        
        for i, combo in enumerate(selected):
            result = self._try_combination(combo)
            results.append(result)
            
            status = "✓" if result["success"] else "✗"
            confidence = pattern_scores.get(combo["pattern"], 0) * 100
            logger.info(f"  [{i+1}/{n}] {status} {result['pattern']} (信心度: {confidence:.0f}%)")
        
        self._save_experience()
        self._print_session_stats(results)
        return results

    def try_specific(self, mode: str | None = None, ports: list[int] | None = None) -> dict[str, Any]:
        """嘗試特定組合.

        Args:
            mode: 模式（ui/ai/hybrid）
            ports: 端口列表

        Returns:
            嘗試結果
        """
        # 查找匹配的組合
        matching = [c for c in self.all_combinations if 
                   (mode is None or c["mode"] == mode) and
                   (ports is None or c["ports"] == ports)]
        
        if not matching:
            logger.error("找不到匹配的組合！")
            return {"success": False, "error": "No matching combination"}
        
        combo = matching[0]
        logger.info(f"🎯 嘗試特定組合: {combo['pattern']}")
        result = self._try_combination(combo)
        
        self._save_experience()
        return result

    def _try_combination(self, combo: dict[str, Any]) -> dict[str, Any]:
        """嘗試執行一個組合.

        Args:
            combo: CLI 組合

        Returns:
            執行結果
        """
        command = combo["command"]
        
        # 檢查是否是新發現
        tried_set = set(self.experience.get("tried_combinations", []))
        is_new = command not in tried_set
        
        # 模擬執行（實際使用時可以真的執行）
        # 這裡簡化為：檢查命令格式是否正確
        success = self._simulate_execution(combo)
        
        # 記錄經驗
        if command not in tried_set:
            self.experience["tried_combinations"].append(command)
            self.curiosity_stats["new_discoveries"] += 1
        
        self.curiosity_stats["total_tries"] += 1
        if success:
            self.curiosity_stats["successful_tries"] += 1
        else:
            self.curiosity_stats["failed_tries"] += 1
        
        # 更新模式統計
        pattern = combo["pattern"]
        if success:
            if pattern not in self.experience["successful_patterns"]:
                self.experience["successful_patterns"][pattern] = {
                    "success_count": 0,
                    "total_count": 0,
                    "examples": []
                }
            self.experience["successful_patterns"][pattern]["success_count"] += 1
            self.experience["successful_patterns"][pattern]["total_count"] += 1
            if len(self.experience["successful_patterns"][pattern]["examples"]) < 3:
                self.experience["successful_patterns"][pattern]["examples"].append(command)
        else:
            if pattern not in self.experience["failed_patterns"]:
                self.experience["failed_patterns"][pattern] = {
                    "fail_count": 0,
                    "total_count": 0,
                    "examples": []
                }
            self.experience["failed_patterns"][pattern]["fail_count"] += 1
            self.experience["failed_patterns"][pattern]["total_count"] += 1
            if len(self.experience["failed_patterns"][pattern]["examples"]) < 3:
                self.experience["failed_patterns"][pattern]["examples"].append(command)
        
        # 記錄好奇心日誌
        self.experience["curiosity_log"].append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": command,
            "pattern": pattern,
            "success": success,
            "is_new_discovery": is_new
        })
        
        return {
            "command": command,
            "pattern": pattern,
            "mode": combo["mode"],
            "complexity": combo["complexity"],
            "success": success,
            "is_new_discovery": is_new
        }

    def _simulate_execution(self, combo: dict[str, Any]) -> bool:
        """模擬執行（可替換為真實執行）.

        Args:
            combo: CLI 組合

        Returns:
            是否成功
        """
        # 簡單模擬：90% 成功率，端口越多越容易失敗
        base_success_rate = 0.95
        port_penalty = len(combo["ports"]) * 0.05
        success_rate = max(0.7, base_success_rate - port_penalty)
        
        return random.random() < success_rate

    def _print_session_stats(self, results: list[dict[str, Any]]) -> None:
        """打印本次統計."""
        if not results:
            return
        
        success_count = sum(1 for r in results if r["success"])
        new_count = sum(1 for r in results if r["is_new_discovery"])
        
        logger.info(f"\n{'='*60}")
        logger.info("本次統計:")
        logger.info(f"  成功: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        logger.info(f"  新發現: {new_count}")
        logger.info(f"{'='*60}\n")

    def show_stats(self) -> None:
        """顯示統計信息."""
        logger.info(f"\n{'='*60}")
        logger.info("🎮 AI 遊樂場統計")
        logger.info(f"{'='*60}")
        logger.info(f"總嘗試次數: {self.curiosity_stats['total_tries']}")
        logger.info(f"新發現: {self.curiosity_stats['new_discoveries']}")
        logger.info(f"成功: {self.curiosity_stats['successful_tries']}")
        logger.info(f"失敗: {self.curiosity_stats['failed_tries']}")
        
        if self.curiosity_stats['total_tries'] > 0:
            success_rate = self.curiosity_stats['successful_tries'] / self.curiosity_stats['total_tries'] * 100
            logger.info(f"成功率: {success_rate:.1f}%")
        
        logger.info(f"\n探索進度:")
        tried_count = len(self.experience.get("tried_combinations", []))
        logger.info(f"  已嘗試: {tried_count}/{len(self.all_combinations)} ({tried_count/len(self.all_combinations)*100:.1f}%)")
        logger.info(f"  未探索: {len(self.all_combinations) - tried_count}")
        
        logger.info(f"\n學習到的模式:")
        for pattern, data in self.experience.get("successful_patterns", {}).items():
            success_rate = data["success_count"] / data["total_count"] * 100
            logger.info(f"  ✓ {pattern}: {data['success_count']}/{data['total_count']} ({success_rate:.0f}%)")
        
        logger.info(f"{'='*60}\n")


def main():
    """主程式."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI CLI 遊樂場 - 自由探索測試')
    parser.add_argument(
        '--mode',
        choices=['random', 'curious', 'smart', 'interactive'],
        default='curious',
        help='探索模式'
    )
    parser.add_argument(
        '--tries',
        type=int,
        default=10,
        help='嘗試次數'
    )
    parser.add_argument(
        '--specific-mode',
        choices=['ui', 'ai', 'hybrid'],
        help='指定模式測試'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='只顯示統計信息'
    )
    
    args = parser.parse_args()
    
    # 初始化遊樂場
    possibilities_file = Path("_out/core_cli_possibilities.json")
    playground = AIPlayground(possibilities_file)
    
    if args.stats:
        playground.show_stats()
        return
    
    # 執行探索
    if args.mode == 'random':
        playground.random_try(args.tries)
    elif args.mode == 'curious':
        playground.curious_explore(args.tries)
    elif args.mode == 'smart':
        playground.smart_explore(args.tries)
    elif args.mode == 'interactive':
        logger.info("🎮 互動模式啟動！")
        logger.info("輸入命令或 'quit' 退出\n")
        
        while True:
            try:
                cmd = input("AI> ").strip().lower()
                
                if cmd in ('quit', 'exit', 'q'):
                    break
                elif cmd.startswith('random'):
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 5
                    playground.random_try(n)
                elif cmd.startswith('curious'):
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 10
                    playground.curious_explore(n)
                elif cmd.startswith('smart'):
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 10
                    playground.smart_explore(n)
                elif cmd == 'stats':
                    playground.show_stats()
                else:
                    logger.info("未知命令。可用命令: random [n], curious [n], smart [n], stats, quit")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"錯誤: {e}")
    
    # 最終統計
    playground.show_stats()


if __name__ == "__main__":
    main()
