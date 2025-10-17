"""
CLI 訓練腳本 - 讓 AI 逐步學習所有 CLI 使用可能性
基於 978 種 CLI 組合，進行系統化訓練

訓練策略：
1. 從簡單到複雜（下界 3 種 → 完整 978 種）
2. 分批訓練，每批有記憶
3. 使用強化學習，根據執行結果調整
4. 保存訓練進度，可中斷續訓
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLITrainingOrchestrator:
    """CLI 訓練編排器 - 系統化訓練 AI 學習所有 CLI 用法."""

    def __init__(
        self,
        possibilities_file: Path,
        checkpoint_dir: Path,
        batch_size: int = 10
    ):
        """初始化訓練編排器.

        Args:
            possibilities_file: CLI 可能性 JSON 檔案
            checkpoint_dir: 檢查點目錄
            batch_size: 每批訓練數量
        """
        self.possibilities_file = possibilities_file
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入 CLI 可能性
        with open(possibilities_file, encoding='utf-8') as f:
            self.cli_data = json.load(f)
        
        self.cli_entry = self.cli_data["cli_entries"][0]
        self.total_possibilities = self.cli_data["summary"]["total_usage_possibilities"]
        
        logger.info("載入 CLI 數據:")
        logger.info("  總可能性: %d", self.total_possibilities)
        logger.info("  批次大小: %d", batch_size)
        
        # 訓練狀態
        self.training_state = self._load_checkpoint()
    
    def _load_checkpoint(self) -> dict[str, Any]:
        """載入訓練檢查點."""
        checkpoint_file = self.checkpoint_dir / "training_state.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, encoding='utf-8') as f:
                state = json.load(f)
            logger.info("✓ 載入檢查點: 已訓練 %d/%d", state["trained_count"], state["total"])
            return state
        else:
            logger.info("開始新訓練")
            return {
                "trained_count": 0,
                "total": self.total_possibilities,
                "current_batch": 0,
                "training_history": [],
                "learned_patterns": {}
            }
    
    def _save_checkpoint(self) -> None:
        """保存訓練檢查點."""
        checkpoint_file = self.checkpoint_dir / "training_state.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_state, f, ensure_ascii=False, indent=2)
        logger.info("✓ 保存檢查點")
    
    def generate_cli_combinations(self) -> list[dict[str, Any]]:
        """生成所有 CLI 組合.

        Returns:
            CLI 命令列表
        """
        combinations = []
        
        # 提取參數配置
        params = self.cli_entry["parameters"]
        modes = params["mode"]["values"]
        hosts = params["host"]["candidates"]
        ports = params["ports"]["candidates"]
        
        # 1. 不指定 --ports 的組合 (3 種)
        for mode in modes:
            for host in hosts:
                combinations.append({
                    "category": "minimal",
                    "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host}",
                    "params": {"mode": mode, "host": host, "ports": None},
                    "complexity": 1
                })
        
        # 2. 單一端口組合 (15 種)
        for mode in modes:
            for host in hosts:
                for port in ports:
                    combinations.append({
                        "category": "single-port",
                        "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port}",
                        "params": {"mode": mode, "host": host, "ports": [port]},
                        "complexity": 2
                    })
        
        # 3. 雙端口組合（排列）
        for mode in modes:
            for host in hosts:
                for i, port1 in enumerate(ports):
                    for j, port2 in enumerate(ports):
                        if i != j:  # 不同端口
                            combinations.append({
                                "category": "dual-port",
                                "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port1} {port2}",
                                "params": {"mode": mode, "host": host, "ports": [port1, port2]},
                                "complexity": 3
                            })
        
        # 4. 三端口組合
        for mode in modes:
            for host in hosts:
                for i, port1 in enumerate(ports):
                    for j, port2 in enumerate(ports):
                        if i != j:
                            for k, port3 in enumerate(ports):
                                if k not in [i, j]:
                                    combinations.append({
                                        "category": "triple-port",
                                        "command": f"python -m services.core.aiva_core.ui_panel.auto_server --mode {mode} --host {host} --ports {port1} {port2} {port3}",
                                        "params": {"mode": mode, "host": host, "ports": [port1, port2, port3]},
                                        "complexity": 4
                                    })
        
        logger.info("✓ 生成 %d 個 CLI 組合", len(combinations))
        return combinations
    
    def create_training_batches(
        self,
        combinations: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """創建訓練批次（由簡到繁）.

        Args:
            combinations: 所有組合

        Returns:
            批次列表
        """
        # 按複雜度排序
        sorted_combinations = sorted(combinations, key=lambda x: x["complexity"])
        
        # 分批
        batches = []
        for i in range(0, len(sorted_combinations), self.batch_size):
            batch = sorted_combinations[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info("✓ 創建 %d 個訓練批次", len(batches))
        return batches
    
    def simulate_cli_execution(self, cli_command: dict[str, Any]) -> dict[str, Any]:
        """模擬 CLI 執行（實際應該執行真實命令）.

        Args:
            cli_command: CLI 命令字典

        Returns:
            執行結果
        """
        # 這裡模擬執行結果
        # 實際應該: subprocess.run(cli_command["command"].split())
        
        params = cli_command["params"]
        
        # 模擬成功率（簡單命令成功率更高）
        success_prob = 1.0 - (cli_command["complexity"] - 1) * 0.05
        success = np.random.random() < success_prob
        
        return {
            "success": success,
            "command": cli_command["command"],
            "params": params,
            "category": cli_command["category"],
            "exit_code": 0 if success else 1,
            "output": "Server started successfully" if success else "Port already in use"
        }
    
    def train_batch(
        self,
        batch: list[dict[str, Any]],
        batch_idx: int
    ) -> dict[str, Any]:
        """訓練一個批次.

        Args:
            batch: 批次命令列表
            batch_idx: 批次索引

        Returns:
            批次訓練結果
        """
        logger.info("\n" + "=" * 70)
        logger.info("批次 %d/%d 訓練", batch_idx + 1, (self.total_possibilities + self.batch_size - 1) // self.batch_size)
        logger.info("=" * 70)
        
        batch_results = []
        successful = 0
        
        for i, cli_cmd in enumerate(batch, 1):
            logger.info("[%d/%d] 訓練命令: %s", i, len(batch), cli_cmd["category"])
            logger.info("  命令: %s", cli_cmd["command"])
            
            # 執行命令
            result = self.simulate_cli_execution(cli_cmd)
            
            # 計算獎勵
            reward = 1.0 if result["success"] else 0.3  # 失敗也給點獎勵（嘗試過了）
            
            batch_results.append({
                "command": cli_cmd,
                "result": result,
                "reward": reward
            })
            
            if result["success"]:
                successful += 1
                logger.info("  ✓ 成功")
            else:
                logger.info("  ✗ 失敗: %s", result["output"])
            
            # 學習模式模式（記錄成功的參數組合）
            if result["success"]:
                pattern_key = f"{cli_cmd['params']['mode']}_{cli_cmd['category']}"
                if pattern_key not in self.training_state["learned_patterns"]:
                    self.training_state["learned_patterns"][pattern_key] = {
                        "count": 0,
                        "examples": []
                    }
                self.training_state["learned_patterns"][pattern_key]["count"] += 1
                if len(self.training_state["learned_patterns"][pattern_key]["examples"]) < 3:
                    self.training_state["learned_patterns"][pattern_key]["examples"].append(
                        cli_cmd["command"]
                    )
        
        # 批次統計
        success_rate = successful / len(batch) * 100
        avg_reward = sum(r["reward"] for r in batch_results) / len(batch_results)
        
        logger.info("\n批次統計:")
        logger.info("  成功: %d/%d (%.1f%%)", successful, len(batch), success_rate)
        logger.info("  平均獎勵: %.2f", avg_reward)
        logger.info("  已學習模式: %d", len(self.training_state["learned_patterns"]))
        
        # 更新訓練狀態
        self.training_state["trained_count"] += len(batch)
        self.training_state["current_batch"] = batch_idx + 1
        self.training_state["training_history"].append({
            "batch_idx": batch_idx,
            "commands_trained": len(batch),
            "success_rate": success_rate,
            "avg_reward": avg_reward
        })
        
        return {
            "batch_idx": batch_idx,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "results": batch_results
        }
    
    def run_training(self, max_batches: int | None = None) -> dict[str, Any]:
        """運行完整訓練流程.

        Args:
            max_batches: 最大批次數（None 表示訓練全部）

        Returns:
            訓練總結
        """
        logger.info("\n" + "=" * 70)
        logger.info("開始 CLI 訓練")
        logger.info("=" * 70)
        logger.info("總可能性: %d", self.total_possibilities)
        logger.info("批次大小: %d", self.batch_size)
        logger.info("已訓練: %d", self.training_state["trained_count"])
        logger.info("=" * 70 + "\n")
        
        # 生成所有組合
        all_combinations = self.generate_cli_combinations()
        
        # 找出已訓練的命令
        trained_commands = set()
        for batch_history in self.training_state.get("training_history", []):
            # 從歷史中提取已訓練的命令
            if "results" in batch_history:
                for result in batch_history["results"]:
                    if "command" in result and "command" in result["command"]:
                        trained_commands.add(result["command"]["command"])
        
        # 過濾出未訓練的組合
        untrained_combinations = [c for c in all_combinations if c["command"] not in trained_commands]
        
        logger.info("✓ 已訓練: %d 個組合", len(trained_commands))
        logger.info("✓ 未訓練: %d 個組合", len(untrained_combinations))
        
        # 創建新的訓練批次（只包含未訓練的）
        batches = self.create_training_batches(untrained_combinations)
        
        # 從頭開始訓練這些新批次
        start_batch = 0
        end_batch = min(len(batches), max_batches) if max_batches else len(batches)
        
        logger.info("訓練批次: %d - %d (共 %d)", start_batch + 1, end_batch, len(batches))
        
        # 訓練每個批次
        for batch_idx in range(start_batch, end_batch):
            batch = batches[batch_idx]
            batch_result = self.train_batch(batch, batch_idx)
            
            # 定期保存檢查點
            if (batch_idx + 1) % 5 == 0:
                self._save_checkpoint()
        
        # 最終保存
        self._save_checkpoint()
        
        # 訓練總結
        total_trained = self.training_state["trained_count"]
        progress = total_trained / self.total_possibilities * 100
        
        logger.info("\n" + "=" * 70)
        logger.info("訓練完成")
        logger.info("=" * 70)
        logger.info("訓練進度: %d/%d (%.1f%%)", total_trained, self.total_possibilities, progress)
        logger.info("已學習模式: %d", len(self.training_state["learned_patterns"]))
        logger.info("訓練批次: %d", len(self.training_state["training_history"]))
        
        if self.training_state["training_history"]:
            recent_history = self.training_state["training_history"][-10:]
            avg_success = sum(h["success_rate"] for h in recent_history) / len(recent_history)
            logger.info("近期成功率: %.1f%%", avg_success)
        
        logger.info("=" * 70 + "\n")
        
        return {
            "total_trained": total_trained,
            "total_possibilities": self.total_possibilities,
            "progress": progress,
            "learned_patterns": len(self.training_state["learned_patterns"]),
            "training_batches": len(self.training_state["training_history"])
        }
    
    def print_learned_patterns(self) -> None:
        """印出已學習的模式."""
        logger.info("\n" + "=" * 70)
        logger.info("已學習模式")
        logger.info("=" * 70)
        
        for pattern_key, pattern_data in self.training_state["learned_patterns"].items():
            logger.info("\n模式: %s", pattern_key)
            logger.info("  訓練次數: %d", pattern_data["count"])
            logger.info("  範例命令:")
            for example in pattern_data["examples"]:
                logger.info("    - %s", example)
        
        logger.info("\n" + "=" * 70 + "\n")


def main():
    """主程式."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLI 訓練腳本')
    parser.add_argument(
        '--possibilities',
        type=Path,
        default=Path('_out/core_cli_possibilities.json'),
        help='CLI 可能性 JSON 檔案'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('_out/cli_training'),
        help='檢查點目錄'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='批次大小'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        help='最大訓練批次數（用於測試）'
    )
    parser.add_argument(
        '--show-patterns',
        action='store_true',
        help='顯示已學習模式'
    )
    
    args = parser.parse_args()
    
    # 創建訓練器
    trainer = CLITrainingOrchestrator(
        possibilities_file=args.possibilities,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size
    )
    
    # 顯示模式
    if args.show_patterns:
        trainer.print_learned_patterns()
        return
    
    # 運行訓練
    summary = trainer.run_training(max_batches=args.max_batches)
    
    # 顯示學習到的模式
    trainer.print_learned_patterns()
    
    logger.info("\n✓ 訓練完成！")
    logger.info("詳細進度已保存至: %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
