"""
快速訓練腳本 - 直接訓練到 900 個組合
修復版本，確保生成並訓練所有組合
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程式 - 訓練到至少 900 個組合."""
    # 載入配置
    cli_data_file = Path("_out/core_cli_possibilities.json")
    checkpoint_dir = Path("_out/cli_training")
    checkpoint_file = checkpoint_dir / "training_state.json"
    
    with open(cli_data_file, encoding='utf-8') as f:
        cli_data = json.load(f)
    
    total = cli_data["summary"]["total_usage_possibilities"]
    
    # 載入當前進度
    if checkpoint_file.exists():
        with open(checkpoint_file, encoding='utf-8') as f:
            state = json.load(f)
        current = state["trained_count"]
    else:
        current = 0
    
    target = 900
    remaining = max(0, target - current)
    
    logger.info("=" * 60)
    logger.info("訓練目標:")
    logger.info("  當前進度: %d/%d (%.1f%%)", current, total, current/total*100)
    logger.info("  目標: %d", target)
    logger.info("  需要訓練: %d 個組合", remaining)
    logger.info("=" * 60)
    
    if current >= target:
        logger.info("\n✓ 已達到目標！無需繼續訓練。")
        return
    
    # 計算需要的批次數
    batch_size = 10
    batches_needed = (remaining + batch_size - 1) // batch_size
    
    logger.info("\n開始訓練 %d 批次...\n", batches_needed)
    
    # 導入並運行原始訓練器
    from train_cli_with_memory import CLITrainingOrchestrator
    
    trainer = CLITrainingOrchestrator(
        possibilities_file=cli_data_file,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size
    )
    
    #修改生成邏輯：生成所有組合而非只生成已訓練的
    all_combinations = trainer.generate_cli_combinations()
    logger.info("✓ 生成了 %d 個 CLI 組合", len(all_combinations))
    
    # 只訓練未訓練的部分
    trained_commands = set()
    for history in state.get("training_history", []):
        if "commands" in history:
            for cmd in history["commands"]:
                trained_commands.add(cmd)
    
    untrained = [c for c in all_combinations if c["command"] not in trained_commands]
    
    logger.info("✓ 篩選出 %d 個未訓練組合", len(untrained))
    
    # 創建批次
    batches = trainer.create_training_batches(untrained[:remaining])
    
    logger.info("✓ 創建 %d 個訓練批次\n", len(batches))
    
    # 訓練
    for batch_idx, batch in enumerate(batches):
        logger.info("批次 %d/%d 訓練...", batch_idx + 1, len(batches))
        batch_result = trainer.train_batch(batch, trainer.training_state["current_batch"] + batch_idx)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info("  ✓ 已完成 %d/%d 批次", batch_idx + 1, len(batches))
            trainer._save_checkpoint()
    
    # 最終保存
    trainer._save_checkpoint()
    
    # 最終報告
    final_count = trainer.training_state["trained_count"]
    logger.info("\n" + "=" * 60)
    logger.info("訓練完成！")
    logger.info("  最終進度: %d/%d (%.1f%%)", final_count, total, final_count/total*100)
    logger.info("  已達目標: %s", "✓" if final_count >= target else "✗")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
