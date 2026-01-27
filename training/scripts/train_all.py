#!/usr/bin/env python
"""
AIVA 訓練快速啟動腳本

一鍵完成所有訓練步驟：
1. 構建安全詞彙表
2. 生成蒸餾數據集
3. 訓練 Student Model
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """執行命令並處理錯誤"""
    logger.info(f"{'='*60}")
    logger.info(f"▶ {description}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(result.stdout)
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 執行失敗: {description}")
        logger.error(f"錯誤輸出:\n{e.stderr}")
        return False


def main():
    """主流程"""
    scripts_dir = Path(__file__).parent
    
    logger.info("\n")
    logger.info("🎓 AIVA 知識蒸餾訓練 - 完整流程")
    logger.info("="*60)
    logger.info("此腳本將依序執行：")
    logger.info("  1. 構建安全詞彙表")
    logger.info("  2. 生成蒸餾訓練數據集")
    logger.info("  3. 訓練 AIVA 5M Student Model")
    logger.info("="*60)
    
    input("按 Enter 鍵開始訓練...")
    
    # 步驟 1: 構建詞彙表
    step1_cmd = [
        sys.executable,
        str(scripts_dir / "build_security_vocabulary.py"),
        "--docs-dir", r"C:\Users\User\Downloads\新增資料夾",
        "--output-dir", str(scripts_dir.parent / "data" / "security_vocabulary")
    ]
    
    if not run_command(step1_cmd, "步驟 1/3: 構建安全詞彙表"):
        logger.error("訓練中止")
        return 1
    
    # 步驟 2: 生成數據集
    step2_cmd = [
        sys.executable,
        str(scripts_dir / "generate_distillation_dataset.py"),
        "--vocab-dir", str(scripts_dir.parent / "data" / "security_vocabulary"),
        "--output-dir", str(scripts_dir.parent / "data" / "distillation_dataset"),
        "--samples-per-type", "100"
    ]
    
    if not run_command(step2_cmd, "步驟 2/3: 生成蒸餾數據集"):
        logger.error("訓練中止")
        return 1
    
    # 步驟 3: 訓練模型
    step3_cmd = [
        sys.executable,
        str(scripts_dir / "train_student_model.py")
    ]
    
    if not run_command(step3_cmd, "步驟 3/3: 訓練 Student Model"):
        logger.error("訓練中止")
        return 1
    
    # 完成
    logger.info("\n")
    logger.info("="*60)
    logger.info("✅ 訓練完成！")
    logger.info("="*60)
    logger.info("📁 輸出位置:")
    logger.info(f"  - 詞彙表: {scripts_dir.parent / 'data' / 'security_vocabulary'}")
    logger.info(f"  - 數據集: {scripts_dir.parent / 'data' / 'distillation_dataset'}")
    logger.info(f"  - 模型: {scripts_dir.parent / 'models'}")
    logger.info("="*60)
    logger.info("\n下一步:")
    logger.info("  1. 查看訓練報告: cat ../data/distillation_dataset/dataset_report.md")
    logger.info("  2. 部署模型: cp ../models/best_model.pt ../../services/.../weights/")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
