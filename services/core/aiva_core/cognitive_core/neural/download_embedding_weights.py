"""
下載並本地化 AIVA Embedding 模型權重

這個腳本會：
1. 從 Hugging Face Hub 下載 all-MiniLM-L6-v2 的完整權重
2. 保存到本地目錄供離線使用
3. 驗證下載的完整性
"""

import logging
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_embedding_weights(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    save_dir: Path | str = "./weights/all-MiniLM-L6-v2"
) -> None:
    """
    下載並保存 Embedding 模型權重到本地
    
    Args:
        model_name: Hugging Face 模型名稱
        save_dir: 本地保存目錄
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📥 開始下載模型: {model_name}")
    logger.info(f"💾 保存位置: {save_dir.absolute()}")
    
    try:
        # 下載 Tokenizer
        logger.info("⏳ 下載 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        logger.info("✅ Tokenizer 下載完成")
        
        # 下載 Model
        logger.info("⏳ 下載模型權重 (~90MB)...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        logger.info("✅ 模型權重下載完成")
        
        # 驗證
        logger.info("🔍 驗證文件完整性...")
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.txt"
        ]
        
        missing_files = [f for f in required_files if not (save_dir / f).exists()]
        
        if missing_files:
            logger.error(f"❌ 缺少文件: {missing_files}")
            return
        
        # 統計信息
        total_size = sum(f.stat().st_size for f in save_dir.glob("*") if f.is_file())
        file_count = len(list(save_dir.glob("*")))
        
        logger.info("=" * 60)
        logger.info("✅ 模型下載完成！")
        logger.info(f"📁 文件數量: {file_count}")
        logger.info(f"💾 總大小: {total_size / 1024 / 1024:.2f} MB")
        logger.info(f"📍 位置: {save_dir.absolute()}")
        logger.info("=" * 60)
        logger.info("\n使用方式：")
        logger.info(f'AIVAEmbedding(model_name_or_path="{save_dir.absolute()}")')
        
    except Exception as e:
        logger.error(f"❌ 下載失敗: {str(e)}")
        raise


def verify_local_weights(weights_dir: Path | str) -> bool:
    """
    驗證本地權重的完整性
    
    Args:
        weights_dir: 權重目錄路徑
    
    Returns:
        True if valid, False otherwise
    """
    weights_dir = Path(weights_dir)
    
    if not weights_dir.exists():
        logger.error(f"❌ 目錄不存在: {weights_dir}")
        return False
    
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    missing = [f for f in required_files if not (weights_dir / f).exists()]
    
    if missing:
        logger.error(f"❌ 缺少必要文件: {missing}")
        return False
    
    logger.info(f"✅ 權重文件完整: {weights_dir}")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下載 AIVA Embedding 模型權重")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face 模型名稱"
    )
    parser.add_argument(
        "--save-dir",
        default="./weights/all-MiniLM-L6-v2",
        help="本地保存目錄"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="僅驗證現有權重，不下載"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_local_weights(args.save_dir)
    else:
        download_embedding_weights(args.model, args.save_dir)
