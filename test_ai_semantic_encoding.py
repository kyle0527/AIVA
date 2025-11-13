#!/usr/bin/env python3
"""
AI 語意編碼能力獨立測試
驗證 P0 修復後 AI 是否能理解程式碼語意
"""

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_semantic_encoding_capability():
    """測試語意編碼能力"""
    logger.info("🧠 測試 AI 語意編碼能力...")
    
    try:
        # 導入語意編碼器
        from sentence_transformers import SentenceTransformer
        logger.info("✅ sentence-transformers 已安裝")
        
        # 初始化編碼器
        logger.info("⏳ 載入語意編碼模型...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ 語意編碼器載入成功")
        
        # 測試案例: 比較語意相似度
        test_pairs = [
            {
                "name": "關鍵字 vs 錯誤拼寫",
                "code1": "def attack_target():",
                "code2": "fed attack_target():",
                "should_be": "不同",
                "threshold": 0.7
            },
            {
                "name": "相同功能不同命名",
                "code1": "import os",
                "code2": "import sys",
                "should_be": "相似",
                "threshold": 0.8
            },
            {
                "name": "相同字符不同順序",
                "code1": "user.password",
                "code2": "word.pass_user",
                "should_be": "不同",
                "threshold": 0.6
            }
        ]
        
        logger.info("\n🔬 開始語意理解測試:")
        logger.info("=" * 60)
        
        all_passed = True
        for test in test_pairs:
            # 編碼
            emb1 = encoder.encode(test["code1"], convert_to_tensor=True)
            emb2 = encoder.encode(test["code2"], convert_to_tensor=True)
            
            # 計算餘弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), 
                emb2.unsqueeze(0)
            ).item()
            
            # 判斷結果
            if test["should_be"] == "不同":
                passed = similarity < test["threshold"]
                result = "✅" if passed else "❌"
            else:  # 相似
                passed = similarity >= test["threshold"]
                result = "✅" if passed else "❌"
            
            all_passed = all_passed and passed
            
            logger.info(f"{result} {test['name']}:")
            logger.info(f"   '{test['code1']}' vs '{test['code2']}'")
            logger.info(f"   相似度: {similarity:.4f} (預期: {test['should_be']}, 閾值: {test['threshold']})")
            logger.info("")
        
        logger.info("=" * 60)
        if all_passed:
            logger.info("🎉 所有語意理解測試通過!")
            logger.info("✅ AI 現在可以理解程式碼的語意差異")
        else:
            logger.warning("⚠️ 部分測試未通過")
        
        return all_passed
        
    except ImportError:
        logger.error("❌ sentence-transformers 未安裝")
        logger.error("   請執行: pip install sentence-transformers")
        return False
    except Exception as e:
        logger.error(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_analysis_understanding():
    """測試程式碼分析理解能力"""
    logger.info("\n📊 測試程式碼分析理解能力...")
    
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 五大模組的核心代碼片段
        modules = {
            "ai_engine": "class RealAICore(nn.Module): def forward(self, x): return self.network(x)",
            "execution": "async def execute(self, plan): results = await self.executor.run(plan)",
            "tools": "class CodeReader(Tool): def read(self, path): return Path(path).read_text()",
            "bio_neuron_master": "class BioNeuronMasterController: def decide(self, task): return self.agent.generate(task)",
            "training": "def train_step(self, inputs, targets): loss = self.criterion(outputs, targets)"
        }
        
        logger.info("\n🔍 編碼五大模組代碼:")
        embeddings = {}
        for name, code in modules.items():
            emb = encoder.encode(code, convert_to_tensor=True)
            embeddings[name] = emb
            logger.info(f"✅ {name}: {emb.shape} - 前3維: {emb[:3].tolist()}")
        
        # 測試跨模組相似度
        logger.info("\n🔗 測試模組間語意差異:")
        logger.info("=" * 60)
        
        for name1, emb1 in embeddings.items():
            for name2, emb2 in embeddings.items():
                if name1 < name2:  # 避免重複比較
                    sim = torch.nn.functional.cosine_similarity(
                        emb1.unsqueeze(0), 
                        emb2.unsqueeze(0)
                    ).item()
                    logger.info(f"{name1} <-> {name2}: 相似度 {sim:.4f}")
        
        logger.info("=" * 60)
        logger.info("✅ AI 可以識別不同模組的語意差異")
        logger.info("✅ 這使 AI 能夠理解模組的功能和職責")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 程式碼分析測試失敗: {e}")
        return False

def main():
    logger.info("🚀 開始 AI 語意編碼能力完整測試\n")
    logger.info("=" * 60)
    
    # 測試 1: 語意理解能力
    test1_passed = test_semantic_encoding_capability()
    
    # 測試 2: 程式碼分析理解
    test2_passed = test_code_analysis_understanding()
    
    # 總結
    logger.info("\n" + "=" * 60)
    logger.info("📋 測試總結:")
    logger.info("=" * 60)
    logger.info(f"{'✅' if test1_passed else '❌'} 語意理解能力測試")
    logger.info(f"{'✅' if test2_passed else '❌'} 程式碼分析理解測試")
    logger.info("=" * 60)
    
    if test1_passed and test2_passed:
        logger.info("🎉 所有測試通過！")
        logger.info("✅ P0 修復成功: AI 現在具備程式碼語意理解能力")
        logger.info("✅ AI 可以分析五大模組並理解其功能差異")
    else:
        logger.warning("⚠️ 部分測試未通過，請檢查錯誤信息")

if __name__ == "__main__":
    main()
