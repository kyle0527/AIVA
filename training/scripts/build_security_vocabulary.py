"""
安全領域詞彙表構建工具

將 AIVA 的安全知識文檔轉換為增強詞彙表，
使 Embedding 層能夠理解專業安全術語。

功能：
1. 從4份安全知識文檔中提取關鍵術語
2. 構建安全領域專用詞彙表
3. 與基礎詞彙表（all-MiniLM-L6-v2）整合
4. 生成訓練語料以微調 Embedding 層
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityVocabularyBuilder:
    """安全領域詞彙表構建器"""
    
    # 安全知識文檔路徑
    KNOWLEDGE_DOCS = {
        "sqli_xss_ssrf_idor": "AI 掃描器漏洞判斷邏輯資料庫.md",
        "cve_identification": "AI 識別高危險 CVE 模組.md", 
        "waf_bypass": "WAF 繞過技術字典生成.md",
        "web_architecture": "Web 架構安全漏洞檢測指南.md"
    }
    
    # 關鍵安全術語模式（正則）
    SECURITY_PATTERNS = {
        # CVE 編號
        "cve": r"CVE-\d{4}-\d{4,7}",
        
        # 漏洞類型
        "vulnerability": r"(?:SQL[i\s](?:injection|注入)|XSS|SSRF|IDOR|CSRF|XXE|SSTI|"
                        r"RCE|LFI|RFI|BOLA|Mass\s+Assignment|JWT|GraphQL|"
                        r"Insecure\s+Deserialization|Path\s+Traversal)",
        
        # 攻擊技術
        "technique": r"(?:UNION\s+SELECT|WAITFOR\s+DELAY|SLEEP\(\)|pg_sleep|"
                     r"dbms_pipe|BENCHMARK|JNDI|OGNL|SpEL|"
                     r"\$\{jndi:|Polyglots?|Obfuscation)",
        
        # 數據庫系統
        "database": r"(?:MySQL|MariaDB|PostgreSQL|MSSQL|Oracle|MongoDB|Redis)",
        
        # WAF 廠商
        "waf": r"(?:Cloudflare|AWS\s+WAF|Imperva|ModSecurity|F5)",
        
        # 協議/架構
        "protocol": r"(?:GraphQL|REST\s+API|WebSocket|gRPC|SOAP|JWT|OAuth)",
        
        # 錯誤訊息關鍵字
        "error_keyword": r"(?:syntax\s+error|unterminated\s+quoted|"
                        r"ORA-\d+|ERROR:\s|mysql_fetch|Warning:|Fatal\s+error)",
        
        # 漏洞利用函數
        "exploit_function": r"(?:eval\(|exec\(|system\(|passthru\(|"
                           r"shell_exec\(|file_get_contents\(|"
                           r"document\.write\(|innerHTML)",
        
        # HTTP Headers
        "http_header": r"(?:User-Agent|X-Forwarded-For|Referer|Cookie|"
                      r"Authorization|Content-Type|X-F5-Auth-Token)",
    }
    
    def __init__(self, docs_dir: Path, output_dir: Path):
        """
        初始化詞彙表構建器
        
        Args:
            docs_dir: 知識文檔目錄路徑
            output_dir: 輸出目錄路徑
        """
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 術語統計
        self.security_terms: Counter = Counter()
        self.term_contexts: Dict[str, List[str]] = {}  # 術語 -> 上下文列表
        
    def extract_terms_from_docs(self) -> None:
        """從知識文檔中提取安全術語"""
        logger.info("🔍 開始從知識文檔提取安全術語...")
        
        for doc_type, doc_name in self.KNOWLEDGE_DOCS.items():
            doc_path = self.docs_dir / doc_name
            
            if not doc_path.exists():
                logger.warning(f"⚠️  文檔不存在: {doc_path}")
                continue
            
            logger.info(f"📄 處理文檔: {doc_name}")
            
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用各種模式提取術語
            for pattern_type, pattern in self.SECURITY_PATTERNS.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    term = match.group(0)
                    
                    # 標準化術語（保留大小寫敏感的 CVE）
                    if pattern_type != "cve":
                        term = term.lower()
                    
                    self.security_terms[term] += 1
                    
                    # 提取上下文（前後50字符）
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].replace('\n', ' ')
                    
                    if term not in self.term_contexts:
                        self.term_contexts[term] = []
                    
                    if len(self.term_contexts[term]) < 3:  # 每個術語保留3個上下文
                        self.term_contexts[term].append(context)
            
            logger.info(f"  ✅ 已提取 {sum(self.security_terms.values())} 個術語實例")
        
        logger.info(f"✅ 提取完成！共發現 {len(self.security_terms)} 個唯一安全術語")
    
    def build_security_vocab(self, min_frequency: int = 2) -> Dict[str, int]:
        """
        構建安全領域詞彙表
        
        Args:
            min_frequency: 最小出現頻率（過濾低頻詞）
        
        Returns:
            術語 -> 詞頻的字典
        """
        logger.info(f"🔨 構建安全詞彙表（最小頻率: {min_frequency}）...")
        
        # 過濾低頻詞
        filtered_vocab = {
            term: freq 
            for term, freq in self.security_terms.items() 
            if freq >= min_frequency
        }
        
        # 按頻率排序
        sorted_vocab = dict(
            sorted(filtered_vocab.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"✅ 詞彙表構建完成！共 {len(sorted_vocab)} 個術語")
        return sorted_vocab
    
    def generate_training_corpus(self, vocab: Dict[str, int]) -> List[str]:
        """
        生成訓練語料（用於微調 Embedding）
        
        Args:
            vocab: 安全詞彙表
        
        Returns:
            訓練句子列表
        """
        logger.info("📝 生成訓練語料...")
        
        corpus = []
        
        for term, freq in vocab.items():
            # 為高頻術語添加多個上下文
            contexts = self.term_contexts.get(term, [])
            
            for context in contexts:
                # 清理上下文
                cleaned = re.sub(r'\s+', ' ', context).strip()
                if len(cleaned) > 20:  # 確保有足夠的上下文
                    corpus.append(cleaned)
        
        logger.info(f"✅ 生成 {len(corpus)} 條訓練語料")
        return corpus
    
    def save_outputs(
        self, 
        vocab: Dict[str, int], 
        corpus: List[str]
    ) -> None:
        """
        保存輸出文件
        
        Args:
            vocab: 安全詞彙表
            corpus: 訓練語料
        """
        logger.info("💾 保存輸出文件...")
        
        # 1. 保存詞彙表（JSON格式）
        vocab_file = self.output_dir / "security_vocabulary.json"
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_terms": len(vocab),
                    "total_frequency": sum(vocab.values()),
                    "source_docs": list(self.KNOWLEDGE_DOCS.values()),
                    "created_at": "2026-01-20"
                },
                "vocabulary": vocab
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✅ 詞彙表: {vocab_file}")
        
        # 2. 保存純文本詞彙表（與 vocab.txt 格式兼容）
        vocab_txt_file = self.output_dir / "security_terms.txt"
        with open(vocab_txt_file, 'w', encoding='utf-8') as f:
            for term in vocab.keys():
                f.write(f"{term}\n")
        logger.info(f"  ✅ 術語列表: {vocab_txt_file}")
        
        # 3. 保存訓練語料
        corpus_file = self.output_dir / "security_training_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for sentence in corpus:
                f.write(f"{sentence}\n")
        logger.info(f"  ✅ 訓練語料: {corpus_file}")
        
        # 4. 生成術語-上下文映射（用於理解術語用法）
        context_file = self.output_dir / "term_contexts.json"
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(self.term_contexts, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✅ 術語上下文: {context_file}")
        
        # 5. 生成統計報告
        report = self._generate_report(vocab)
        report_file = self.output_dir / "vocabulary_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"  ✅ 統計報告: {report_file}")
    
    def _generate_report(self, vocab: Dict[str, int]) -> str:
        """生成統計報告"""
        top_terms = list(vocab.items())[:20]
        
        report = f"""# AIVA 安全領域詞彙表統計報告

## 概述
- **總術語數**: {len(vocab)}
- **總出現次數**: {sum(vocab.values())}
- **來源文檔**: {len(self.KNOWLEDGE_DOCS)}

## Top 20 高頻術語

| 排名 | 術語 | 頻率 |
|------|------|------|
"""
        for i, (term, freq) in enumerate(top_terms, 1):
            report += f"| {i} | `{term}` | {freq} |\n"
        
        report += f"""
## 按類別分類

### CVE 編號
{len([t for t in vocab if t.startswith('CVE-')])} 個唯一 CVE

### 數據庫系統
{', '.join([t for t in vocab if t.lower() in ['mysql', 'postgresql', 'mssql', 'oracle', 'mongodb']])}

### 攻擊技術
前10個: {', '.join([f'`{t}`' for t in list(vocab.keys())[:10]])}

## 使用方式

1. **擴展基礎詞彙表**:
   ```python
   # 載入 all-MiniLM-L6-v2 的 tokenizer
   base_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
   
   # 添加安全術語
   with open("security_terms.txt") as f:
       new_terms = [line.strip() for line in f]
   
   base_tokenizer.add_tokens(new_terms)
   ```

2. **微調 Embedding 層**:
   使用 `security_training_corpus.txt` 進行領域適應訓練

3. **直接整合到 AIVA**:
   AIVAEmbedding 會自動使用擴展後的詞彙表
"""
        return report


def main():
    """主函數：執行完整的詞彙表構建流程"""
    import argparse
    
    parser = argparse.ArgumentParser(description="構建 AIVA 安全領域詞彙表")
    parser.add_argument(
        "--docs-dir",
        default=r"C:\Users\User\Downloads\新增資料夾",
        help="知識文檔目錄路徑"
    )
    parser.add_argument(
        "--output-dir", 
        default="../data/security_vocabulary",
        help="輸出目錄路徑"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="最小術語頻率"
    )
    
    args = parser.parse_args()
    
    # 創建構建器
    builder = SecurityVocabularyBuilder(
        docs_dir=Path(args.docs_dir),
        output_dir=Path(args.output_dir)
    )
    
    # 執行構建流程
    logger.info("=" * 60)
    logger.info("🚀 AIVA 安全領域詞彙表構建器")
    logger.info("=" * 60)
    
    # 步驟1: 提取術語
    builder.extract_terms_from_docs()
    
    # 步驟2: 構建詞彙表
    vocab = builder.build_security_vocab(min_frequency=args.min_freq)
    
    # 步驟3: 生成訓練語料
    corpus = builder.generate_training_corpus(vocab)
    
    # 步驟4: 保存輸出
    builder.save_outputs(vocab, corpus)
    
    logger.info("=" * 60)
    logger.info("✅ 完成！詞彙表已保存到: " + str(builder.output_dir))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
