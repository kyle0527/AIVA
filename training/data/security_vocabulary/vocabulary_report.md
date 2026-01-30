# AIVA 安全領域詞彙表統計報告

## 概述
- **總術語數**: 63
- **總出現次數**: 791
- **來源文檔**: 4

## Top 20 高頻術語

| 排名 | 術語 | 頻率 |
|------|------|------|
| 1 | `jwt` | 76 |
| 2 | `graphql` | 68 |
| 3 | `rce` | 59 |
| 4 | `xss` | 56 |
| 5 | `websocket` | 45 |
| 6 | `cloudflare` | 35 |
| 7 | `idor` | 28 |
| 8 | `f5` | 28 |
| 9 | `imperva` | 27 |
| 10 | `authorization` | 25 |
| 11 | `ssrf` | 24 |
| 12 | `oracle` | 20 |
| 13 | `aws waf` | 18 |
| 14 | `bola` | 17 |
| 15 | `sql injection` | 16 |
| 16 | `ognl` | 15 |
| 17 | `mysql` | 13 |
| 18 | `cookie` | 11 |
| 19 | `mass assignment` | 11 |
| 20 | `x-forwarded-for` | 10 |

## 按類別分類

### CVE 編號
11 個唯一 CVE

### 數據庫系統
oracle, mysql, mssql, postgresql

### 攻擊技術
前10個: `jwt`, `graphql`, `rce`, `xss`, `websocket`, `cloudflare`, `idor`, `f5`, `imperva`, `authorization`

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
