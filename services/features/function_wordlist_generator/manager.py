"""
Wordlist Generator Manager
"""

from collections import Counter
from datetime import datetime
import itertools
import logging
import math
import os
import time

from .models import (
    CharsetType,
    GeneratorResult,
    WordlistStats,
)

logger = logging.getLogger(__name__)

# 內建字符集映射
_CHARSETS: dict[str, str] = {
    CharsetType.LOWERCASE: "abcdefghijklmnopqrstuvwxyz",
    CharsetType.UPPERCASE: "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    CharsetType.DIGITS: "0123456789",
    CharsetType.SPECIAL: "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    CharsetType.ALPHA: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    CharsetType.ALPHANUMERIC: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    CharsetType.ALL: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=",
}


class WordlistGeneratorManager:
    """字典生成器管理器"""
    
    def __init__(self):
        logger.info("WordlistGeneratorManager initialized")
    
    async def generate_combination(
        self,
        charset: str,
        min_length: int,
        max_length: int,
        output_file: str
    ) -> GeneratorResult:
        """生成組合字典（使用 itertools.product 窮舉指定字符集和長度範圍）"""
        start = datetime.now()
        t0 = time.monotonic()
        try:
            logger.info(f"Generating combination wordlist: {min_length}-{max_length} chars, charset len={len(charset)}")

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            total_words = 0
            with open(output_file, "w", encoding="utf-8") as fh:
                for length in range(min_length, max_length + 1):
                    for combo in itertools.product(charset, repeat=length):
                        fh.write("".join(combo) + "\n")
                        total_words += 1

            elapsed = time.monotonic() - t0
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"Combination wordlist generated: {total_words} words, {size_mb:.2f} MB, {elapsed:.1f}s")
            return GeneratorResult(
                success=True,
                output_file=output_file,
                total_words=total_words,
                unique_words=total_words,  # itertools.product 不重複
                file_size_mb=round(size_mb, 4),
                generation_time=elapsed,
                start_time=start,
                end_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def generate_cupp_wordlist(
        self,
        name: str,
        birthdate: str | None = None,
        keywords: list[str] = None,
        output_file: str = "cupp_wordlist.txt"
    ) -> GeneratorResult:
        """基於目標資訊生成字典（CUPP 風格：名字/生日/關鍵詞的常見變形組合）

        生成邏輯：
        1. 收集基底詞彙（name、birthdate 片段、keywords）
        2. 對每個基底詞套用 leetspeak 和大小寫轉換
        3. 組合兩個詞彙 + 數字後綴
        不依賴外部 CUPP 工具，純 Python 實作。
        """
        start = datetime.now()
        t0 = time.monotonic()
        try:
            if keywords is None:
                keywords = []
            logger.info(f"Generating CUPP-style wordlist for: {name}")

            # ── 1. 收集基底詞彙 ──
            bases: list[str] = []
            if name:
                parts = name.strip().split()
                bases.extend(parts)
                if len(parts) > 1:
                    bases.append("".join(parts))          # 全名連寫

            if birthdate:
                clean = birthdate.replace("-", "").replace("/", "").replace(".", "")
                if len(clean) >= 4:
                    bases.append(clean)                   # YYYYMMDD
                    bases.append(clean[-4:])              # MMDD 或 YYYY
                    bases.append(clean[:4])               # YYYY

            bases.extend(k.strip() for k in keywords if k.strip())

            # ── 2. 變形函數 ──
            _leet = str.maketrans("aeiost", "431057")

            def variants(word: str) -> list[str]:
                w = word.strip()
                if not w:
                    return []
                return list({
                    w,
                    w.lower(),
                    w.upper(),
                    w.capitalize(),
                    w.lower().translate(_leet),       # leetspeak
                    w.capitalize() + "!",
                    w.lower() + "123",
                    w.capitalize() + "123",
                    w + "@",
                })

            # ── 3. 展開所有變形 ──
            expanded: list[str] = []
            for b in bases:
                expanded.extend(variants(b))

            # ── 4. 兩兩組合 + 數字後綴 ──
            suffixes = ["", "1", "12", "123", "!", "@", "#", "2024", "2025"]
            words: set[str] = set(expanded)
            for a, b in itertools.combinations(bases[:8], 2):  # 限前8個避免爆炸
                for sa, sb in itertools.product(variants(a)[:3], variants(b)[:3]):
                    for suf in suffixes:
                        words.add(sa + sb + suf)
                        words.add(sb + sa + suf)

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as fh:
                for w in sorted(words):
                    fh.write(w + "\n")

            elapsed = time.monotonic() - t0
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"CUPP wordlist generated: {len(words)} words, {elapsed:.2f}s")
            return GeneratorResult(
                success=True,
                output_file=output_file,
                total_words=len(words),
                unique_words=len(words),
                file_size_mb=round(size_mb, 4),
                generation_time=elapsed,
                start_time=start,
                end_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"CUPP generation failed: {str(e)}", exc_info=True)
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def merge_wordlists(
        self,
        input_files: list[str],
        output_file: str,
        deduplicate: bool = True,
        sort: bool = False
    ) -> GeneratorResult:
        """合併多個字典（支援去重與排序）"""
        start = datetime.now()
        t0 = time.monotonic()
        try:
            logger.info(f"Merging {len(input_files)} wordlists → {output_file}")

            missing = [f for f in input_files if not os.path.isfile(f)]
            if missing:
                return GeneratorResult(
                    success=False,
                    output_file=output_file,
                    error=f"Files not found: {missing}",
                )

            words: list[str] | set[str]
            if deduplicate:
                words = set()
                for path in input_files:
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            w = line.rstrip("\n")
                            if w:
                                words.add(w)  # type: ignore[union-attr]
                words_list = sorted(words) if sort else list(words)
            else:
                words_list = []
                for path in input_files:
                    with open(path, encoding="utf-8", errors="ignore") as fh:
                        for line in fh:
                            w = line.rstrip("\n")
                            if w:
                                words_list.append(w)
                if sort:
                    words_list.sort()

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write("\n".join(words_list) + ("\n" if words_list else ""))

            total = len(words_list)
            elapsed = time.monotonic() - t0
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"Merged {len(input_files)} files → {total} words, {size_mb:.2f} MB, {elapsed:.2f}s")
            return GeneratorResult(
                success=True,
                output_file=output_file,
                total_words=total,
                unique_words=total if deduplicate else len(set(words_list)),
                file_size_mb=round(size_mb, 4),
                generation_time=elapsed,
                start_time=start,
                end_time=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Merge failed: {str(e)}", exc_info=True)
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error=str(e)
            )
    
    async def analyze_wordlist(
        self,
        file_path: str
    ) -> WordlistStats:
        """分析字典統計資訊（長度分布、字符集、Shannon entropy）"""
        try:
            logger.info(f"Analyzing wordlist: {file_path}")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Wordlist not found: {file_path}")

            words: list[str] = []
            length_dist: dict[int, int] = {}
            char_counter: Counter = Counter()
            has_lower = has_upper = has_digit = has_special = False

            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    w = line.rstrip("\n")
                    if not w:
                        continue
                    words.append(w)
                    ln = len(w)
                    length_dist[ln] = length_dist.get(ln, 0) + 1
                    char_counter.update(w)
                    if not has_lower and any(c.islower() for c in w):
                        has_lower = True
                    if not has_upper and any(c.isupper() for c in w):
                        has_upper = True
                    if not has_digit and any(c.isdigit() for c in w):
                        has_digit = True
                    if not has_special and any(not c.isalnum() for c in w):
                        has_special = True

            total = len(words)
            unique = len(set(words))
            size_mb = os.path.getsize(file_path) / 1024 / 1024
            min_len = min(length_dist) if length_dist else 0
            max_len = max(length_dist) if length_dist else 0
            avg_len = sum(ln * cnt for ln, cnt in length_dist.items()) / total if total else 0.0

            # Shannon entropy（以字符為單位）
            total_chars = sum(char_counter.values())
            entropy = 0.0
            if total_chars > 0:
                for cnt in char_counter.values():
                    p = cnt / total_chars
                    entropy -= p * math.log2(p)

            # 複雜度分數（0-1）：字符集種類 / 4
            charset_used: list[str] = []
            if has_lower:
                charset_used.append("lowercase")
            if has_upper:
                charset_used.append("uppercase")
            if has_digit:
                charset_used.append("digits")
            if has_special:
                charset_used.append("special")
            complexity = len(charset_used) / 4.0

            logger.info(f"Analysis complete: {total} words, {unique} unique, entropy={entropy:.2f}")
            return WordlistStats(
                file_path=file_path,
                total_words=total,
                unique_words=unique,
                file_size_mb=round(size_mb, 4),
                min_length=min_len,
                max_length=max_len,
                avg_length=round(avg_len, 2),
                length_distribution=length_dist,
                charset_used=charset_used,
                has_lowercase=has_lower,
                has_uppercase=has_upper,
                has_digits=has_digit,
                has_special=has_special,
                complexity_score=round(complexity, 4),
                entropy_score=round(entropy, 4),
            )

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
