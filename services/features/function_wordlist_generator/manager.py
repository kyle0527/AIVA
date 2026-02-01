"""
Wordlist Generator Manager
"""

import os
import logging
import aiofiles
from typing import List, Optional
from datetime import datetime

from .models import (
    GenerationStrategy,
    CharsetType,
    GeneratorConfig,
    GeneratorResult,
    WordlistStats
)

logger = logging.getLogger(__name__)


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
        """生成組合字典"""
        try:
            logger.info(f"Generating combination wordlist: {min_length}-{max_length} chars")
            
            # TODO: 實現組合生成邏輯
            
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error="Implementation pending"
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
        birthdate: Optional[str] = None,
        keywords: List[str] = None,
        output_file: str = "cupp_wordlist.txt"
    ) -> GeneratorResult:
        """基於目標資訊生成字典 (CUPP)"""
        try:
            logger.info(f"Generating CUPP wordlist for: {name}")
            
            # TODO: 整合 CUPP 工具
            
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error="Implementation pending"
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
        input_files: List[str],
        output_file: str,
        deduplicate: bool = True,
        sort: bool = False
    ) -> GeneratorResult:
        """合併多個字典"""
        try:
            logger.info(f"Merging {len(input_files)} wordlists")
            
            # TODO: 實現合併邏輯
            
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error="Implementation pending"
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
        """分析字典統計資訊"""
        try:
            logger.info(f"Analyzing wordlist: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            total_words = 0
            unique_words_set = set()
            min_length = float('inf')
            max_length = 0
            total_length = 0

            has_lowercase = False
            has_uppercase = False
            has_digits = False
            has_special = False
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                async for line in f:
                    word = line.strip()
                    if not word:
                        continue

                    total_words += 1
                    unique_words_set.add(word)

                    length = len(word)
                    min_length = min(min_length, length)
                    max_length = max(max_length, length)
                    total_length += length

                    # Check character types if not all found yet
                    if not (has_lowercase and has_uppercase and has_digits and has_special):
                        if not has_lowercase and any(c.islower() for c in word):
                            has_lowercase = True
                        if not has_uppercase and any(c.isupper() for c in word):
                            has_uppercase = True
                        if not has_digits and any(c.isdigit() for c in word):
                            has_digits = True
                        if not has_special and not word.isalnum():
                            has_special = True

            if total_words == 0:
                min_length = 0
                avg_length = 0.0
            else:
                avg_length = total_length / total_words

            complexity_score = 0.0
            if has_lowercase: complexity_score += 0.25
            if has_uppercase: complexity_score += 0.25
            if has_digits: complexity_score += 0.25
            if has_special: complexity_score += 0.25

            return WordlistStats(
                file_path=file_path,
                total_words=total_words,
                unique_words=len(unique_words_set),
                file_size_mb=round(file_size_mb, 4),
                min_length=min_length,
                max_length=max_length,
                avg_length=round(avg_length, 2),
                has_lowercase=has_lowercase,
                has_uppercase=has_uppercase,
                has_digits=has_digits,
                has_special=has_special,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
