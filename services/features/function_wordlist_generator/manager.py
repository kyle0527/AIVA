"""
Wordlist Generator Manager
"""

import os
import itertools
import logging
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
        start_time = datetime.now()
        try:
            logger.info(f"Generating combination wordlist: {min_length}-{max_length} chars")
            
            if not charset:
                raise ValueError("Charset cannot be empty")
            if min_length < 1:
                raise ValueError("min_length must be at least 1")
            if max_length < min_length:
                raise ValueError("max_length must be greater than or equal to min_length")

            # Deduplicate and sort charset for consistent results
            unique_charset = "".join(sorted(list(set(charset))))

            total_words = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for length in range(min_length, max_length + 1):
                    for combo in itertools.product(unique_charset, repeat=length):
                        word = "".join(combo)
                        f.write(word + '\n')
                        total_words += 1

            end_time = datetime.now()
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            generation_time = (end_time - start_time).total_seconds()

            return GeneratorResult(
                success=True,
                output_file=output_file,
                total_words=total_words,
                unique_words=total_words,
                file_size_mb=file_size_mb,
                generation_time=generation_time,
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return GeneratorResult(
                success=False,
                output_file=output_file,
                error=str(e),
                start_time=start_time,
                end_time=datetime.now()
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
            
            # TODO: 實現分析邏輯
            
            return WordlistStats(
                file_path=file_path,
                total_words=0
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
