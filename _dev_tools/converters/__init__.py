#!/usr/bin/env python3
"""
AIVA Converters Plugin
======================

Plugin entry point for AIVA conversion and generation tools.
"""

__version__ = "1.0.0"
__author__ = "AIVA Architecture Team"
__license__ = "MIT"

from pathlib import Path

# Plugin metadata
PLUGIN_NAME = "aiva_converters"
PLUGIN_VERSION = __version__
PLUGIN_DESCRIPTION = "Comprehensive conversion and generation tools for AIVA"
PLUGIN_ROOT = Path(__file__).parent

# Export main classes and functions
# 暫時註釋掉有問題的導入,只導出可用的
try:
    from .core.schema_codegen_tool import SchemaCodeGenerator
except ImportError:
    SchemaCodeGenerator = None

try:
    from .core.typescript_generator import TypeScriptGenerator
except ImportError:
    TypeScriptGenerator = None

try:
    from .core.cross_language_validator import CrossLanguageValidator
except ImportError:
    CrossLanguageValidator = None

__all__ = [
    "SchemaCodeGenerator",
    "TypeScriptGenerator", 
    "CrossLanguageValidator",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PLUGIN_DESCRIPTION"
]