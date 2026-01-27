#!/usr/bin/env python3
"""
Legacy import bridge - moved to _dev_tools/converters/core/schema_codegen_tool.py

AIVA Schema Code Generation Tool - Legacy Bridge

This file provides backward compatibility for existing imports.
The actual implementation has been moved to the AIVA Dev Tools.

Original Location: services/aiva_common/tools/schema_codegen_tool.py
New Location: _dev_tools/converters/core/schema_codegen_tool.py

Legacy Usage (still works):
    python services/aiva_common/tools/schema_codegen_tool.py --generate-all

New Usage (recommended):
    python _dev_tools/converters/core/schema_codegen_tool.py --generate-all
"""

import sys
import warnings
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

# Issue deprecation warning
warnings.warn(
    "Importing from 'services.aiva_common.tools.schema_codegen_tool' is deprecated. "
    "Please use '_dev_tools.converters.core.schema_codegen_tool' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Placeholder class to avoid import errors when module is loaded
SchemaCodeGenerator = None  # type: ignore[assignment]

# Try to import from the new dev_tools location
try:
    dev_tools_path = Path(__file__).parent.parent.parent.parent / "_dev_tools" / "converters"
    sys.path.insert(0, str(dev_tools_path))
    
    from core.schema_codegen_tool import SchemaCodeGenerator as _SchemaCodeGenerator  # type: ignore[import-not-found]
    SchemaCodeGenerator = _SchemaCodeGenerator
    
except ImportError as e:
    # Silently fail - the class will be None
    # This prevents static analysis errors while allowing runtime detection
    pass


def main():
    """Main entry point for CLI usage"""
    print("[WARN] DEPRECATION NOTICE")
    print("此檔案已移至 _dev_tools/converters/core/schema_codegen_tool.py")
    print("請使用新的位置執行")
    print()
    
    # Try to delegate to the new location
    try:
        dev_tools_path = Path(__file__).parent.parent.parent.parent / "_dev_tools" / "converters"
        sys.path.insert(0, str(dev_tools_path))
        from core.schema_codegen_tool import main as dev_main  # type: ignore[import-not-found]
        dev_main()
    except ImportError:
        print("[ERROR] Cannot find implementation at _dev_tools/converters/core/")
        sys.exit(1)


if __name__ == "__main__":
    main()
