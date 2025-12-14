#!/usr/bin/env python3
"""
Legacy import bridge - moved to plugins/aiva_converters/core/schema_codegen_tool.py

AIVA Schema Code Generation Tool - Legacy Bridge

This file provides backward compatibility for existing imports.
The actual implementation has been moved to the AIVA Converters Plugin.

Original Location: services/aiva_common/tools/schema_codegen_tool.py
New Location: plugins/aiva_converters/core/schema_codegen_tool.py

Legacy Usage (still works):
    python services/aiva_common/tools/schema_codegen_tool.py --generate-all

New Usage (recommended):
    python plugins/aiva_converters/core/schema_codegen_tool.py --generate-all
"""

import sys
import warnings
import importlib.util
from pathlib import Path

# Issue deprecation warning
warnings.warn(
    "Importing from 'services.aiva_common.tools.schema_codegen_tool' is deprecated. "
    "Please use 'plugins.aiva_converters.core.schema_codegen_tool' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Try to import from the new plugin location
try:
    plugin_path = Path(__file__).parent.parent.parent.parent / "plugins"
    sys.path.insert(0, str(plugin_path))
    
    from aiva_converters.core.schema_codegen_tool import SchemaCodeGenerator
    
    print("✅ Successfully loaded SchemaCodeGenerator from AIVA Converters Plugin")
    
except ImportError as e:
    print(f"❌ Plugin import failed: {e}")
    print("🔄 Trying backup implementation...")
    
    # Load backup if plugin fails
    backup_file = Path(__file__).parent / "schema_codegen_tool_backup.py"
    if backup_file.exists():
        try:
            spec = importlib.util.spec_from_file_location("schema_codegen_backup", backup_file)
            if spec and spec.loader:
                backup_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(backup_module)
                SchemaCodeGenerator = backup_module.SchemaCodeGenerator
                print("⚠️  Using backup implementation")
            else:
                raise ImportError("Failed to load backup module spec")
        except Exception as backup_error:
            print(f"❌ Backup import also failed: {backup_error}")
            raise ImportError(
                "Cannot import SchemaCodeGenerator from plugin or backup. "
                "Please ensure the AIVA Converters Plugin is properly installed."
            ) from e
    else:
        raise ImportError(
            f"Plugin not found and no backup available at {backup_file}. "
            "Please ensure the AIVA Converters Plugin is properly installed."
        ) from e


def main():
    """Main entry point for CLI usage"""
    print("⚠️  DEPRECATION NOTICE ⚠️")
    print("此檔案已移至 AIVA Converters Plugin。")
    print("請使用新的插件位置：plugins/aiva_converters/core/schema_codegen_tool.py")
    print()
    
    # Try to delegate to the new location
    try:
        from aiva_converters.core.schema_codegen_tool import main as plugin_main
        plugin_main()
    except ImportError:
        print("❌ Cannot find plugin implementation")
        sys.exit(1)


if __name__ == "__main__":
    main()
