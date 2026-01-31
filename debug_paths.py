import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, r"C:\D\fold7\AIVA-git")

try:
    from services.integration.data.internal_exploration.paths_config import ExternalPaths, PROJECT_ROOT
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print("Resolved Paths for Python:")
    for path in ExternalPaths.ANALYSIS_INPUTS['python']:
        print(f"Path: {path} | Exists: {path.exists()}")
except Exception as e:
    print(f"Error: {e}")
