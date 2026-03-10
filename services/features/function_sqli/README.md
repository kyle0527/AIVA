# AIVA SQL Injection Module (`function_sqli`)

Advanced SQL Injection detection module for the AIVA platform.

## 🚀 Overview

This module provides comprehensive SQL Injection detection capabilities, ranging from basic payload testing to advanced, smart scanning techniques.

**Key Features (v2.0 Refactor):**

-   **Smart Scanning**: Includes page stability analysis and backend database fingerprinting before launching attacks.
-   **WAF Evasion**: Built-in `PayloadWrapperEncoder` with Tamper Mixin supports various obfuscation levels (0-3).
-   **Fuzzy Logic Detection**: Uses sequence similarity (diff) to detect boolean-based injections on dynamic pages.
-   **Four-Tier Architecture**:
    1.  **Orchestrator**: `SmartDetectionManager`
    2.  **Encoder/Tamper**: `PayloadWrapperEncoder`
    3.  **Engines**: `engines/` (e.g., `BooleanDetectionEngine`, `ErrorDetectionEngine`)
    4.  **Models**: `detection_models.py`

## 📂 Directory Structure

```
services/features/function_sqli/
├── config.py                  # Centralized configuration (SqliConfig)
├── smart_detection_manager.py # Main entry point & orchestrator
├── payload_wrapper_encoder.py # Payload encoding & Tamper logic
├── detection_models.py        # Shared data models (DetectionResult)
├── backend_db_fingerprinter.py# Database fingerprinting logic
└── engines/                   # Detection Engines
    ├── base_detector.py       # Base class for all detectors
    ├── boolean_detection_engine.py
    ├── error_detection_engine.py
    └── ...
```

## ⚙️ Configuration

Key parameters in `config.py`:

-   `waf_evasion_level`:
    -   `0`: None (Default)
    -   `1`: Low (Random Case)
    -   `2`: Medium (Space2Comment, Version Comments)
    -   `3`: High (Double URL Encode, Between, etc.)
-   `stability_threshold`: Float (0.0 - 1.0). Minimum similarity score to consider a page "stable".
-   `fuzzy_similarity_threshold`: Float (0.0 - 1.0). Threshold for boolean false positive reduction.

## 🛠️ Usage

### Basic Usage

```python
from services.features.function_sqli.smart_detection_manager import SmartDetectionManager
from services.features.function_sqli.config import SqliConfig

# Initialize
config = SqliConfig(waf_evasion_level=1)
manager = SmartDetectionManager(config)

# Register Detectors
from services.features.function_sqli.engines.boolean_detection_engine import BooleanDetectionEngine
# Note: Detectors usually instantiated inside manager or via factory in full implementation
```

## 🧩 Architecture Details

### Smart Scanner Flow

1.  **Stability Check**: The scanner sends multiple identical requests to the target URL to establish a baseline stability score. If unstable, thresholds are adjusted.
2.  **Fingerprinting**: Attempts to identify the backend database (MySQL, PostgreSQL, Oracle, etc.) to optimize payload selection.
3.  **Detection**: Iterates through registered detection engines (Boolean, Error, etc.).

### Tamper Logic

Located in `PayloadWrapperEncoder`.

-   **Mixins**: `PayloadTamperMixin` provides methods like `_tamper_space2comment`, `_tamper_randomcase`.
-   **Application**: Controlled by `waf_evasion_level` in `SqliConfig`.
