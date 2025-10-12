"""
Deduplication utilities for AIVA Platform.

Provides:
- DeDup: Fingerprint-based deduplication system
- compute_fingerprint: Smart fingerprinting for findings
"""

from .dedupe import DeDup, compute_fingerprint

__all__ = ["DeDup", "compute_fingerprint"]
