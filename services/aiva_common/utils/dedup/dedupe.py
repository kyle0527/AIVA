"""
Deduplication system for vulnerability findings.

Integrated from Warprecon-M project for AIVA Platform.

Provides efficient fingerprint-based deduplication to avoid reporting
the same vulnerability multiple times.

Features:
- SQLite-based fingerprint tracking
- TTL-based automatic expiration
- Thread-safe operations
- WAL mode for better performance
- Smart fingerprinting algorithm

Example:
    from services.aiva_common.utils.dedup import DeDup

    dedupe = DeDup(
        db_path="data/findings.db",
        ttl_days=30
    )

    if not dedupe.seen_before(finding):
        dedupe.mark(finding)
        # Process new finding
    else:
        # Skip duplicate
        pass
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import weakref

logger = logging.getLogger(__name__)


def compute_fingerprint(item: Mapping[str, Any]) -> str:
    """
    Compute a deterministic fingerprint for a finding.

    The fingerprint includes protocol, location, parameter, and payload
    to uniquely identify a vulnerability. Uses JSON serialization and
    SHA256 hashing to avoid collisions.

    Args:
        item: Finding dictionary with keys like:
            - target: Target URL/endpoint
            - vector: Dict with protocol, location, param
            - payload: Attack payload used
            - mode: Optional detection mode

    Returns:
        SHA256 hex digest (64 characters)

    Example:
        >>> finding = {
        ...     "target": "https://example.com/api",
        ...     "vector": {
        ...         "protocol": "http",
        ...         "location": "query",
        ...         "param": "id"
        ...     },
        ...     "payload": "' OR 1=1--",
        ...     "mode": "sqli"
        ... }
        >>> fp = compute_fingerprint(finding)
        >>> len(fp)
        64
    """
    vector = item.get("vector", {}) or {}
    base_obj = {
        "target": item.get("target", "") or "",
        "protocol": vector.get("protocol", "") or "",
        "location": vector.get("location", "") or "",
        "param": vector.get("param", "") or "",
        "payload": item.get("payload", "") or "",
    }
    mode = item.get("mode")
    if mode:
        base_obj["mode"] = mode
    base = json.dumps(base_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


class DeDup:
    """
    Track fingerprints of findings using a lightweight SQLite store.

    The store keeps a bounded window of fingerprints governed by ttl_days
    so lookups remain efficient even when processing large campaigns.
    When db_path is not provided, an in-memory database is used.

    Features:
    - Efficient SQLite storage with WAL mode
    - TTL-based automatic expiration
    - Thread-safe operations
    - Minimal memory footprint
    - Automatic database cleanup

    Example:
        # Persistent storage
        dedupe = DeDup(db_path="findings.db", ttl_days=30)

        # In-memory (for testing)
        dedupe = DeDup()

        # Check and mark
        if not dedupe.seen_before(finding):
            # New finding
            pass

        # Manual cleanup
        dedupe.close()
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        ttl_days: float | None = 30.0,
    ) -> None:
        """
        Initialize deduplication tracker.

        Args:
            db_path: Path to SQLite database file.
                    Use None or ":memory:" for in-memory database.
            ttl_days: Time-to-live in days for fingerprints.
                     Use None or 0 for no expiration.

        Example:
            # Persistent with 30-day TTL
            dedupe = DeDup("findings.db", ttl_days=30)

            # In-memory, no expiration
            dedupe = DeDup(":memory:", ttl_days=None)
        """
        # Convert TTL days to seconds with explicit None/zero
        # handling to satisfy type checkers
        if ttl_days is None or ttl_days == 0:
            self._ttl_seconds = 0.0
        else:
            # mypy: ttl_days is float here
            days: float = float(ttl_days)
            self._ttl_seconds = max(days, 0.0) * 86400.0
        self._lock = threading.Lock()

        # Determine database location
        if isinstance(db_path, Path):
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_location = str(db_path)
        elif isinstance(db_path, str) and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            db_location = db_path
        elif db_path:
            db_location = str(db_path)
        else:
            db_location = ":memory:"

        # Initialize SQLite connection
        self._conn = sqlite3.connect(
            db_location, isolation_level=None, check_same_thread=False
        )
        self._finalizer = weakref.finalize(self, self._conn.close)

        # Enable WAL mode for better concurrency
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            logger.warning("Failed to enable WAL mode, using default journal")

        # Create schema
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fingerprints (
                fingerprint TEXT PRIMARY KEY,
                expires REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fingerprints_expires "
            "ON fingerprints(expires)"
        )

        logger.debug(
            "Initialized DeDup with db=%s, ttl_days=%s",
            db_location,
            ttl_days,
        )

    def _prune(self, now: float) -> None:
        """Remove expired fingerprints."""
        if self._ttl_seconds > 0:
            deleted = self._conn.execute(
                "DELETE FROM fingerprints WHERE expires IS NOT NULL AND expires <= ?",
                (now,),
            ).rowcount
            if deleted > 0:
                logger.debug("Pruned %d expired fingerprints", deleted)

    def seen_before(self, item: Mapping[str, Any]) -> bool:
        """
        Check if item has been seen before and mark it.

        This method is atomic - it checks and marks in a single operation.
        Returns True if the item was already seen (duplicate), False if new.

        Args:
            item: Finding dictionary to check

        Returns:
            True if duplicate (seen before), False if new

        Example:
            if dedupe.seen_before(finding):
                logger.debug("Skipping duplicate finding")
                return

            # Process new finding
            process_finding(finding)
        """
        digest = compute_fingerprint(item)
        now = time.time()

        with self._lock:
            self._prune(now)

            # Check if fingerprint exists and is not expired
            if self._ttl_seconds > 0:
                row = self._conn.execute(
                    "SELECT 1 FROM fingerprints WHERE fingerprint = ? "
                    "AND (expires IS NULL OR expires > ?)",
                    (digest, now),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT 1 FROM fingerprints WHERE fingerprint = ?",
                    (digest,),
                ).fetchone()

            if row:
                return True

            # Mark as seen
            expires = (now + self._ttl_seconds) if self._ttl_seconds > 0 else None
            self._conn.execute(
                "INSERT OR REPLACE INTO fingerprints (fingerprint, expires) VALUES (?, ?)",
                (digest, expires),
            )
            return False

    def mark(self, item: Mapping[str, Any]) -> None:
        """
        Explicitly mark an item as seen without checking.

        Use this when you want to pre-populate the dedup database
        or update expiration time for existing items.

        Args:
            item: Finding dictionary to mark

        Example:
            # Pre-populate from existing findings
            for finding in historical_findings:
                dedupe.mark(finding)
        """
        digest = compute_fingerprint(item)
        now = time.time()
        expires = (now + self._ttl_seconds) if self._ttl_seconds > 0 else None

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO fingerprints "
                "(fingerprint, expires) VALUES (?, ?)",
                (digest, expires),
            )

    def count(self) -> int:
        """
        Get the number of fingerprints currently tracked.

        Returns:
            Number of active fingerprints

        Example:
            >>> dedupe.count()
            1234
        """
        with self._lock:
            self._prune(time.time())
            row = self._conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()
            return row[0] if row else 0

    def clear(self) -> None:
        """
        Clear all fingerprints from the database.

        Use with caution - this removes all deduplication history.

        Example:
            # Start fresh campaign
            dedupe.clear()
        """
        with self._lock:
            self._conn.execute("DELETE FROM fingerprints")
            logger.info("Cleared all fingerprints from dedup database")

    def close(self) -> None:
        """
        Close the underlying SQLite connection.

        This is automatically called on object destruction, but can be
        called explicitly for immediate cleanup.

        Example:
            dedupe = DeDup("findings.db")
            try:
                # Use dedupe
                pass
            finally:
                dedupe.close()
        """
        with self._lock:
            if self._finalizer.alive:
                self._finalizer()
                logger.debug("Closed DeDup database connection")


# Backward compatibility alias
fp = compute_fingerprint
