"""
Rate limiter with Token Bucket algorithm, adaptive limits, and persistence.

Integrated from Warprecon-M project for AIVA Platform.
Provides intelligent rate limiting with:
- Global and per-host dual-layer limits
- Adaptive rate adjustment based on response feedback
- Retry-After header parsing
- Persistent state management
- Host TTL and automatic cleanup
- Cooldown mechanism

Example:
    limiter = RateLimiter(
        global_rps=10.0,
        per_host_rps=5.0,
        state_file="data/ratelimit.json"
    )

    await limiter.acquire(host)
    # ... make request ...
    limiter.update_from_response(
        host,
        status_code=response.status_code,
        headers=dict(response.headers),
        latency=response.elapsed.total_seconds()
    )
"""

import asyncio
import contextlib
import json
import logging
import os
import re
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token Bucket implementation for rate limiting."""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.capacity = burst
        self.tokens: float = float(burst)  # Use float for token calculations
        self.lock = threading.Lock()
        self.updated = time.monotonic()

    def take(self, n: int = 1) -> bool:
        """Try to consume n tokens. Returns True if successful."""
        with self.lock:
            now = time.monotonic()
            delta = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


class RateLimiter:
    """
    Async rate limiter supporting persistence and adaptive limits.

    Features:
    - Global + per-host dual-layer rate limiting
    - Adaptive rate adjustment based on 429/503 responses
    - Retry-After header parsing
    - Persistent state across restarts
    - Host TTL for automatic cleanup
    - Cooldown mechanism for temporary restrictions
    """

    def __init__(
        self,
        global_rps: float,
        per_host_rps: float,
        jitter_ms: int = 0,
        state_file: str | None = None,
        *,
        host_ttl: float | None = 3600.0,
        cleanup_interval: float = 60.0,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            global_rps: Global requests per second limit
            per_host_rps: Per-host requests per second limit
            jitter_ms: Random jitter in milliseconds to add to delays
            state_file: Path to persist state (None for no persistence)
            host_ttl: Host entry TTL in seconds (None for no expiry)
            cleanup_interval: Interval for host cleanup in seconds
        """
        self.global_rps = float(global_rps)
        self.per_host_rps = float(per_host_rps)
        burst_g = max(1, int(self.global_rps))
        burst_h = max(1, int(self.per_host_rps))
        self.global_bucket = TokenBucket(self.global_rps, burst_g)
        self.default_host_burst = burst_h
        self.host_buckets: dict[str, TokenBucket] = {}
        self.host_overrides: dict[str, float] = {}
        self.host_last_used: dict[str, float] = {}
        self.jitter = jitter_ms / 1000.0
        self.state_file = Path(state_file) if state_file else None
        self.state_lock = threading.Lock()
        self.config_lock = threading.Lock()
        self._host_usage_lock = threading.Lock()
        self.min_global_rps = 0.05
        self.max_global_rps = max(self.global_rps * 10.0, self.global_rps + 10.0)
        self.min_per_host_rps = 0.05
        self.max_per_host_rps = max(self.per_host_rps * 10.0, self.per_host_rps + 10.0)
        self._state_write_interval = 1.0
        self._next_state_write = 0.0
        try:
            ttl_val = float(host_ttl) if host_ttl is not None else None
        except (TypeError, ValueError):
            ttl_val = None
        self.host_ttl = ttl_val if ttl_val and ttl_val > 0 else None
        try:
            cleanup_val = float(cleanup_interval)
        except (TypeError, ValueError):
            cleanup_val = 60.0
        self._cleanup_interval = cleanup_val if cleanup_val > 0 else 0.0
        self._next_cleanup = (
            time.monotonic() + self._cleanup_interval if self._cleanup_interval else 0.0
        )

        # Initialize cooldown tracking and logging
        self._cooldown_until: dict[str, float] = {}
        self._log = logging.getLogger(__name__)

        if self.state_file:
            self._load_state()

    def _host_rate(self, host: str) -> float:
        """Get the rate limit for a specific host."""
        return self.host_overrides.get(host, self.per_host_rps)

    def _bucket_for(self, host: str) -> TokenBucket:
        """Get or create Token Bucket for a host."""
        self._maybe_cleanup_hosts()
        b = self.host_buckets.get(host)
        if b is None:
            rate = self._host_rate(host)
            burst = max(1, int(rate))
            b = TokenBucket(rate, burst)
            self.host_buckets[host] = b
        self._touch_host(host)
        return b

    async def acquire(self, host: str) -> None:
        """
        Acquire permission to send a request to the host.

        Blocks until both global and host tokens are available.
        Respects cooldown periods from Retry-After headers.

        Args:
            host: Target host for the request
        """
        # Wait for cooldown to expire
        while not self.should_send(host):
            await asyncio.sleep(0.1 + self.jitter)

        while True:
            took_global = self.global_bucket.take()
            took_host = self._bucket_for(host).take()
            if took_global and took_host:
                self._write_state()
                return
            await asyncio.sleep(0.01 + self.jitter)

    def penalize(
        self,
        host: str,
        penalty: float,
        *,
        status: int | None = None,
        latency: float | None = None,
    ) -> None:
        """
        Manually penalize a host by removing tokens.

        Args:
            host: Host to penalize
            penalty: Number of tokens to remove
            status: Optional status code for adaptive adjustment
            latency: Optional latency for adaptive adjustment
        """
        b = self._bucket_for(host)
        with b.lock:
            b.tokens = max(0.0, b.tokens - penalty)
        if status is not None or latency is not None:
            self.update_from_response(host, status_code=status, latency=latency)
        self._write_state()

    def update_from_response(
        self,
        host: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
        latency: float | None = None,
    ) -> None:
        """
        Adjust global and per-host rates based on response feedback.

        Handles:
        - Retry-After headers (sets cooldown)
        - 429/503/5xx responses (decreases rate)
        - 2xx responses with good latency (increases rate)
        - High latency (decreases rate)

        Args:
            host: Host that sent the response
            status_code: HTTP status code
            headers: Response headers (for Retry-After)
            latency: Response time in seconds
        """
        self._touch_host(host)

        # Phase 1: Handle Retry-After
        retry_after_until = self._handle_retry_after(host, headers)

        # Phase 2: Calculate adjustments
        adjustments = self._calculate_adjustments(status_code, latency, retry_after_until)

        # Phase 3: Apply adjustments
        self._apply_adjustments(host, adjustments)

        # Phase 4: Cleanup cooldown if success
        self._cleanup_cooldown_if_success(host, status_code, latency)

    def _handle_retry_after(self, host: str, headers: dict[str, str] | None) -> float | None:
        """Parse and apply Retry-After header."""
        ra_until = self._parse_retry_after(host, headers)
        if ra_until is None:
            return None

        prev = self._cooldown_until.get(host)
        self._cooldown_until[host] = max(prev or 0.0, ra_until)
        with contextlib.suppress(Exception):
            self._log.info(
                "ratelimiter.cooldown_begin host=%s until=%.3f source=Retry-After",
                host,
                ra_until,
            )
        return ra_until

    def _parse_retry_after(self, host: str, headers: dict[str, str] | None) -> float | None:
        """Parse Retry-After header value."""
        if not headers or not any(k.lower() == "retry-after" for k in headers):
            return None

        try:
            ra_val = next((headers[k] for k in headers if k.lower() == "retry-after"), None)
            if ra_val is None:
                return None

            ra_val = str(ra_val).strip()
            if re.fullmatch(r"\d+", ra_val):
                return time.monotonic() + float(ra_val)

            return self._parse_http_date(host, ra_val)
        except (KeyError, AttributeError, ValueError) as e:
            self._log.debug("Failed to process Retry-After header for host %s: %s", host, str(e))
            return None

    def _parse_http_date(self, host: str, ra_val: str) -> float | None:
        """Parse HTTP-date format in Retry-After."""
        try:
            dt = parsedate_to_datetime(ra_val)
            return time.monotonic() + max(0.0, (dt.timestamp() - time.time()))
        except (ValueError, TypeError, OverflowError) as e:
            self._log.debug(
                "Failed to parse HTTP-date in Retry-After header for host %s: %s",
                host,
                str(e),
            )
            return None

    def _calculate_adjustments(
        self, status_code: int | None, latency: float | None, retry_after_until: float | None
    ) -> dict[str, tuple[float, float, float]]:
        """Calculate penalty/boost adjustments from response signals."""
        penalty = [1.0, 1.0, 1.0]  # [global, host, base]
        boost = [1.0, 1.0, 1.0]

        if retry_after_until is not None:
            penalty[2] *= 1.5  # base
            penalty[1] *= 2.0  # host

        self._adjust_for_status_code(status_code, penalty, boost)
        self._adjust_for_latency(latency, penalty, boost)

        return {"penalty": tuple(penalty), "boost": tuple(boost)}

    def _adjust_for_status_code(
        self, status_code: int | None, penalty: list[float], boost: list[float]
    ) -> None:
        """Adjust penalty/boost based on status code."""
        if status_code is None:
            return

        if status_code == 429:
            penalty[0] = min(penalty[0], 0.7)   # global
            penalty[1] = min(penalty[1], 0.4)   # host
            penalty[2] = min(penalty[2], 0.85)  # base
        elif status_code in {503, 504}:
            penalty[0] = min(penalty[0], 0.8)
            penalty[1] = min(penalty[1], 0.6)
            penalty[2] = min(penalty[2], 0.9)
        elif 500 <= status_code < 600:
            penalty[0] = min(penalty[0], 0.9)
            penalty[1] = min(penalty[1], 0.75)
            penalty[2] = min(penalty[2], 0.95)
        elif 200 <= status_code < 300:
            boost[0] = max(boost[0], 1.05)
            boost[1] = max(boost[1], 1.08)
            boost[2] = max(boost[2], 1.02)

    def _adjust_for_latency(
        self, latency: float | None, penalty: list[float], boost: list[float]
    ) -> None:
        """Adjust penalty/boost based on latency."""
        if latency is None:
            return

        if latency > 5.0:
            penalty[0] = min(penalty[0], 0.6)
            penalty[1] = min(penalty[1], 0.5)
            penalty[2] = min(penalty[2], 0.85)
        elif latency > 2.5:
            penalty[0] = min(penalty[0], 0.75)
            penalty[1] = min(penalty[1], 0.65)
            penalty[2] = min(penalty[2], 0.9)
        elif latency < 0.3:
            boost[0] = max(boost[0], 1.08)
            boost[1] = max(boost[1], 1.1)
            boost[2] = max(boost[2], 1.04)
        elif latency < 0.6:
            boost[0] = max(boost[0], 1.03)
            boost[1] = max(boost[1], 1.05)
            boost[2] = max(boost[2], 1.02)

    def _apply_adjustments(
        self, host: str, adjustments: dict[str, tuple[float, float, float]]
    ) -> None:
        """Apply calculated penalty or boost adjustments."""
        penalty = adjustments["penalty"]
        boost = adjustments["boost"]

        # Apply penalties first (higher priority)
        if any(p < 1.0 for p in penalty):
            if penalty[0] < 1.0:
                self._scale_global(penalty[0])
            if penalty[2] < 1.0:
                self._scale_per_host_base(penalty[2])
            if penalty[1] < 1.0:
                self._scale_host(host, penalty[1])
            return

        # Apply boosts if no penalties
        if boost[0] > 1.0:
            self._scale_global(boost[0])
        if boost[2] > 1.0:
            self._scale_per_host_base(boost[2])
        if boost[1] > 1.0:
            self._scale_host(host, boost[1])

    def _cleanup_cooldown_if_success(
        self, host: str, status_code: int | None, latency: float | None
    ) -> None:
        """End cooldown early if response is successful with good latency."""
        try:
            if not (status_code and 200 <= int(status_code) < 300 and latency is not None):
                return

            until = self._cooldown_until.get(host)
            if not until:
                return

            now = time.monotonic()
            if now + max(0.0, latency) * 2 < until:
                self._cooldown_until.pop(host, None)
                self._log.info("ratelimiter.cooldown_end host=%s reason=success", host)
        except (KeyError, ValueError, TypeError) as e:
            self._log.debug("Error processing response update for host %s: %s", host, str(e))

    def should_send(self, host: str) -> bool:
        """
        Check if requests to host are currently allowed.

        Returns False if host is in cooldown due to Retry-After.

        Args:
            host: Host to check

        Returns:
            True if sending is allowed, False if in cooldown
        """
        now = time.monotonic()
        until = self._cooldown_until.get(host)
        return not (until is not None and now < until)

    def save_state(self) -> None:
        """Force write state to disk immediately."""
        self._write_state(force=True)

    def load_state(self) -> None:
        """Reload state from disk."""
        self._load_state()

    def _write_state(self, *, force: bool = False) -> None:
        """Write state to persistent storage."""
        if not self.state_file:
            return
        now_mono = time.monotonic()
        if not force and now_mono < self._next_state_write:
            return
        self._maybe_cleanup_hosts(now_mono=now_mono)
        with self.state_lock:
            now_mono = time.monotonic()
            if not force and now_mono < self._next_state_write:
                return
            self._maybe_cleanup_hosts(now_mono=now_mono)
            now_wall = time.time()
            data: dict[str, Any] = {
                "saved_at": now_wall,
                "global": self._bucket_state(
                    self.global_bucket, self.global_rps, now_wall, now_mono
                ),
                "per_host_rps": self.per_host_rps,
                "hosts": {},
                "host_overrides": dict(self.host_overrides),
            }
            for host, bucket in list(self.host_buckets.items()):
                data["hosts"][host] = self._bucket_state(
                    bucket, self._host_rate(host), now_wall, now_mono
                )

            tmp_path = self.state_file.with_suffix(".tmp")
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                os.replace(tmp_path, self.state_file)
            except OSError as exc:
                logger.warning(
                    "Failed to persist rate limiter state to %s: %s",
                    self.state_file,
                    exc,
                )
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp_path)
            else:
                self._next_state_write = time.monotonic() + self._state_write_interval

    def _load_state(self) -> None:
        """Load state from persistent storage."""
        if not self.state_file or not self.state_file.exists():
            return

        with self.state_lock:
            data = self._read_state_file()
            if data is None:
                return

            time_context = self._calculate_time_context(data)
            self._restore_global_bucket(data, time_context)
            self._clear_host_state()
            self._restore_host_overrides(data)
            self._restore_per_host_base(data)
            self._restore_host_buckets(data, time_context)

    def _read_state_file(self) -> dict[str, Any] | None:
        """Read and parse state file."""
        try:
            with self.state_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load rate limiter state from %s: %s", self.state_file, exc)
            return None

    def _calculate_time_context(self, data: dict[str, Any]) -> dict[str, float]:
        """Calculate timing context for state restoration."""
        now_wall = time.time()
        now_mono = time.monotonic()
        saved_at_f = self._safe_float(data.get("saved_at"), 0.0)
        base_elapsed = max(0.0, now_wall - saved_at_f) if saved_at_f else 0.0

        return {
            "now_wall": now_wall,
            "now_mono": now_mono,
            "base_elapsed": base_elapsed,
        }

    def _restore_global_bucket(self, data: dict[str, Any], time_context: dict[str, float]) -> None:
        """Restore global bucket from saved state."""
        global_state = data.get("global")
        if not isinstance(global_state, dict):
            return

        rate = self._safe_float(global_state.get("rate"), self.global_rps)
        self._set_global_rate(rate)

        tokens = self._safe_float(global_state.get("tokens"), self.global_bucket.tokens)
        updated_at = self._safe_float(global_state.get("updated"), None)

        elapsed = (
            time_context["base_elapsed"]
            if updated_at is None
            else max(0.0, time_context["now_wall"] - updated_at)
        )

        with self.global_bucket.lock:
            refill = tokens + elapsed * self.global_bucket.rate
            self.global_bucket.tokens = min(self.global_bucket.capacity, refill)
            self.global_bucket.updated = max(time_context["now_mono"] - elapsed, 0.0)

    def _clear_host_state(self) -> None:
        """Clear existing host state."""
        self.host_buckets.clear()
        with self._host_usage_lock:
            self.host_last_used.clear()

    def _restore_host_overrides(self, data: dict[str, Any]) -> None:
        """Restore host-specific rate overrides."""
        overrides = data.get("host_overrides")
        if not isinstance(overrides, dict):
            self.host_overrides = {}
            return

        new_overrides: dict[str, float] = {}
        for host, value in overrides.items():
            safe_value = self._safe_float(value, None)
            if safe_value is not None:
                new_overrides[str(host)] = safe_value
        self.host_overrides = new_overrides

    def _restore_per_host_base(self, data: dict[str, Any]) -> None:
        """Restore per-host base rate."""
        per_host_base = data.get("per_host_rps")
        if per_host_base is not None:
            with contextlib.suppress(TypeError, ValueError):
                self._set_per_host_base(float(per_host_base))

    def _restore_host_buckets(self, data: dict[str, Any], time_context: dict[str, float]) -> None:
        """Restore all host-specific buckets."""
        hosts = data.get("hosts")
        if not isinstance(hosts, dict):
            return

        for host, bucket_data in hosts.items():
            if isinstance(bucket_data, dict):
                self._restore_single_host_bucket(
                    str(host), bucket_data, time_context
                )

    def _restore_single_host_bucket(
        self, host: str, bucket_data: dict[str, Any], time_context: dict[str, float]
    ) -> None:
        """Restore a single host bucket."""
        rate = self._safe_float(bucket_data.get("rate"), self._host_rate(host))
        burst = max(1, int(rate))
        bucket = TokenBucket(rate, burst)

        tokens = self._safe_float(bucket_data.get("tokens"), burst)
        updated_at = self._safe_float(bucket_data.get("updated"), None)

        elapsed = (
            time_context["base_elapsed"]
            if updated_at is None
            else max(0.0, time_context["now_wall"] - updated_at)
        )

        refill = tokens + elapsed * rate
        bucket.tokens = min(bucket.capacity, refill)
        bucket.updated = max(time_context["now_mono"] - elapsed, 0.0)

        self.host_buckets[host] = bucket
        self._touch_host(host, time_context["now_mono"])

    def _safe_float(self, value: Any, default: float | None) -> float | None:
        """Safely convert value to float, return default on failure."""
        try:
            return float(value) if value is not None else default
        except (TypeError, ValueError):
            return default

    def _bucket_state(
        self,
        bucket: TokenBucket,
        rate: float,
        wall_now: float,
        mono_now: float,
    ) -> dict[str, Any]:
        """Capture current bucket state for persistence."""
        with bucket.lock:
            elapsed = max(0.0, mono_now - bucket.updated)
            updated_wall = max(0.0, wall_now - elapsed)
            return {
                "rate": rate,
                "capacity": bucket.capacity,
                "tokens": min(bucket.capacity, bucket.tokens),
                "updated": updated_wall,
            }

    def _set_global_rate(self, rate: float) -> None:
        """Update global rate with bounds checking."""
        clamped = max(self.min_global_rps, min(self.max_global_rps, rate))
        burst = max(1, int(clamped))
        with self.global_bucket.lock:
            self.global_rps = clamped
            self.global_bucket.rate = clamped
            self.global_bucket.capacity = burst
            self.global_bucket.tokens = min(
                self.global_bucket.capacity, self.global_bucket.tokens
            )
            self.global_bucket.updated = time.monotonic()

    def _set_per_host_base(self, rate: float) -> None:
        """Update per-host base rate with bounds checking."""
        clamped = max(self.min_per_host_rps, min(self.max_per_host_rps, rate))
        with self.config_lock:
            self.per_host_rps = clamped
            self.default_host_burst = max(1, int(clamped))
            unaffected = [h for h in self.host_buckets if h not in self.host_overrides]
        for host in unaffected:
            self._apply_rate(self.host_buckets[host], clamped)

    def _scale_global(self, factor: float) -> None:
        """Scale global rate by a factor."""
        if factor <= 0:
            return
        self._set_global_rate(self.global_rps * factor)

    def _scale_per_host_base(self, factor: float) -> None:
        """Scale per-host base rate by a factor."""
        if factor <= 0:
            return
        self._set_per_host_base(self.per_host_rps * factor)

    def _scale_host(self, host: str, factor: float) -> None:
        """Scale specific host rate by a factor."""
        if factor <= 0:
            return
        with self.config_lock:
            current = self.host_overrides.get(host, self.per_host_rps)
            new_rate = max(
                self.min_per_host_rps, min(self.max_per_host_rps, current * factor)
            )
            self.host_overrides[host] = new_rate
        bucket = self._bucket_for(host)
        self._apply_rate(bucket, new_rate)

    def _apply_rate(self, bucket: TokenBucket, rate: float) -> None:
        """Apply new rate to a bucket."""
        burst = max(1, int(rate))
        with bucket.lock:
            bucket.rate = rate
            bucket.capacity = burst
            bucket.tokens = min(bucket.capacity, bucket.tokens)
            bucket.updated = time.monotonic()

    def _touch_host(self, host: str, now_mono: float | None = None) -> None:
        """Update last used timestamp for a host."""
        if now_mono is None:
            now_mono = time.monotonic()
        with self._host_usage_lock:
            self.host_last_used[host] = now_mono

    def _maybe_cleanup_hosts(self, now_mono: float | None = None) -> None:
        """Trigger host cleanup if interval has elapsed."""
        if self.host_ttl is None:
            return
        if now_mono is None:
            now_mono = time.monotonic()
        if (
            self._cleanup_interval
            and self._next_cleanup
            and now_mono < self._next_cleanup
        ):
            return
        self.cleanup_expired_hosts(now_mono=now_mono)
        if self._cleanup_interval:
            self._next_cleanup = now_mono + self._cleanup_interval
        else:
            self._next_cleanup = now_mono

    def cleanup_expired_hosts(
        self,
        ttl_seconds: float | None = None,
        *,
        now_mono: float | None = None,
    ) -> None:
        """Remove hosts that haven't been used recently."""
        ttl = self.host_ttl if ttl_seconds is None else ttl_seconds
        if ttl is None:
            return
        try:
            ttl_val = float(ttl)
        except (TypeError, ValueError):
            return
        if ttl_val <= 0:
            return
        if now_mono is None:
            now_mono = time.monotonic()
        cutoff = now_mono - ttl_val
        with self._host_usage_lock:
            stale_hosts = [
                host for host, seen in self.host_last_used.items() if seen < cutoff
            ]
        if not stale_hosts:
            return
        with self._host_usage_lock:
            for host in stale_hosts:
                self.host_last_used.pop(host, None)
        with self.config_lock:
            for host in stale_hosts:
                self.host_overrides.pop(host, None)
        for host in stale_hosts:
            self.host_buckets.pop(host, None)
