"""
Retry and backoff utilities for network requests.

Integrated from Warprecon-M project for AIVA Platform.
Provides intelligent retry mechanisms with exponential backoff to improve
network resilience and handle transient errors.

Features:
- Jittered exponential backoff to avoid thundering herd
- RetryingAsyncClient: Drop-in replacement for httpx.AsyncClient
- Configurable retry attempts and exceptions
- Custom backoff strategies

Example:
    from services.aiva_common.utils.network import RetryingAsyncClient, jitter_backoff

    client = RetryingAsyncClient(
        retries=3,
        backoff=jitter_backoff,
        timeout=10.0
    )

    response = await client.get("https://example.com/api")
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import logging
import random
from typing import Any

import httpx

logger = logging.getLogger(__name__)


BackoffCallable = Callable[..., float]


def jitter_backoff(
    base: float = 0.5,
    factor: float = 2.0,
    max_time: float = 8.0,
    attempt: int = 0,
) -> float:
    """
    Return a jittered exponential delay for the given attempt.

    Jitter helps prevent thundering herd problem when multiple clients
    retry simultaneously.

    Args:
        base: Base delay in seconds
        factor: Exponential backoff factor
        max_time: Maximum delay cap in seconds
        attempt: Current retry attempt number (0-indexed)

    Returns:
        Delay in seconds with random jitter

    Example:
        >>> jitter_backoff(attempt=0)  # ~0.25-0.5s
        >>> jitter_backoff(attempt=1)  # ~0.5-1.0s
        >>> jitter_backoff(attempt=2)  # ~1.0-2.0s
    """
    t = min(max_time, base * (factor**attempt))
    return t * (0.5 + random.random())


def _resolve_backoff(backoff: BackoffCallable | None, attempt: int) -> float:
    """
    Evaluate backoff callable for attempt and normalize the delay.

    Args:
        backoff: Backoff function or None
        attempt: Current retry attempt number

    Returns:
        Normalized delay in seconds (>= 0.0)
    """
    if backoff is None:
        return 0.0

    try:
        delay = backoff(attempt=attempt)
    except TypeError:
        # Fallback for callables that accept positional-only parameter
        delay = backoff(attempt)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "Backoff callable %r failed on attempt %s: %s", backoff, attempt, exc
        )
        return 0.0

    try:
        return max(0.0, float(delay))
    except (TypeError, ValueError):
        logger.warning(
            "Backoff callable %r returned non-numeric delay %r on attempt %s",
            backoff,
            delay,
            attempt,
        )
        return 0.0


class RetryingAsyncClient(httpx.AsyncClient):
    """
    httpx.AsyncClient with automatic retry and backoff support.

    This is a drop-in replacement for httpx.AsyncClient that adds intelligent
    retry logic with exponential backoff. Particularly useful for handling
    transient network errors.

    Features:
    - Configurable retry attempts
    - Customizable backoff strategy
    - Selective exception retry
    - Detailed logging

    Example:
        client = RetryingAsyncClient(
            retries=3,
            backoff=jitter_backoff,
            retry_on=(httpx.RequestError, httpx.TimeoutException)
        )

        try:
            response = await client.get("https://example.com")
        except httpx.RequestError:
            # Failed after all retries
            pass
    """

    def __init__(
        self,
        *args: Any,
        retries: int = 0,
        backoff: BackoffCallable | None = None,
        retry_on: Sequence[type[BaseException]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize RetryingAsyncClient.

        Args:
            *args: Positional arguments for httpx.AsyncClient
            retries: Maximum number of retry attempts (default: 0)
            backoff: Backoff function (default: jitter_backoff)
            retry_on: Exceptions to retry on (default: httpx.RequestError)
            **kwargs: Keyword arguments for httpx.AsyncClient
        """
        super().__init__(*args, **kwargs)
        self._max_retries = max(0, int(retries))
        self._backoff = backoff if backoff is not None else jitter_backoff
        if retry_on is None:
            retry_excs: Sequence[type[BaseException]] = (httpx.RequestError,)
        else:
            retry_excs = tuple(retry_on)
        self._retry_exceptions: tuple[type[BaseException], ...] = tuple(retry_excs)

    async def request(
        self, method: str, url: httpx.URL | str, **kwargs: Any
    ) -> httpx.Response:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            **kwargs: Additional request parameters

        Returns:
            HTTP response

        Raises:
            Exception from retry_on list if all retries exhausted
        """
        attempt = 0
        while True:
            try:
                return await super().request(method, url, **kwargs)
            except self._retry_exceptions as exc:
                if attempt >= self._max_retries:
                    raise

                delay = _resolve_backoff(self._backoff, attempt)
                attempt += 1
                if delay > 0:
                    logger.debug(
                        "Retrying %s %s in %.3fs after %s (attempt %s/%s)",
                        method,
                        url,
                        delay,
                        exc,
                        attempt,
                        self._max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.debug(
                        "Retrying %s %s immediately after %s (attempt %s/%s)",
                        method,
                        url,
                        exc,
                        attempt,
                        self._max_retries,
                    )
