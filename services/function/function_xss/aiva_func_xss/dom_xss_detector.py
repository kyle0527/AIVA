from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomDetectionResult:
    """Represents a DOM based trigger that was identified."""

    payload: str
    snippet: str


class DomXssDetector:
    """Lightweight DOM-based XSS detector.

    The implementation favours portability – it operates on raw HTML documents
    returned by HTTP requests and attempts to infer whether the provided
    payload was evaluated in a DOM context.  This is not a full browser
    execution engine, but it gives the worker an automated way to elevate the
    confidence of findings that echo payloads inside executable contexts such
    as script blocks or event handler attributes.
    """

    def analyze(self, *, payload: str, document: str) -> DomDetectionResult | None:
        """Return a DOM detection result when the payload is wrapped in script context."""

        if not payload or not document:
            return None

        index = document.find(payload)
        if index == -1:
            return None

        window_start = max(0, index - 160)
        window_end = min(len(document), index + len(payload) + 160)
        window = document[window_start:window_end]
        lower_window = window.lower()

        if (
            "<script" in lower_window
            or "onload=" in lower_window
            or "onclick=" in lower_window
        ):
            snippet = window.strip()
            return DomDetectionResult(payload=payload, snippet=snippet)

        return None
