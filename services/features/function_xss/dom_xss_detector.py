

from dataclasses import dataclass
import re


@dataclass
class DomDetectionResult:
    """Represents a DOM based trigger that was identified."""
    payload: str
    snippet: str
    source: str
    sink: str
    confidence: str # High, Medium, Low


class DomXssDetector:
    """
    Native DOM-based XSS detector using Static Analysis.

    Identifies potential DOM XSS vulnerabilities by scanning for dangerous
    Source-to-Sink patterns in the JavaScript code within the HTML document.
    """

    def __init__(self):
        # Common DOM XSS Sources (Inputs)
        self.sources_regex = r"(location\.(hash|search|href|pathname)|document\.(URL|referrer|cookie)|window\.name)"

        # Common DOM XSS Sinks (Execution Points)
        self.sinks_regex = r"(innerHTML|outerHTML|document\.write|document\.writeln|eval\(|setTimeout\(|setInterval\(|Function\(|execScript\()"

    def analyze(self, *, payload: str, document: str) -> DomDetectionResult | None:
        """
        Analyze the document for DOM XSS vulnerabilities.

        It looks for:
        1. Presence of the payload in a script context (basic reflection).
        2. Dangerous Source-to-Sink data flows (static analysis).
        """

        if not document:
            return None

        # 1. Basic Reflection Check (Is payload in script?)
        # This is weak but still useful for verifying if input reached JS
        if payload and payload in document:
            # Check if it's inside a <script> block
            script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", document, re.DOTALL | re.IGNORECASE)
            for block in script_blocks:
                if payload in block:
                    return DomDetectionResult(
                        payload=payload,
                        snippet=block[:100] + "...",
                        source="Reflected in Script",
                        sink="Unknown",
                        confidence="Medium"
                    )

        # 2. Static Source-to-Sink Analysis
        # Check if the document contains code that reads from a Source and writes to a Sink
        # This is a heuristic scan

        # Extract all scripts
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", document, re.DOTALL | re.IGNORECASE)
        # Also check event handlers
        event_handlers = re.findall(r"on\w+\s*=\s*['\"](.*?)['\"]", document, re.IGNORECASE)

        all_js = scripts + event_handlers

        for js_code in all_js:
            # Check for Source
            source_match = re.search(self.sources_regex, js_code)
            # Check for Sink
            sink_match = re.search(self.sinks_regex, js_code)

            if source_match and sink_match:
                # Found both Source and Sink in the same block.
                # Ideally we check if they are connected, but that requires AST.
                # For now, proximity check or just co-existence is a "Potential" finding.

                # Simple check: is Source passed to Sink? e.g. eval(location.hash)
                # This is very naive but catches obvious cases

                # Removing newlines for simple regex
                js_linear = js_code.replace('\n', ' ')

                # Check for direct flow attempt (very rough)
                if re.search(rf"{sink_match.group(1)}.*{source_match.group(1)}", js_linear):
                     return DomDetectionResult(
                        payload="(Static Analysis)",
                        snippet=js_code[:200].strip(),
                        source=source_match.group(1),
                        sink=sink_match.group(1),
                        confidence="Medium"
                    )

        return None
