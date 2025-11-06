"""
Resource ID Extractor

Extracts resource identifiers from URLs and generates test variations.
Supports multiple ID patterns: numeric, UUID, hash, mixed.
"""



from dataclasses import dataclass
import re
from typing import Literal

IdPattern = Literal["numeric", "uuid", "hash", "mixed", "unknown"]


@dataclass
class ResourceId:
    """
    Resource identifier information extracted from URL.

    Attributes:
        value: The actual ID value (e.g., "123", "abc-def-456")
        pattern: The detected pattern type
        position: Where the ID was found ("path", "query", "body")
        parameter_name: Parameter name if found in query/body, None otherwise
    """

    value: str
    pattern: IdPattern
    position: Literal["path", "query", "body"]
    parameter_name: str | None = None


class ResourceIdExtractor:
    """
    Extracts resource identifiers from URLs and generates test variations.

    Supports detection of:
    - Numeric IDs (e.g., /users/123)
    - UUIDs (e.g., /orders/a1b2c3d4-e5f6-7890-abcd-1234567890ab)
    - Hashes (e.g., /sessions/5f4dcc3b5aa765d61d8327deb882cf99)
    - Mixed alphanumeric (e.g., /items/abc123def456)

    Example:
        >>> extractor = ResourceIdExtractor()
        >>> ids = extractor.extract_from_url("https://api.example.com/users/123")
        >>> print(ids[0].value)
        '123'
        >>> print(ids[0].pattern)
        'numeric'
    """

    # ID pattern regular expressions
    PATTERNS: dict[IdPattern, str] = {
        "numeric": r"\b\d{1,19}\b",  # Pure numeric ID (1-19 digits)
        "uuid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "hash": r"\b[0-9a-f]{32,64}\b",  # MD5 (32) or SHA256 (64)
        "mixed": r"\b[a-zA-Z0-9_-]{8,}\b",  # Mixed alphanumeric
    }

    def extract_from_url(self, url: str) -> list[ResourceId]:
        """
        Extract resource IDs from URL path and query parameters.

        Args:
            url: The URL to extract IDs from

        Returns:
            List of ResourceId objects found in the URL

        Example:
            >>> extractor = ResourceIdExtractor()
            >>> ids = extractor.extract_from_url("https://api.example.com/users/123?order=456")
            >>> len(ids)
            2
            >>> ids[0].value
            '123'
            >>> ids[1].value
            '456'
        """
        ids: list[ResourceId] = []

        # Extract from path
        path = url.split("?")[0]
        path_parts = path.split("/")

        for part in path_parts:
            if not part:
                continue

            # Try each pattern
            for pattern_name, regex in self.PATTERNS.items():
                if re.fullmatch(regex, part, re.IGNORECASE):
                    ids.append(
                        ResourceId(value=part, pattern=pattern_name, position="path")
                    )
                    break

        # Extract from query parameters
        if "?" in url:
            query = url.split("?", 1)[1]
            for param in query.split("&"):
                if "=" not in param:
                    continue

                key, value = param.split("=", 1)

                # Try each pattern
                for pattern_name, regex in self.PATTERNS.items():
                    if re.fullmatch(regex, value, re.IGNORECASE):
                        ids.append(
                            ResourceId(
                                value=value,
                                pattern=pattern_name,
                                position="query",
                                parameter_name=key,
                            )
                        )
                        break

        return ids

    def generate_test_ids(self, original_id: ResourceId, count: int = 5) -> list[str]:
        """
        Generate test ID variations based on the original ID pattern.

        Args:
            original_id: The original ResourceId to base variations on
            count: Maximum number of test IDs to generate (default: 5)

        Returns:
            List of test ID strings

        Example:
            >>> extractor = ResourceIdExtractor()
            >>> rid = ResourceId(value="123", pattern="numeric", position="path")
            >>> test_ids = extractor.generate_test_ids(rid, count=3)
            >>> len(test_ids)
            3
            >>> int(test_ids[0]) != 123
            True
        """
        test_ids: list[str] = []

        if original_id.pattern == "numeric":
            # Numeric ID: generate nearby numbers
            try:
                base = int(original_id.value)
                # Generate IDs with various offsets
                offsets = [-2, -1, 1, 2, 10, 100, 1000]
                for offset in offsets:
                    new_id = base + offset
                    if new_id > 0:  # Ensure positive ID
                        test_ids.append(str(new_id))
                    if len(test_ids) >= count:
                        break
            except ValueError:
                pass

        elif original_id.pattern == "uuid":
            # UUID: modify parts of the UUID
            import random

            parts = original_id.value.split("-")
            for _ in range(count):
                # Modify the first segment
                modified = parts.copy()
                modified[0] = f"{random.randint(0, 0xFFFFFFFF):08x}"
                test_ids.append("-".join(modified))

        elif original_id.pattern == "hash":
            # Hash: generate random hashes of same length
            import hashlib
            import random

            length = len(original_id.value)
            for i in range(count):
                # Generate random hash
                random_str = f"test_{random.randint(0, 999999)}_{i}"
                if length == 32:  # MD5
                    test_hash = hashlib.md5(random_str.encode()).hexdigest()
                else:  # SHA256
                    test_hash = hashlib.sha256(random_str.encode()).hexdigest()
                test_ids.append(test_hash)

        elif original_id.pattern == "mixed":
            # Mixed: increment numbers in the string
            import random

            unique_ids: set[str] = set()
            attempts = 0
            max_attempts = count * 10  # Prevent infinite loop

            while len(unique_ids) < count and attempts < max_attempts:
                result = ""
                for char in original_id.value:
                    if char.isdigit():
                        result += str(random.randint(0, 9))
                    else:
                        result += char
                if result != original_id.value:
                    unique_ids.add(result)
                attempts += 1

            test_ids.extend(list(unique_ids))

        return test_ids[:count]

    def replace_id_in_url(self, url: str, old_id: str, new_id: str) -> str:
        """
        Replace an ID in URL with a new ID.

        Args:
            url: Original URL
            old_id: ID to replace
            new_id: New ID value

        Returns:
            URL with replaced ID

        Example:
            >>> extractor = ResourceIdExtractor()
            >>> url = "https://api.example.com/users/123"
            >>> new_url = extractor.replace_id_in_url(url, "123", "456")
            >>> "456" in new_url
            True
        """
        # Replace only exact path segments or query parameter values matching old_id
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        parsed = urlparse(url)
        # Replace in path segments
        path_parts = parsed.path.split("/")
        path_parts = [new_id if part == old_id else part for part in path_parts]
        new_path = "/".join(path_parts)

        # Replace in query parameters
        query_params = parse_qsl(parsed.query, keep_blank_values=True)
        new_query_params = [(k, new_id if v == old_id else v) for k, v in query_params]
        new_query = urlencode(new_query_params)

        # Reconstruct the URL
        new_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                new_path,
                parsed.params,
                new_query,
                parsed.fragment,
            )
        )
        return new_url
