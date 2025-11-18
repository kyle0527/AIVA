

from services.aiva_common.schemas import Authentication


class AuthenticationManager:
    """Supply authentication contexts to HTTP client (e.g., headers/cookies)."""

    def __init__(self, auth: Authentication) -> None:
        self.auth = auth
