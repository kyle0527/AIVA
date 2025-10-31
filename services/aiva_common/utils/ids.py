import uuid


def new_id(prefix: str) -> str:
    """Generate a short unique identifier with a given prefix."""
    return f"{prefix}-{uuid.uuid4()}"
