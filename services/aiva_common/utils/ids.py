import uuid


def new_id(prefix: str) -> str:
    """Generate a short unique identifier with a given prefix.
    
    Args:
        prefix: The prefix for the ID (e.g., 'finding', 'task', 'scan')
        
    Returns:
        A unique ID in the format: prefix_uuid (with underscore)
        
    Examples:
        >>> new_id('finding')  # Returns: finding_a1b2c3d4-...
        >>> new_id('task')     # Returns: task_a1b2c3d4-...
    """
    return f"{prefix}_{uuid.uuid4()}"
