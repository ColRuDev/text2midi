"""
Adapter-specific exceptions for domain isolation.

These exception types wrap external SDK errors, allowing the domain
layer to handle failures without depending on specific infrastructure.
"""


class LLMTranslationError(Exception):
    """
    Raised when LLM translation fails due to SDK or configuration errors.

    This exception wraps all errors from the LLM SDK layer, providing
    domain isolation and consistent error handling.

    Attributes:
        message: Human-readable error description.
        __cause__: The original exception that triggered this error.

    Example:
        >>> try:
        ...     response = client.models.generate_content(...)
        ... except Exception as e:
        ...     raise LLMTranslationError(f"Translation failed: {e}") from e
    """

    def __init__(self, message: str):
        super().__init__(message)
