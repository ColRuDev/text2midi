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


class ConfigurationError(Exception):
    """
    Raised when a required configuration or dependency is missing.

    This exception wraps configuration-related errors, providing
    domain isolation and consistent error handling.

    Attributes:
        message: Human-readable error description.
        __cause__: The original exception that triggered this error.

    Example:
        >>> if not shutil.which("fluidsynth"):
        ...     raise ConfigurationError("fluidsynth executable not found")
    """

    def __init__(self, message: str):
        super().__init__(message)


class GeneratorError(Exception):
    """
    Raised when MIDI generation fails due to model or inference errors.

    This exception wraps all errors from the generator adapter layer,
    providing domain isolation and consistent error handling.

    Attributes:
        message: Human-readable error description.
        __cause__: The original exception that triggered this error.

    Example:
        >>> try:
        ...     model = AutoModelForCausalLM.from_pretrained(path)
        ... except Exception as e:
        ...     raise GeneratorError(f"Model loading failed: {e}") from e
    """

    def __init__(self, message: str):
        super().__init__(message)
