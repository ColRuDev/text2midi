"""
Unit tests for adapter exceptions.

Tests for custom exception types used across adapters.
"""

import pytest

from adapters.exceptions import (
    ConfigurationError,
    GeneratorError,
    LLMTranslationError,
)


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_configuration_error_is_exception(self):
        """ConfigurationError is a proper exception."""
        error = ConfigurationError("Test error message")
        
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"

    def test_configuration_error_can_be_raised(self):
        """ConfigurationError can be raised and caught."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Missing configuration")
        
        assert "Missing configuration" in str(exc_info.value)

    def test_configuration_error_with_cause(self):
        """ConfigurationError preserves cause chain."""
        original = ValueError("Original error")
        
        try:
            raise original
        except ValueError as e:
            error = ConfigurationError("Wrapped error")
            error.__cause__ = e
            
            assert error.__cause__ is original
            assert isinstance(error.__cause__, ValueError)

    def test_configuration_error_message_format(self):
        """ConfigurationError formats message correctly."""
        error = ConfigurationError("fluidsynth executable not found")
        
        assert "fluidsynth" in str(error)
        assert "not found" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_exception(self):
        """All custom exceptions inherit from Exception."""
        assert issubclass(ConfigurationError, Exception)
        assert issubclass(GeneratorError, Exception)
        assert issubclass(LLMTranslationError, Exception)

    def test_exceptions_are_distinct(self):
        """Each exception type is distinct."""
        assert ConfigurationError is not GeneratorError
        assert ConfigurationError is not LLMTranslationError
        assert GeneratorError is not LLMTranslationError
