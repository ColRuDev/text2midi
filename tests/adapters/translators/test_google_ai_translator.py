"""
Unit tests for Google AI Translator adapter.

These tests mock the google-genai SDK to verify adapter behavior
without making real API calls.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from adapters.translators.google_ai_translator import (
    GoogleAIConfig,
    GoogleAITranslator,
)
from adapters.exceptions import LLMTranslationError
from domain.entities import Intent


class TestGoogleAIConfig:
    """Tests for GoogleAIConfig dataclass."""

    def test_default_values(self):
        """Config uses correct defaults."""
        config = GoogleAIConfig()

        assert config.model_name == "gemma-4-31b-it"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_output_tokens == 1024
        # Default path is relative to the adapter module
        assert config.system_prompt_path.name == "system_prompt.md"
        assert "prompts" in str(config.system_prompt_path)

    def test_custom_values(self):
        """Config accepts custom values."""
        config = GoogleAIConfig(
            model_name="custom-model",
            temperature=0.5,
            top_p=0.8,
            max_output_tokens=2048,
            system_prompt_path=Path("custom_prompt.md"),
        )

        assert config.model_name == "custom-model"
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.max_output_tokens == 2048
        assert config.system_prompt_path == Path("custom_prompt.md")


class TestGoogleAITranslatorInit:
    """Tests for GoogleAITranslator initialization."""

    @pytest.fixture
    def mock_genai_client(self):
        """Mock the genai.Client class."""
        with patch(
            "adapters.translators.google_ai_translator.genai.Client"
        ) as mock:
            yield mock

    @pytest.fixture
    def temp_sysprompt(self, tmp_path):
        """Create a temporary system prompt file."""
        prompt_file = tmp_path / "sysprompt.md"
        prompt_file.write_text("Test system prompt for MIDI generation.")
        return prompt_file

    @pytest.fixture
    def mock_env_api_key(self, monkeypatch):
        """Set a mock GOOGLE_API_KEY environment variable."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-12345")

    def test_init_loads_system_prompt(
        self, mock_genai_client, temp_sysprompt, mock_env_api_key
    ):
        """Translator loads system prompt at init."""
        config = GoogleAIConfig(system_prompt_path=temp_sysprompt)
        translator = GoogleAITranslator(config)

        assert translator._system_prompt == "Test system prompt for MIDI generation."

    def test_init_missing_system_prompt_raises_filenotfounderror(
        self, mock_genai_client, mock_env_api_key
    ):
        """Missing system prompt file raises FileNotFoundError."""
        config = GoogleAIConfig(system_prompt_path=Path("nonexistent_prompt.md"))

        with pytest.raises(FileNotFoundError) as exc_info:
            GoogleAITranslator(config)

        assert "System prompt not found" in str(exc_info.value)

    def test_init_missing_api_key_raises_error(
        self, temp_sysprompt, monkeypatch
    ):
        """Missing GOOGLE_API_KEY raises LLMTranslationError."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        config = GoogleAIConfig(system_prompt_path=temp_sysprompt)

        with pytest.raises(LLMTranslationError) as exc_info:
            GoogleAITranslator(config)

        assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_init_creates_client_with_api_key(
        self, mock_genai_client, temp_sysprompt, mock_env_api_key
    ):
        """Client is created with the API key from environment."""
        config = GoogleAIConfig(system_prompt_path=temp_sysprompt)
        GoogleAITranslator(config)

        mock_genai_client.assert_called_once_with(api_key="test-api-key-12345")


class TestGoogleAITranslatorTranslate:
    """Tests for the translate method."""

    @pytest.fixture
    def mock_client(self):
        """Create a fully mocked genai client."""
        with patch(
            "adapters.translators.google_ai_translator.genai.Client"
        ) as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def temp_sysprompt(self, tmp_path):
        """Create a temporary system prompt file."""
        prompt_file = tmp_path / "sysprompt.md"
        prompt_file.write_text("Test system prompt.")
        return prompt_file

    @pytest.fixture
    def translator(self, mock_client, temp_sysprompt):
        """Create a translator with mocked client."""
        config = GoogleAIConfig(system_prompt_path=temp_sysprompt)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            return GoogleAITranslator(config)

    def test_translate_returns_list_of_prompts(self, translator, mock_client):
        """Happy path: translate returns list of prompt strings."""
        mock_response = Mock()
        mock_response.text = "tempo:80 key:C_major instruments:piano"
        mock_client.models.generate_content.return_value = mock_response

        result = translator.translate(Intent("A sunrise at the beach"), 3)

        assert len(result) == 3
        assert all(isinstance(p, str) for p in result)
        assert all(p == "tempo:80 key:C_major instruments:piano" for p in result)

    def test_zero_variations_returns_empty_list(self, translator):
        """Edge case: num_variations=0 returns empty list without API call."""
        result = translator.translate(Intent("test"), 0)

        assert result == []

    def test_translate_makes_correct_api_calls(
        self, translator, mock_client, temp_sysprompt
    ):
        """API calls are made with correct parameters."""
        mock_response = Mock()
        mock_response.text = "generated prompt"
        mock_client.models.generate_content.return_value = mock_response

        intent = Intent("A melancholic piano melody")
        translator.translate(intent, 2)

        # Should be called twice (for 2 variations)
        assert mock_client.models.generate_content.call_count == 2

        # Check the call arguments
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "gemma-4-31b-it"
        assert call_args.kwargs["contents"] == "A melancholic piano melody"

    def test_translate_wraps_sdk_exceptions(self, translator, mock_client):
        """SDK exceptions are wrapped in LLMTranslationError."""
        mock_client.models.generate_content.side_effect = Exception(
            "Network error"
        )

        with pytest.raises(LLMTranslationError) as exc_info:
            translator.translate(Intent("test"), 1)

        assert "LLM translation failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_translate_uses_custom_config(self, mock_client, temp_sysprompt):
        """Custom config values are passed to API calls."""
        config = GoogleAIConfig(
            model_name="custom-model",
            temperature=0.5,
            top_p=0.8,
            max_output_tokens=2048,
            system_prompt_path=temp_sysprompt,
        )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            translator = GoogleAITranslator(config)

        mock_response = Mock()
        mock_response.text = "result"
        mock_client.models.generate_content.return_value = mock_response

        translator.translate(Intent("test"), 1)

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == "custom-model"

        # Check config was passed to GenerateContentConfig
        gen_config = call_args.kwargs["config"]
        assert gen_config.temperature == 0.5
        assert gen_config.top_p == 0.8
        assert gen_config.max_output_tokens == 2048

    def test_negative_variations_raises_valueerror(self, translator):
        """Edge case: num_variations < 0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            translator.translate(Intent("test"), -1)

        assert "num_variations must be >= 0" in str(exc_info.value)

    def test_empty_response_raises_llmtranslationerror(self, translator, mock_client):
        """Empty or blocked LLM response raises LLMTranslationError."""
        mock_response = Mock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        with pytest.raises(LLMTranslationError) as exc_info:
            translator.translate(Intent("test"), 1)

        assert "Empty or blocked response" in str(exc_info.value)


class TestGoogleAITranslatorIntegration:
    """Integration-style tests with mocked SDK."""

    @pytest.fixture
    def temp_sysprompt(self, tmp_path):
        """Create a temporary system prompt file."""
        prompt_file = tmp_path / "sysprompt.md"
        prompt_file.write_text(
            "You are a MIDI prompt generator. "
            "Convert user intent to technical music parameters."
        )
        return prompt_file

    def test_full_translation_workflow(self, temp_sysprompt):
        """Test the complete translation workflow with mocked SDK."""
        with patch(
            "adapters.translators.google_ai_translator.genai.Client"
        ) as mock_client_class:
            # Setup mock client
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Mock responses for 3 variations
            responses = [
                Mock(text=f"tempo:80 key:C_major instruments:piano variation{i}")
                for i in range(3)
            ]
            mock_client.models.generate_content.side_effect = responses

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                config = GoogleAIConfig(system_prompt_path=temp_sysprompt)
                translator = GoogleAITranslator(config)

                result = translator.translate(Intent("A peaceful sunrise"), 3)

            assert len(result) == 3
            assert "variation0" in result[0]
            assert "variation1" in result[1]
            assert "variation2" in result[2]
