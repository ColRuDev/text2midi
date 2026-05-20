"""
Tests for configuration profiles.

Tests validate that predefined profiles exist and can be loaded,
and that PRD 09 strict_instruments is properly configured.
"""

import unittest

from domain.entities import GenerationProfile


class TestPredefinedProfiles(unittest.TestCase):
    """Test suite for predefined configuration profiles."""

    def test_one_shot_profile_exists_and_is_generation_profile(self):
        """
        AC1: ONE_SHOT profile must exist and be a GenerationProfile instance.
        """
        from config.profiles import ONE_SHOT

        self.assertIsInstance(ONE_SHOT, GenerationProfile)

    def test_balanced_profile_exists_and_is_generation_profile(self):
        """
        AC1: BALANCED profile must exist and be a GenerationProfile instance.
        """
        from config.profiles import BALANCED

        self.assertIsInstance(BALANCED, GenerationProfile)

    def test_deep_search_profile_exists_and_is_generation_profile(self):
        """
        AC1: DEEP_SEARCH profile must exist and be a GenerationProfile instance.
        """
        from config.profiles import DEEP_SEARCH

        self.assertIsInstance(DEEP_SEARCH, GenerationProfile)

    def test_one_shot_has_lower_token_count_than_balanced(self):
        """
        ONE_SHOT should be faster (fewer tokens) than BALANCED.
        """
        from config.profiles import BALANCED, ONE_SHOT

        self.assertLess(ONE_SHOT.max_tokens, BALANCED.max_tokens)

    def test_deep_search_has_higher_token_count_than_balanced(self):
        """
        DEEP_SEARCH should explore more (more tokens) than BALANCED.
        """
        from config.profiles import BALANCED, DEEP_SEARCH

        self.assertGreater(DEEP_SEARCH.max_tokens, BALANCED.max_tokens)


class TestProfilesStrictInstruments(unittest.TestCase):
    """Test suite for strict_instruments in predefined profiles (PRD 09)."""

    def test_one_shot_has_strict_instruments_false(self):
        """
        PRD 09: ONE_SHOT profile must have strict_instruments=False by default.
        """
        from config.profiles import ONE_SHOT

        self.assertFalse(ONE_SHOT.strict_instruments)

    def test_balanced_has_strict_instruments_false(self):
        """
        PRD 09: BALANCED profile must have strict_instruments=False by default.
        """
        from config.profiles import BALANCED

        self.assertFalse(BALANCED.strict_instruments)

    def test_deep_search_has_strict_instruments_false(self):
        """
        PRD 09: DEEP_SEARCH profile must have strict_instruments=False by default.
        """
        from config.profiles import DEEP_SEARCH

        self.assertFalse(DEEP_SEARCH.strict_instruments)


class TestProfilesDictionary(unittest.TestCase):
    """Test suite for the PROFILES dictionary."""

    def test_profiles_dict_contains_one_shot(self):
        """
        AC2: PROFILES dict must contain 'one-shot' key.
        """
        from config.profiles import PROFILES

        self.assertIn("one-shot", PROFILES)

    def test_profiles_dict_contains_balanced(self):
        """
        AC2: PROFILES dict must contain 'balanced' key.
        """
        from config.profiles import PROFILES

        self.assertIn("balanced", PROFILES)

    def test_profiles_dict_contains_deep_search(self):
        """
        AC2: PROFILES dict must contain 'deep-search' key.
        """
        from config.profiles import PROFILES

        self.assertIn("deep-search", PROFILES)

    def test_profiles_values_are_generation_profiles(self):
        """
        All PROFILES values must be GenerationProfile instances.
        """
        from config.profiles import PROFILES

        for key, profile in PROFILES.items():
            with self.subTest(key=key):
                self.assertIsInstance(profile, GenerationProfile)

    def test_profiles_dict_has_three_entries(self):
        """
        PROFILES should have exactly 3 entries.
        """
        from config.profiles import PROFILES

        self.assertEqual(len(PROFILES), 3)


if __name__ == "__main__":
    unittest.main()
