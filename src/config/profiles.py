"""
Configuration Profiles - Predefined GenerationProfile instances.

This module provides predefined configuration profiles for different
use cases:
- ONE_SHOT: Quick generation with minimal exploration
- BALANCED: Default profile with moderate exploration
- DEEP_SEARCH: Maximum quality with extensive exploration

Architecture:
    - Pure configuration module (no dependencies on adapters)
    - Uses GenerationProfile from domain.entities
    - Provides factory functions to return fresh profile instances
    - Prevents accidental mutation of global profile objects
"""

from domain.entities import GenerationProfile


def one_shot() -> GenerationProfile:
    """
    Create a ONE_SHOT profile - Quick generation for fast results.
    
    Low token count, fewer beams - optimized for speed.
    
    Returns:
        A fresh GenerationProfile instance.
    """
    return GenerationProfile(
        token_batch_size=300,
        num_beams=2,
        top_k=1,
        max_tokens=600,
        clap_weight=0.3,
        key_weight=0.35,
        note_weight=0.35,
        strict_instruments=False,
    )


def balanced() -> GenerationProfile:
    """
    Create a BALANCED profile - Default with moderate exploration.
    
    Good balance between quality and speed.
    
    Returns:
        A fresh GenerationProfile instance.
    """
    return GenerationProfile(
        token_batch_size=500,
        num_beams=5,
        top_k=2,
        max_tokens=2000,
        clap_weight=0.4,
        key_weight=0.3,
        note_weight=0.3,
        strict_instruments=False,
    )


def deep_search() -> GenerationProfile:
    """
    Create a DEEP_SEARCH profile - Maximum quality with extensive exploration.
    
    More tokens, more beams - optimized for quality.
    
    Returns:
        A fresh GenerationProfile instance.
    """
    return GenerationProfile(
        token_batch_size=500,
        num_beams=8,
        top_k=3,
        max_tokens=4000,
        clap_weight=0.5,
        key_weight=0.25,
        note_weight=0.25,
        strict_instruments=False,
    )


# Dictionary mapping profile names to factory functions
PROFILE_FACTORIES = {
    "one-shot": one_shot,
    "balanced": balanced,
    "deep-search": deep_search,
}


def get_profile(name: str) -> GenerationProfile:
    """
    Get a profile by name.
    
    Args:
        name: Profile name ("one-shot", "balanced", or "deep-search").
    
    Returns:
        A fresh GenerationProfile instance.
    
    Raises:
        KeyError: If profile name is not found.
    """
    if name not in PROFILE_FACTORIES:
        raise KeyError(
            f"Unknown profile: {name}. "
            f"Available profiles: {list(PROFILE_FACTORIES.keys())}"
        )
    return PROFILE_FACTORIES[name]()


# Backwards compatibility: module-level profile instances
# DEPRECATED: Use factory functions instead to avoid mutation issues
ONE_SHOT = one_shot()
BALANCED = balanced()
DEEP_SEARCH = deep_search()

# Dictionary mapping profile names to profile instances
# DEPRECATED: Use PROFILE_FACTORIES and get_profile() instead
PROFILES = {
    "one-shot": ONE_SHOT,
    "balanced": BALANCED,
    "deep-search": DEEP_SEARCH,
}
