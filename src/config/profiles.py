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
    - Provides PROFILES dictionary for easy lookup by name
"""

from domain.entities import GenerationProfile

# ONE_SHOT: Quick generation for fast results
# Low token count, fewer beams - optimized for speed
ONE_SHOT = GenerationProfile(
    token_batch_size=300,
    num_beams=2,
    top_k=1,
    max_tokens=600,
    clap_weight=0.3,
    key_weight=0.35,
    note_weight=0.35,
    strict_instruments=False,
)

# BALANCED: Default profile with moderate exploration
# Good balance between quality and speed
BALANCED = GenerationProfile(
    token_batch_size=500,
    num_beams=5,
    top_k=2,
    max_tokens=2000,
    clap_weight=0.4,
    key_weight=0.3,
    note_weight=0.3,
    strict_instruments=False,
)

# DEEP_SEARCH: Maximum quality with extensive exploration
# More tokens, more beams - optimized for quality
DEEP_SEARCH = GenerationProfile(
    token_batch_size=500,
    num_beams=8,
    top_k=3,
    max_tokens=4000,
    clap_weight=0.5,
    key_weight=0.25,
    note_weight=0.25,
    strict_instruments=False,
)

# Dictionary mapping profile names to profile instances
PROFILES = {
    "one-shot": ONE_SHOT,
    "balanced": BALANCED,
    "deep-search": DEEP_SEARCH,
}
