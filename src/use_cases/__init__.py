"""
Use Cases Layer - Application-specific business rules.

This layer contains orchestrators that coordinate domain objects
and interfaces to accomplish user-facing tasks.

Use cases are:
- Pure Python with no external dependencies
- Framework-agnostic
- Focused on a single user goal
- Orchestrators, not implementers
"""

from .best_of_n_search import BestOfNSearch
from .progressive_search import ProgressiveSearch

__all__ = ["BestOfNSearch", "ProgressiveSearch"]
