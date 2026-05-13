"""
Adapters layer - Infrastructure implementations of domain interfaces.

This module contains all adapters that implement domain interfaces.
Adapters handle external I/O (APIs, databases, file systems) while
depending only on domain interfaces for testability.

Hexagonal Architecture: Adapters depend inward on domain, never outward.
"""
