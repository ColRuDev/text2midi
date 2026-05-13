"""
Pytest configuration for integration tests.

Loads environment variables from .env file before running tests.
"""

import os
from pathlib import Path


def pytest_configure(config):
    """Load .env file before running tests."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key, value)
