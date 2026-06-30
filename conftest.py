"""Pytest sees this file automatically and runs it before collecting tests.

Puts src/ on sys.path so test files can import the analysis modules directly
(e.g. `from calibrate_judge import parse_tail`) without needing a packaged layout.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
