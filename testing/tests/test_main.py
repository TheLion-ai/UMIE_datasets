"""
test_main.

Objective:
This test checks whether main.py runs correctly.
"""

import pytest


def test_main_no_exceptions():
    """Test that checks if main.py doesn't raise any exceptions."""
    try:
        import main
    except Exception as e:
        pytest.fail(f'Trying to import main.py raised an exception: "{e}"')
