"""Unit tests for process_in_parallel: deterministic, order-preserving results."""

from utils.parallel import process_in_parallel


def _square(value: int) -> int:
    """Return ``value`` squared (a top-level, picklable worker for the process pool)."""
    return value * value


def test_sequential_and_parallel_results_match_and_are_ordered():
    """num_workers=1 and num_workers=4 give identical, input-ordered results."""
    items = list(range(20))
    expected = [v * v for v in items]

    sequential = process_in_parallel(_square, items, num_workers=1)
    parallel = process_in_parallel(_square, items, num_workers=4)

    assert sequential == expected
    assert parallel == expected
    assert parallel == sequential


def test_empty_input_returns_empty():
    """An empty item list returns an empty result for any worker count."""
    assert process_in_parallel(_square, [], num_workers=1) == []
    assert process_in_parallel(_square, [], num_workers=4) == []
