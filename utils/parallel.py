"""Deterministic parallel map helper for the opt-in multiprocessing steps (Theme H, Task 29).

:func:`process_in_parallel` maps a function over a sequence of items, optionally across a
process pool, while guaranteeing the result order matches the input order regardless of the
worker count. It is the single concurrency primitive the pipeline steps should use; the
number of workers is driven by :attr:`base.pipeline.ExportConfig.num_workers`.

Concurrency contract (read before using a worker count > 1):
    * Worker functions must be *pure*: they take one item and RETURN data. They must not
      mutate shared state and must not write to the shared JSONL or any shared output file.
      With a process pool each worker runs in its own process, so concurrent writes to a
      single JSONL would interleave and clobber records.
    * The MAIN process owns all serial JSONL / file writes. The intended pattern is to do
      the CPU-bound, side-effect-free transform in the workers, collect the ordered results
      here, then have the caller (in the main process) write them out sequentially.
    * Because results are returned in input order and writes happen serially in the main
      process, output is byte-identical regardless of ``num_workers``.
    * Worker functions and their arguments must be picklable (define them at module top
      level, not as closures/lambdas), as required by :mod:`multiprocessing`.

A wall-clock benchmark on a large real dataset is a follow-up (it needs real data); this
module only establishes the deterministic, order-preserving contract.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Callable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def process_in_parallel(
    func: Callable[[T], R],
    items: Sequence[T],
    num_workers: int = 1,
    ordered: bool = True,
) -> List[R]:
    """Map ``func`` over ``items``, optionally in parallel, preserving input order.

    When ``num_workers <= 1`` the items are processed sequentially in the current process
    (no pool is created). When ``num_workers > 1`` a :class:`ProcessPoolExecutor` runs the
    work across processes; with ``ordered=True`` the returned list is in the same order as
    ``items`` no matter how many workers are used (deterministic).

    Args:
        func (Callable[[T], R]): A pure, picklable function applied to each item.
        items (Sequence[T]): The items to process.
        num_workers (int): Number of worker processes; ``<= 1`` runs sequentially.
        ordered (bool): If ``True`` (default), results follow input order.

    Returns:
        List[R]: The results of applying ``func`` to each item.
    """
    item_list = list(items)
    if num_workers <= 1 or len(item_list) <= 1:
        return [func(item) for item in item_list]

    # "spawn" avoids the fork()-in-a-multi-threaded-process deadlock risk (and the matching
    # DeprecationWarning) that bites the default fork start method on Linux.
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context("spawn")) as executor:
        if ordered:
            # Executor.map yields results in submission (input) order, giving deterministic
            # output regardless of which worker finishes first.
            return list(executor.map(func, item_list))
        # Unordered: return results as they complete (faster, but order is non-deterministic;
        # only use when the caller does not rely on positional correspondence with `items`).
        futures = [executor.submit(func, item) for item in item_list]
        return [future.result() for future in as_completed(futures)]
