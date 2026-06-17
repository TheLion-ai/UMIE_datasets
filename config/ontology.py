"""Hierarchical, multi-axis ontology support shared by ``labels`` and ``masks`` (Theme L).

This module adds three additive, opt-in layers on top of the flat ``Label`` / ``Mask``
dataclasses, **without changing any existing id, RadLex name, or emitted value**:

* **Task 36 - hierarchy.** ``Label``/``Mask`` gain an optional ``radlex_parent_id`` pointing at
  the RadLex id of their parent concept, so the entries form the RadLex tree from
  ``RadLex Terms map.md`` (e.g. Neoplasm -> Malignant -> Adenocarcinoma -> Renal adenocarcinoma
  -> Clear-cell). The generic query helpers below (``ancestors`` / ``descendants`` /
  ``group_by_level`` / ``children`` / ``roots``) let callers query at any granularity.
* **Task 37 - multi-axis.** Each finding can be decomposed into ``anatomy`` / ``pathology`` /
  ``modifier`` axes, each a dual-coded :class:`OntologyTerm` (RadLex + source).
* **Task 39 - secondary ontologies.** Optional ``snomed_id`` / ``fma_id`` / ``uberon_id``
  cross-codes alongside the primary RadLex code.

The query helpers are intentionally generic - they operate on any iterable of objects exposing
``radlex_name`` / ``radlex_id`` / ``radlex_parent_id`` - so the same code serves both the label
and the mask registries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, TypeVar


@dataclass
class OntologyTerm:
    """A single dual-coded ontology term: the RadLex concept plus the source name it came from.

    Used for the multi-axis (anatomy / pathology / modifier) decomposition (Task 37). Every
    field is optional so an axis may be partially specified (e.g. RadLex known, source blank).
    """

    radlex_name: str = ""
    radlex_id: str = ""
    source_name: str = ""  # how the originating dataset referred to this axis term, if at all

    def is_empty(self) -> bool:
        """Return True when no part of the term is populated."""
        return not (self.radlex_name or self.radlex_id or self.source_name)


class OntologyNode(Protocol):
    """Structural type for anything the hierarchy helpers can walk (a ``Label`` or a ``Mask``)."""

    radlex_name: str
    radlex_id: str
    radlex_parent_id: Optional[str]


T = TypeVar("T", bound=OntologyNode)


def index_by_id(entries: Iterable[T]) -> dict[str, list[T]]:
    """Group entries by their RadLex id (a list per id so duplicate ids are preserved, not lost)."""
    index: dict[str, list[T]] = {}
    for entry in entries:
        if entry.radlex_id:
            index.setdefault(entry.radlex_id, []).append(entry)
    return index


def index_by_name(entries: Iterable[T]) -> dict[str, T]:
    """Map ``radlex_name`` -> entry (names are the stable, human-facing identity within a registry)."""
    return {entry.radlex_name: entry for entry in entries}


def get_by_name(name: str, entries: Iterable[T]) -> Optional[T]:
    """Return the entry with the given RadLex name, or ``None``."""
    return index_by_name(entries).get(name)


def parent(entry: T, entries: Iterable[T]) -> Optional[T]:
    """Resolve ``entry``'s parent by matching its ``radlex_parent_id`` to another entry's ``radlex_id``.

    Returns ``None`` for roots (no parent id) or dangling parents (id not present in the registry).
    If several entries share the parent id, the first is returned; :func:`validate` flags such
    duplicate ids separately so the ambiguity is surfaced rather than hidden.
    """
    if not entry.radlex_parent_id:
        return None
    matches = index_by_id(entries).get(entry.radlex_parent_id, [])
    return matches[0] if matches else None


def children(entry: T, entries: Iterable[T]) -> list[T]:
    """Return the direct children of ``entry`` (entries whose parent id is ``entry``'s RadLex id)."""
    if not entry.radlex_id:
        return []
    return [e for e in entries if e.radlex_parent_id == entry.radlex_id]


def ancestors(entry: T, entries: Iterable[T]) -> list[T]:
    """Return ``entry``'s ancestors from nearest parent to root (cycle-safe)."""
    entries = list(entries)
    chain: list[T] = []
    seen = {id(entry)}
    current = parent(entry, entries)
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = parent(current, entries)
    return chain


def descendants(entry: T, entries: Iterable[T]) -> list[T]:
    """Return all transitive descendants of ``entry`` (breadth-first, cycle-safe)."""
    entries = list(entries)
    out: list[T] = []
    seen = {id(entry)}
    frontier = children(entry, entries)
    while frontier:
        node = frontier.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        frontier.extend(children(node, entries))
    return out


def descendants_of(name: str, entries: Iterable[T]) -> list[T]:
    """Convenience: all descendants of the entry named ``name`` (empty list if the name is unknown)."""
    entries = list(entries)
    root = get_by_name(name, entries)
    return descendants(root, entries) if root is not None else []


def roots(entries: Iterable[T]) -> list[T]:
    """Return the top-level entries (no parent, or a parent id absent from the registry)."""
    entries = list(entries)
    return [e for e in entries if parent(e, entries) is None]


def depth(entry: T, entries: Iterable[T]) -> int:
    """Return the number of ancestors above ``entry`` (0 for a root)."""
    return len(ancestors(entry, entries))


def group_by_level(entries: Iterable[T]) -> dict[int, list[T]]:
    """Group entries by their depth in the hierarchy (0 == roots, 1 == their children, ...)."""
    entries = list(entries)
    levels: dict[int, list[T]] = {}
    for entry in entries:
        levels.setdefault(depth(entry, entries), []).append(entry)
    return dict(sorted(levels.items()))


def validate(entries: Iterable[T]) -> list[str]:
    """Audit the registry and return a list of human-readable problems (Task 38).

    Reports duplicate RadLex ids, empty RadLex ids, dangling ``radlex_parent_id`` references,
    and parent cycles. Returns an empty list when the registry is clean.
    """
    entries = list(entries)
    problems: list[str] = []

    by_id = index_by_id(entries)
    for radlex_id, group in by_id.items():
        if len(group) > 1:
            names = ", ".join(sorted(e.radlex_name for e in group))
            problems.append(f"Duplicate RadLex id {radlex_id} shared by: {names}")

    for entry in entries:
        if not entry.radlex_id:
            problems.append(f"Empty RadLex id for '{entry.radlex_name}'")
        if entry.radlex_parent_id and entry.radlex_parent_id not in by_id:
            problems.append(
                f"Dangling parent id {entry.radlex_parent_id} on '{entry.radlex_name}' "
                "(no entry in the registry has that RadLex id)"
            )

    for entry in entries:
        seen = {id(entry)}
        current = parent(entry, entries)
        while current is not None:
            if id(current) in seen:
                problems.append(f"Parent cycle detected at '{entry.radlex_name}'")
                break
            seen.add(id(current))
            current = parent(current, entries)

    return problems
