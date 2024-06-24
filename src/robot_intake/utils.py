"""General utility functions."""

from graphlib import CycleError, TopologicalSorter
from typing import Collection, Tuple, TypeVar, List

from robot_intake.structs import HashableComparable

_T = TypeVar("_T", bound=HashableComparable)


def topological_sort(l: Collection[_T], pairs: Collection[Tuple[_T, _T]]) -> List[_T]:
    """Create an ordered verison of l that obeys pairwise > relations."""
    # Create the TopologicalSorter object.
    ts: TopologicalSorter[_T] = TopologicalSorter()
    for node in l:
        ts.add(node)
    for x, y in pairs:
        ts.add(x, y)
    try:
        sorted_list = list(ts.static_order())
    except CycleError:
        raise ValueError("The given constraints form a cycle and cannot be satisfied")

    return sorted_list
