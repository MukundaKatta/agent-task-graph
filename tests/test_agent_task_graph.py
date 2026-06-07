"""Tests for agent-task-graph."""

from __future__ import annotations

import pytest

from agent_task_graph import (
    CycleError,
    TaskGraph,
    TaskNode,
    TaskNotFoundError,
    TaskStatus,
)
from agent_task_graph.core import (
    DuplicateTaskError,
    InvalidTransitionError,
    MissingDependencyError,
)

# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


def test_status_values():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.DONE.value == "done"
    assert TaskStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# TaskNode — construction and serialisation
# ---------------------------------------------------------------------------


def test_task_node_minimal():
    node = TaskNode(id="t1", name="fetch")
    assert node.id == "t1"
    assert node.name == "fetch"
    assert node.status is TaskStatus.PENDING
    assert node.deps == set()
    assert node.metadata == {}


def test_task_node_to_dict():
    node = TaskNode(id="t1", name="fetch", deps={"t0"}, metadata={"k": "v"})
    d = node.to_dict()
    assert d["id"] == "t1"
    assert d["status"] == "pending"
    assert d["deps"] == ["t0"]
    assert d["metadata"] == {"k": "v"}


def test_task_node_from_dict_round_trip():
    node = TaskNode(
        id="t2",
        name="parse",
        status=TaskStatus.DONE,
        deps={"t1"},
        metadata={"x": 1},
    )
    restored = TaskNode.from_dict(node.to_dict())
    assert restored.id == node.id
    assert restored.name == node.name
    assert restored.status is node.status
    assert restored.deps == node.deps
    assert restored.metadata == node.metadata


def test_task_node_repr():
    node = TaskNode(id="abc", name="n", deps={"x"})
    r = repr(node)
    assert "abc" in r
    assert "pending" in r


# ---------------------------------------------------------------------------
# TaskGraph — add
# ---------------------------------------------------------------------------


def test_add_task():
    g = TaskGraph()
    node = g.add("t1", "Task one")
    assert node.id == "t1"
    assert node.name == "Task one"
    assert "t1" in g


def test_add_with_deps():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    assert g.get("t2").deps == {"t1"}


def test_add_duplicate_raises():
    g = TaskGraph()
    g.add("t1", "A")
    with pytest.raises(DuplicateTaskError) as exc_info:
        g.add("t1", "duplicate")
    assert exc_info.value.task_id == "t1"


def test_add_missing_dep_raises():
    g = TaskGraph()
    with pytest.raises(MissingDependencyError) as exc_info:
        g.add("t1", "A", deps=["nonexistent"])
    assert exc_info.value.dep_id == "nonexistent"


def test_add_with_metadata():
    g = TaskGraph()
    node = g.add("t1", "A", metadata={"priority": 5})
    assert node.metadata == {"priority": 5}


# ---------------------------------------------------------------------------
# TaskGraph — remove
# ---------------------------------------------------------------------------


def test_remove_task():
    g = TaskGraph()
    g.add("t1", "A")
    g.remove("t1")
    assert "t1" not in g


def test_remove_cleans_deps():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    g.remove("t1")
    assert "t1" not in g.get("t2").deps


def test_remove_missing_raises():
    g = TaskGraph()
    with pytest.raises(TaskNotFoundError):
        g.remove("nope")


# ---------------------------------------------------------------------------
# TaskGraph — transitions
# ---------------------------------------------------------------------------


def test_start():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    assert g.get("t1").status is TaskStatus.RUNNING


def test_complete():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    g.complete("t1")
    assert g.get("t1").status is TaskStatus.DONE


def test_fail():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    g.fail("t1")
    assert g.get("t1").status is TaskStatus.FAILED


def test_reset_from_failed():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    g.fail("t1")
    g.reset("t1")
    assert g.get("t1").status is TaskStatus.PENDING


def test_reset_from_running():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    g.reset("t1")
    assert g.get("t1").status is TaskStatus.PENDING


def test_invalid_transition_raises():
    g = TaskGraph()
    g.add("t1", "A")
    with pytest.raises(InvalidTransitionError) as exc_info:
        g.complete("t1")  # PENDING → DONE not allowed
    assert exc_info.value.task_id == "t1"


def test_transition_unknown_raises():
    g = TaskGraph()
    with pytest.raises(TaskNotFoundError):
        g.start("nope")


# ---------------------------------------------------------------------------
# TaskGraph — ready
# ---------------------------------------------------------------------------


def test_ready_no_deps():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B")
    ready_ids = {n.id for n in g.ready()}
    assert ready_ids == {"t1", "t2"}


def test_ready_with_unmet_deps():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    ready_ids = {n.id for n in g.ready()}
    assert ready_ids == {"t1"}
    assert "t2" not in ready_ids


def test_ready_after_completing_dep():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    g.start("t1")
    g.complete("t1")
    ready_ids = {n.id for n in g.ready()}
    assert "t2" in ready_ids


def test_ready_excludes_running():
    g = TaskGraph()
    g.add("t1", "A")
    g.start("t1")
    assert g.ready() == []


def test_ready_diamond():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    g.add("c", "C", deps=["a"])
    g.add("d", "D", deps=["b", "c"])
    g.start("a")
    g.complete("a")
    ready_ids = {n.id for n in g.ready()}
    assert ready_ids == {"b", "c"}
    g.start("b")
    g.complete("b")
    g.start("c")
    g.complete("c")
    assert {n.id for n in g.ready()} == {"d"}


# ---------------------------------------------------------------------------
# TaskGraph — blocked_by / dependents_of
# ---------------------------------------------------------------------------


def test_blocked_by():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B")
    g.add("t3", "C", deps=["t1", "t2"])
    blockers = {n.id for n in g.blocked_by("t3")}
    assert blockers == {"t1", "t2"}


def test_blocked_by_done_dep_excluded():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B")
    g.add("t3", "C", deps=["t1", "t2"])
    g.start("t1")
    g.complete("t1")
    blockers = {n.id for n in g.blocked_by("t3")}
    assert blockers == {"t2"}


def test_dependents_of():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    g.add("t3", "C", deps=["t1"])
    dep_ids = {n.id for n in g.dependents_of("t1")}
    assert dep_ids == {"t2", "t3"}


# ---------------------------------------------------------------------------
# TaskGraph — topological sort
# ---------------------------------------------------------------------------


def test_topological_sort_linear():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    g.add("c", "C", deps=["b"])
    order = [n.id for n in g.topological_sort()]
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_sort_diamond():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    g.add("c", "C", deps=["a"])
    g.add("d", "D", deps=["b", "c"])
    order = [n.id for n in g.topological_sort()]
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


# ---------------------------------------------------------------------------
# TaskGraph — roots and leaves
# ---------------------------------------------------------------------------


def test_roots():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    assert [n.id for n in g.roots()] == ["a"]


def test_leaves():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    assert [n.id for n in g.leaves()] == ["b"]


def test_single_node_is_both_root_and_leaf():
    g = TaskGraph()
    g.add("only", "Only")
    assert [n.id for n in g.roots()] == ["only"]
    assert [n.id for n in g.leaves()] == ["only"]


# ---------------------------------------------------------------------------
# TaskGraph — queries
# ---------------------------------------------------------------------------


def test_by_status():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    g.start("a")
    assert len(g.pending()) == 1
    assert len(g.running()) == 1


def test_count():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    assert g.count() == 2
    assert g.count(TaskStatus.PENDING) == 2
    assert g.count(TaskStatus.DONE) == 0


def test_len():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    assert len(g) == 2


def test_is_complete_false():
    g = TaskGraph()
    g.add("a", "A")
    assert not g.is_complete()


def test_is_complete_true():
    g = TaskGraph()
    g.add("a", "A")
    g.start("a")
    g.complete("a")
    assert g.is_complete()


def test_has_failures():
    g = TaskGraph()
    g.add("a", "A")
    g.start("a")
    g.fail("a")
    assert g.has_failures()


def test_all():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    assert len(g.all()) == 2


# ---------------------------------------------------------------------------
# TaskGraph — clear and serialisation
# ---------------------------------------------------------------------------


def test_clear():
    g = TaskGraph()
    g.add("a", "A")
    g.clear()
    assert len(g) == 0


def test_to_dict_round_trip():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B", deps=["a"])
    g.start("a")
    g.complete("a")

    restored = TaskGraph.from_dict(g.to_dict())
    assert len(restored) == 2
    assert restored.get("a").status is TaskStatus.DONE
    assert restored.get("b").status is TaskStatus.PENDING
    assert "a" in restored.get("b").deps


def test_repr():
    g = TaskGraph()
    g.add("a", "A")
    r = repr(g)
    assert "TaskGraph" in r
    assert "1" in r


# ---------------------------------------------------------------------------
# TaskGraph — cycle detection
# ---------------------------------------------------------------------------


def test_topological_sort_detects_cycle():
    # A cycle cannot be built via add(), so construct one through from_dict.
    data = {
        "order": ["a", "b"],
        "nodes": [
            {"id": "a", "name": "A", "status": "pending", "deps": ["b"]},
            {"id": "b", "name": "B", "status": "pending", "deps": ["a"]},
        ],
    }
    g = TaskGraph.from_dict(data)
    with pytest.raises(CycleError):
        g.topological_sort()


def test_topological_sort_disconnected_nodes():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    order = [n.id for n in g.topological_sort()]
    assert set(order) == {"a", "b"}


# ---------------------------------------------------------------------------
# TaskGraph — misc query / transition edge cases
# ---------------------------------------------------------------------------


def test_dependents_of_unknown_raises():
    g = TaskGraph()
    with pytest.raises(TaskNotFoundError):
        g.dependents_of("nope")


def test_blocked_by_unknown_raises():
    g = TaskGraph()
    with pytest.raises(TaskNotFoundError):
        g.blocked_by("nope")


def test_reset_from_pending_raises():
    g = TaskGraph()
    g.add("t1", "A")
    with pytest.raises(InvalidTransitionError):
        g.reset("t1")  # PENDING → PENDING not allowed


def test_contains():
    g = TaskGraph()
    g.add("a", "A")
    assert "a" in g
    assert "missing" not in g


def test_failed_dep_blocks_ready():
    g = TaskGraph()
    g.add("t1", "A")
    g.add("t2", "B", deps=["t1"])
    g.start("t1")
    g.fail("t1")
    # t2 must not be ready while its dependency is FAILED.
    assert {n.id for n in g.ready()} == set()


def test_from_dict_preserves_order():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    g.add("c", "C")
    restored = TaskGraph.from_dict(g.to_dict())
    assert [n.id for n in restored.all()] == ["a", "b", "c"]


def test_done_and_failed_queries():
    g = TaskGraph()
    g.add("a", "A")
    g.add("b", "B")
    g.start("a")
    g.complete("a")
    g.start("b")
    g.fail("b")
    assert [n.id for n in g.done()] == ["a"]
    assert [n.id for n in g.failed()] == ["b"]
