"""DAG-based task dependency manager for agent workflows.

:class:`TaskGraph` holds :class:`TaskNode` objects arranged in a directed
acyclic graph (DAG).  Each node has an explicit set of dependency IDs that
must reach ``DONE`` status before the node itself is considered *ready* to
run.

The typical lifecycle of a node is::

    PENDING → RUNNING → DONE
                      ↘ FAILED

Use :meth:`~TaskGraph.ready` to poll for tasks whose dependencies are all
satisfied, :meth:`~TaskGraph.start` to begin them, and
:meth:`~TaskGraph.complete` / :meth:`~TaskGraph.fail` to finish them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle status of a task node."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskNotFoundError(KeyError):
    """Raised when a task ID is not found in the graph."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id!r} not found.")


class CycleError(ValueError):
    """Raised when adding a dependency would create a cycle."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Adding dependencies for {task_id!r} would create a cycle.")


class DuplicateTaskError(ValueError):
    """Raised when a task ID is already registered."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"Task {task_id!r} is already in the graph.")


class MissingDependencyError(ValueError):
    """Raised when a declared dependency does not exist in the graph."""

    def __init__(self, task_id: str, dep_id: str) -> None:
        self.task_id = task_id
        self.dep_id = dep_id
        super().__init__(
            f"Dependency {dep_id!r} of task {task_id!r} is not in the graph."
        )


class InvalidTransitionError(RuntimeError):
    """Raised when a status transition is not allowed."""

    def __init__(
        self, task_id: str, from_status: TaskStatus, to_status: TaskStatus
    ) -> None:
        self.task_id = task_id
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Cannot transition task {task_id!r}"
            f" from {from_status.value!r} to {to_status.value!r}."
        )


@dataclass
class TaskNode:
    """A single node in the task graph.

    Attributes:
        id: Unique string identifier.
        name: Human-readable label.
        status: Current lifecycle status.
        deps: IDs of tasks that must be DONE before this one is ready.
        metadata: Arbitrary extra data.
    """

    id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    deps: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "deps": sorted(self.deps),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskNode:
        """Reconstruct a :class:`TaskNode` from a plain dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            deps=set(data.get("deps", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            f"TaskNode(id={self.id!r},"
            f" status={self.status.value!r},"
            f" deps={sorted(self.deps)!r})"
        )


class TaskGraph:
    """An in-memory DAG of tasks with dependency tracking.

    Tasks are added with explicit dependency IDs.  A task is *ready* when
    all its dependencies have status ``DONE`` and it is still ``PENDING``.

    Args:
        allow_failed_deps: If ``True``, a task is still considered blocked
            (not ready) when a dependency has ``FAILED``.  Default ``False``
            means failed deps permanently block downstream tasks.

    Example::

        graph = TaskGraph()
        graph.add("fetch",  "Fetch data")
        graph.add("parse",  "Parse data",   deps=["fetch"])
        graph.add("upload", "Upload result", deps=["parse"])

        graph.start("fetch")
        graph.complete("fetch")
        assert graph.ready() == [graph.get("parse")]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}
        self._order: list[str] = []  # insertion order

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        task_id: str,
        name: str,
        *,
        deps: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskNode:
        """Add a task to the graph.

        Args:
            task_id: Unique identifier.
            name: Human-readable label.
            deps: IDs of prerequisite tasks (all must exist already).
            metadata: Arbitrary payload.

        Returns:
            The new :class:`TaskNode`.

        Raises:
            DuplicateTaskError: If *task_id* is already registered.
            MissingDependencyError: If any dep ID is not in the graph.
            CycleError: If adding these deps would create a cycle.
        """
        if task_id in self._nodes:
            raise DuplicateTaskError(task_id)
        dep_set = set(deps or [])
        for dep_id in dep_set:
            if dep_id not in self._nodes:
                raise MissingDependencyError(task_id, dep_id)
        # Cycle check: can task_id be reached from any of its own deps?
        # Since task_id is new (not yet in graph), we just need to verify
        # that none of the deps transitively depend on task_id — impossible
        # since task_id does not exist yet.  No cycle is possible on insert.
        node = TaskNode(
            id=task_id,
            name=name,
            deps=dep_set,
            metadata=dict(metadata or {}),
        )
        self._nodes[task_id] = node
        self._order.append(task_id)
        return node

    def remove(self, task_id: str) -> None:
        """Remove a task from the graph.

        Also removes *task_id* from the dep sets of all other tasks.

        Raises:
            TaskNotFoundError: If *task_id* is not in the graph.
        """
        self.get(task_id)  # raises if missing
        del self._nodes[task_id]
        self._order.remove(task_id)
        for node in self._nodes.values():
            node.deps.discard(task_id)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _transition(
        self,
        task_id: str,
        to_status: TaskStatus,
        allowed_from: set[TaskStatus],
    ) -> None:
        node = self.get(task_id)
        if node.status not in allowed_from:
            raise InvalidTransitionError(task_id, node.status, to_status)
        node.status = to_status

    def start(self, task_id: str) -> None:
        """Mark *task_id* as RUNNING.  Must be PENDING."""
        self._transition(task_id, TaskStatus.RUNNING, {TaskStatus.PENDING})

    def complete(self, task_id: str) -> None:
        """Mark *task_id* as DONE.  Must be RUNNING."""
        self._transition(task_id, TaskStatus.DONE, {TaskStatus.RUNNING})

    def fail(self, task_id: str) -> None:
        """Mark *task_id* as FAILED.  Must be RUNNING."""
        self._transition(task_id, TaskStatus.FAILED, {TaskStatus.RUNNING})

    def reset(self, task_id: str) -> None:
        """Reset *task_id* back to PENDING.  Can be FAILED or RUNNING."""
        self._transition(
            task_id,
            TaskStatus.PENDING,
            {TaskStatus.FAILED, TaskStatus.RUNNING},
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, task_id: str) -> TaskNode:
        """Return the node for *task_id*.

        Raises:
            TaskNotFoundError: If not found.
        """
        if task_id not in self._nodes:
            raise TaskNotFoundError(task_id)
        return self._nodes[task_id]

    def ready(self) -> list[TaskNode]:
        """Return PENDING tasks whose deps are all DONE, in insertion order."""
        result = []
        for tid in self._order:
            node = self._nodes[tid]
            if node.status is not TaskStatus.PENDING:
                continue
            if all(
                self._nodes[d].status is TaskStatus.DONE
                for d in node.deps
                if d in self._nodes
            ):
                result.append(node)
        return result

    def blocked_by(self, task_id: str) -> list[TaskNode]:
        """Return dep nodes that are blocking *task_id* (not yet DONE)."""
        node = self.get(task_id)
        return [
            self._nodes[d]
            for d in sorted(node.deps)
            if d in self._nodes and self._nodes[d].status is not TaskStatus.DONE
        ]

    def dependents_of(self, task_id: str) -> list[TaskNode]:
        """Return tasks that directly depend on *task_id*, in insertion order."""
        self.get(task_id)  # validate exists
        return [
            self._nodes[tid] for tid in self._order if task_id in self._nodes[tid].deps
        ]

    def topological_sort(self) -> list[TaskNode]:
        """Return nodes in topological order (deps before dependents).

        Raises:
            CycleError: If the graph contains a cycle.
        """
        in_deg: dict[str, int] = {tid: 0 for tid in self._order}
        for tid in self._order:
            for dep in self._nodes[tid].deps:
                if dep in in_deg:
                    in_deg[tid] += 1  # tid depends on dep
        # Actually recompute properly: in_degree = number of deps in graph
        in_deg = {}
        for tid in self._order:
            in_deg[tid] = sum(1 for dep in self._nodes[tid].deps if dep in self._nodes)
        queue = [tid for tid in self._order if in_deg[tid] == 0]
        result: list[TaskNode] = []
        while queue:
            tid = queue.pop(0)
            result.append(self._nodes[tid])
            for dep_of in self._order:
                if tid in self._nodes[dep_of].deps:
                    in_deg[dep_of] -= 1
                    if in_deg[dep_of] == 0:
                        queue.append(dep_of)
        if len(result) != len(self._nodes):
            # Find a node involved in the cycle for the error message
            visited = {n.id for n in result}
            cycle_node = next(t for t in self._order if t not in visited)
            raise CycleError(cycle_node)
        return result

    def by_status(self, status: TaskStatus) -> list[TaskNode]:
        """Return nodes with *status* in insertion order."""
        return [
            self._nodes[tid] for tid in self._order if self._nodes[tid].status is status
        ]

    def pending(self) -> list[TaskNode]:
        """All PENDING nodes in insertion order."""
        return self.by_status(TaskStatus.PENDING)

    def running(self) -> list[TaskNode]:
        """All RUNNING nodes in insertion order."""
        return self.by_status(TaskStatus.RUNNING)

    def done(self) -> list[TaskNode]:
        """All DONE nodes in insertion order."""
        return self.by_status(TaskStatus.DONE)

    def failed(self) -> list[TaskNode]:
        """All FAILED nodes in insertion order."""
        return self.by_status(TaskStatus.FAILED)

    def roots(self) -> list[TaskNode]:
        """Nodes with no dependencies, in insertion order."""
        return [self._nodes[tid] for tid in self._order if not self._nodes[tid].deps]

    def leaves(self) -> list[TaskNode]:
        """Nodes that no other node depends on, in insertion order."""
        has_dependents = {dep for tid in self._order for dep in self._nodes[tid].deps}
        return [self._nodes[tid] for tid in self._order if tid not in has_dependents]

    def all(self) -> list[TaskNode]:
        """All nodes in insertion order."""
        return [self._nodes[tid] for tid in self._order]

    def count(self, status: TaskStatus | None = None) -> int:
        """Total nodes, optionally filtered by *status*."""
        if status is None:
            return len(self._nodes)
        return sum(1 for node in self._nodes.values() if node.status is status)

    def is_complete(self) -> bool:
        """Return ``True`` when every node is DONE."""
        return all(node.status is TaskStatus.DONE for node in self._nodes.values())

    def has_failures(self) -> bool:
        """Return ``True`` when at least one node is FAILED."""
        return any(node.status is TaskStatus.FAILED for node in self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._nodes

    # ------------------------------------------------------------------
    # Serialisation / reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all nodes."""
        self._nodes.clear()
        self._order.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a plain dict."""
        return {
            "order": list(self._order),
            "nodes": [self._nodes[tid].to_dict() for tid in self._order],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGraph:
        """Reconstruct a :class:`TaskGraph` from a plain dict."""
        graph = cls()
        for d in data.get("nodes", []):
            node = TaskNode.from_dict(d)
            graph._nodes[node.id] = node
        order = data.get("order", [n["id"] for n in data.get("nodes", [])])
        graph._order = [tid for tid in order if tid in graph._nodes]
        return graph

    def __repr__(self) -> str:
        done = self.count(TaskStatus.DONE)
        return (
            f"TaskGraph(total={len(self._nodes)},"
            f" done={done},"
            f" ready={len(self.ready())})"
        )
