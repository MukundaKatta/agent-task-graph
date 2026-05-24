# agent-task-graph

DAG-based task dependency manager for agent workflows.

Define tasks with explicit dependencies. Poll `ready()` to find tasks whose dependencies are all done. A task can only start when its deps are complete.

## Install

```bash
pip install agent-task-graph
```

## Quick start

```python
from agent_task_graph import TaskGraph

graph = TaskGraph()
graph.add("fetch",  "Fetch raw data")
graph.add("parse",  "Parse response",  deps=["fetch"])
graph.add("upload", "Upload result",   deps=["parse"])

# Only "fetch" is ready now
graph.start("fetch")
graph.complete("fetch")

# Now "parse" is ready
item = graph.ready()[0]
graph.start(item.id)
graph.complete(item.id)
```

## API

### `TaskGraph`

| Method | Description |
|---|---|
| `add(id, name, *, deps, metadata)` | Add a task node |
| `remove(id)` | Remove a task (and drop it from other nodes' deps) |
| `start(id)` | PENDING → RUNNING |
| `complete(id)` | RUNNING → DONE |
| `fail(id)` | RUNNING → FAILED |
| `reset(id)` | FAILED/RUNNING → PENDING |
| `ready()` | PENDING tasks with all deps DONE |
| `blocked_by(id)` | Dep nodes not yet DONE |
| `dependents_of(id)` | Tasks that directly depend on this one |
| `topological_sort()` | Nodes in dependency order |
| `roots()` | Nodes with no deps |
| `leaves()` | Nodes nothing depends on |
| `is_complete()` | All nodes DONE? |
| `has_failures()` | Any FAILED? |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

### `TaskStatus`

`PENDING` · `RUNNING` · `DONE` · `FAILED`

### Errors

- `TaskNotFoundError` — task id not found
- `DuplicateTaskError` — id already registered
- `MissingDependencyError` — declared dep not in graph
- `InvalidTransitionError` — illegal status transition
- `CycleError` — cycle detected during topological sort

## License

MIT
