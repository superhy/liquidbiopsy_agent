from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from .state import TaskStatus
from .task import Task, TaskRecord
from liquidbiopsy_agent.utils.io import write_json


class GraphState(TypedDict):
    records: Dict[str, TaskRecord]
    resume_failed_only: bool


class DAGExecutor:
    def __init__(
        self,
        tasks: Dict[str, Task],
        edges: Dict[str, List[str]],
        run_dir: Path,
        config_hash: str,
        decisions=None,
    ):
        self.tasks = tasks
        self.edges = edges
        self.run_dir = run_dir
        self.config_hash = config_hash
        self.state_path = run_dir / "logs" / "state.json"
        self.decisions = decisions
        self.deps = self._build_deps()

    def _build_deps(self) -> Dict[str, List[str]]:
        rev = defaultdict(list)
        for src, dsts in self.edges.items():
            for d in dsts:
                rev[d].append(src)
        return dict(rev)

    def _roots(self) -> List[str]:
        all_nodes = set(self.tasks)
        non_roots = set()
        for _, dsts in self.edges.items():
            non_roots.update(dsts)
        return sorted(all_nodes - non_roots)

    def _leaves(self) -> List[str]:
        all_nodes = set(self.tasks)
        non_leaves = set(self.edges.keys())
        return sorted(all_nodes - non_leaves)

    def save_state(self, records: Dict[str, TaskRecord]) -> None:
        payload = {name: rec.__dict__ for name, rec in records.items()}
        write_json(self.state_path, payload)

    def load_state(self) -> Dict[str, TaskRecord]:
        if not self.state_path.exists():
            return {}
        with open(self.state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[str, TaskRecord] = {}
        for name, rec in data.items():
            out[name] = TaskRecord(**rec)
        return out

    def _node_runner(self, name: str):
        def _run(state: GraphState) -> GraphState:
            records = state["records"]
            deps = [records[d].status for d in self.deps.get(name, []) if d in records]
            if any(status == TaskStatus.FAILED for status in deps):
                return state
            if state["resume_failed_only"] and name in records and records[name].status != TaskStatus.FAILED:
                return state
            task = self.tasks[name]
            rec = task.run(self.run_dir, self.config_hash)
            records[name] = rec
            self.save_state(records)
            return {"records": records, "resume_failed_only": state["resume_failed_only"]}

        return _run

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(GraphState)
        graph.add_node("start", lambda state: state)
        for name in self.tasks:
            graph.add_node(name, self._node_runner(name))
        roots = self._roots()
        for root in roots:
            graph.add_edge("start", root)
        for src, dsts in self.edges.items():
            for dst in dsts:
                graph.add_edge(src, dst)
        for leaf in self._leaves():
            graph.add_edge(leaf, END)
        graph.set_entry_point("start")
        return graph

    def run(self, resume_failed_only: bool = False) -> Dict[str, TaskRecord]:
        records: Dict[str, TaskRecord] = self.load_state()
        state: GraphState = {"records": records, "resume_failed_only": resume_failed_only}
        graph = self._build_graph().compile()
        result = graph.invoke(state)
        records = result["records"]
        failed_errors = [r.error for r in records.values() if r.status == TaskStatus.FAILED and r.error]
        if failed_errors and self.decisions:
            self.decisions.failure_plan(failed_errors)
        return records
