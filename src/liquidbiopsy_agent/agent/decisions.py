from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import LLMClient, safe_parse_json
from liquidbiopsy_agent.utils.io import write_json


class DecisionEngine:
    def __init__(self, run_dir: Path, enable_llm: bool = False, instruction: str = ""):
        self.run_dir = run_dir
        self.enable_llm = enable_llm
        self.client = LLMClient() if enable_llm else None
        self.decisions_dir = run_dir / "logs" / "decisions"
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir = run_dir / "logs" / "agent"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.agent_dir / "memory.json"
        self.instruction = instruction.strip()
        if self.instruction:
            self._append_memory({"event": "user_instruction", "payload": {"text": self.instruction}})

    def _run_llm(self, prompt: str, schema_keys: List[str]) -> Dict[str, Any]:
        if not self.client or not self.client.enabled:
            return {}
        raw = self.client.complete(prompt)
        parsed = safe_parse_json(raw)
        if not parsed:
            return {}
        if not all(k in parsed for k in schema_keys):
            return {}
        return parsed

    def record(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = self.decisions_dir / f"{name}.json"
        write_json(path, payload)
        self._append_memory({"event": name, "payload": payload})
        return payload

    def assay_type_rules(self, filenames: List[str]) -> Dict[str, Any]:
        heuristics = {"rules": [".*wgs.*", ".*atac.*"], "default": "unknown"}
        prompt = (
            "Suggest regex rules to infer assay type from filenames: "
            + ",".join(filenames[:10])
            + " Return JSON with keys rules (list) and default."
        )
        llm_resp = self._run_llm(prompt, ["rules", "default"])
        return self.record("D1_assay_type", llm_resp or heuristics)

    def schema_policy(self, audit: List[Dict[str, Any]]) -> Dict[str, Any]:
        heuristics = {"prefer_columns": [0, 1, 2], "fallback_length": True}
        prompt = "Based on schema audit, choose columns for chrom,start,end. Return prefer_columns list."
        llm_resp = self._run_llm(prompt, ["prefer_columns"])
        return self.record("D2_schema", llm_resp or heuristics)

    def qc_thresholds(self, qc_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        heuristics = {"min_fragments": 10000, "max_chrY_frac": 0.1}
        prompt = "Given QC metrics, propose thresholds for min_fragments and max_chrY_frac as JSON."
        llm_resp = self._run_llm(prompt, ["min_fragments", "max_chrY_frac"])
        return self.record("D3_qc_thresholds", llm_resp or heuristics)

    def cnv_policy(self, n_fragments: int, uniformity: float) -> Dict[str, Any]:
        heuristics = {"bin_size": 1_000_000, "run_segmentation": False}
        prompt = (
            "Given fragment count and uniformity CV, pick bin_size (int) and run_segmentation (bool)."
        )
        llm_resp = self._run_llm(prompt, ["bin_size", "run_segmentation"])
        return self.record("D4_cnv", llm_resp or heuristics)

    def frag_feature_policy(self, sample_count: int) -> Dict[str, Any]:
        heuristics = {"bins": [i for i in range(50, 501, 5)], "extra_features": []}
        prompt = "Pick fragment length histogram bins (list of ints) and extra_features list."
        llm_resp = self._run_llm(prompt, ["bins", "extra_features"])
        return self.record("D5_frag", llm_resp or heuristics)

    def meth_panel_policy(self, panels: List[str]) -> Dict[str, Any]:
        heuristics = {"panels": panels}
        prompt = "Select which methylation proxy panels to run from list; return JSON key panels."
        llm_resp = self._run_llm(prompt, ["panels"])
        return self.record("D6_meth", llm_resp or heuristics)

    def failure_plan(self, errors: List[str]) -> Dict[str, Any]:
        heuristics = {"action": "rerun_failed", "notes": errors[:3]}
        prompt = "Given recent errors, propose retry plan JSON with action and notes."
        llm_resp = self._run_llm(prompt, ["action", "notes"])
        return self.record("D7_failure", llm_resp or heuristics)

    def plot_policy(self, node: str, sample_count: int, config: Dict[str, Any]) -> Dict[str, Any]:
        heuristics = {
            "enable": bool(config.get("advanced_plots", False)),
            "qc_bin_heatmap": bool(config.get("qc_bin_heatmap", False)),
            "cnv_chrom_plot": bool(config.get("cnv_chrom_plot", False)),
            "frag_violin": bool(config.get("frag_violin", False)),
            "max_samples": int(config.get("max_samples", 20)),
            "max_bins": int(config.get("max_bins", 200)),
        }
        if self.instruction and (not self.client or not self.client.enabled):
            self._append_memory(
                {
                    "event": "instruction_ignored",
                    "payload": {"reason": "LLM disabled; falling back to config", "node": node},
                }
            )
        prompt = (
            f"User instruction: {self.instruction or 'none'}. "
            f"Decide whether to enable advanced plots for node {node} with sample_count={sample_count}. "
            "Return JSON with keys enable, qc_bin_heatmap, cnv_chrom_plot, frag_violin, max_samples, max_bins. "
            "Prefer user instruction; otherwise choose sensible defaults."
        )
        llm_resp = self._run_llm(
            prompt,
            ["enable", "qc_bin_heatmap", "cnv_chrom_plot", "frag_violin", "max_samples", "max_bins"],
        )
        result = llm_resp or heuristics
        self.record(f"D8_plot_policy_{node}", result)
        return result

    def agent_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        heuristics = self._heuristic_agent_summary(payload)
        prompt = (
            "Given cohort QC, feature summaries, and drift flags, produce JSON with keys "
            "parameter_tuning (list), rerun_samples (list), explanation (string). "
            "Use British English for the explanation."
        )
        llm_resp = self._run_llm(prompt, ["parameter_tuning", "rerun_samples", "explanation"])
        result = llm_resp or heuristics
        self._append_memory({"event": "agent_summary", "payload": result})
        return result

    def _heuristic_agent_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tuning = []
        reruns = []
        qc = payload.get("qc", [])
        cnv = payload.get("cnv", [])
        drift = payload.get("drift", {})
        if qc:
            low_frag = [r["sample_id"] for r in qc if r.get("n_fragments", 0) < 10000]
            for sid in low_frag:
                reruns.append({"sample_id": sid, "reason": "Low fragment count"})
            uniformity = [r.get("coverage_uniformity_cv") for r in qc if r.get("coverage_uniformity_cv") is not None]
            if uniformity and sum(uniformity) / len(uniformity) > 1.0:
                tuning.append(
                    {
                        "section": "cnv",
                        "key": "bin_size",
                        "suggested": 2000000,
                        "reason": "High coverage uniformity CV suggests coarser bins may stabilise counts.",
                    }
                )
        if cnv:
            vars_ = [r.get("cnv_var") for r in cnv if r.get("cnv_var") is not None]
            if vars_ and sum(vars_) / len(vars_) > 1e5:
                tuning.append(
                    {
                        "section": "cnv",
                        "key": "run_segmentation",
                        "suggested": False,
                        "reason": "High variance could reflect noise; keep segmentation off for now.",
                    }
                )
        if drift:
            for metric, flagged in drift.items():
                for sid in flagged.keys():
                    reruns.append({"sample_id": sid, "reason": f"Outlier in {metric}"})
        explanation = (
            "This run completed with a mix of passing and borderline samples. "
            "Where fragment counts are low or uniformity is poor, a rerun or re-export of BEDs is advisable. "
            "CNV variance appears sensitive to binning; consider coarser bins if noise dominates. "
            "Drift flags highlight samples worth manual inspection before downstream modelling."
        )
        return {"parameter_tuning": tuning, "rerun_samples": reruns, "explanation": explanation}

    def _append_memory(self, event: Dict[str, Any]) -> None:
        if self.memory_path.exists():
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        else:
            data = []
        data.append(event)
        write_json(self.memory_path, data)
