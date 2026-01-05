from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from liquidbiopsy_agent.agent.task import Task
from liquidbiopsy_agent.utils.io import read_json


def encode_image(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def make_task(run_dir: Path, config: Dict[str, Any]) -> Task:
    project_root = Path(__file__).resolve().parents[4]
    template_dir = project_root / "templates"
    template_path = template_dir / "report.html.j2"
    report_path = run_dir / "reports" / "report.html"

    def _run(inputs: Dict[str, Any], config_section: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
        manifest = pd.read_parquet(run_dir / "data" / "manifest" / "manifest.parquet")
        qc = pd.read_parquet(run_dir / "qc" / "sample_qc.parquet") if (run_dir / "qc" / "sample_qc.parquet").exists() else pd.DataFrame()
        features = pd.read_parquet(run_dir / "features" / "merged" / "features_table.parquet") if (run_dir / "features" / "merged" / "features_table.parquet").exists() else pd.DataFrame()
        drift = read_json(run_dir / "analysis" / "drift_flags.json") if (run_dir / "analysis" / "drift_flags.json").exists() else {}
        explanation_path = run_dir / "analysis" / "agent" / "explanation.txt"
        agent_summary_path = run_dir / "analysis" / "agent" / "agent_summary.json"
        agent_summary = read_json(agent_summary_path) if agent_summary_path.exists() else {}
        explanation_text = explanation_path.read_text(encoding="utf-8") if explanation_path.exists() else ""

        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape())
        template = env.get_template("report.html.j2")

        plots_dir = run_dir / "qc" / "plots"
        frag_plots_dir = run_dir / "features" / "frag" / "plots"
        cnv_plots_dir = run_dir / "features" / "cnv" / "plots"
        cnv_preview = ""
        if not manifest.empty:
            first_sample = manifest.iloc[0]["sample_id"]
            cnv_plot = cnv_plots_dir / f"{first_sample}_cnv_profile.png"
            if cnv_plot.exists():
                cnv_preview = encode_image(cnv_plot)
        context = {
            "n_samples": len(manifest),
            "qc_table": qc.head(20).to_dict(orient="records") if not qc.empty else [],
            "features_table": features.head(20).to_dict(orient="records") if not features.empty else [],
            "drift": drift,
            "length_plot": encode_image(plots_dir / "cohort_length_box.png"),
            "coverage_heatmap": encode_image(plots_dir / "coverage_uniformity_heatmap.png"),
            "coverage_bin_heatmap": encode_image(plots_dir / "coverage_bin_heatmap.png"),
            "length_peaks": encode_image(frag_plots_dir / "length_peak_comparison.png"),
            "length_violin": encode_image(frag_plots_dir / "length_violin.png"),
            "cnv_preview": cnv_preview,
            "cnv_chrom_preview": encode_image(cnv_plots_dir / f"{first_sample}_cnv_chrom_plot.png") if cnv_preview else "",
            "pca_plot": encode_image(run_dir / "analysis" / "plots" / "pca.png"),
            "agent_summary": agent_summary,
            "agent_explanation": explanation_text,
        }
        html = template.render(**context)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        return {"report": str(report_path)}

    return Task(
        name="report",
        inputs={"manifest": run_dir / "data" / "manifest" / "manifest.parquet"},
        outputs=[report_path],
        config_section=config,
        run_fn=_run,
        retries=int(config.get("retries", 0)),
    )
