from __future__ import annotations

import argparse
import sys
from pathlib import Path

from liquidbiopsy_agent.config import Config
from liquidbiopsy_agent.logging import setup_logging
from liquidbiopsy_agent.pipeline.pipeline import build_pipeline, resume_pipeline
from liquidbiopsy_agent.utils.storage import resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiquidBiopsy Agent pipeline")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run pipeline")
    run_p.add_argument("--input", required=True, help="Input directory or tar")
    run_p.add_argument("--output", required=True, help="Output directory")
    run_p.add_argument("--config", default="configs/default.yaml", help="Config YAML")
    run_p.add_argument("--instruction", default="", help="Natural language instruction for agent")

    status_p = sub.add_parser("status", help="Show status")
    status_p.add_argument("--run-dir", required=True, help="Run directory")

    resume_p = sub.add_parser("resume", help="Resume failed nodes")
    resume_p.add_argument("--run-dir", required=True, help="Run directory")
    resume_p.add_argument("--config", default="configs/default.yaml", help="Config YAML")
    resume_p.add_argument("--instruction", default="", help="Natural language instruction for agent")

    clean_p = sub.add_parser("clean-cache", help="Clean cache")
    clean_p.add_argument("--run-dir", required=True, help="Run directory")

    return parser.parse_args()


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    input_path = resolve_data_path(args.input, path_kind="pipeline input", must_exist=True)
    output_dir = resolve_data_path(args.output, path_kind="pipeline output directory", must_exist=False)
    pipeline = build_pipeline(input_path, output_dir, cfg, instruction=args.instruction)
    setup_logging(pipeline.run_dir / "logs")
    pipeline.run()
    print(f"Run directory: {pipeline.run_dir}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = resolve_data_path(args.run_dir, path_kind="run directory", must_exist=True)
    state_path = run_dir / "logs" / "state.json"
    if not state_path.exists():
        print("No state found", file=sys.stderr)
        return 1
    import json

    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for name, rec in data.items():
        print(f"{name}: {rec['status']}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    cfg = Config.load(Path(args.config))
    run_dir = resolve_data_path(args.run_dir, path_kind="run directory", must_exist=True)
    pipeline = resume_pipeline(run_dir, cfg, instruction=args.instruction)
    setup_logging(run_dir / "logs")
    pipeline.run(resume_failed_only=True)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    run_dir = resolve_data_path(args.run_dir, path_kind="run directory", must_exist=True)
    import shutil

    cache_dir = run_dir / "logs" / "nodes"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("Cache cleared")
    else:
        print("No cache found")
    return 0


def main() -> None:
    args = parse_args()
    if not args.command:
        print("No command provided", file=sys.stderr)
        sys.exit(1)
    if args.command == "run":
        sys.exit(cmd_run(args))
    if args.command == "status":
        sys.exit(cmd_status(args))
    if args.command == "resume":
        sys.exit(cmd_resume(args))
    if args.command == "clean-cache":
        sys.exit(cmd_clean(args))


if __name__ == "__main__":
    main()
