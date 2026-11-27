"""Demo/live command line and run artifact output."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from client import Client, HttpTransport


def main(robot_class):
    parser = argparse.ArgumentParser(
        description="CUMCM 2026 B robot; demo is local, live uses official HTTP API"
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("demo", "live"):
        p = sub.add_parser(mode)
        p.add_argument("--problem", type=int, choices=(3,), default=3)
        p.add_argument("--output", type=Path, default=Path("runs"))
        p.add_argument("--fast-measures", type=int, default=12)
        p.add_argument("--strategy", choices=("auto", "coverage-grid"), default="auto")
        p.add_argument("--quiet", action="store_true")
        if mode == "demo":
            p.add_argument("--seed", type=int, default=20260910)
            p.add_argument("--profile", choices=("practice",), default="practice")
        else:
            p.add_argument("--robot-id", required=True, help="Currently logged-in competition team number")
            p.add_argument("--url", default="http://127.0.0.1:2026")
            p.add_argument("--wait-ready", type=float, default=60.0)
    args = parser.parse_args()
    if not 0 <= args.fast_measures <= 20:
        parser.error("--fast-measures must be in 0..20")
    if not args.output.is_absolute():
        args.output = Path(__file__).resolve().parent / args.output
    run_dir = args.output / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    arena = None
    if args.mode == "demo":
        from local_simulator import LocalTransport, build_arena

        arena = build_arena(args.seed, args.problem, args.profile)
        transport, robot_id, wait = LocalTransport(arena), "local-test", 0
    else:
        transport, robot_id, wait = HttpTransport(args.url), args.robot_id, args.wait_ready
    client = Client(transport, robot_id, run_dir / "actions.jsonl", wait_ready=wait)
    robot = robot_class(
        client,
        args.problem,
        fast_measures=args.fast_measures,
        progress=None if args.quiet else lambda msg: print(msg, flush=True),
        strategy=args.strategy,
    )
    try:
        summary = robot.run()
        summary["mode"] = args.mode
        summary["configuration"] = {"fast_measures": args.fast_measures, "fast_distance_m": 3000.0}
        if arena is not None:
            # Test truth is accessed ONLY after the robot has finished.
            summary.update(
                seed=args.seed,
                local_environment=arena.environment,
                source_count=len(arena.sources),
                actual_cleared_count=sum(s.cleared for s in arena.sources.values()),
                clear_fraction=sum(s.cleared for s in arena.sources.values()) / len(arena.sources),
                actual_all_cleared=all(s.cleared for s in arena.sources.values()),
            )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Run output: {run_dir.resolve()}")
        return (
            0
            if (
                summary["completed"]
                and summary["exit_confirmed"]
                and not summary["error"]
                and summary.get("actual_all_cleared", True)
            )
            else 2
        )
    finally:
        client.close()
