"""Run a core experiment from saved seeds: python code/run.py 2 --count 10."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def task_case(question, seed):
    from local_simulator import LocalTransport, build_arena
    if question == 3:
        from client import Client
        from robot import Robot
    else:
        from protocol import Client
        from strategy import Robot
    arena = build_arena(seed, question, "practice")
    client = Client(LocalTransport(arena), "local-test", wait_ready=0)
    robot = Robot(client, question, fast_measures=12, progress=None,
                  strategy="auto" if question == 3 else "directional-informed")
    try:
        result = robot.run()
        # Read simulator truth only after the controller finishes.
        return {
            "seed": seed,
            "source_count": len(arena.sources),
            "remaining": sum(not source.cleared for source in arena.sources.values()),
            "result": result,
        }
    finally:
        client.close()


def summarize(question, rows):
    summary = {"question": question, "cases": len(rows)}
    if question == 2:
        summary["rmse_m"] = {
            name: math.sqrt(statistics.mean(row["policies"][name]["error_m"] ** 2 for row in rows))
            for name in ("first_only", "random_second", "proposed")
        }
    else:
        results = [row["result"] for row in rows]
        summary.update(
            all_clear_exit=sum(row["remaining"] == 0 and result["completed"]
                               and result["exit_confirmed"] and result["error"] is None
                               for row, result in zip(rows, results)),
            remaining_sources=sum(row["remaining"] for row in rows),
            mean_full_exit_s=statistics.mean(result["virtual_time_s"] for result in results),
            mean_per_source_s=statistics.mean(result["virtual_time_s"] / row["source_count"]
                                              for row, result in zip(rows, results)),
            failed_clear_count=sum(result["failed_clear_count"] for result in results),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="复现核心实验；不指定 --count 时运行全部保存种子。")
    parser.add_argument("question", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--count", type=int, help="运行前 N 个场景")
    parser.add_argument("--output", type=Path, help="新结果文件；默认 results/qN.jsonl")
    args = parser.parse_args()
    if args.count is not None and args.count <= 0:
        parser.error("--count 必须为正整数")
    if args.question == 1:
        return subprocess.call([sys.executable, str(ROOT / "q1/q1_geometry.py")])
    folder = ROOT / f"q{args.question}"
    sys.path.insert(0, str(folder))
    seeds = json.loads((folder / "seeds.json").read_text(encoding="utf-8"))
    if args.count is not None:
        seeds = seeds[:args.count]
    target = args.output or ROOT.parent / "results" / f"q{args.question}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with target.open("x", encoding="utf-8") as output:
        for i, seed in enumerate(seeds, 1):
            if args.question == 2:
                from benchmark_helpers import replay
                index = seed["scene_index"]
                row = replay(seed["seed"], index, f"q2-selection-comparison-heldout-20260912:{index}")
                row["seed"] = seed["seed"]
            else:
                row = task_case(args.question, seed)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if i % 100 == 0:
                print(f"{i}/{len(seeds)}", file=sys.stderr, flush=True)
    summary = summarize(args.question, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {target.resolve()}")
    return 0 if args.question == 2 or summary["all_clear_exit"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
