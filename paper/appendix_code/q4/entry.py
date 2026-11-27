import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from protocol import Client, HttpTransport
from strategy import Robot
Q4_ROOT = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser(description='T4.8 question-four robot')
    modes = parser.add_subparsers(dest='mode', required=True)
    for mode in ('demo', 'live'):
        p = modes.add_parser(mode)
        p.add_argument('--problem', type=int, choices=(4,), default=4)
        p.add_argument('--output', type=Path, default=Path('runs'))
        p.add_argument('--fast-measures', type=int, default=12)
        p.add_argument('--strategy', choices=('auto',), default='auto')
        p.add_argument('--quiet', action='store_true')
        if mode == 'demo':
            p.add_argument('--seed', type=int, default=20260910)
            p.add_argument('--profile', choices=('practice',), default='practice')
            p.add_argument('--error-mode', choices=('smooth', 'spatial', 'positive', 'negative'))
        else:
            p.add_argument('--robot-id', required=True, help='Currently logged-in competition team number')
            p.add_argument('--url', default='http://127.0.0.1:2026')
            p.add_argument('--wait-ready', type=float, default=60.0)
    args = parser.parse_args()
    if not 0 <= args.fast_measures <= 20:
        parser.error('--fast-measures must be in 0..20')
    output = args.output if args.output.is_absolute() else Q4_ROOT / args.output
    run_dir = output / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    arena = None
    if args.mode == 'demo':
        from local_simulator import LocalTransport, build_arena
        arena = build_arena(args.seed, 4, 'practice', error_mode=args.error_mode)
        transport, robot_id, wait = (LocalTransport(arena), 'local-test', 0)
    else:
        transport, robot_id, wait = (HttpTransport(args.url), args.robot_id, args.wait_ready)
    client = Client(transport, robot_id, run_dir / 'actions.jsonl', wait_ready=wait)
    robot = Robot(client, 4, fast_measures=args.fast_measures, progress=None if args.quiet else lambda message: print(message, flush=True), strategy='directional-informed')
    try:
        summary = robot.run()
        summary.update(mode=args.mode, release_version='T4.8', strategy='directional-refined', frozen_candidate='refined20', source_domain_m=1770.0, configuration={'fast_measures': args.fast_measures, 'fast_distance_m': 3000.0})
        if arena is not None:
            summary.update(seed=args.seed, local_environment=arena.environment, source_count=len(arena.sources), actual_cleared_count=sum((s.cleared for s in arena.sources.values())), clear_fraction=sum((s.cleared for s in arena.sources.values())) / len(arena.sources), actual_all_cleared=all((s.cleared for s in arena.sources.values())))
        (run_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Run output: {run_dir.resolve()}')
        return 0 if summary['completed'] and summary['exit_confirmed'] and (not summary['error']) and summary.get('actual_all_cleared', True) else 2
    finally:
        client.close()
if __name__ == '__main__':
    raise SystemExit(main())
