from functools import lru_cache
import json
from pathlib import Path
from coverage_geometry import OmniCoverage
from geometry import distance, open_visit_order
@lru_cache(maxsize=1)
def layouts():
    candidates = json.loads(Path(__file__).with_name('rotation-layouts.json').read_text())
    unique = []
    seen = set()
    for layout in candidates:
        key = tuple(sorted(((tuple(point), axis) for point, axis in zip(layout['points'], layout['axes']))))
        if key in seen:
            continue
        seen.add(key)
        unique.append(layout)
    return unique
def replay_coverage(layout, negative_positions):
    coverage = OmniCoverage(refine_for=layout['points'])
    for channel, positions in negative_positions.items():
        for point in positions:
            coverage.no_signal(point, channel)
    return coverage
def choose(robot, targets, unknown):
    choices = layouts()
    current = robot.client.position
    best = choices[0]
    best_cost = float('inf')
    scores = []
    for layout in choices if targets and unknown else choices[:1]:
        tasks = {('target', channel): point for channel, point in targets.items()}
        if unknown:
            tasks.update({('anchor', i): tuple(point) for i, point in enumerate(layout['points']) if i})
        order = open_visit_order(current, tasks, relocate=True, multistart=True)
        previous = current
        cost = 0.0
        for key in order:
            cost += distance(previous, tasks[key])
            previous = tasks[key]
        scores.append(dict(name=layout['name'], route_m=cost))
        if cost < best_cost - 1e-06:
            best, best_cost = (layout, cost)
    report = dict(selected=best['name'], baseline_route_m=scores[0]['route_m'], selected_route_m=best_cost, predicted_saving_m=scores[0]['route_m'] - best_cost, pending_count=len(targets), scores=scores)
    return (best, replay_coverage(best, robot.negative_positions), report)
