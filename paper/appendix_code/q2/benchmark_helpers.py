from pathlib import Path
import hashlib
import json
import math
import random
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE
FROZEN = HERE
sys.path.insert(0, str(FROZEN))
from policies import Observation, initial_polygon, estimate, choose_station, world
from geometry_snapshot import covering_circle
from practice_snapshot import generate_practice, round_away, error_degrees, quantize_bearing_hundredths
OFFSET = (963.0, 508.5)
NAMES = ('first_only', 'random_second', 'proposed')
def sector_distance(p):
    x, y = p
    a = math.radians(1)
    angle = math.atan2(y, x)
    if -a <= angle <= a:
        length = math.hypot(x, y)
        return max(5 - length, length - 1500, 0.0)
    return min((math.hypot(x - t * math.cos(theta), y - t * math.sin(theta)) for theta in (-a, a) for t in [min(1500.0, max(5.0, x * math.cos(theta) + y * math.sin(theta)))]))
def random_station(first, seed_key):
    assert first.position == (0, 0)
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_key.encode()).digest(), 'big'))
    a = math.radians(1)
    while True:
        point = (rng.uniform(5 * math.cos(a) - 1500, 3000), rng.uniform(-1500 * (1 + math.sin(a)), 1500 * (1 + math.sin(a))))
        if sector_distance(point) <= 1500:
            return choose_station(first, point)
def score(first, jammer, noise_seed, seed_key, frozen_proposed=None):
    poly = initial_polygon(first)
    c, radius = covering_circle(poly)
    truth = (jammer['x_um'] / 1000000.0, jammer['y_um'] / 1000000.0)
    prediction = world(c, first)
    records = {'first_only': dict(estimate=prediction, error_m=math.dist(prediction, truth), posterior_radius_m=radius, second_kind='not_taken', fallback=False)}
    for name in NAMES[1:]:
        if name == 'proposed' and frozen_proposed is not None:
            records[name] = frozen_proposed
            continue
        station = random_station(first, seed_key) if name == 'random_second' else choose_station(first, OFFSET)
        second = measure(jammer, noise_seed, station)
        result = estimate(first, second, poly)
        destination = tuple((round_away(x * 1000000.0) / 1000000.0 for x in result['position']))
        error = math.dist(destination, truth)
        records[name] = dict(second_position=station, second_kind=second.kind, second_bearing=second.bearing, estimate=destination, error_m=error, posterior_radius_m=result['radius'], fallback=result['fallback'], enclosure_violation=error > result['radius'] + 0.0001)
    return records
def measure(jammer, noise_seed, position):
    target = (jammer['x_um'] / 1000000.0, jammer['y_um'] / 1000000.0)
    d = math.dist(target, position)
    if d > jammer['max_receive_um'] / 1000000.0:
        return Observation(position, jammer['channel'], 'no_signal')
    if d <= 5:
        return Observation(position, jammer['channel'], 'near')
    exact = math.degrees(math.atan2(target[1] - position[1], target[0] - position[0])) % 360
    error = error_degrees(noise_seed, jammer['channel'], *position)
    beta = quantize_bearing_hundredths(exact, error) / 100
    assert abs((beta - exact + 180) % 360 - 180) <= 1 + 1e-10
    return Observation(position, jammer['channel'], 'direction', beta)
def replay(seed_hex, index, random_key):
    seed = bytes.fromhex(seed_hex)
    scene = generate_practice(seed, 3)
    chooser = random.Random(int.from_bytes(hashlib.sha256(seed + b'/selection').digest(), 'big'))
    jammer = scene['jammers'][chooser.randrange(len(scene['jammers']))]
    first = measure(jammer, scene['noise_seed'], (0, 0))
    assert first.kind == 'direction'
    policies = score(first, jammer, scene['noise_seed'], random_key)
    return dict(scene_index=index, first_bearing=first.bearing, channel=jammer['channel'], policies=policies)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='第二问固定场景下的三种选点规则求解')
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--output', type=Path, default=Path('q2_results_new.jsonl'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (HERE / 'results.jsonl').open(encoding='utf-8') as source, args.output.open('w', encoding='utf-8') as output:
        for i, line in enumerate(source):
            if args.count is not None and i >= args.count:
                break
            case = json.loads(line)
            index = case['scene_index']
            result = replay(case['seed'], index, f'q2-selection-comparison-heldout-20260912:{index}')
            result['seed'] = case['seed']
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
