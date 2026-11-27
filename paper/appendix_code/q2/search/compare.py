import geometry
from model_snapshot import evaluate, P1, point_polygon_distance
import concurrent.futures
import json
import math
from pathlib import Path
import time
OUT = Path(__file__).resolve().parent
D = math.radians(1)
OLD_RADIUS = 1000 - math.sqrt(1500 ** 2 + 750 ** 2 - 2 * 1500 * 750 * math.cos(D))
DOMAINS = ('old_disk', 'full_safe', 'expanded')
def min_projection(s):
    x, y = s
    values = [x * math.cos(D) - abs(y) * math.sin(D)]
    opposite = (math.atan2(y, x) + math.pi + math.pi) % (2 * math.pi) - math.pi
    if abs(opposite) <= D:
        values.append(-math.hypot(x, y))
    return min(values)
def safe_slack(s):
    projection = min_projection(s)
    return min((max(1000, r) ** 2 - (s[0] ** 2 + s[1] ** 2 + r * r - 2 * r * projection) for r in (5, 1000, 1500)))
def member(s, domain):
    if domain == 'old_disk':
        return math.hypot(s[0] - 750, s[1]) <= OLD_RADIUS + 1e-08
    if domain == 'full_safe':
        return safe_slack(s) >= -1e-07
    return point_polygon_distance(s, P1) <= 1500 + 1e-08
def boundary(domain, angle_deg):
    a = math.radians(angle_deg)
    if domain == 'old_disk':
        return (750 + OLD_RADIUS * math.cos(a), OLD_RADIUS * math.sin(a))
    m = math.cos(a) * math.cos(D) - abs(math.sin(a)) * math.sin(D)
    if m < 0:
        return (0.0, 0.0)
    t = min(2000 * m, 3000 * m, 5 * m + math.sqrt(1000 ** 2 - 25 * (1 - m * m)))
    t = max(0.0, t - 1e-07)
    return (t * math.cos(a), t * math.sin(a))
def key(s):
    return tuple((round(v, 9) for v in s))
def task(args):
    s, step = args
    row = evaluate(s, angle_step=step)
    row['travel_m'] = math.hypot(*s)
    row['safe_slack_m2'] = safe_slack(s)
    row['domains'] = [d for d in DOMAINS if member(s, d)]
    return row
def seeds(rows, domain, count=4, separation=40):
    selected = []
    for row in sorted((r for r in rows if domain in r['domains']), key=lambda r: r['J']):
        if all((math.dist(row['point'], r['point']) >= separation for r in selected)):
            selected.append(row)
        if len(selected) == count:
            break
    return selected
def local_candidates(rows, half_width, spatial_step, boundary_width, boundary_step):
    points = set()
    for domain in DOMAINS:
        for row in seeds(rows, domain, separation=max(2, half_width / 2)):
            x, y = row['point']
            n = round(half_width / spatial_step)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    s = (x + i * spatial_step, y + j * spatial_step)
                    if s[1] >= 0 and member(s, domain):
                        points.add(key(s))
            if domain != 'expanded':
                origin_x = 750 if domain == 'old_disk' else 0
                a = math.degrees(math.atan2(y, x - origin_x))
                n = round(boundary_width / boundary_step)
                for j in range(-n, n + 1):
                    angle = a + j * boundary_step
                    if 0 <= angle <= (180 if domain == 'old_disk' else 89):
                        points.add(key(boundary(domain, angle)))
    points.update((key(r['point']) for d in DOMAINS for r in seeds(rows, d)))
    return sorted(points)
def main():
    start = time.perf_counter()
    points = {(float(x), float(y)) for x in range(-1500, 3001, 100) for y in range(0, 1601, 100) if member((x, y), 'expanded')}
    for domain, end in [('old_disk', 180), ('full_safe', 89)]:
        points.update((key(boundary(domain, a)) for a in range(end + 1)))
    points.update([(750.0, 200.0), (750.0, 600.0), (956.0, 520.0)])
    stages = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for name, step, spec in [('coarse', 0.1, None), ('local', 0.05, (100, 20, 8, 0.25)), ('fine', 0.025, (12, 2, 2, 0.05)), ('final', 0.005, (2, 0.5, 0.3, 0.01))]:
            if spec is not None:
                points = local_candidates(stages[-1]['rows'], *spec)
            rows = list(pool.map(task, ((p, step) for p in sorted(points)), chunksize=8))
            best = {d: seeds(rows, d, 1)[0] for d in DOMAINS}
            stages.append(dict(name=name, angle_step_deg=step, rows=rows, best=best))
            print(json.dumps(dict(stage=name, count=len(rows), best=best, wall_s=time.perf_counter() - start)), flush=True)
            (OUT / 'result.json').write_text(json.dumps(dict(stages=stages, old_radius_m=OLD_RADIUS, wall_s=time.perf_counter() - start), indent=2), encoding='utf-8')
        finalists = [tuple(stages[-1]['best'][d]['point']) for d in DOMAINS]
        checks = list(pool.map(task, ((p, step) for p in finalists for step in (0.0025, 0.001))))
    for row, (_, step) in zip(checks, ((p, step) for p in finalists for step in (0.0025, 0.001))):
        row['angle_step_deg'] = step
    result = dict(stages=stages, angle_checks=checks, old_radius_m=OLD_RADIUS, wall_s=time.perf_counter() - start, scope='origin first station, bearing zero, upper half-plane by symmetry; finite search', note='Same geometry and feedback grid precision across domains within each stage. No continuous global-optimality certificate. near<=5 surrogate cannot affect the winning scores, which all exceed 5 m. Expanded search uses an outer polygon of the exact source region; final safe membership is analytic.')
    (OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(final=stages[-1]['best'], checks=checks, wall_s=result['wall_s'])), flush=True)
if __name__ == '__main__':
    main()
