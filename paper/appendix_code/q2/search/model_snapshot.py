import json
import math
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from geometry import clip, covering_circle, convex_hull, distance, disk_outer, point_polygon_distance
def outside_disk_hull(poly, station, radius):
    if not poly:
        return []
    sx, sy = station
    points = [p for p in poly if distance(p, station) >= radius]
    for a, b in zip(poly, poly[1:] + poly[:1]):
        dx, dy = (b[0] - a[0], b[1] - a[1])
        ax, ay = (a[0] - sx, a[1] - sy)
        aa = dx * dx + dy * dy
        if aa < 1e-20:
            continue
        bb = 2 * (ax * dx + ay * dy)
        cc = ax * ax + ay * ay - radius * radius
        disc = bb * bb - 4 * aa * cc
        if disc < 0:
            continue
        root = math.sqrt(disc)
        for t in ((-bb - root) / (2 * aa), (-bb + root) / (2 * aa)):
            if 0 <= t <= 1:
                points.append((a[0] + t * dx, a[1] + t * dy))
    return convex_hull(points)
def first_region():
    poly = [(-1800.0, -1800.0), (1800.0, -1800.0), (1800.0, 1800.0), (-1800.0, 1800.0)]
    for angle, sign in ((-1, 1), (1, -1)):
        a = math.radians(angle)
        poly = clip(poly, sign * math.sin(a), -sign * math.cos(a), 0)
    poly = clip(poly, -1, 0, -5 * math.cos(math.radians(1)))
    for angle in (-1, -0.5, 0, 0.5, 1):
        a = math.radians(angle)
        poly = clip(poly, math.cos(a), math.sin(a), 1500)
    return poly
P1 = first_region()
def outcome_range(poly, station):
    if point_polygon_distance(station, poly) < 1e-07:
        return (-180.0, 180.0)
    angles = sorted((math.degrees(math.atan2(p[1] - station[1], p[0] - station[0])) % 360 for p in poly))
    gaps = [(angles[(i + 1) % len(angles)] + (360 if i == len(angles) - 1 else 0) - angles[i], i) for i in range(len(angles))]
    _, i = max(gaps)
    low = angles[(i + 1) % len(angles)]
    high = angles[i]
    if high < low:
        high += 360
    return (low - 1, high + 1)
def evaluate(station, angle_step=0.25, cutoff=None):
    sx, sy = station
    if sx == 0 and sy == 0:
        radius = covering_circle(P1)[1]
        return dict(point=station, J=radius, no_signal=0.0, direction=radius, beta=0.0, cut=False)
    negative = clip(P1, sx, sy, (sx * sx + sy * sy) / 2) if sx or sy else []
    negative = outside_disk_hull(negative, station, 1000)
    no_radius = covering_circle(negative)[1] if negative else 0.0
    near_radius = 5.0 if point_polygon_distance(station, P1) <= 5 else 0.0
    worst = max(no_radius, near_radius)
    if cutoff is not None and worst >= cutoff:
        return dict(point=station, J=worst, no_signal=no_radius, direction=None, beta=None, cut=True)
    base = disk_outer(P1, station, 1500, sides=720)
    direction = 0.0
    beta_worst = None
    if base:
        low, high = outcome_range(base, station)
        count = max(1, math.ceil((high - low) / angle_step))
        for i in range(count + 1):
            beta = low + (high - low) * i / count
            poly = base
            for angle, sign in ((beta - 1, 1), (beta + 1, -1)):
                a = math.radians(angle)
                nx, ny = (sign * math.sin(a), -sign * math.cos(a))
                poly = clip(poly, nx, ny, nx * sx + ny * sy)
            if not poly:
                continue
            if point_polygon_distance(station, poly) <= 5:
                poly = outside_disk_hull(poly, station, 5)
            if not poly:
                continue
            radius = covering_circle(poly)[1]
            if radius > direction:
                direction, beta_worst = (radius, beta % 360)
            if cutoff is not None and direction >= cutoff:
                return dict(point=station, J=max(worst, direction), no_signal=no_radius, direction=direction, beta=beta_worst, cut=True)
    return dict(point=station, J=max(worst, direction), no_signal=no_radius, direction=direction, beta=beta_worst, cut=False)
def main():
    start = time.perf_counter()
    output = Path(__file__).resolve().parent
    coarse = []
    for sx in range(-1500, 3001, 100):
        for sy in range(0, 1601, 100):
            if point_polygon_distance((sx, sy), P1) > 1500:
                continue
            coarse.append(evaluate((sx, sy), angle_step=0.5, cutoff=120))
    best = min(coarse, key=lambda r: r['J'])
    print('coarse best', best, flush=True)
    bx, by = best['point']
    fine = []
    for sx in range(int(bx) - 150, int(bx) + 151, 10):
        for sy in range(max(0, int(by) - 150), int(by) + 151, 10):
            fine.append(evaluate((sx, sy), angle_step=0.1, cutoff=best['J'] + 15))
    best = min(fine, key=lambda r: r['J'])
    print('fine best', best, flush=True)
    bx, by = best['point']
    final = []
    for sx in range(int(bx) - 12, int(bx) + 13, 2):
        for sy in range(int(by) - 12, int(by) + 13, 2):
            final.append(evaluate((sx, sy), angle_step=0.025, cutoff=best['J'] + 3))
    best = min(final, key=lambda r: r['J'])
    checks = [evaluate(best['point'], step) for step in (0.1, 0.025, 0.005)]
    table = [evaluate(p, 0.025) for p in ((0, 0), (750, 0), (750, 200), (750, 600), (1000, 600), best['point'])]
    value = dict(assumptions=dict(first_station=[0, 0], first_bearing_deg=0, angular_bound_deg=1, receive_radius_m=[1000, 1500], source_disk_m=1800, observation_model='bounded-error relaxation with direction/near/no_signal', caveat='Approximate continuous geometry and sampled bearings; no global or Delta certificate'), first_region=P1, prior_radius=covering_circle(P1)[1], coarse=coarse, fine=fine, refinement=final, best=best, angle_refinement=checks, table=table, elapsed_s=time.perf_counter() - start)
    with (output / 'result.json').open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2)
    print(json.dumps(dict(best=best, checks=checks, table=table, elapsed_s=value['elapsed_s']), indent=2), flush=True)
if __name__ == '__main__':
    main()
