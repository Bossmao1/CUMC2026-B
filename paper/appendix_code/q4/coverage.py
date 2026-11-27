import gzip
import json
import math
from functools import lru_cache
from pathlib import Path
from geometry import convex_hull, point_polygon_distance, clip, distance, open_visit_order
def clear_cover_points(poly, center, degrees):
    a = math.radians(degrees)
    ex, ey = (math.cos(a), math.sin(a))
    local = [((p[0] - center[0]) * ex + (p[1] - center[1]) * ey, -(p[0] - center[0]) * ey + (p[1] - center[1]) * ex) for p in poly]
    xmin, xmax = (min((p[0] for p in local)) - 20, max((p[0] for p in local)) + 20)
    ymin, ymax = (min((p[1] for p in local)) - 20, max((p[1] for p in local)) + 20)
    step = 34.0
    height = step * math.sqrt(3) / 2
    points = []
    for j in range(math.floor(ymin / height), math.ceil(ymax / height) + 1):
        shift = j % 2 * step / 2
        for i in range(math.floor((xmin - shift) / step), math.ceil((xmax - shift) / step) + 1):
            x, y = (i * step + shift, j * height)
            point = (center[0] + x * ex - y * ey, center[1] + x * ey + y * ex)
            if point_polygon_distance(point, poly) <= 20 + 1e-05:
                points.append(point)
    return points
@lru_cache(maxsize=1)
def _proofs():
    return json.loads(gzip.decompress(Path(__file__).with_name('coverage_proofs.json.gz').read_bytes()))
class Coverage:
    def __init__(self, rotation_degrees=0.0):
        angle = float(rotation_degrees) % 360.0
        if angle not in range(0, 360, 15):
            raise ValueError('The frozen strategy uses the 24 certified orientations')
        proof = _proofs()[str(int(angle))]
        self.points = tuple((tuple(p) for p in proof['points']))
        self.cells = tuple((tuple((tuple(v) for v in cell)) for cell in proof['cells']))
        self.parents = tuple((frozenset((tuple(p) for p in parent)) for parent in proof['parents']))
        self.distance_margin_squared = 1.0
        self.remaining = set(range(len(self.cells)))
        self.stations = set()
        self.witnesses = [[] for _ in self.cells]
        self.cache = {}
    def eligible(self, point):
        point = tuple(point)
        if point not in self.cache:
            self.cache[point] = tuple((i for i, cell in enumerate(self.cells) if all(((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 < 1000 ** 2 - getattr(self, 'distance_margin_squared', 0.001) for p in cell))))
        return self.cache[point]
    def add_survey(self, point):
        point = tuple(point)
        if point in self.stations:
            return not self.remaining
        self.stations.add(point)
        for i in self.eligible(point):
            if i not in self.remaining:
                continue
            self.witnesses[i].append(point)
            if self.parents[i] <= self.stations:
                self.remaining.remove(i)
                continue
            if strict_hull(self.cells[i], self.witnesses[i]):
                self.remaining.remove(i)
        return not self.remaining
    def anchor_needed(self, point):
        return any((point in self.parents[i] for i in self.remaining))
def strict_hull(cell, points):
    hull = convex_hull(points)
    return len(hull) >= 3 and all(((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 1e-05 * math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(hull, hull[1:] + hull[:1]) for p in cell))
def route_length(start, nodes):
    order = open_visit_order(start, nodes, relocate=True)
    return sum((distance(a, b) for a, b in zip([start] + [nodes[k] for k in order], [nodes[k] for k in order])))
def certify_cell(cell, points, depth=0):
    eligible = [p for p in points if all((distance(p, v) ** 2 < 999999.0 for v in cell))]
    if strict_hull(cell, eligible):
        return [(tuple(cell), tuple(convex_hull(eligible)))]
    if depth >= 3:
        return None
    xmin, xmax = (min((p[0] for p in cell)), max((p[0] for p in cell)))
    ymin, ymax = (min((p[1] for p in cell)), max((p[1] for p in cell)))
    xm, ym = ((xmin + xmax) / 2, (ymin + ymax) / 2)
    leaves = []
    for a, b in ((xmin, xm), (xm, xmax)):
        for c, d in ((ymin, ym), (ym, ymax)):
            child = list(cell)
            for nx, ny, limit in ((1, 0, b), (-1, 0, -a), (0, 1, d), (0, -1, -c)):
                child = clip(child, nx, ny, limit)
            if not child:
                continue
            proof = certify_cell(child, points, depth + 1)
            if proof is None:
                return None
            leaves.extend(proof)
    return leaves or None
def move_one(coverage, start, index, anchors, tasks):
    old = tuple(anchors[index])
    others = set(coverage.stations) | {tuple(p) for k, p in anchors.items() if k != index}
    eligible = {p: set(coverage.eligible(p)) for p in others}
    updates = {}
    constraints = []
    for i in sorted(coverage.remaining):
        if coverage.parents[i] <= others:
            continue
        witnesses = [p for p in others if i in eligible[p]]
        if strict_hull(coverage.cells[i], witnesses):
            updates[i] = [(coverage.cells[i], tuple(convex_hull(witnesses)))]
        else:
            constraints.append(i)
    if not constraints:
        return None
    def certify(p):
        found = {}
        for i in constraints:
            proof = certify_cell(coverage.cells[i], list(others) + [tuple(p)])
            if proof is None:
                return None
            found[i] = proof
        return found
    remaining = {k: p for k, p in tasks.items() if k != ('anchor', index)}
    following = remaining[open_visit_order(old, remaining, relocate=True)[0]] if remaining else start
    dx, dy = (following[0] - start[0], following[1] - start[1])
    den = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((old[0] - start[0]) * dx + (old[1] - start[1]) * dy) / den)) if den else 0.0
    ideals = [(start[0] + t * dx, start[1] + t * dy)] + [p for k, p in sorted(remaining.items(), key=lambda kv: distance(old, kv[1])) if k[0] == 'target' and distance(old, p) <= 600.0][:2]
    best = old
    best_proof = None
    best_cost = route_length(start, tasks)
    for ideal in ideals:
        proof = certify(ideal)
        candidate = ideal
        if proof is None:
            lo, hi = (0.0, 1.0)
            for _ in range(10):
                f = (lo + hi) / 2
                p = tuple((a + f * (b - a) for a, b in zip(old, ideal)))
                if certify(p) is not None:
                    lo = f
                else:
                    hi = f
            candidate = tuple((a + 0.98 * lo * (b - a) for a, b in zip(old, ideal)))
            proof = certify(candidate)
        if proof is None:
            continue
        changed = dict(tasks)
        changed['anchor', index] = candidate
        cost = route_length(start, changed)
        if cost < best_cost - 1.0:
            best, best_proof, best_cost = (candidate, proof, cost)
    if best_proof is None:
        return old
    updates.update(best_proof)
    cells = []
    parents = []
    for i, cell in enumerate(coverage.cells):
        for child, parent in updates.get(i, [(cell, tuple(coverage.parents[i]))]):
            cells.append(child)
            parents.append(frozenset(parent))
    actual = tuple(coverage.stations)
    coverage.cells = tuple(cells)
    coverage.parents = tuple(parents)
    coverage.remaining = set(range(len(cells)))
    coverage.stations = set()
    coverage.witnesses = [[] for _ in cells]
    coverage.cache = {}
    for p in actual:
        coverage.add_survey(p)
    return best
