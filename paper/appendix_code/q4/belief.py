import math
from functools import lru_cache
import numpy as np
from geometry import bearing_region, covering_circle, distance, enroute_measurement, open_visit_order as quick_order
from routes import route_ils24
from gridline import symmetric_stations
@lru_cache(maxsize=1)
def hypotheses():
    positions = []
    for i in range(384):
        r = 1770 * math.sqrt((i + 0.5) / 384)
        a = i * math.pi * (3 - math.sqrt(5))
        positions.append((r * math.cos(a), r * math.sin(a)))
    rows = [(x, y, r, math.cos(a * math.pi / 8), math.sin(a * math.pi / 8), omni) for x, y in positions for r in (1000.0, 1250.0, 1500.0) for a in range(16) for omni in (False, True)]
    return np.asarray(rows, dtype=np.float64)
@lru_cache(maxsize=4096)
def visible_mask(point):
    h = hypotheses()
    dx, dy = (point[0] - h[:, 0], point[1] - h[:, 1])
    visible = (dx * dx + dy * dy <= h[:, 2] ** 2) & ((h[:, 5] > 0) | (dx * h[:, 3] + dy * h[:, 4] >= 0))
    return int.from_bytes(np.packbits(visible, bitorder='little').tobytes(), 'little')
@lru_cache(maxsize=512)
def visible_x_group(point):
    delta = hypotheses()[:, :2] - np.asarray(point)
    mask = np.abs(delta[:, 1]) >= np.abs(delta[:, 0])
    packed = np.packbits(mask, bitorder='little').tobytes()
    return int.from_bytes(packed, 'little') & visible_mask(point)
def refine_detour(point, following, axes):
    plans = []
    for axis in axes:
        midpoint = list(point)
        coordinate = 0 if axis == 'x' else 1
        midpoint[coordinate] = round(midpoint[coordinate] / 150) * 150.0
        p, q = symmetric_stations(midpoint, axis, 6.0)
        if math.dist(point, q) < math.dist(point, p):
            p, q = (q, p)
        plans.append((math.dist(point, p), p, q))
    previous = point
    extra = 0.0
    for _, p, q in sorted(plans):
        extra += math.dist(previous, q) if tuple(p) == tuple(point) else math.dist(previous, p) + math.dist(p, q)
        previous = q
    if following is not None:
        extra += math.dist(previous, following) - math.dist(point, following)
    return extra
class Belief:
    def __init__(self):
        self.total = len(hypotheses())
        self.unseen = (1 << self.total) - 1
    def observe_full_survey(self, point):
        self.unseen &= ~visible_mask(tuple(point))
    def expected_remaining(self, known_count):
        if known_count >= 16 or not self.unseen:
            return 0.0
        q = self.unseen.bit_count() / self.total
        weights = [(n, math.comb(n, known_count) * q ** (n - known_count)) for n in range(max(10, known_count), 17)]
        normalizer = sum((w for _, w in weights))
        return sum(((n - known_count) * w for n, w in weights)) / normalizer if normalizer else 0.0
    def _base_cost(self, start, order, tasks, known_count, unknown_count, detour_weight):
        path = np.asarray([start] + [tasks[k] for k in order], dtype=float)
        edges = np.sqrt(((path[1:] - path[:-1]) ** 2).sum(axis=1))
        total = float(edges.sum())
        initial_count = self.unseen.bit_count()
        remaining = self.expected_remaining(known_count)
        if not initial_count:
            return total + 30 * unknown_count * sum((k[0] == 'anchor' for k in order))
        unseen = self.unseen
        for j, key in enumerate(order, 1):
            if key[0] != 'anchor':
                continue
            outstanding = unknown_count - remaining * (1 - unseen.bit_count() / initial_count)
            total += 30 * outstanding
            newly_seen = unseen & visible_mask(tuple(tasks[key]))
            unseen &= ~newly_seen
        return total
    def route_cost(self, start, order, tasks, known_count, unknown_count, detour_weight):
        score = self._base_cost(start, order, tasks, known_count, unknown_count, detour_weight)
        total = self.unseen.bit_count()
        expected = self.expected_remaining(known_count)
        if not total or expected <= 0:
            return score
        unseen = self.unseen
        for i, key in enumerate(order):
            if key[0] != 'anchor':
                continue
            point = tuple(tasks[key])
            found = unseen & visible_mask(point)
            xcount = (found & visible_x_group(point)).bit_count()
            ycount = found.bit_count() - xcount
            px, py = (xcount / total, ycount / total)
            none = max(0.0, 1 - px - py) ** expected
            only_x = (1 - py) ** expected - none
            only_y = (1 - px) ** expected - none
            both = max(0.0, 1 - none - only_x - only_y)
            following = tasks[order[i + 1]] if i + 1 < len(order) else None
            score += only_x * refine_detour(point, following, ('x',))
            score += only_y * refine_detour(point, following, ('y',))
            score += both * refine_detour(point, following, ('x', 'y'))
            unseen &= ~found
        return score
def belief_first_task(start, tasks, regions, observers, anchors, clear_threshold, *, belief, known_count, unknown_count, detour_weight):
    order = route_ils24(start, tasks, relocate=True)
    selected = order[0]
    best_score = belief.route_cost(start, order, tasks, known_count, unknown_count, detour_weight)
    for key, anchor in tasks.items():
        if key[0] != 'anchor':
            continue
        benefit = 0.0
        if key in anchors:
            choice = enroute_measurement(start, anchor, regions, observers, clear_threshold)
            if choice is not None:
                station, channels = choice
                for channel in channels:
                    center, radius = covering_circle(regions[channel])
                    angle = math.degrees(math.atan2(center[1] - station[1], center[0] - station[0]))
                    posterior = bearing_region(regions[channel], station, angle)
                    if posterior:
                        benefit += min(radius - covering_circle(posterior)[1], 300) - 30
        remaining = {k: p for k, p in tasks.items() if k != key}
        after = quick_order(anchor, remaining, relocate=True)
        candidate = [key] + after
        score = belief.route_cost(start, candidate, tasks, known_count, unknown_count, detour_weight) - 0.5 * benefit
        if score < best_score:
            best_score, selected = (score, key)
    return selected
