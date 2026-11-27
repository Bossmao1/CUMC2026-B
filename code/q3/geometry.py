"""Conservative bearing geometry. Distances are metres, angles are degrees."""

from itertools import combinations


import math


DELTA = 1.01


EPS = 1e-7


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip(poly, a, b, bound):
    """Intersect a convex polygon with a*x+b*y <= bound + EPS."""
    if not poly:
        return []
    bound += EPS  # Inflate, never shrink, a feasible set for numerical tolerance.
    result = []
    prev = poly[-1]
    vp = a * prev[0] + b * prev[1] - bound
    for cur in poly:
        vc = a * cur[0] + b * cur[1] - bound
        if (vc <= 0) != (vp <= 0):
            t = vp / (vp - vc)
            result.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
        if vc <= 0:
            result.append(cur)
        prev, vp = cur, vc
    clean = []
    for point in result:
        if not clean or distance(point, clean[-1]) > 1e-9:
            clean.append(point)
    if len(clean) > 1 and distance(clean[0], clean[-1]) < 1e-9:
        clean.pop()
    return clean


def disk_outer(poly, center, radius, sides=32):
    """Circumscribed regular polygon: never an inscribed disk approximation."""
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        a, b = math.cos(angle), math.sin(angle)
        poly = clip(poly, a, b, radius + a * center[0] + b * center[1])
    return poly


def initial_region():
    return disk_outer(
        [(-1770.0, -1770.0), (1770.0, -1770.0), (1770.0, 1770.0), (-1770.0, 1770.0)], (0, 0), 1770
    )


def bearing_region(poly, position, degrees, delta=DELTA):
    low, high = (math.radians(degrees + sign * delta) for sign in (-1, 1))
    # cross(v_low, g-s)>=0; cross(v_high, g-s)<=0.
    for a, b in ((math.sin(low), -math.cos(low)), (-math.sin(high), math.cos(high))):
        poly = clip(poly, a, b, a * position[0] + b * position[1])
    return disk_outer(poly, position, 1500)


def diameter(poly):
    return max((distance(a, b) for a, b in combinations(poly, 2)), default=0.0)


def covering_circle(poly):
    """Minimum enclosing circle of vertices; return a verified safe radius."""
    if not poly:
        raise ValueError("Empty region has no target covering circle")
    if len(poly) == 1:
        return poly[0], 0.0
    # Always start with a valid covering circle, including degenerate cases.
    center = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    best = max(distance(center, p) for p in poly)

    def consider(c, radius):
        nonlocal center, best
        if radius >= best or not all(math.isfinite(x) for x in (*c, radius)):
            return
        actual = max(distance(c, p) for p in poly)
        if actual <= radius + 1e-6 and actual < best:
            center, best = c, actual

    for a, b in combinations(poly, 2):
        c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        consider(c, distance(a, b) / 2)
    for a, b, c in combinations(poly, 3):
        bx, by, cx, cy = b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]
        det = 2 * (bx * cy - by * cx)
        if abs(det) < 1e-10:
            continue
        ub, uc = bx * bx + by * by, cx * cx + cy * cy
        center3 = (a[0] + (cy * ub - by * uc) / det, a[1] + (bx * uc - cx * ub) / det)
        consider(center3, distance(a, center3))
    return center, max(distance(center, p) for p in poly)


def point_polygon_distance(point, poly):
    if not poly:
        return math.inf
    if len(poly) == 1:
        return distance(point, poly[0])
    inside = len(poly) >= 3
    best = math.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        cross = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
        inside = inside and cross >= -1e-6
        norm = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / norm)) if norm else 0.0
        best = min(best, distance(point, (a[0] + t * dx, a[1] + t * dy)))
    return 0.0 if inside else best


def fallback_points(position, degrees):
    angle = math.radians(degrees)
    ex, ey = math.cos(angle), math.sin(angle)
    for row, z in enumerate((-30, -10, 10, 30)):
        ts = range(0, 1501, 20) if row % 2 == 0 else range(1500, -1, -20)
        for t in ts:
            yield (position[0] + t * ex - z * ey, position[1] + t * ey + z * ex)


def select_measurement(poly, current, first, history, budget):
    """T3.11 sampled move-plus-continuation score, with the original ordering.

    Samples rank candidate moves; they never certify a source or its absence.
    """
    center, radius = covering_circle(poly)
    origin, degrees = first
    angle = math.radians(degrees)
    e, normal = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
    middle = (origin[0] + 750 * e[0], origin[1] + 750 * e[1])
    candidates = [center, middle]
    candidates += [(middle[0] + z * normal[0], middle[1] + z * normal[1]) for z in (-220, 220)]
    step = max(30.0, min(240.0, radius * 0.65))
    for i in range(8):
        a = i * math.pi / 4
        candidates.append((center[0] + step * math.cos(a), center[1] + step * math.sin(a)))
    for fraction in (0.75, 1.0):
        base = (
            current[0] + fraction * (center[0] - current[0]),
            current[1] + fraction * (center[1] - current[1]),
        )
        candidates += [(base[0] + z * normal[0], base[1] + z * normal[1]) for z in (-100, -40, 40, 100)]
    samples = poly[:: max(1, len(poly) // 8)] + [center]
    best = None
    for s in candidates:
        travel = distance(current, s)
        if travel > budget or any(distance(s, p) < 2 for p in history):
            continue
        if max(distance(s, p) for p in poly) > 1000 - 1e-5:
            continue
        worst = 0.0
        continuation = []
        for target in samples:
            if distance(target, s) <= 5:
                worst = max(worst, 5.0)
                continuation.append(0.0)
                continue
            true_angle = math.degrees(math.atan2(target[1] - s[1], target[0] - s[0]))
            for error in (-DELTA, 0.0, DELTA):
                posterior = bearing_region(poly, s, true_angle + error)
                if posterior:
                    posterior_center, posterior_radius = covering_circle(posterior)
                    worst = max(worst, posterior_radius)
                    continuation.append(distance(s, posterior_center) + 2 * max(0.0, posterior_radius - 20))
        score = travel + sum(continuation) / len(continuation) if continuation else worst + 0.08 * travel
        if best is None or score < best[0]:
            best = (score, s)
    return best[1] if best else None


def _visit_order(current, points, relocate=False, first=None):
    """Nearest-neighbour initialization and open-path 2-opt; no true locations."""
    todo = dict(points)
    order, position = [], current
    if first is not None:
        order.append(first)
        position = todo.pop(first)
    while todo:
        key = min(todo, key=lambda k: distance(position, todo[k]))
        position = todo.pop(key)
        order.append(key)
    for _ in range(20):
        best = None
        for i in range(len(order) - 1):
            before = current if i == 0 else points[order[i - 1]]
            for j in range(i + 1, len(order)):
                a, b = points[order[i]], points[order[j]]
                delta = distance(before, b) - distance(before, a)
                if j + 1 < len(order):
                    after = points[order[j + 1]]
                    delta += distance(a, after) - distance(b, after)
                if delta < -1e-7 and (best is None or delta < best[0]):
                    best = delta, i, j
        if best is None:
            break
        _, i, j = best
        order[i : j + 1] = reversed(order[i : j + 1])
    if relocate:

        def path_length(keys):
            path = [current] + [points[k] for k in keys]
            return sum(distance(a, b) for a, b in zip(path, path[1:]))

        for _ in range(5):
            best_length, best_order = path_length(order), None
            for i, key in enumerate(order):
                reduced = order[:i] + order[i + 1 :]
                for j in range(len(order)):
                    candidate = reduced[:j] + [key] + reduced[j:]
                    candidate_length = path_length(candidate)
                    if candidate_length < best_length - 1e-7:
                        best_length, best_order = candidate_length, candidate
            if best_order is None:
                break
            order = best_order
    return order


def open_visit_order(current, points, relocate=False, multistart=False):
    best = _visit_order(current, points, relocate)
    if not multistart or len(points) < 3:
        return best

    def length(order):
        path = [current] + [points[k] for k in order]
        return sum(distance(a, b) for a, b in zip(path, path[1:]))

    best_length = length(best)
    # Every first task is an alternative initializer, not a forced final choice.
    for first in points:
        candidate = _visit_order(current, points, relocate, first)
        value = length(candidate)
        if value < best_length - 1e-7:
            best, best_length = candidate, value
    return best


def clear_approach(poly, current, center, limit=20 - 1e-4):
    """Closest point on the approach segment that contains P in a clear disk."""

    def safe(point):
        return max(distance(point, p) for p in poly) <= limit

    if not safe(center):
        return center
    if safe(current):
        return current
    lo, hi = 0.0, 1.0
    best = center
    for _ in range(25):
        mid = (lo + hi) / 2
        point = (current[0] + mid * (center[0] - current[0]), current[1] + mid * (center[1] - current[1]))
        if safe(point):
            hi, best = mid, point
        else:
            lo = mid
    return best if safe(best) else center


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def exclude_negative(poly, stations, sides=32):
    """Outer convex hull of P minus an inscribed polygon in each 1000m disk.

    Removing an inner approximation of each excluded disk retains every possible
    target. Taking a convex hull may restore excluded points; it cannot delete
    any feasible point. Directions alone still define the positive outer region.
    """
    apothem = 1000 * math.cos(math.pi / sides)
    for sx, sy in stations:
        if not poly:
            return []
        pieces = []
        for i in range(sides):
            angle = (i + 0.5) * 2 * math.pi / sides
            nx, ny = math.cos(angle), math.sin(angle)
            pieces.extend(clip(poly, -nx, -ny, -apothem - nx * sx - ny * sy))
        poly = convex_hull(pieces)
    return poly


def range_weighted_center(poly, positives, negatives):
    """Reception-history weighted route proxy; never alters the feasible region.

    Uses an area quadrature and a uniform 1000-1500 m radius design prior.
    This omits the joint spatial bearing-error likelihood; it is not a calibrated
    posterior. Small or unsupported regions keep the original circle center.
    """
    center, radius = covering_circle(poly)
    if radius <= 35 or len(poly) < 3:
        return center
    origin = tuple(sum(p[k] for p in poly) / len(poly) for k in (0, 1))
    total = sx = sy = 0.0
    subdivisions = 5
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ax, ay = a[0] - origin[0], a[1] - origin[1]
        bx, by = b[0] - origin[0], b[1] - origin[1]
        area = abs(ax * by - ay * bx) / 2 / subdivisions**2
        for i in range(subdivisions):
            for j in range(subdivisions - i):
                offsets = (1 / 3, 2 / 3) if i + j < subdivisions - 1 else (1 / 3,)
                for offset in offsets:
                    u, v = (i + offset) / subdivisions, (j + offset) / subdivisions
                    p = (origin[0] + u * ax + v * bx, origin[1] + u * ay + v * by)
                    lower = max([1000.0] + [distance(p, q) for q in positives])
                    upper = min([1500.0] + [distance(p, q) for q in negatives])
                    weight = area * max(0.0, upper - lower)
                    total += weight
                    sx += weight * p[0]
                    sy += weight * p[1]
    return (sx / total, sy / total) if total > 1e-10 else center
