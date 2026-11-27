import math
from geometry import convex_hull, distance, open_visit_order
def nearest_lens(current, centres, radius=999.999):
    def safe(p):
        return all((distance(p, c) <= radius + 1e-07 for c in centres))
    if safe(current):
        return current
    for i, a in enumerate(centres):
        if any((distance(a, b) > 2 * radius + 1e-07 for b in centres[i + 1:])):
            return None
    candidates = []
    for a in centres:
        d = distance(a, current)
        if d > 1e-09:
            candidates.append((a[0] + radius * (current[0] - a[0]) / d, a[1] + radius * (current[1] - a[1]) / d))
    for i, a in enumerate(centres):
        for b in centres[i + 1:]:
            d = distance(a, b)
            if d < 1e-09 or d > 2 * radius:
                continue
            h = math.sqrt(max(0.0, radius * radius - d * d / 4))
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            normal = (-(b[1] - a[1]) / d, (b[0] - a[0]) / d)
            for sign in (-1, 1):
                candidates.append((mid[0] + sign * h * normal[0], mid[1] + sign * h * normal[1]))
    for point in sorted(candidates, key=lambda p: distance(current, p)):
        if safe(point):
            return point
    return None
def final_station(bot, anchors, unknown):
    if not unknown:
        return None
    required = 0
    for channel in unknown:
        required |= bot.coverage.remaining[channel]
    if not required:
        return None
    corners = []
    mask = required
    while mask:
        bit = mask & -mask
        i = bit.bit_length() - 1
        x, y = bot.coverage.cells[i]
        h = bot.coverage.halves[i]
        corners.extend(((x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)))
        mask -= bit
    current = bot.client.position
    point = nearest_lens(current, convex_hull(corners))
    if point is None:
        return None
    point = tuple((round(x, 6) for x in point))
    if required & ~bot.coverage.mask(point):
        return None
    useful = {i: p for i, p in anchors.items() if bot.coverage.mask(p) & required}
    order = open_visit_order(current, useful, relocate=True, multistart=True)
    path = [current] + [useful[k] for k in order]
    before = sum((distance(a, b) for a, b in zip(path, path[1:]))) + 30 * len(unknown) * len(order)
    after = distance(current, point) + 30 * len(unknown)
    return point if after < before - 1e-05 else None
