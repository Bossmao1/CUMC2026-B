from itertools import combinations
import math
import numpy as np
from scipy.optimize import linprog
TOL = 1e-07
def constraints(stations, bearings, delta=1.0):
    rows, rhs = ([], [])
    for (x, y), degrees in zip(stations, bearings):
        lo, hi = (math.radians(degrees - delta), math.radians(degrees + delta))
        for a, b in ((math.sin(lo), -math.cos(lo)), (-math.sin(hi), math.cos(hi))):
            rows.append((a, b))
            rhs.append(a * x + b * y)
    return (np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
def hull(points):
    points = sorted(set(((float(x), float(y)) for x, y in points)))
    clean = []
    for p in points:
        if not clean or math.dist(p, clean[-1]) > 1e-06:
            clean.append(p)
    if len(clean) <= 1:
        return np.asarray(clean)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    upper = []
    for p in clean:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 1e-09:
            lower.pop()
        lower.append(p)
    for p in reversed(clean):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 1e-09:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])
def intersection(A, b):
    if len(b) == 0:
        return ('unbounded', np.empty((0, 2)))
    def lp(c):
        return linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 2, method='highs', options={'primal_feasibility_tolerance': 1e-08})
    feasible = lp([0.0, 0.0])
    if feasible.status == 2:
        return ('empty', np.empty((0, 2)))
    if not feasible.success:
        raise RuntimeError(feasible.message)
    for c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r = lp(c)
        if r.status == 3:
            return ('unbounded', np.empty((0, 2)))
        if not r.success:
            raise RuntimeError(r.message)
    vertices = []
    for i, j in combinations(range(len(b)), 2):
        pair = A[[i, j]]
        if abs(np.linalg.det(pair)) < 1e-12:
            continue
        p = np.linalg.solve(pair, b[[i, j]])
        if np.max(A @ p - b) <= TOL:
            vertices.append(p)
    if not vertices:
        raise RuntimeError('Bounded LP but no stable vertex: inspect numerical degeneracy')
    return ('bounded', hull(vertices))
def enclosing_circle(poly):
    if len(poly) == 1:
        return (poly[0].copy(), 0.0)
    center = np.mean(poly, axis=0)
    best = max(np.linalg.norm(poly - center, axis=1))
    def consider(c, r):
        nonlocal center, best
        if not np.all(np.isfinite(c)) or r >= best:
            return
        actual = max(np.linalg.norm(poly - c, axis=1))
        if actual <= r + TOL and actual < best:
            center, best = (c, actual)
    for a, b in combinations(poly, 2):
        consider((a + b) / 2, math.dist(a, b) / 2)
    for a, b, c in combinations(poly, 3):
        B = b - a
        C = c - a
        det = 2 * (B[0] * C[1] - B[1] * C[0])
        if abs(det) < 1e-12:
            continue
        q = a + np.array([C[1] * np.dot(B, B) - B[1] * np.dot(C, C), B[0] * np.dot(C, C) - C[0] * np.dot(B, B)]) / det
        consider(q, math.dist(q, a))
    return (center, float(best))
def describe(A, b):
    status, poly = intersection(A, b)
    result = dict(status=status, vertices=poly.tolist())
    if status != 'bounded':
        return result
    a, bp = max(combinations(range(len(poly)), 2), key=lambda ij: math.dist(poly[ij[0]], poly[ij[1]])) if len(poly) > 1 else (0, 0)
    D = math.dist(poly[a], poly[bp])
    mid = (poly[a] + poly[bp]) / 2
    maxdist = float(max(np.linalg.norm(poly - mid, axis=1)))
    c, r = enclosing_circle(poly)
    area = abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))) / 2
    result.update(vertex_count=len(poly), diameter=D, diameter_indices=[a, bp], diameter_midpoint=mid.tolist(), diameter_circle_max_distance=maxdist, diameter_circle_covers=maxdist <= D / 2 + TOL, enclosing_center=c.tolist(), enclosing_radius=r, area=area, constraint_residual=float(np.max(A @ poly.T - b[:, None])))
    return result
def locate(stations, bearings, delta=1.0):
    A, b = constraints(stations, bearings, delta)
    return describe(A, b)
if __name__ == '__main__':
    import json
    from pathlib import Path
    data = json.loads(Path(__file__).with_name('cases.json').read_text(encoding='utf-8'))
    example = data['paper_example']
    result = {'paper_example': {}, 'constructed_cases': []}
    for name, count in (('two', 2), ('three', 3)):
        result['paper_example'][name] = locate(example['stations'][:count], example['bearings_deg'][:count], example['delta_deg'])
    result['paper_example']['triangle'] = locate(example['triangle_stations'], example['triangle_bearings_deg'], example['delta_deg'])
    for case in data['constructed_cases']:
        result['constructed_cases'].append({'id': case['id'], 'result': locate(case['stations'], case['bearings_deg'], case['delta_deg'])})
    print(json.dumps(result, ensure_ascii=False, indent=2))
