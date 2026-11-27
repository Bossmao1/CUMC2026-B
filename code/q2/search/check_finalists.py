"""Check final chosen points at finer circle geometry; preserve search outputs."""
import geometry
import model_snapshot as model
from compare import OUT, DOMAINS, D
import json
import math
import time


def refined_first_region():
    p = [(-1800.,-1800.),(1800.,-1800.),(1800.,1800.),(-1800.,1800.)]
    for angle, sign in ((-1,1),(1,-1)):
        a = math.radians(angle)
        p = geometry.clip(p,sign*math.sin(a),-sign*math.cos(a),0)
    p = geometry.clip(p,-1,0,-5*math.cos(D))
    for i in range(41):
        a = math.radians(-1+i*.05)
        p = geometry.clip(p,math.cos(a),math.sin(a),1500)
    return p


def main():
    start = time.perf_counter()
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    assert 'angle_checks' in result, 'Wait until comparison has completed.'
    model.P1 = refined_first_region()
    original_disk = geometry.disk_outer
    model.disk_outer = lambda poly,center,radius,sides=720: original_disk(poly,center,radius,sides=1440)
    rows = {}
    for domain in DOMAINS:
        point = result['stages'][-1]['best'][domain]['point']
        row = model.evaluate(tuple(point), angle_step=.001)
        row['travel_m'] = math.hypot(*point)
        row['move_plus_measure_s'] = row['travel_m']/5+5
        rows[domain] = row
        print(domain,json.dumps(row),flush=True)
    output = dict(rows=rows,first_circle_tangent_step_deg=.05,
                  second_circle_sides=1440,feedback_step_deg=.001,
                  wall_s=time.perf_counter()-start,
                  note='Chosen-point convergence check, not a new global search or certified bound.')
    (OUT/'geometry_check.json').write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
