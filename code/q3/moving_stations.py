"""Move future stations while retaining coverage of all real remaining cells."""
import math
from geometry import convex_hull,distance,open_visit_order

RADIUS=999.999

def line_interval(corners,axis,fixed):
    fixed_index=0 if axis=='x' else 1
    free_index=1-fixed_index
    low,high=-math.inf,math.inf
    for point in corners:
        residual=RADIUS**2-(point[fixed_index]-fixed)**2
        if residual<0:return None
        width=math.sqrt(residual)
        low=max(low,point[free_index]-width)
        high=min(high,point[free_index]+width)
        if low>high:return None
    return low,high

def on_line(axis,fixed,free):
    return (fixed,free) if axis=='x' else (free,fixed)

def best_on_line(corners,axis,fixed,previous,following):
    interval=line_interval(corners,axis,fixed)
    if interval is None:return None
    low,high=interval
    def length(free):
        point=on_line(axis,fixed,free)
        return distance(previous,point)+(distance(point,following) if following is not None else 0.)
    for _ in range(42):
        one=(2*low+high)/3
        two=(low+2*high)/3
        if length(one)<=length(two):high=two
        else:low=one
    free=round((low+high)/2,6)
    point=on_line(axis,float(fixed),free)
    if max(distance(point,p) for p in corners)>1000-1e-5:return None
    return point,length(free)

def adjust(robot,points,remaining,targets,unknown):
    if not unknown or not remaining:return
    stats=getattr(robot,'moving_stats',None)
    if stats is None:
        robot.moving_stats=stats=dict(evaluations=0,accepted=0,removed=0,predicted_saving_m=0.)
    coverage=robot.coverage
    required=0
    for channel in unknown:required|=coverage.remaining[channel]
    masks={i:coverage.mask(points[i]) for i in remaining}
    tasks={('target',k):p for k,p in targets.items()}
    tasks.update({('anchor',i):points[i] for i in sorted(remaining)})
    order=open_visit_order(robot.client.position,tasks,relocate=True,multistart=True)
    # A single coordinate pass follows the current complete route. The route
    # is recomputed by the caller after this pass using only observed targets.
    for offset,key in enumerate(order):
        if key[0]!='anchor':continue
        index=key[1]
        others=0
        for j,mask in masks.items():
            if j!=index:others|=mask
        own=required & ~others
        if own==0:
            remaining.remove(index)
            masks.pop(index)
            stats['removed']+=1
            continue
        if own & ~masks[index]:raise ValueError('Existing future coverage lost')
        corners=[]
        bits=own
        while bits:
            bit=bits & -bits
            j=bit.bit_length()-1
            bits-=bit
            (x,y),half=coverage.cells[j],coverage.halves[j]
            corners.extend((x+dx,y+dy) for dx in (-half,half) for dy in (-half,half))
        corners=convex_hull(corners)
        earlier=[k for k in order[:offset] if k[0]=='target' or k[1] in remaining]
        later=[k for k in order[offset+1:] if k[0]=='target' or k[1] in remaining]
        previous=tasks[earlier[-1]] if earlier else robot.client.position
        following=tasks[later[0]] if later else None
        best=points[index]
        best_axis=robot.rotation_axes[index]
        initial=distance(previous,best)+(distance(best,following) if following is not None else 0.)
        best_cost=initial
        stats['evaluations']+=1
        for axis,coordinate in (('x',0),('y',1)):
            low=max(p[coordinate]-RADIUS for p in corners)
            high=min(p[coordinate]+RADIUS for p in corners)
            for k in range(math.ceil((low+6)/150),math.floor((high+6)/150)+1):
                result=best_on_line(corners,axis,150*k-6,previous,following)
                if result is None:continue
                point,cost=result
                if cost>=best_cost-1.:continue
                if own & ~coverage.mask(point):continue
                best,best_axis,best_cost=point,axis,cost
        if best_cost<initial-1.:
            points[index]=best
            tasks[key]=best
            robot.rotation_axes[index]=best_axis
            masks[index]=coverage.mask(best)
            stats['accepted']+=1
            stats['predicted_saving_m']+=initial-best_cost
    union=0
    for mask in masks.values():union|=mask
    if required & ~union:raise ValueError('Moved stations lack complete coverage')
