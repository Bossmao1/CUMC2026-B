"""Open Euclidean routes with alternating reversals and short-chain relocation."""
import math
import random
from geometry import open_visit_order as original_order


def route_ils24(current,points,relocate=False,multistart=False):
    keys = list(points)
    if len(keys)<3:
        return original_order(current,points,relocate,multistart)
    coordinates = [current]+[points[k] for k in keys]
    n = len(keys)
    matrix = [[math.dist(a,b) for b in coordinates]+[0.] for a in coordinates]
    matrix.append([0.]*(n+2))
    end = n+1
    indices = {k:i+1 for i,k in enumerate(keys)}

    def length(path):
        return sum(matrix[a][b] for a,b in zip(path,path[1:]))

    def improve(path):
        for _ in range(60):
            best_delta,best = -1e-7,None
            for i in range(1,n):
                a,b = path[i-1],path[i]
                for j in range(i+1,n+1):
                    c,d = path[j],path[j+1]
                    delta = matrix[a][c]+matrix[b][d]-matrix[a][b]-matrix[c][d]
                    if delta<best_delta:
                        best_delta,best = delta,('reverse',i,j)
            for block in (1,2,3):
                for i in range(1,n-block+2):
                    a,b,c,d = path[i-1],path[i],path[i+block-1],path[i+block]
                    removal = matrix[a][d]-matrix[a][b]-matrix[c][d]
                    reduced = path[:i]+path[i+block:]
                    for j in range(len(reduced)-1):
                        e,f = reduced[j],reduced[j+1]
                        delta = removal+matrix[e][b]+matrix[c][f]-matrix[e][f]
                        if delta<best_delta:
                            best_delta,best = delta,('chain',i,block,j)
            if best is None:
                break
            if best[0]=='reverse':
                _,i,j = best
                path = path[:i]+list(reversed(path[i:j+1]))+path[j+1:]
            else:
                _,i,block,j = best
                chain = path[i:i+block]
                reduced = path[:i]+path[i+block:]
                path = reduced[:j+1]+chain+reduced[j+1:]
        return path

    old = original_order(current,points,relocate=relocate)
    best = improve([0]+[indices[k] for k in old]+[end])
    best_length = length(best)
    if n>=8:
        rng = random.Random(94721+n)
        for _ in range(24):
            a,b,c,d = sorted(rng.sample(range(1,n),4))
            nodes = best[1:-1]
            perturbed = nodes[:a]+nodes[c:d]+nodes[b:c]+nodes[a:b]+nodes[d:]
            candidate = improve([0]+perturbed+[end])
            candidate_length = length(candidate)
            if candidate_length<best_length-1e-7:
                best,best_length = candidate,candidate_length
    return [keys[node-1] for node in best[1:-1]]
