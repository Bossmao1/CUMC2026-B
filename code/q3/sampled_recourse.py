"""Approximate prior recourse for route choice; never an absence certificate."""
import bisect,math,random
from functools import lru_cache
from geometry import distance,open_visit_order

# Equal-area deterministic disk quadrature. This is independent of arena seeds.
POINTS=tuple((1770*math.sqrt((i+.5)/2048)*math.cos(i*math.pi*(3-math.sqrt(5))),
              1770*math.sqrt((i+.5)/2048)*math.sin(i*math.pi*(3-math.sqrt(5)))) for i in range(2048))

@lru_cache(maxsize=256)
def spatial(history):
    highs=[min([1500.]+[distance(p,s) for s in history]) for p in POINTS]
    weights=[max(0.,h-1000.) for h in highs]
    cdf=[];total=0.
    for w in weights:total+=w;cdf.append(total)
    return total/(500*len(POINTS)),cdf,highs

def count_distribution(known,qs):
    # Elementary symmetric sums of per-channel no-signal probabilities.
    e=[[1.]]
    for q in qs:
        prev=e[-1];n=len(prev)
        e.append([(prev[m] if m<n else 0.)+(q*prev[m-1] if m else 0.) for m in range(n+1)])
    values=[]
    for m in range(max(0,10-known),min(len(qs),16-known)+1):
        values.append((m,e[-1][m]/math.comb(20,known+m)))
    total=sum(w for _,w in values)
    return ([(m,w/total) for m,w in values] if total else []),e

def worlds(bot,tasks,rng):
    known=bot.cleared|set(k[1] for k in tasks if k[0]=='target')
    unknown=sorted(set(range(1,21))-known-bot.excluded)
    distributions=[spatial(tuple(bot.negative_positions[k])) for k in unknown]
    probabilities,e=count_distribution(len(known),[d[0] for d in distributions])
    if not probabilities:return [],len(unknown)
    result=[]
    for _ in range(bot.sample_count):
        u=rng.random();m=probabilities[-1][0]
        for count,p in probabilities:
            u-=p
            if u<0:m=count;break
        selected=[]
        for j in range(len(unknown),0,-1):
            if not m:break
            q=distributions[j-1][0]
            inclusion=q*e[j-1][m-1]/e[j][m] if m<=j and e[j][m]>0 else 0.
            if rng.random()<inclusion:selected.append(j-1);m-=1
        assert m==0
        sample=[]
        for j in selected:
            q,cdf,highs=distributions[j]
            index=min(len(POINTS)-1,bisect.bisect_right(cdf,rng.random()*cdf[-1]))
            radius=1000.+rng.random()*(highs[index]-1000.)
            sample.append((unknown[j],POINTS[index],radius))
        rng.shuffle(sample)
        result.append(sample)
    return result,len(unknown)

def length(current,tasks,order):
    total=0.
    for key in order:total+=distance(current,tasks[key]);current=tasks[key]
    return total

def route_score(current,tasks,order,samples,unknown_count):
    """Insert each sampled source only after a real anchor could discover it.

    Known targets use observed estimates. Anchor positions are held fixed in
    the forecast; localization of a future discovery is optimistic. Neither
    approximation is used by the real localization or stopping conditions.
    """
    path=[(key,tasks[key]) for key in order]
    base=length(current,tasks,order)
    anchor_count=sum(k[0]=='anchor' for k in order)
    # Five seconds per measurement plus a one-second switch proxy, in metres.
    extra=30.*anchor_count*(unknown_count-len(samples))
    for channel,point,radius in samples:
        discovery=None;seen=0
        for i,(key,position) in enumerate(path):
            if key[0]!='anchor':continue
            seen+=1
            if distance(position,point)<=radius:discovery=i;break
        if discovery is None:
            # Finite quadrature and moved stations can make this proxy invalid.
            # Reject the forecast, not the source/channel or physical task.
            return None
        extra+=30.*seen
        best=None
        for after in range(discovery,len(path)):
            a=path[after][1]
            delta=distance(a,point)
            if after+1<len(path):
                b=path[after+1][1];delta+=distance(point,b)-distance(a,b)
            if best is None or delta<best[0]:best=(delta,after+1)
        delta,index=best
        path.insert(index,(('sample',channel),point));extra+=delta
    return base+extra

def choose(bot,tasks):
    current=bot.client.position
    original=open_visit_order(current,tasks,relocate=True,multistart=True)
    if len(tasks)<2 or not any(k[0]=='anchor' for k in tasks):return original
    if not hasattr(bot,'recourse_rng'):
        bot.recourse_rng=random.Random(64017031)
        bot.recourse_stats=dict(decisions=0,changed=0,invalid_forecasts=0,sampled_sources=0)
    stats=bot.recourse_stats;stats['decisions']+=1
    samples,unknown_count=worlds(bot,tasks,bot.recourse_rng)
    stats['sampled_sources']+=sum(map(len,samples))
    if not samples or not any(samples):return original
    plans=[original]
    for first in tasks:
        rest={k:p for k,p in tasks.items() if k!=first}
        plan=[first]+open_visit_order(tasks[first],rest,relocate=True,multistart=False)
        if plan!=original:plans.append(plan)
    scores=[]
    for plan in plans:
        values=[route_score(current,tasks,plan,s,unknown_count) for s in samples]
        if any(v is None for v in values):
            stats['invalid_forecasts']+=1
            return original
        scores.append(sum(values)/len(values))
    best=min(range(len(plans)),key=lambda i:scores[i])
    result=plans[best]
    stats['changed']+=int(result[0]!=original[0])
    return result
