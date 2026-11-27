"""Conservative two-bearing bounds from the public 150m smooth noise grid.

The caller must pair readings from the same channel in the same case. No noise
seed, true source coordinates, reception radius or scenario state is accessed.
"""
from dataclasses import dataclass
import math
from itertools import combinations
import numpy as np


def smoothstep(t):
    return t*t*(3-2*t)


def noise_weights(position):
    x,y=position
    gx,gy=x/150.,y/150.
    ix,iy=math.floor(gx),math.floor(gy)
    sx,sy=smoothstep(gx-ix),smoothstep(gy-iy)
    return {(ix,iy):(1-sx)*(1-sy), (ix+1,iy):sx*(1-sy),
            (ix,iy+1):(1-sx)*sy, (ix+1,iy+1):sx*sy}


def error_difference_bound(first,second,quantization_guard_deg=.01):
    """Deterministic |reported difference - true difference| bound in degrees.

    Each grid value is in [-1,1] degrees. For raw field error e, the final
    reported error lies within clip(e,-.995,.995) +/- .005 degrees. Since
    clipping is 1-Lipschitz, a pair needs only .01 degrees of quantum slack.
    The additional 1e-10 degree is numerical slack, not a physical noise model.
    """
    a,b=noise_weights(first),noise_weights(second)
    return sum(abs(a.get(key,0.)-b.get(key,0.)) for key in a.keys() | b.keys())+quantization_guard_deg+1e-10


def _outward(low,high):
    return math.nextafter(low,-math.inf),math.nextafter(high,math.inf)


def _trig_interval(low,high,kind):
    f=math.sin if kind=='sin' else math.cos
    values=[f(low),f(high)]
    step=math.pi/2
    for i in range(math.ceil(low/step),math.floor(high/step)+1):
        values.append(f(i*step))
    return _outward(min(values),max(values))


def _multiply(a,b):
    products=[x*y for x in a for y in b]
    return _outward(min(products),max(products))


@dataclass(frozen=True)
class PairBox:
    midpoint: tuple
    tangent: tuple
    normal: tuple
    u: tuple
    v: tuple
    difference_deg: tuple
    mean_deg: tuple
    difference_guard_deg: float
    nominal: tuple

    def to_world(self,u,v):
        return (self.midpoint[0]+u*self.tangent[0]+v*self.normal[0],
                self.midpoint[1]+u*self.tangent[1]+v*self.normal[1])

    @property
    def center(self):
        return self.to_world(sum(self.u)/2,sum(self.v)/2)

    @property
    def covering_radius(self):
        return math.hypot(self.u[1]-self.u[0],self.v[1]-self.v[0])/2

    @property
    def range_interval(self):
        def nearest(interval):
            return 0. if interval[0]<=0<=interval[1] else min(abs(v) for v in interval)
        lower=math.hypot(nearest(self.u),nearest(self.v))
        upper=math.hypot(max(abs(v) for v in self.u),max(abs(v) for v in self.v))
        return _outward(lower,upper)

    @property
    def vertices(self):
        return tuple(self.to_world(u,v) for u,v in ((self.u[0],self.v[0]),(self.u[1],self.v[0]),
                                                   (self.u[1],self.v[1]),(self.u[0],self.v[1])))

    def contains(self,point,tolerance=1e-6):
        dx,dy=point[0]-self.midpoint[0],point[1]-self.midpoint[1]
        u=dx*self.tangent[0]+dy*self.tangent[1]
        v=dx*self.normal[0]+dy*self.normal[1]
        return self.u[0]-tolerance<=u<=self.u[1]+tolerance and self.v[0]-tolerance<=v<=self.v[1]+tolerance


def paired_bearing_box(first_position,first_degrees,second_position,second_degrees,mean_guard_deg=1.01,quantization_guard_deg=.01):
    """Return a conservative rotated rectangle, or None for unresolved parallax.

    Individual bearing sectors and the arena/receive-radius region should still
    be intersected separately. This rectangle is an additional outer constraint.
    A raw nominal intersection is a route estimate, never a clear certificate.
    """
    values=(*first_position,*second_position,first_degrees,second_degrees)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Finite coordinates and bearings are required')
    dx,dy=second_position[0]-first_position[0],second_position[1]-first_position[1]
    length=math.hypot(dx,dy)
    if length<=1e-8:
        return None
    half=length/2
    tangent=(dx/length,dy/length)
    normal=(-tangent[1],tangent[0])
    midpoint=((first_position[0]+second_position[0])/2,(first_position[1]+second_position[1])/2)
    delta=(second_degrees-first_degrees+180.)%360.-180.
    guard=error_difference_bound(first_position,second_position,quantization_guard_deg)
    dl,dh=delta-guard,delta+guard
    if dl<=0<=dh or dl<=-180 or dh>=180:
        return None
    heading=math.atan2(dy,dx)
    mean=math.radians(first_degrees+delta/2)-heading
    ml,mh=mean-math.radians(mean_guard_deg),mean+math.radians(mean_guard_deg)
    sine=_trig_interval(math.radians(dl),math.radians(dh),'sin')
    if sine[0]<=0<=sine[1]:
        return None
    reciprocal=_outward(1/sine[1],1/sine[0])
    sin2mean=_trig_interval(2*ml,2*mh,'sin')
    cos2mean=_trig_interval(2*ml,2*mh,'cos')
    cosdelta=_trig_interval(math.radians(dl),math.radians(dh),'cos')
    numerator=_outward(cosdelta[0]-cos2mean[1],cosdelta[1]-cos2mean[0])
    u=_multiply((half,half),_multiply(sin2mean,reciprocal))
    v=_multiply((half,half),_multiply(numerator,reciprocal))
    v=(max(0.,v[0]),v[1]) if dl>0 else (v[0],min(0.,v[1]))
    # Outward metre slack accommodates accumulated elementary-function rounding.
    u=(u[0]-1e-7,u[1]+1e-7)
    v=(v[0]-1e-7,v[1]+1e-7)
    if any(not math.isfinite(x) for x in (*u,*v)):
        return None
    d=math.radians(delta)
    un=half*math.sin(2*mean)/math.sin(d)
    vn=half*(math.cos(d)-math.cos(2*mean))/math.sin(d)
    nominal=(midpoint[0]+un*tangent[0]+vn*normal[0],midpoint[1]+un*tangent[1]+vn*normal[1])
    return PairBox(midpoint,tangent,normal,u,v,(dl,dh),(math.degrees(ml),math.degrees(mh)),guard,nominal)


def clip_pair_box(poly,bound,clip_function):
    """Intersect an existing conservative polygon using its usual clip function.

    clip_function(poly,a,b,c) must retain a*x+b*y <= c with outward tolerance.
    Passing geometry.clip from the current robot preserves its numerical policy.
    """
    if bound is None:
        return poly
    for axis,interval in ((bound.tangent,bound.u),(bound.normal,bound.v)):
        offset=axis[0]*bound.midpoint[0]+axis[1]*bound.midpoint[1]
        poly=clip_function(poly,axis[0],axis[1],offset+interval[1])
        poly=clip_function(poly,-axis[0],-axis[1],-offset-interval[0])
    return poly


def symmetric_stations(grid_center,axis,half_baseline=10.):
    """Plan a pair across one verified public grid line; no movement is executed."""
    if axis not in ('x','y') or not 0<half_baseline<=150:
        raise ValueError('axis x/y and half-baseline in (0,150] are required')
    index=0 if axis=='x' else 1
    if abs(grid_center[index]/150-round(grid_center[index]/150))>1e-9:
        raise ValueError('The baseline midpoint must lie on the selected 150m grid line')
    first,second=list(grid_center),list(grid_center)
    first[index]-=half_baseline
    second[index]+=half_baseline
    return tuple(first),tuple(second)


def position_covariance(box,observations):
    pair=next(((p,q) for (p,_),(q,_) in combinations(observations,2)
               if math.dist(p,q)>1e-5 and math.dist(((p[0]+q[0])/2,(p[1]+q[1])/2),box.midpoint)<1e-7),None)
    if pair is None:
        return None
    p,q=pair
    weights=[noise_weights(p),noise_weights(q)]
    keys=set(weights[0])|set(weights[1])
    w=np.asarray([[row.get(k,0.) for k in keys] for row in weights])
    angular=(w@w.T)/3+np.eye(2)*(.01**2/12)
    jacobian=[]
    for station in pair:
        dx,dy=box.nominal[0]-station[0],box.nominal[1]-station[1]
        squared=dx*dx+dy*dy
        if squared<1e-4:
            return None
        jacobian.append([-dy/squared*180/math.pi,dx/squared*180/math.pi])
    try:
        inverse=np.linalg.inv(jacobian)
    except np.linalg.LinAlgError:
        return None
    return inverse@angular@inverse.T
