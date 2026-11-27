"""Q2 policies consume public observations only, never generated scene objects."""
from dataclasses import dataclass
import math
from geometry_snapshot import clip, disk_outer, covering_circle, distance, convex_hull
from practice_snapshot import round_away


@dataclass(frozen=True)
class Observation:
    position: tuple
    channel: int
    kind: str
    bearing: float | None = None


def local(p, first):
    a=math.radians(first.bearing);c,s=math.cos(a),math.sin(a)
    x,y=p[0]-first.position[0],p[1]-first.position[1]
    return (c*x+s*y,-s*x+c*y)


def world(p, first):
    a=math.radians(first.bearing);c,s=math.cos(a),math.sin(a)
    return (first.position[0]+c*p[0]-s*p[1],first.position[1]+s*p[0]+c*p[1])


def choose_station(first, offset):
    assert first.kind=='direction'
    p=world(offset,first)
    return tuple(round_away(v*1e6)/1e6 for v in p)


def initial_polygon(first):
    poly=[(-1800.,-1800.),(1800.,-1800.),(1800.,1800.),(-1800.,1800.)]
    for angle,sign in ((-1,1),(1,-1)):
        a=math.radians(angle)
        poly=clip(poly,sign*math.sin(a),-sign*math.cos(a),0)
    poly=clip(poly,-1,0,-5*math.cos(math.radians(1)))
    for angle in (-1,-.5,0,.5,1):
        a=math.radians(angle);poly=clip(poly,math.cos(a),math.sin(a),1500)
    # Known task boundary, never the hidden 1770 m generator support.
    if first.position != (0,0):
        poly=disk_outer(poly,local((0,0),first),1800,sides=720)
    return poly


def outside_disk_hull(poly,station,radius):
    if not poly:return []
    points=[p for p in poly if distance(p,station)>=radius]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1];ax,ay=a[0]-station[0],a[1]-station[1]
        aa=dx*dx+dy*dy
        if aa<1e-20:continue
        bb=2*(ax*dx+ay*dy);cc=ax*ax+ay*ay-radius*radius
        disc=bb*bb-4*aa*cc
        if disc<0:continue
        for t in ((-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa)):
            if 0<=t<=1:points.append((a[0]+t*dx,a[1]+t*dy))
    return convex_hull(points)


def estimate(first, second, initial=None):
    assert first.kind=='direction' and first.channel==second.channel
    if second.kind=='near':
        return dict(position=second.position,radius=5.,fallback=False)
    poly=initial_polygon(first) if initial is None else initial
    original=poly
    sx,sy=local(second.position,first)
    if second.kind=='no_signal':
        poly=clip(poly,sx,sy,(sx*sx+sy*sy)/2)
        poly=outside_disk_hull(poly,(sx,sy),1000)
    elif second.kind=='direction':
        beta=(second.bearing-first.bearing)%360
        for angle,sign in ((beta-1,1),(beta+1,-1)):
            a=math.radians(angle);nx,ny=sign*math.sin(a),-sign*math.cos(a)
            poly=clip(poly,nx,ny,nx*sx+ny*sy)
        poly=disk_outer(poly,(sx,sy),1500,sides=720)
        poly=outside_disk_hull(poly,(sx,sy),5)
    else:raise ValueError(second.kind)
    fallback=not poly
    center,radius=covering_circle(original if fallback else poly)
    return dict(position=world(center,first),radius=radius,fallback=fallback)
