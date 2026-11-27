"""T4.8: T4.7 policy plus certified adaptive movement of its original 20 stations."""
import math
import time
import numpy as np
from protocol import ProtocolError
from geometry import (bearing_region,clip,covering_circle,distance,bearing_history_region,
    point_polygon_distance,select_measurement,clear_approach,fallback_points,enroute_measurement,
    open_visit_order as quick_order)
from gridline import paired_bearing_box,clip_pair_box,symmetric_stations
from coverage import Coverage,clear_cover_points,move_one
from routes import route_ils24
from belief import Belief,belief_first_task
from gridline import position_covariance


class Robot:
    initial_survey_point=(-6.,-6.)
    orientation_candidates=24
    orientation_span=360.
    detour_weight=0.
    clear_offset=12.

    def __init__(self,client,problem=4,fast_measures=12,fast_distance=3000.,progress=None,
                 strategy='directional-informed'):
        if problem!=4:
            raise ValueError('The frozen strategy is for question four')
        self.client,self.problem=client,problem
        self.fast_measures,self.fast_distance=fast_measures,fast_distance
        self.progress=progress or (lambda message: None)
        self.strategy=strategy
        self.clear_threshold=80.
        self.excluded,self.cleared=set(),set()
        self.scanned={k:set() for k in range(1,21)}
        self.last_clear_time=self.completion_reason=None
        self.coverage_surveys=self.fallback_count=self.geometry_warnings=0
        self.enroute_probe_count=self.enroute_replan_count=0
        self.shortened_survey_legs=0
        self.planned_leg_savings_m=0.
        self.route_boxes={}
        self.pair_probes=self.pair_clear_attempts=self.pair_clear_successes=0

    def measure(self,point,channel):
        return self.client.request('/measure',point,channel)

    def _clear_action(self, point, channel):
        result = self.client.request('/clear', point, channel)
        success = result['clear_result'] == 'success'
        if success:
            self.cleared.add(channel)
            self.last_clear_time = self.client.virtual_time
            self.progress(f'Cleared channel {channel}; total={len(self.cleared)}; virtual={self.client.virtual_time:.1f}s')
        return success

    def route_region(self,channel,poly):
        # Used for task order and probing decisions only. Actual localization
        # and finite fallback retain the original physical bearing geometry.
        box = self.route_boxes.get(channel)
        if box is not None:
            tighter = clip_pair_box(poly,box,clip)
            if tighter:
                return tighter
        return poly

    def refine_survey(self,point,anchor,pending):
        if not self.fast_measures:
            return
        cache = {(tuple(p),ch):reading for ch,observed in pending.items() for p,reading in observed}
        channels = []
        for channel,observed in pending.items():
            if observed[-1][0] != point:
                continue
            poly = bearing_history_region(observed)
            poly = self.route_region(channel,poly)
            if poly and covering_circle(poly)[1]>300:
                channels.append(channel)
        if not channels:
            return
        groups = {'x':[],'y':[]}
        for channel in channels:
            a = math.radians(pending[channel][-1][1]['svd_deg'])
            groups['x' if abs(math.sin(a))>=abs(math.cos(a)) else 'y'].append(channel)
        plans = []
        for axis,group in groups.items():
            if not group:
                continue
            midpoint = list(point)
            coordinate = 0 if axis=='x' else 1
            midpoint[coordinate] = round(midpoint[coordinate]/150)*150.
            p,q = symmetric_stations(tuple(midpoint),axis,6.)
            if distance(self.client.position,q)<distance(self.client.position,p):
                p,q = q,p
            plans.append((distance(self.client.position,p),p,q,group))
        for _,p,q,group in sorted(plans):
            readings = {channel:[] for channel in group}
            for station in (p,q):
                for channel in group:
                    if channel in self.cleared:
                        continue
                    result = cache.get((tuple(station),channel))
                    if result is None:
                        result = self.measure(station,channel)
                    if result['measure_result']=='near':
                        if not self.clear(station,channel):
                            raise ProtocolError('Shared grid near followed by failed clear')
                        pending.pop(channel,None)
                    elif result['measure_result']=='direction':
                        pending[channel].append((station,result))
                        readings[channel].append((station,result))
            for channel,pair in readings.items():
                if len(pair)!=2:
                    continue
                bound = paired_bearing_box(pair[0][0],pair[0][1]['svd_deg'],pair[1][0],pair[1][1]['svd_deg'])
                if bound is not None:
                    self.route_boxes[channel] = bound

    def choose_orientation(self,coverage,regions,belief):
        targets = {}
        for channel,poly in regions.items():
            if poly:
                center,radius = covering_circle(poly)
                if radius<=300:
                    targets[('target',channel)] = center
        known = len(self.cleared | regions.keys())
        unknown = 20-known-len(self.excluded)
        best,best_score = coverage,math.inf
        for i in range(self.orientation_candidates):
            candidate = coverage if i==0 else Coverage(i*self.orientation_span/self.orientation_candidates)
            tasks = {('anchor',j):p for j,p in enumerate(candidate.points) if j}
            tasks.update(targets)
            order = route_ils24(self.client.position,tasks,relocate=True)
            score = belief.route_cost(self.client.position,order,tasks,known,unknown,0.)
            if score<best_score:
                best,best_score = candidate,score
        # Only the actual initial full survey belongs to the rotated certificate.
        if best is not coverage:
            best.add_survey(getattr(self,'initial_survey_point',(0.,0.)))
        return best

    def localize(self,channel,observation,observations=None):
        box=self.route_boxes.get(channel)
        self.radial_covariance=position_covariance(box,observations or []) if box is not None else None
        self.radial_channel=channel
        self.radial_used=False
        self.approach_channel=channel
        self.approach_available=True
        try:
            return self._localize_grid(channel,observation,observations)
        finally:
            self.approach_available=False


    def _localize_grid(self,channel,observation,observations=None):
        bound = self.route_boxes.get(channel)
        if bound is not None:
            observed = list(observations or [(self.client.position,observation)])
            if all(r['measure_result']=='direction' for _,r in observed):
                poly = bearing_history_region(observed)
                estimated = self.route_region(channel,poly)
                if estimated and covering_circle(estimated)[1]<=150:
                    center = bound.nominal
                    if point_polygon_distance(center,estimated)>1e-5:
                        center = covering_circle(estimated)[0]
                    self.pair_clear_attempts += 1
                    if self.clear(center,channel):
                        self.pair_clear_successes += 1
                        return
                    # A failed guess never narrows the hard fallback region.
                    return self._localize_bearings(channel,observation,observed)
        return self._localize_pair(channel,observation,observations)

    def _localize_pair(self, channel, observation, observations=None):
        observed = list(observations or [(self.client.position, observation)])
        if self.fast_measures < 2 or any(r['measure_result']=='near' for _,r in observed):
            return self._localize_bearings(channel, observation, observed)
        poly = bearing_history_region(observed)
        if not poly or covering_circle(poly)[1] <= 300.:
            return self._localize_bearings(channel, observation, observed)

        choices = []
        for old, reading in observed:
            angle = math.radians(reading['svd_deg'])
            for axis, quality in (('x',abs(math.sin(angle))),('y',abs(math.cos(angle)))):
                if quality < .7:
                    continue
                center = list(old)
                coordinate = 0 if axis=='x' else 1
                center[coordinate] = round(center[coordinate]/150.)*150.
                first, second = symmetric_stations(tuple(center), axis, half_baseline=6.)
                if distance(self.client.position, second)<distance(self.client.position, first):
                    first, second = second, first
                travel = distance(self.client.position, first)+distance(first, second)
                score = travel+200*(1-quality)+distance(old, center)
                choices.append((score, first, second))
        if not choices:
            return self._localize_bearings(channel, observation, observed)
        _, first, second = min(choices)
        self.pair_probes += 1
        pair = []
        for point in (first, second):
            stored = next((r for p,r in observed if distance(p,point)<1e-8),None)
            reading = stored if stored is not None else self.measure(point,channel)
            if reading['measure_result']=='near':
                if not self.clear(point,channel):
                    raise ProtocolError('Grid pair near followed by failed clear')
                return
            if reading['measure_result']=='direction':
                pair.append((point,reading))
                if stored is None:
                    observed.append((point,reading))
                    poly = bearing_region(poly,point,reading['svd_deg'])
        if len(pair)==2 and poly:
            bound = paired_bearing_box(pair[0][0],pair[0][1]['svd_deg'],
                                       pair[1][0],pair[1][1]['svd_deg'])
            if bound is not None:
                estimated = clip_pair_box(poly,bound,clip)
                if estimated:
                    center,radius = covering_circle(estimated)
                    if radius <= 150:
                        target = bound.nominal
                        if point_polygon_distance(target,estimated)>1e-5:
                            target = center
                        self.pair_clear_attempts += 1
                        if self.clear(target,channel):
                            self.pair_clear_successes += 1
                            return
        return self._localize_bearings(channel, observed[-1][1], observed)

    def _localize_bearings(self, channel, observation, observations=None):
        c = self.client
        observations = observations if observations is not None else [(c.position, observation)]
        near = next((position for position, result in observations
                     if result['measure_result'] == 'near'), None)
        if near is not None:
            if not self.clear(near, channel):
                raise ProtocolError('Stored near observation followed by failed clear')
            return
        if observation['measure_result'] == 'near':
            if not self.clear(c.position, channel):
                raise ProtocolError('near followed by failed clear at the same position')
            return
        first = observations[-1][0], observations[-1][1]['svd_deg']
        poly = bearing_history_region(observations)
        history, failed = [position for position, _ in observations], []
        positive_positions = list(history)
        used_distance, measures, tries, no_signal = 0., 0, 0, 0
        if (poly and self.fast_measures
                and covering_circle(poly)[1] > self.clear_threshold and min(distance(c.position, p) for p in history) >= 100
                and distance(c.position, covering_circle(poly)[0]) <= 1250):
            # Pay for a stationary bearing before committing to another detour.
            # No signal does not erase the earlier positive observation.
            probe = self.measure(c.position, channel)
            measures += 1
            history.append(c.position)
            if probe['measure_result'] == 'near':
                if not self.clear(c.position, channel):
                    raise ProtocolError('near followed by failed clear during stationary probe')
                return
            if probe['measure_result'] == 'direction':
                positive_positions.append(c.position)
                poly = bearing_region(poly, c.position, probe['svd_deg'])
        while poly and measures < self.fast_measures:
            center, radius = covering_circle(poly)
            center = clear_approach(poly, c.position, center)
            if radius <= self.clear_threshold and tries < 5 and distance(c.position, center)+used_distance <= self.fast_distance:
                if all(distance(center, old)>2 for old in failed):
                    used_distance += distance(c.position, center)
                    tries += 1
                    if self.clear(center, channel):
                        return
                    failed.append(center)
                    if radius <= 20-1e-5:
                        # Do not continue trusting a region contradicted by a guaranteed clear.
                        poly = []
                        self.geometry_warnings += 1
                        break
            next_point = self._select_measurement(channel,poly, c.position, first, history,
                                            self.problem, self.fast_distance-used_distance,
                                            objective='cost',
                                            directional_observers=positive_positions,
                                            visibility_penalty=500.)
            if next_point is None:
                break
            used_distance += distance(c.position, next_point)
            history.append(next_point)
            result = self.measure(next_point, channel)
            measures += 1
            if result['measure_result'] == 'near':
                if not self.clear(c.position, channel):
                    raise ProtocolError('near followed by failed clear')
                return
            if result['measure_result'] == 'direction':
                positive_positions.append(c.position)
                poly = bearing_region(poly, c.position, result['svd_deg'])
                if not poly:
                    self.geometry_warnings += 1
                no_signal = 0
            else:
                no_signal += 1
                if no_signal >= 3:
                    break
        self.fallback_count += 1
        self.progress(f'Finite clear coverage for channel {channel}')
        if poly:
        
            center=covering_circle(poly)[0]
            points=clear_cover_points(poly,center,first[1])
            # Start at the estimate, then visit nearby uncovered grid centers.
            position=center
            while points:
                point=min(points,key=lambda p:distance(p,position))
                points.remove(point)
                if any(distance(point,p)<1e-6 for p in failed):
                    continue
                if self.clear(point,channel):
                    return
                failed.append(point)
                position=point
        # Prune only by a conservative polygon or a previously failed 20m disk.
        # Sampling never supplies a completeness certificate.
        for point in fallback_points(*first):
            if poly and point_polygon_distance(point, poly) > 20+1e-5:
                continue
            if any(distance(point, old)<1e-6 for old in failed):
                continue
            if self.clear(point, channel):
                return
        raise ProtocolError(f'Finite clear coverage exhausted for channel {channel}; physical/geometry inconsistency')

    def _select_measurement(self,channel,poly,*args,**kwargs):
        box=self.route_boxes.get(channel)
        if box is not None:
            proxy=clip_pair_box(poly,box,clip)
            poly=proxy or poly
        return select_measurement(poly,*args,**kwargs)


    def _clear_approach(self,point,channel):
        box=self.route_boxes.get(channel)
        if (getattr(self,'approach_available',False) and channel==self.approach_channel
                and box is not None and distance(point,box.nominal)<1e-6):
            self.approach_available=False
            incoming=self.client.position
            following=getattr(self,'clear_following_point',None)
            vectors=[]
            for other in (incoming,following):
                if other is not None and distance(point,other)>1e-6:
                    length=distance(point,other)
                    vectors.append(tuple((v-p)/length for v,p in zip(other,point)))
            vector=tuple(sum(v[i] for v in vectors) for i in (0,1))
            norm=math.hypot(*vector)
            if norm>1e-6:
                unit=tuple(v/norm for v in vector)
                offset=self.clear_offset
                trial=tuple(p+offset*u for p,u in zip(point,unit))
                saving=distance(incoming,point)-distance(incoming,trial)
                if following is not None:
                    saving+=distance(point,following)-distance(trial,following)
                if saving>=5.:
                    if self._clear_action(trial,channel):
                        return True
                    # Failure is an ordinary paid trial. Immediately retry the
                    # original nominal point before invoking measurement recovery.
        return self._clear_action(point,channel)

    def clear(self,point,channel):
        success=self._clear_approach(point,channel)
        box=self.route_boxes.get(channel)
        if (success or channel!=getattr(self,'radial_channel',None) or self.radial_used
                or box is None or distance(point,box.nominal)>1e-6 or self.radial_covariance is None):
            return success
        self.radial_used=True
        variance,vectors=np.linalg.eigh(self.radial_covariance)
        minor,major=math.sqrt(max(0.,variance[0])),math.sqrt(max(0.,variance[1]))
        if not (5<=major<=40 and minor<=12 and major>=2*minor):
            return False
        axis=vectors[:,1]
        following=getattr(self,'clear_following_point',None)
        if following is not None and sum(v*(f-p) for v,f,p in zip(axis,following,point))<0:
            axis=-axis
        for sign in (1.,-1.):
            trial=tuple(p+sign*25*v for p,v in zip(point,axis))
            if self._clear_approach(trial,channel):
                return True
        # All trials are paid and verified; unchanged measurement/finite
        # coverage recovery remains available after the two additional misses.
        return False

    def execute(self):
        coverage = Coverage()
        belief = Belief()
        points = coverage.points
        remaining = set(range(len(points)))
        pending = {}
        enroute_attempted=set()
        c = self.client
        c.request('/enter')

        def unknown_channels():
            return set(range(1,21))-self.cleared-pending.keys()-self.excluded

        def region(channel):
            poly = bearing_history_region(pending[channel])
            return self.route_region(channel,poly)

        def survey(point, anchor=None):
            self.coverage_surveys += 1
            channels = [c.channel]+[k for k in range(1,21) if k != c.channel]
            for channel in channels:
                if channel in self.cleared or channel in self.excluded:
                    continue
                if channel in pending:
                    poly = region(channel)
                    if poly:
                        continue
                reading = self.measure(point,channel)
                if anchor is not None:
                    self.scanned[channel].add(anchor)
                if reading['measure_result'] == 'near':
                    if not self.clear(point,channel):
                        raise ProtocolError('near followed by failed clear')
                    pending.pop(channel,None)
                elif reading['measure_result'] == 'direction':
                    pending.setdefault(channel,[]).append((point,reading))
            # Every STILL unknown channel was measured here and at all earlier
            # full surveys. Known positive channels never use this absence proof.
            self.refine_survey(point,anchor,pending)
            belief.observe_full_survey(point)
            if coverage.add_survey(point):
                newly_excluded = unknown_channels()
                self.excluded.update(newly_excluded)
                c.log('directional_coverage_certificate',channels=sorted(newly_excluded),
                      cells=len(coverage.cells),stations=len(coverage.stations))

        survey(points[0],0)
        remaining.remove(0)
        coverage = self.choose_orientation(coverage,{k:region(k) for k in pending},belief)
        points = list(coverage.points)
        while True:
            if len(self.cleared | pending.keys())==16:
                # Sixteen DISTINCT known channels exhaust the public source bound.
                # This certifies only the other channels; pending sources still
                # require actual successful clears before the run can finish.
                self.excluded.update(unknown_channels())
            if len(self.cleared)==16:
                self.completion_reason='source_count_upper_bound'
                return
            if len(self.cleared | self.excluded)==20:
                self.completion_reason='continuous_directional_coverage'
                return
            unknown = unknown_channels()
            tasks = {}
            for channel in pending:
                poly = region(channel)
                center,radius=covering_circle(poly) if poly else (pending[channel][-1][0],0.)
                if (unknown and radius>300 and
                        any(distance(center,points[i])<1000 for i in remaining)):
                    # A future mandatory survey can provide another baseline.
                    # This only postpones a known target; never discards it.
                    continue
                tasks[('target',channel)] = center
            if unknown:
                tasks.update({('anchor',i):points[i] for i in sorted(remaining)})
            if not tasks:
                raise ProtocolError('Directional coverage ended with unresolved channels')
            if unknown and self.fast_measures:
                kind,index=belief_first_task(c.position,tasks,
                    {k:region(k) for k in pending},
                    {k:[p for p,_ in readings] for k,readings in pending.items()},
                    {('anchor',i):points[i] for i in sorted(remaining-enroute_attempted)},
                    self.clear_threshold,belief=belief,
                    known_count=len(self.cleared | pending.keys()),unknown_count=len(unknown),
                    detour_weight=self.detour_weight)
            else:
                kind,index = route_ils24(c.position,tasks,relocate=True)[0]
            if kind=='anchor':
                if self.fast_measures and pending and index not in enroute_attempted:
                    choice=enroute_measurement(c.position,points[index],
                        {k:region(k) for k in pending},
                        {k:[p for p,_ in observations] for k,observations in pending.items()},
                        self.clear_threshold)
                    if choice is not None:
                        station,channels=choice
                        enroute_attempted.add(index)
                        self.enroute_probe_count+=1
                        c.log('enroute_probe',anchor=index,station=station,channels=channels)
                        informative=False
                        for channel in channels:
                            reading=self.measure(station,channel)
                            if reading['measure_result']=='near':
                                if not self.clear(station,channel):
                                    raise ProtocolError('Enroute near followed by failed clear')
                                pending.pop(channel)
                                informative=True
                            elif reading['measure_result']=='direction':
                                before=covering_circle(region(channel))[1]
                                pending[channel].append((station,reading))
                                after=region(channel)
                                informative |= bool(after) and covering_circle(after)[1]<.8*before
                        if informative:
                            self.enroute_replan_count+=1
                            c.log('enroute_replan',unvisited_anchor=index)
                            # The anchor stays unvisited. Only full surveys may
                            # contribute to unknown-channel coverage evidence.
                            continue
                point=move_one(coverage,c.position,index,{i:points[i] for i in remaining},tasks)
                if point is None:
                    remaining.remove(index)
                    continue
                self.station_motion=getattr(self,'station_motion',0.)+distance(points[index],point)
                points[index]=point
                survey(points[index],index)
                remaining.remove(index)
            else:
                others={k:p for k,p in tasks.items() if k!=('target',index)}
                self.clear_following_point=(others[quick_order(tasks[('target',index)],others,relocate=True)[0]]
                                            if others else None)
                observations = pending.pop(index)
                self.localize(index,observations[-1][1],observations)


    def run(self):
        error = None
        try:
            self.execute()
        except (Exception, KeyboardInterrupt) as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.client.log('failure', error=error)
        finally:
            if self.client.entered and not self.client.exited:
                try:
                    self.client.request('/exit')
                except (Exception, KeyboardInterrupt) as exc:
                    exit_error = f'exit unconfirmed: {type(exc).__name__}: {exc}'
                    error = f'{error}; {exit_error}' if error else exit_error
        c = self.client
        predicted = (c.counters['movement_time_us']/1e6+c.counters['switches']+5*c.counters['measures']
                     +3*c.counters['clear_attempts']+2*c.counters['cleared'])
        return dict(problem=self.problem, strategy=self.strategy, completed=self.completion_reason is not None,
                    completion_reason=self.completion_reason, exit_confirmed=c.exited,
                    error=error, cleared_channels=sorted(self.cleared), cleared_count=len(self.cleared),
                    virtual_time_s=c.virtual_time,
                    average_clear_time_s=c.virtual_time/len(self.cleared) if self.cleared else None,
                    real_elapsed_s=time.monotonic()-c.start if c.start is not None else None,
                    counters=c.counters, failed_clear_count=c.counters['clear_attempts']-c.counters['cleared'],
                    predicted_virtual_time_s=predicted,
                    time_residual_s=c.virtual_time-predicted, fallback_count=self.fallback_count,
                    geometry_warnings=self.geometry_warnings,
                    excluded_channels=sorted(self.excluded), survey_count=self.coverage_surveys,
                    enroute_probe_count=self.enroute_probe_count,enroute_replan_count=self.enroute_replan_count,
                    shortened_survey_legs=self.shortened_survey_legs,
                    planned_leg_savings_m=self.planned_leg_savings_m,
                    last_clear_virtual_time_s=self.last_clear_time,
                    tail_time_s=c.virtual_time-self.last_clear_time if self.last_clear_time is not None else None,
                    scan_counts={str(k): len(v) for k, v in self.scanned.items()})
