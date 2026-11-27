"""Local rule-based test double. This is NOT the official competition simulator."""
from dataclasses import dataclass
import hashlib
import json
import math
import random
import time

from geometry import distance
from action_timing import movement_microseconds
from practice_reference import error_degrees, generate_practice, quantize_bearing_hundredths


@dataclass
class Source:
    channel: int
    position: tuple
    radius: float = 1000.
    direction: float | None = None
    cleared: bool = False


def generate_case(seed, problem, count=None):
    """Legacy synthetic distribution, retained only for explicit legacy/stress tests."""
    rng = random.Random(seed)
    count = count if count is not None else rng.randint(10, 16)
    sources = []
    for index, channel in enumerate(rng.sample(range(1, 21), count)):
        angle, radius = rng.random()*2*math.pi, 1800*math.sqrt(rng.random())
        direction = None
        if problem == 4 and (index == 0 or (index > 1 and rng.random() < .65)):
            direction = rng.random()*360
        sources.append(Source(channel, (radius*math.cos(angle), radius*math.sin(angle)),
                              rng.uniform(1000, 1500), direction))
    return sources


class LocalArena:
    def __init__(self, sources, robot_id='local-test', error_mode='spatial', real_limit=1200,
                 noise_seed=0, bearing_quantization='legacy', timing_mode='microseconds'):
        self.sources = {s.channel: s for s in sources}
        self.robot_id = robot_id
        self.error_mode = error_mode
        self.noise_seed = noise_seed
        self.bearing_quantization = bearing_quantization
        if timing_mode not in ('microseconds', 'legacy-float'):
            raise ValueError('Unknown timing mode')
        self.timing_mode = timing_mode
        self.virtual_time_us = 0
        self.environment = {'profile': 'custom', 'error_mode': error_mode,
                            'bearing_quantization': bearing_quantization, 'timing_mode': timing_mode}
        self.real_limit = real_limit
        self.position = (0., 0.)
        self.channel = 1
        self.virtual_time = 0.
        self.entered = self.closed = False
        self.start = None
        self.cache = {}
        self.counters = dict(measures=0, switches=0, clear_attempts=0, cleared=0, distance=0.)

    def response(self, accepted, **extra):
        return dict(accepted=accepted, real_timestamp_ms=int(time.time()*1000),
                    virtual_time_s=round(self.virtual_time, 6) if accepted else 0, **extra)

    def bearing_error(self, position, channel):
        if self.error_mode == 'smooth':
            return error_degrees(self.noise_seed, channel, *position)
        if self.error_mode == 'positive':
            return 1.
        if self.error_mode == 'negative':
            return -1.
        key = json.dumps([*position, channel], separators=(',', ':')).encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
        return 2*(value/(2**64-1))-1

    def handle(self, path, payload):
        basic = {'arena_id', 'robot_id', 'request_id'}
        if path not in ('/enter', '/exit', '/measure', '/clear'):
            return 404, self.response(False)
        expected = basic | ({'position', 'channel'} if path in ('/measure', '/clear') else set())
        if not isinstance(payload, dict) or not expected <= payload.keys():
            return 400, self.response(False)
        if payload.keys() != expected or payload['arena_id'] != 'default' or payload['robot_id'] != self.robot_id:
            return 200, self.response(False)
        rid = payload['request_id']
        if not isinstance(rid, str) or not rid:
            return 400, self.response(False)
        if path in ('/measure', '/clear'):
            pos, channel = payload['position'], payload['channel']
            if not isinstance(pos, dict) or set(pos) != {'x', 'y'}:
                return 400, self.response(False)
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v)>2e6 for v in pos.values()):
                return 400, self.response(False)
            if type(channel) not in (int, float) or not 1<=channel<=20 or channel != int(channel):
                return 400, self.response(False)
        signature = json.dumps([path, payload], sort_keys=True, allow_nan=False)
        if rid in self.cache:
            old_signature, result = self.cache[rid]
            return result if signature == old_signature else (409, self.response(False))
        if self.closed or (self.start is not None and time.monotonic()-self.start >= self.real_limit):
            return 200, self.response(False)
        if path == '/enter':
            if self.entered:
                return 200, self.response(False)
            self.entered, self.start = True, time.monotonic()
            result = self.response(True, max_virtual_duration_s=360000,
                                   max_real_duration_s=1200, remaining_real_duration_s=self.real_limit)
        elif not self.entered:
            return 200, self.response(False)
        elif path == '/exit':
            self.closed = True
            result = self.response(True, exit_reason='user_exit')
        else:
            self._move((float(payload['position']['x']), float(payload['position']['y'])))
            channel = int(payload['channel'])
            source = self.sources.get(channel)
            exists = source is not None and not source.cleared
            d = distance(self.position, source.position) if exists else math.inf
            if path == '/clear':
                self.counters['clear_attempts'] += 1
                self._add_seconds(3)
                success = exists and d <= 20
                if success:
                    source.cleared = True
                    self.counters['cleared'] += 1
                    self._add_seconds(2)
                result = self.response(True, clear_result='success' if success else 'no_target_in_range')
            else:
                self.counters['measures'] += 1
                self._add_seconds(5 + (channel != self.channel))
                self.counters['switches'] += int(channel != self.channel)
                self.channel = channel
                visible = exists and d <= source.radius
                if visible and source.direction is not None:
                    a = math.radians(source.direction)
                    visible = ((self.position[0]-source.position[0])*math.cos(a)
                               +(self.position[1]-source.position[1])*math.sin(a)) >= -1e-10
                if not visible:
                    result = self.response(True, measure_result='no_signal')
                elif d <= 5:
                    result = self.response(True, measure_result='near')
                else:
                    a = math.degrees(math.atan2(source.position[1]-self.position[1], source.position[0]-self.position[0]))
                    error = self.bearing_error(self.position, channel)
                    if self.bearing_quantization == 'practice':
                        a = quantize_bearing_hundredths(a, error)/100
                    else:
                        a = round((a+error) % 360, 2) % 360
                    result = self.response(True, measure_result='direction', svd_deg=a)
        if self.virtual_time >= 360000:
            self.closed = True
        result = (200, result)
        self.cache[rid] = signature, result
        return result

    def _add_seconds(self, seconds):
        if self.timing_mode == 'microseconds':
            self.virtual_time_us += seconds * 1_000_000
            self.virtual_time = self.virtual_time_us / 1_000_000
        else:
            self.virtual_time += seconds

    def _move(self, position):
        length = distance(self.position, position)
        self.counters['distance'] += length
        if self.timing_mode == 'microseconds':
            self.virtual_time_us += movement_microseconds(length)
            self.virtual_time = self.virtual_time_us / 1_000_000
        else:
            self.virtual_time += length/5
        self.position = position


class LocalTransport:
    def __init__(self, arena):
        self.arena = arena

    def send(self, path, payload, timeout):
        return self.arena.handle(path, payload)


def build_arena(seed, problem, profile='practice', count=None, error_mode=None):
    """Independent synthetic seeds only; never consumes any official case or seed.

    practice reproduces the supplied static reference, not verified Go bit-level
    equivalence or the unobserved distribution of official server cases.
    """
    if profile == 'legacy':
        arena = LocalArena(generate_case(seed, problem, count), error_mode=error_mode or 'spatial',
                           timing_mode='legacy-float')
        arena.environment = dict(profile='legacy', seed=seed, error_mode=arena.error_mode,
                                 bearing_quantization='legacy', source_disk_radius_m=1800,
                                 timing_mode=arena.timing_mode)
        return arena
    if profile != 'practice':
        raise ValueError('Unknown simulation profile')
    if count is not None:
        raise ValueError('Practice profile derives count from the seed; use --profile legacy for --count')
    seed_bytes = hashlib.sha256(f'cumcm-local-validation-v1:{seed}'.encode('ascii')).digest()
    case = generate_practice(seed_bytes, problem)
    sources = [Source(j['channel'], (j['x_um']/1e6, j['y_um']/1e6), j['max_receive_um']/1e6,
                      None if j['direction_udeg'] is None else j['direction_udeg']/1e6)
               for j in case['jammers']]
    arena = LocalArena(sources, error_mode=error_mode or 'smooth', noise_seed=case['noise_seed'],
                       bearing_quantization='practice')
    fingerprint = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    arena.environment = dict(profile='practice-reference-v1', seed=seed,
                             timing_mode=arena.timing_mode, engine_revision='per-move-us-v2',
                             scenario_fingerprint=fingerprint,
                             error_mode=arena.error_mode, bearing_quantization='practice',
                             noise_grid_m=150, source_disk_radius_m=1770,
                             directional_count=sum(s.direction is not None for s in sources),
                             provenance='User-supplied static reconstruction; no dynamic EXE comparison')
    return arena
