"""Static reconstruction of practice generation and bearing noise.

Not the original source. No networking, activation, or official case access.
Floating-point trig/rounding have not been compared with the running Go program.
Use fresh independent seeds when evaluating strategies.
"""
import hashlib
import hmac
import math


def round_away(x):
    # Avoid Python round(), whose half-way rule is ties-to-even.
    whole = math.floor(abs(x))
    return int(math.copysign(whole + (abs(x) - whole >= 0.5), x))


def grid(seed, channel, ix, iy):
    """bearingnoise.grid, VA 0x1404eeac0: BLAKE2b digest_size=8, NOT truncated 64-byte hash."""
    if not 0 <= seed < 2**64:
        raise ValueError('noise seed must be uint64')
    payload = f'{seed}:{channel}:{ix}:{iy}'.encode('ascii')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, 'big')
    return (float(value) / math.ldexp(1.0, 64)) * 2.0 - 1.0


def error_degrees(seed, channel, x_m, y_m):
    """bearingnoise.ErrorDegrees, VA 0x1404ee680."""
    gx, gy = x_m / 150.0, y_m / 150.0
    ix, iy = math.floor(gx), math.floor(gy)
    ux, uy = min(1.0, max(0.0, gx-ix)), min(1.0, max(0.0, gy-iy))
    sx, sy = ux*ux*(3.0-2.0*ux), uy*uy*(3.0-2.0*uy)
    g00, g10 = grid(seed, channel, ix, iy), grid(seed, channel, ix+1, iy)
    g01, g11 = grid(seed, channel, ix, iy+1), grid(seed, channel, ix+1, iy+1)
    low = g00 + (g10-g00)*sx
    high = g01 + (g11-g01)*sx
    return low + (high-low)*sy


def quantize_bearing_hundredths(exact, error):
    """bearingnoise.QuantizeBearingHundredths, VA 0x1404ee900."""
    q = round_away((exact+error)*100.0)
    q = max(math.ceil((exact-1.0)*100.0), q)
    q = min(math.floor((exact+1.0)*100.0), q)
    return q % 36000


class CounterSource:
    """scenario.counterSource: separate counter per label; HMAC-SHA256."""
    def __init__(self, seed):
        if len(seed) != 32:
            raise ValueError('generation seed must contain 32 bytes')
        self.seed, self.counters = seed, {}

    def next(self, label):
        n = self.counters.get(label, 0)
        if n == 2**64-1:
            raise OverflowError('counter exhausted')
        payload = b'practice-case-v1\0' + label.encode() + b'\0' + n.to_bytes(8, 'big')
        digest = hmac.new(self.seed, payload, hashlib.sha256).digest()
        self.counters[label] = n+1
        return int.from_bytes(digest[:8], 'big')

    def uint_n(self, label, n):
        if not 0 < n <= 2**63:
            raise ValueError('invalid range')
        threshold = (2**64-n) % n
        while True:
            value = self.next(label)
            if value >= threshold:
                return value % n

    def float01(self, label):
        return float(self.next(label) >> 11) * 2**-53

    def shuffle(self, label, values):
        for i in range(len(values)-1, 0, -1):
            j = self.uint_n(label, i+1)
            values[i], values[j] = values[j], values[i]


def generate_practice(seed, problem):
    """Core scene fields only; excludes official envelope/JSON schema and case code.

    Reconstructed from GeneratePractice VA 0x1404eeda0. Uses observed default
    generation parameters. No source-source separation criterion exists here.
    """
    if problem not in (3, 4):
        raise ValueError('problem must be 3 or 4')
    source = CounterSource(seed)
    count = 10 + source.uint_n('count', 7)
    channels = list(range(1, 21))
    source.shuffle('channels', channels)
    channels = sorted(channels[:count])
    directional = set()
    if problem == 4:
        k = 1 + source.uint_n('directional-count', count)
        candidates = list(channels)
        source.shuffle('directional-channels', candidates)
        directional = set(candidates[:k])
    jammers = []
    for ch in channels:
        while True:
            radius_um = 1770000000 * math.sqrt(source.float01(f'jammer/{ch}/radius'))
            theta = source.float01(f'jammer/{ch}/theta') * (2.0*math.pi)
            x_um = round_away(radius_um * math.cos(theta))
            y_um = round_away(radius_um * math.sin(theta))
            if x_um*x_um + y_um*y_um <= 1770000000**2:
                break
        receive = 1000000000 + source.uint_n(f'jammer/{ch}/receive', 500000001)
        direction = source.uint_n(f'jammer/{ch}/direction', 360000000) if ch in directional else None
        jammers.append(dict(channel=ch, x_um=x_um, y_um=y_um,
                            max_receive_um=receive,
                            kind='directional' if ch in directional else 'omni',
                            direction_udeg=direction))
    return dict(problem=problem, jammers=jammers, noise_seed=source.next('noise-seed'))
