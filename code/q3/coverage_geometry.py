"""Conservative continuous-area certificates for omnidirectional sources only."""
class OmniCoverage:
    """Each bit is a WHOLE square, not a sampled point.

    Squares intersecting the 1770m source-center disk cover it. A no-signal observation excludes
    a square only if all its corners are strictly within 1000m of the station.
    Thus an empty remaining mask proves absence over the continuous target area.
    Start with 50 m cells; refine only cells not certified by supplied anchors.
    """
    def __init__(self, refine_for=None):
        half = 25.
        self.cells = []
        self.halves = []

        def add_cell(x, y, half, depth=0):
            closest_x, closest_y = max(0., abs(x)-half), max(0., abs(y)-half)
            if closest_x**2+closest_y**2 > 1770**2+1e-6:
                return
            if refine_for and not any((abs(sx-x)+half)**2+(abs(sy-y)+half)**2 < 1000**2-1e-3
                                      for sx, sy in refine_for):
                if depth >= 8:
                    raise ValueError('Refined cells cannot certify the proposed anchor ring')
                for dx in (-half/2, half/2):
                    for dy in (-half/2, half/2):
                        add_cell(x+dx, y+dy, half/2, depth+1)
            else:
                self.cells.append((float(x), float(y)))
                self.halves.append(half)
        for x in range(-1775, 1800, 50):
            for y in range(-1775, 1800, 50):
                closest_x, closest_y = max(0., abs(x)-half), max(0., abs(y)-half)
                if closest_x**2 + closest_y**2 <= 1770**2 + 1e-6:
                    add_cell(x, y, half)
        self.full = (1 << len(self.cells))-1
        self.remaining = {k: self.full for k in range(1, 21)}
        self.cache = {}

    def mask(self, station):
        station = tuple(station)
        if station not in self.cache:
            x, y = station
            mask = 0
            for index, ((cx, cy), half) in enumerate(zip(self.cells, self.halves)):
                # The farthest rectangle corner has these axis distances.
                dx, dy = abs(x-cx)+half, abs(y-cy)+half
                if dx*dx+dy*dy < 1000**2-1e-3:
                    mask |= 1 << index
            self.cache[station] = mask
        return self.cache[station]

    def no_signal(self, station, channel):
        self.remaining[channel] &= ~self.mask(station)
        return self.remaining[channel] == 0

    def gain(self, station, channels):
        mask = self.mask(station)
        return sum((self.remaining[k] & mask).bit_count() for k in channels)

    def unresolved_cells(self, channels):
        combined = 0
        for k in channels:
            combined |= self.remaining[k]
        return combined.bit_count()

    def nearer_station(self, current, anchor, channels):
        """Stop along current->anchor as soon as its remaining cells are covered.

        Every corner of every required square must still be inside the new
        1000m disk. Convexity makes the feasible interval along the segment
        contiguous. Binary search keeps a verified feasible upper endpoint.
        This is conservative shortening, not a point-sampled coverage claim.
        """
        combined = 0
        for channel in channels:
            combined |= self.remaining[channel]
        required = combined & self.mask(anchor)
        if not required:
            return tuple(current)
        cells = [(*cell, self.halves[i]) for i, cell in enumerate(self.cells) if required & (1 << i)]

        def safe(point):
            x, y = point
            return all((abs(x-cx)+half)**2+(abs(y-cy)+half)**2 < 1000**2-1e-3
                       for cx, cy, half in cells)

        if safe(current):
            return tuple(current)
        low, high = 0., 1.
        result = tuple(anchor)
        for _ in range(24):
            mid = (low+high)/2
            point = (current[0]+mid*(anchor[0]-current[0]),
                     current[1]+mid*(anchor[1]-current[1]))
            if safe(point):
                high, result = mid, point
            else:
                low = mid
        # Verify using the same certificate calculation used after real feedback.
        return result if required & ~self.mask(result) == 0 else tuple(anchor)
