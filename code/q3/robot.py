"""Fixed T3.13 controller for problem 3. Run: python robot.py demo."""

import math
import sys
import time
from client import Client, HttpTransport, ProtocolError, TimeBudgetError
from bearing_correlation import clip_pair_ranges, grid_scan_points, paired_bearing_box, clip_pair_box
from coverage_geometry import OmniCoverage
from sampled_recourse import choose as recourse_choose
from geometry import (
    bearing_region,
    covering_circle,
    distance,
    fallback_points,
    initial_region,
    point_polygon_distance,
    select_measurement,
    clip,
    open_visit_order,
    clear_approach,
    exclude_negative,
    range_weighted_center,
)


class Robot:

    sample_count = 64

    def __init__(
        self, client, problem=3, fast_measures=12, fast_distance=3000.0, progress=None, strategy="auto"
    ):
        if problem != 3:
            raise ValueError("This release supports problem 3 only")
        if strategy not in ("auto", "coverage-grid"):
            raise ValueError("This release contains the fixed T3.13 strategy only")
        self.client, self.problem = (client, 3)
        self.strategy = "coverage-grid"
        self.fast_measures, self.fast_distance = (fast_measures, fast_distance)
        self.progress = progress or (lambda message: None)
        self.clear_threshold = 100.0
        self.correlation_readings = {}
        self.negative_positions = {k: [] for k in range(1, 21)}
        self.excluded = set()
        self.coverage = OmniCoverage(refine_for=grid_scan_points())
        self.last_clear_time = None
        self.coverage_surveys = 0
        self.cleared = set()
        self.localize_replanned = set()
        self._allow_localize_replan = False
        self.scanned = {k: set() for k in range(1, 21)}
        self.fallback_count = self.geometry_warnings = 0
        self.completion_reason = None

    def measure(self, point, channel):
        result = self.client.request("/measure", point, channel)
        if result["measure_result"] == "direction":
            self.correlation_readings.setdefault(channel, []).append((tuple(point), result["svd_deg"]))
        if result["measure_result"] == "no_signal":
            station = tuple(point)
            if station not in self.negative_positions[channel]:
                self.negative_positions[channel].append(station)
        return result

    def _negative_region(self, poly, channel):
        poly = exclude_negative(poly, self.negative_positions[channel])
        readings = self.correlation_readings.get(channel, [])
        # A fixed receiver radius gives linear positive/negative constraints.
        for positive, _ in readings:
            for negative in self.negative_positions[channel]:
                nx = 2 * (negative[0] - positive[0])
                ny = 2 * (negative[1] - positive[1])
                upper = negative[0] ** 2 + negative[1] ** 2 - positive[0] ** 2 - positive[1] ** 2
                poly = clip(poly, nx, ny, upper)
        # Same-channel grid weights bound angle differences without a noise seed.
        for i, (first, a) in enumerate(readings):
            for second, b in readings[i + 1 :]:
                if distance(first, second) > 100:
                    continue
                bound = paired_bearing_box(first, a, second, b)
                if bound is not None:
                    poly = clip_pair_box(poly, bound, clip)
                    poly = clip_pair_ranges(poly, first, a, second, b, bound, clip)
        return poly

    def clear(self, point, channel):
        result = self.client.request("/clear", point, channel)
        success = result["clear_result"] == "success"
        if success:
            self.cleared.add(channel)
            self.last_clear_time = self.client.virtual_time
            self.progress(
                f"Cleared channel {channel}; total={len(self.cleared)}; virtual={self.client.virtual_time:.1f}s"
            )
        return success

    def localize(self, channel, observation, observations=None):
        c = self.client
        observations = observations if observations is not None else [(c.position, observation)]
        near = next(
            (position for position, result in observations if result["measure_result"] == "near"), None
        )
        if near is not None:
            if not self.clear(near, channel):
                raise ProtocolError("Stored near observation followed by failed clear")
            return
        if observation["measure_result"] == "near":
            if not self.clear(c.position, channel):
                raise ProtocolError("near followed by failed clear at the same position")
            return
        first = (observations[-1][0], observations[-1][1]["svd_deg"])
        poly = initial_region()
        for position, result in observations:
            poly = bearing_region(poly, position, result["svd_deg"])
        poly = self._negative_region(poly, channel)
        history, failed = ([position for position, _ in observations], [])
        used_distance, measures, tries, no_signal = (0.0, 0, 0, 0)
        if (
            poly
            and self.fast_measures
            and (covering_circle(poly)[1] > self.clear_threshold)
            and (min((distance(c.position, p) for p in history)) >= 100)
            and (distance(c.position, covering_circle(poly)[0]) <= 1250)
        ):
            # A failed probe never discards the earlier positive observation.
            probe = self.measure(c.position, channel)
            measures += 1
            history.append(c.position)
            if probe["measure_result"] == "near":
                if not self.clear(c.position, channel):
                    raise ProtocolError("near followed by failed clear during stationary probe")
                return
            if probe["measure_result"] == "direction":
                if self._allow_localize_replan and channel not in self.localize_replanned:
                    self.localize_replanned.add(channel)
                    return False  # New observation; let the caller reorder tasks.
                poly = bearing_region(poly, c.position, probe["svd_deg"])
            poly = self._negative_region(poly, channel)
        while poly and measures < self.fast_measures:
            center, radius = covering_circle(poly)
            center = clear_approach(poly, c.position, center)
            if (
                radius <= self.clear_threshold
                and tries < 5
                and (distance(c.position, center) + used_distance <= self.fast_distance)
            ):
                if all((distance(center, old) > 2 for old in failed)):
                    used_distance += distance(c.position, center)
                    tries += 1
                    if self.clear(center, channel):
                        return
                    failed.append(center)
                    # A failed guaranteed clear contradicts the current region.
                    if radius <= 20 - 1e-05:
                        poly = []
                        self.geometry_warnings += 1
                        break
            next_point = select_measurement(
                poly, c.position, first, history, self.fast_distance - used_distance
            )
            if next_point is None:
                break
            used_distance += distance(c.position, next_point)
            history.append(next_point)
            result = self.measure(next_point, channel)
            measures += 1
            if result["measure_result"] == "near":
                if not self.clear(c.position, channel):
                    raise ProtocolError("near followed by failed clear")
                return
            if result["measure_result"] == "direction":
                if self._allow_localize_replan and channel not in self.localize_replanned:
                    self.localize_replanned.add(channel)
                    return False  # New observation; let the caller reorder tasks.
                poly = bearing_region(poly, c.position, result["svd_deg"])
                poly = self._negative_region(poly, channel)
                if not poly:
                    self.geometry_warnings += 1
                no_signal = 0
            else:
                poly = self._negative_region(poly, channel)
                no_signal += 1
                if no_signal >= 3:
                    break
        self.fallback_count += 1
        self.progress(f"Finite clear coverage for channel {channel}")
        # Finite fallback prunes only by conservative regions and actual failures.
        for point in fallback_points(*first):
            if poly and point_polygon_distance(point, poly) > 20 + 1e-05:
                continue
            if any((distance(point, old) < 1e-06 for old in failed)):
                continue
            if self.clear(point, channel):
                return
        raise ProtocolError(
            f"Finite clear coverage exhausted for channel {channel}; physical/geometry inconsistency"
        )

    def execute(self):
        points = grid_scan_points()
        remaining = set(range(len(points)))
        pending = {}
        survey_locations = []
        c = self.client
        c.request("/enter")

        def survey(point, anchor=None, known_only=False, channel_subset=None):
            self.progress(f"Survey at {point}; anchors remaining={len(remaining)}")
            survey_locations.append(point)
            self.coverage_surveys += 1
            # Preserve channel order: it determines switch costs and later decisions.
            channels = [c.channel] + [k for k in range(1, 21) if k != c.channel]
            for channel in channels:
                if channel_subset is not None and channel not in channel_subset:
                    continue
                if channel in self.cleared or channel in self.excluded:
                    continue
                if known_only and channel not in pending:
                    continue
                if channel in pending and (not known_only):
                    continue
                if channel in pending:
                    region = initial_region()
                    for previous, reading in pending[channel]:
                        region = bearing_region(region, previous, reading["svd_deg"])
                    region = self._negative_region(region, channel)
                    if region and covering_circle(region)[1] <= 35:
                        continue
                result = self.measure(point, channel)
                if anchor is not None:
                    self.scanned[channel].add(anchor)
                if result["measure_result"] == "near":
                    if not self.clear(point, channel):
                        raise ProtocolError("near followed by failed clear")
                    pending.pop(channel, None)
                elif result["measure_result"] == "direction":
                    pending.setdefault(channel, []).append((point, result))
                elif self.coverage.no_signal(point, channel):
                    if channel in pending:
                        raise ProtocolError("Coverage certificate contradicts a known target")
                    self.excluded.add(channel)
                    self.client.log(
                        "absence_certificate",
                        channel=channel,
                        cell_count=len(self.coverage.cells),
                        virtual_time_s=c.virtual_time,
                    )
                if len(self.cleared) == 16:
                    return

        def estimated_center(channel):
            poly = initial_region()
            for position, result in pending[channel]:
                poly = bearing_region(poly, position, result["svd_deg"])
            poly = self._negative_region(poly, channel)
            if poly:
                return range_weighted_center(
                    poly, [p for p, _ in pending[channel]], self.negative_positions[channel]
                )
            center = covering_circle(poly)[0] if poly else pending[channel][-1][0]
            return center

        def estimated_visit_cost(channel):
            return distance(c.position, estimated_center(channel))

        while remaining or pending:
            # The public upper bound proves absence only after 16 observed channels.
            if len(self.cleared | pending.keys()) == 16:
                absent = set(range(1, 21)) - self.cleared - pending.keys() - self.excluded
                self.excluded.update(absent)
                if absent:
                    c.log(
                        "source_count_certificate",
                        channels=sorted(absent),
                        known_channels=16,
                        virtual_time_s=c.virtual_time,
                    )
            if len(self.cleared) == 16:
                self.completion_reason = "source_count_upper_bound"
                return
            if len(self.cleared | self.excluded) == 20:
                self.completion_reason = "continuous_area_coverage"
                return
            if not getattr(self, "rotation_chosen", False) and 0 not in remaining:
                from rotate_selector import choose

                unknown = set(range(1, 21)) - self.cleared - self.excluded - pending.keys()
                layout, coverage, report = choose(self, {k: estimated_center(k) for k in pending}, unknown)
                points = [tuple(p) for p in layout["points"]]
                self.coverage = coverage
                self.rotation_axes = list(layout["axes"])
                self.rotation_info = report
                self.rotation_chosen = True
                c.log("rotation_plan", **report)
            if 0 not in remaining:
                from moving_stations import adjust

                unknown = [k for k in range(1, 21) if k not in self.cleared | self.excluded | pending.keys()]
                adjust(self, points, remaining, {k: estimated_center(k) for k in pending}, unknown)
            if not pending and 0 not in remaining:
                from tail_lens import final_station

                unknown = [k for k in range(1, 21) if k not in self.cleared | self.excluded]
                station = final_station(self, {i: points[i] for i in remaining}, unknown)
                if station is not None:
                    survey(station)
                    continue
            route_anchor = route_target = None
            if pending or 0 not in remaining:
                tasks = {("target", k): estimated_center(k) for k in pending}
                unknown = [k for k in range(1, 21) if k not in self.cleared | self.excluded | pending.keys()]
                tasks.update(
                    {
                        ("anchor", i): points[i]
                        for i in sorted(remaining)
                        if unknown and self.coverage.gain(points[i], unknown) > 0
                    }
                )
                kind, index = recourse_choose(self, tasks)[0]
                if kind == "anchor":
                    route_anchor = index
                else:
                    route_target = index
            if pending and route_anchor is None:
                channel = route_target if route_target is not None else min(pending, key=estimated_visit_cost)
                observations = pending.pop(channel)
                self._allow_localize_replan = True
                try:
                    localized = self.localize(channel, observations[-1][1], observations)
                finally:
                    self._allow_localize_replan = False
                if localized is False:
                    pending[channel] = [
                        (point, dict(measure_result="direction", svd_deg=angle))
                        for point, angle in self.correlation_readings[channel]
                    ]
                    continue
                if len(self.cleared) == 16:
                    self.completion_reason = "source_count_upper_bound"
                    return
            else:
                channels = [k for k in range(1, 21) if k not in self.cleared | self.excluded]
                useful = {i for i in remaining if self.coverage.gain(points[i], channels) > 0}
                if not useful:
                    raise ProtocolError("Remaining cells lack a covering anchor")
                # A remaining square must still have a real future covering station.
                remaining.intersection_update(useful)
                anchor = (
                    route_anchor
                    if route_anchor is not None
                    else min(remaining, key=lambda i: distance(c.position, points[i]))
                )
                station = points[anchor]
                previously_known = set(pending)
                survey(station, anchor)
                remaining.remove(anchor)
                if self.fast_measures and anchor != 0:
                    fresh = set(pending) - previously_known
                    vertical = (
                        self.rotation_axes[anchor] == "y"
                        if hasattr(self, "rotation_axes")
                        else anchor in (1, 4)
                    )
                    fresh = {
                        k
                        for k in fresh
                        if abs(
                            (math.cos if vertical else math.sin)(math.radians(pending[k][-1][1]["svd_deg"]))
                        )
                        >= 0.5
                    }
                    if fresh:
                        probe = (station[0], station[1] + 12) if vertical else (station[0] + 12, station[1])
                        survey(probe, known_only=True, channel_subset=fresh)
                if (
                    self.fast_measures
                    and anchor == 0
                    and (len(survey_locations) == 1)
                    and (len(pending) >= 3)
                ):
                    vertical = {
                        k
                        for k, obs in pending.items()
                        if abs(math.sin(math.radians(obs[-1][1]["svd_deg"])))
                        >= abs(math.cos(math.radians(obs[-1][1]["svd_deg"])))
                    }
                    horizontal = set(pending) - vertical
                    if vertical:
                        survey((6.0, -6.0), known_only=True, channel_subset=vertical)
                    if horizontal:
                        survey((-6.0, 6.0), known_only=True, channel_subset=horizontal)
                    continue
        if len(self.cleared) == 16:
            self.completion_reason = "source_count_upper_bound"
            return
        if len(self.cleared | self.excluded) == 20:
            self.completion_reason = "continuous_area_coverage"
            return
        raise ProtocolError("Retargeted survey tasks ended without a complete area certificate")

    def run(self):
        error = None
        try:
            self.execute()
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.client.log("failure", error=error)
        finally:
            if self.client.entered and not self.client.exited:
                try:
                    self.client.request("/exit")
                except (Exception, KeyboardInterrupt) as exc:
                    exit_error = f"exit unconfirmed: {type(exc).__name__}: {exc}"
                    error = f"{error}; {exit_error}" if error else exit_error
        c = self.client
        predicted = (
            c.counters["movement_time_us"] / 1e6
            + c.counters["switches"]
            + 5 * c.counters["measures"]
            + 3 * c.counters["clear_attempts"]
            + 2 * c.counters["cleared"]
        )
        return dict(
            recourse_stats=getattr(self, "recourse_stats", {}),
            deferred_count=len(self.localize_replanned),
            problem=self.problem,
            strategy=self.strategy,
            completed=self.completion_reason is not None,
            completion_reason=self.completion_reason,
            exit_confirmed=c.exited,
            error=error,
            cleared_channels=sorted(self.cleared),
            cleared_count=len(self.cleared),
            virtual_time_s=c.virtual_time,
            average_clear_time_s=c.virtual_time / len(self.cleared) if self.cleared else None,
            real_elapsed_s=time.monotonic() - c.start if c.start is not None else None,
            counters=c.counters,
            failed_clear_count=c.counters["clear_attempts"] - c.counters["cleared"],
            predicted_virtual_time_s=predicted,
            time_residual_s=c.virtual_time - predicted,
            fallback_count=self.fallback_count,
            geometry_warnings=self.geometry_warnings,
            excluded_channels=sorted(self.excluded),
            survey_count=self.coverage_surveys,
            enroute_probe_count=0,
            enroute_replan_count=0,
            shortened_survey_legs=0,
            planned_leg_savings_m=0.0,
            last_clear_virtual_time_s=self.last_clear_time,
            tail_time_s=c.virtual_time - self.last_clear_time if self.last_clear_time is not None else None,
            scan_counts={str(k): len(v) for k, v in self.scanned.items()},
        )


def main():
    from cli import main as run_cli

    return run_cli(Robot)


if __name__ == "__main__":
    sys.exit(main())
