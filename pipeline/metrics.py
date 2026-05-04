from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

from app.schemas import CalibrationMode, DeliveryMetadata, DeliveryResult, Point2D, Point3D
from pipeline.trajectory_solver import TrajectorySolution

STUMP_TO_STUMP_PITCH_M = 20.12
CREASE_TO_CREASE_M = 17.68
SPEED_LONGITUDINAL_SCALE = CREASE_TO_CREASE_M / STUMP_TO_STUMP_PITCH_M

# Radar-style initial speed: short post-release window in air (not release→bounce / full pitch).
RADAR_T_EPS_S = 0.02
RADAR_T_WIN_MAX_S = 0.24
RADAR_BEFORE_BOUNCE_PAD_S = 0.04
RADAR_D_MIN_M = 3.0
RADAR_D_MAX_M = 8.0
RADAR_V_MIN_KPH = 25.0
RADAR_V_MAX_KPH = 175.0


@dataclass
class PhysicalMetrics:
    speed_kph: float | None
    speed_mph: float | None
    speed_confidence: float
    bounce_distance_from_batter_stumps_m: float | None
    bounce_distance_from_bowler_stumps_m: float | None
    line_meters_from_center: float | None
    line_meters_from_off_stump: float | None
    length_category: str
    line_category: str
    calibration_confidence: float


def _distance_m_pitch(
    p0: Point2D,
    p1: Point2D,
    metadata: DeliveryMetadata,
    pitch_length: float,
    pitch_width: float,
) -> float | None:
    c0 = _pitch_coordinate(p0, metadata)
    c1 = _pitch_coordinate(p1, metadata)
    if c0 is None or c1 is None:
        return None
    longitudinal_m = abs(c1[0] - c0[0]) * pitch_length * SPEED_LONGITUDINAL_SCALE
    lateral_m = (c1[1] - c0[1]) * pitch_width
    d = math.hypot(lateral_m, longitudinal_m)
    return d if d >= 0.015 else None


def _sample_to_point2d(sample: Point2D | Point3D | dict[str, Any]) -> Point2D | None:
    if isinstance(sample, Point2D):
        if sample.t is None:
            return None
        return sample
    if isinstance(sample, Point3D):
        if sample.t is None:
            return None
        return Point2D(x=float(sample.x), y=float(sample.z), t=float(sample.t))
    if isinstance(sample, dict):
        raw_t = sample.get("t")
        if raw_t is None:
            return None
        return Point2D(x=float(sample.get("x", 0.0)), y=float(sample.get("y", 0.0)), t=float(raw_t))
    return None


def _sorted_track(track: Sequence[Point2D | Point3D | dict[str, Any]]) -> list[Point2D]:
    parsed = [_sample_to_point2d(s) for s in track]
    pts = [p for p in parsed if p is not None]
    pts.sort(key=lambda p: float(p.t or 0.0))
    return pts


def estimate_radar_initial_speed_kph(
    track: Sequence[Point2D | Point3D | dict[str, Any]],
    release_t: float,
    release_anchor: Point2D,
    bounce_t: float | None,
    metadata: DeliveryMetadata,
    pitch_length: float,
    pitch_width: float,
) -> tuple[float | None, dict[str, Any]]:
    """
    Approximate radar gun reading: initial air speed from arc length vs time shortly after release.
    Uses pitch-plane distance (not image pixels) and excludes the bounce→stumps segment.
    """
    pts = _sorted_track(track)
    detail: dict[str, Any] = {"radarSpeedMethod": "release_window_regression"}
    if len(pts) < 2:
        detail["radarSpeedFailure"] = "insufficient_track_points"
        return None, detail

    def build_accumulated(t_hi: float) -> tuple[list[tuple[float, float]], Point2D]:
        acc: list[tuple[float, float]] = []
        prev = release_anchor
        s_total = 0.0
        for p in pts:
            t = float(p.t or 0.0)
            if t <= release_t + RADAR_T_EPS_S:
                prev = p
                continue
            if t > t_hi:
                break
            seg = _distance_m_pitch(prev, p, metadata, pitch_length, pitch_width)
            if seg is None:
                prev = p
                continue
            s_total += seg
            acc.append((t - release_t, s_total))
            prev = p
            if s_total >= RADAR_D_MAX_M:
                break
        return acc, prev

    t_hi_primary = release_t + RADAR_T_WIN_MAX_S
    if bounce_t is not None and bounce_t > release_t + RADAR_T_EPS_S + 0.03:
        t_hi_primary = min(t_hi_primary, bounce_t - RADAR_BEFORE_BOUNCE_PAD_S)
    if t_hi_primary <= release_t + RADAR_T_EPS_S + 0.04:
        detail["radarSpeedFailure"] = "window_collapsed_near_bounce"
        return None, detail

    accumulated, _ = build_accumulated(t_hi_primary)
    arc_short = bool(accumulated) and accumulated[-1][1] < RADAR_D_MIN_M
    room_for_extension = bounce_t is None or bounce_t > release_t + 0.34
    if (not accumulated or arc_short) and room_for_extension:
        t_hi_ext = min(release_t + 0.33, bounce_t - RADAR_BEFORE_BOUNCE_PAD_S if bounce_t else release_t + 0.33)
        if t_hi_ext > t_hi_primary + 0.04:
            accumulated, _ = build_accumulated(t_hi_ext)

    if len(accumulated) < 2:
        detail["radarSpeedFailure"] = "no_window_samples"
        return None, detail

    t_rel_vals = [a[0] for a in accumulated]
    s_vals = [a[1] for a in accumulated]
    duration = t_rel_vals[-1] - t_rel_vals[0]
    if duration < 0.05:
        detail["radarSpeedFailure"] = "duration_too_short"
        return None, detail

    n = len(t_rel_vals)
    sum_t = sum(t_rel_vals)
    sum_s = sum(s_vals)
    sum_tt = sum(tt * tt for tt in t_rel_vals)
    sum_ts = sum(t_rel_vals[i] * s_vals[i] for i in range(n))
    denom = n * sum_tt - sum_t * sum_t
    if abs(denom) < 1e-8:
        v_mps = (s_vals[-1] - s_vals[0]) / max(1e-6, t_rel_vals[-1] - t_rel_vals[0])
    else:
        v_mps = (n * sum_ts - sum_t * sum_s) / denom
    if v_mps <= 0.0:
        v_mps = (s_vals[-1] - s_vals[0]) / max(1e-6, t_rel_vals[-1] - t_rel_vals[0])
    if v_mps <= 0.0:
        detail["radarSpeedFailure"] = "non_positive_slope"
        return None, detail

    v_raw_mps = v_mps
    mean_window = float(sum(t_rel_vals) / n)
    if mean_window > 0.16:
        v_mps *= 1.0 + min(0.07, 0.28 * (mean_window - 0.16))

    kph = max(RADAR_V_MIN_KPH, min(RADAR_V_MAX_KPH, v_mps * 3.6))
    detail.update(
        {
            "radarInitialSpeedKph": kph,
            "radarWindowEndSecFromRelease": t_rel_vals[-1],
            "radarArcMeters": s_vals[-1],
            "radarSampleCount": n,
            "radarRegressionMpsRaw": v_raw_mps,
            "radarRegressionMpsAdjusted": v_mps,
        }
    )
    return kph, detail


def compute_speed_kph(solution: TrajectorySolution, metadata: DeliveryMetadata) -> float | None:
    physical = compute_physical_metrics(solution, metadata)
    return physical.speed_kph


def compute_physical_metrics(solution: TrajectorySolution, metadata: DeliveryMetadata) -> PhysicalMetrics:
    calibration_confidence = _calibration_confidence(metadata)
    pitch_length = _pitch_length_meters(metadata)
    pitch_width = metadata.pitchCalibration.pitchWidthMeters

    release_pitch = _pitch_coordinate(solution.release_point, metadata)
    bounce_pitch = _pitch_coordinate(solution.bounce_point, metadata)
    end_pitch = _pitch_coordinate(solution.end_point, metadata)

    speed_kph: float | None = None
    speed_confidence = 0.0
    rt = solution.release_point.t
    if rt is not None:
        bounce_t_opt = float(solution.bounce_point.t) if solution.bounce_point.t is not None else None
        radar_kph, _radar_detail = estimate_radar_initial_speed_kph(
            solution.trajectory,
            float(rt),
            solution.release_point,
            bounce_t_opt,
            metadata,
            pitch_length,
            pitch_width,
        )
        if radar_kph is not None:
            speed_kph = radar_kph
            n_pts = len([p for p in solution.trajectory if p.t is not None])
            speed_confidence = min(
                0.94,
                max(
                    0.12,
                    calibration_confidence * 0.52
                    + solution.confidence * 0.33
                    + min(1.0, n_pts / 26.0) * 0.15,
                ),
            )

    speed_mph = speed_kph * 0.621371 if speed_kph is not None else None

    bounce_distance_from_batter: float | None = None
    bounce_distance_from_bowler: float | None = None
    line_meters_from_center: float | None = None
    line_meters_from_off_stump: float | None = None
    if bounce_pitch is not None:
        longitudinal = max(0.0, min(1.0, bounce_pitch[0]))
        lateral = bounce_pitch[1]
        bounce_distance_from_bowler = longitudinal * pitch_length
        bounce_distance_from_batter = (1.0 - longitudinal) * pitch_length
        line_meters_from_center = (lateral - 0.5) * pitch_width
        # Approximate off stump as half a wicket width from centerline.
        line_meters_from_off_stump = line_meters_from_center - 0.1143

    length_category = classify_length_from_distance(bounce_distance_from_batter, solution.bounce_point)
    length_category = _adjust_length_for_late_bounce(
        length_category=length_category,
        bounce_distance_from_batter=bounce_distance_from_batter,
        bounce_t=solution.bounce_point.t,
        impact_t=solution.end_point.t,
        release_t=solution.release_point.t,
    )
    line_category = classify_line_from_meters(line_meters_from_center, solution.bounce_point)
    return PhysicalMetrics(
        speed_kph=speed_kph,
        speed_mph=speed_mph,
        speed_confidence=speed_confidence,
        bounce_distance_from_batter_stumps_m=bounce_distance_from_batter,
        bounce_distance_from_bowler_stumps_m=bounce_distance_from_bowler,
        line_meters_from_center=line_meters_from_center,
        line_meters_from_off_stump=line_meters_from_off_stump,
        length_category=length_category,
        line_category=line_category,
        calibration_confidence=calibration_confidence,
    )


def _maybe_swap_corridor_depth_video_only(
    d_batter: float | None,
    d_bowler: float | None,
    bp: Point2D,
    metadata: DeliveryMetadata,
) -> tuple[float | None, float | None, bool]:
    """
    Pitch polygon + single camera often maps bounce to the wrong stump end (batter vs bowler).
    Heuristic: huge "from batter" + tiny "from bowler" while the bounce sits low in frame (batter
    half) implies longitudinal fraction is inverted — swap distances for length/speed.
    """
    if metadata.calibrationMode != CalibrationMode.video_only:
        return d_batter, d_bowler, False
    if d_batter is None or d_bowler is None:
        return d_batter, d_bowler, False
    y = float(bp.y)
    strong = float(d_batter) > 13.2 and float(d_bowler) < 5.8
    soft = float(d_batter) > 9.2 and float(d_bowler) < 7.2 and y >= 0.34
    if strong or soft:
        return d_bowler, d_batter, True
    return d_batter, d_bowler, False


def refine_physical_metrics_after_reconstruction(
    result: DeliveryResult,
    metadata: DeliveryMetadata,
    baseline: PhysicalMetrics,
) -> PhysicalMetrics:
    """Recompute speed/length from final release/bounce/impact times and image landmarks."""
    rp, bp, ep = result.releasePoint, result.bouncePoint, result.endPoint
    rt = result.releaseSec if result.releaseSec is not None else rp.t
    bt = result.bounceSec if result.bounceSec is not None else bp.t
    et = result.stumpImpactSec if result.stumpImpactSec is not None else ep.t
    pitch_length = _pitch_length_meters(metadata)
    pitch_width = metadata.pitchCalibration.pitchWidthMeters

    speed_kph = baseline.speed_kph
    speed_mph = baseline.speed_mph
    speed_confidence = baseline.speed_confidence

    track_seq: Sequence[Point2D | Point3D | dict[str, Any]] = result.reconstructedTrajectory or []
    if len(track_seq) < 2 and result.finalTrajectory:
        track_seq = result.finalTrajectory
    if rt is not None and track_seq:
        bounce_for_radar = float(bt) if bt is not None else None
        radar_kph, radar_detail = estimate_radar_initial_speed_kph(
            track_seq,
            float(rt),
            result.releasePoint,
            bounce_for_radar,
            metadata,
            pitch_length,
            pitch_width,
        )
        if radar_kph is not None:
            speed_kph = radar_kph
            speed_mph = radar_kph * 0.621371
            speed_confidence = min(0.95, max(float(speed_confidence), 0.42, baseline.speed_confidence * 0.62 + 0.18))
        result.debug.reconstructionStats.update(radar_detail)

    release_pitch = _pitch_coordinate(rp, metadata)
    bounce_pitch = _pitch_coordinate(bp, metadata)
    end_pitch = _pitch_coordinate(ep, metadata)

    bounce_distance_from_batter: float | None = None
    bounce_distance_from_bowler: float | None = None
    line_meters_from_center: float | None = None
    line_meters_from_off_stump: float | None = None
    if bounce_pitch is not None:
        longitudinal = max(0.0, min(1.0, bounce_pitch[0]))
        lateral = bounce_pitch[1]
        bounce_distance_from_bowler = longitudinal * pitch_length
        bounce_distance_from_batter = (1.0 - longitudinal) * pitch_length
        line_meters_from_center = (lateral - 0.5) * pitch_width
        line_meters_from_off_stump = line_meters_from_center - 0.1143

    corridor_flipped = False
    bounce_distance_from_batter, bounce_distance_from_bowler, corridor_flipped = _maybe_swap_corridor_depth_video_only(
        bounce_distance_from_batter,
        bounce_distance_from_bowler,
        bp,
        metadata,
    )

    length_category = classify_length_from_distance(bounce_distance_from_batter, bp)
    length_category = _adjust_length_for_late_bounce(
        length_category=length_category,
        bounce_distance_from_batter=bounce_distance_from_batter,
        bounce_t=bt,
        impact_t=et,
        release_t=rt,
    )
    line_category = classify_line_from_meters(line_meters_from_center, bp)

    post_dur = float(et - bt) if et is not None and bt is not None and et > bt else 0.0
    pre_dur = float(bt - rt) if bt is not None and rt is not None and bt > rt else 0.0
    video_only = metadata.calibrationMode == CalibrationMode.video_only

    # Video-only: corridor depth inconsistent with timing — adjust length label only (speed is radar window).
    bounce_y_img = float(bp.y)
    if (
        video_only
        and (not corridor_flipped)
        and pre_dur >= 0.18
        and post_dur > 0.0
        and post_dur <= 0.52
        and bounce_y_img >= 0.44
        and (post_dur / pre_dur) <= 0.92
    ):
        absurd_batter = bounce_distance_from_batter is not None and bounce_distance_from_batter > 11.5
        if length_category in {"bouncer", "short_ball"} or absurd_batter:
            length_category = "yorker"

    return PhysicalMetrics(
        speed_kph=speed_kph,
        speed_mph=speed_mph,
        speed_confidence=speed_confidence,
        bounce_distance_from_batter_stumps_m=bounce_distance_from_batter,
        bounce_distance_from_bowler_stumps_m=bounce_distance_from_bowler,
        line_meters_from_center=line_meters_from_center,
        line_meters_from_off_stump=line_meters_from_off_stump,
        length_category=length_category,
        line_category=line_category,
        calibration_confidence=baseline.calibration_confidence,
    )


def classify_line(point: Point2D) -> str:
    if point.x < 0.42:
        return "leg side"
    if point.x > 0.58:
        return "outside off"
    return "stumps"


def classify_length(point: Point2D) -> str:
    if point.y < 0.37:
        return "bouncer"
    if point.y < 0.54:
        return "short_ball"
    if point.y < 0.68:
        return "short_ball"
    if point.y < 0.80:
        return "good_length"
    if point.y < 0.89:
        return "full_length"
    return "yorker"


def classify_line_from_meters(line_meters_from_center: float | None, fallback_point: Point2D) -> str:
    if line_meters_from_center is None:
        return classify_line(fallback_point)
    if line_meters_from_center < -0.23:
        return "leg side"
    if line_meters_from_center > 0.23:
        return "outside off"
    return "stumps"


def classify_length_from_distance(distance_from_batter_m: float | None, fallback_point: Point2D) -> str:
    if distance_from_batter_m is None:
        return classify_length(fallback_point)
    # Typical yorker lands ~1.5–3 m short of the popping crease; include ~2 m band.
    if distance_from_batter_m < 2.85:
        return "yorker"
    if distance_from_batter_m < 4.5:
        return "full_length"
    if distance_from_batter_m < 7.5:
        return "good_length"
    if distance_from_batter_m < 10.5:
        return "short_ball"
    if distance_from_batter_m < 14.0:
        return "short_ball"
    return "bouncer"


def _adjust_length_for_late_bounce(
    length_category: str,
    bounce_distance_from_batter: float | None,
    bounce_t: float | None,
    impact_t: float | None,
    release_t: float | None = None,
) -> str:
    # When the detector briefly drops just before bounce, distance-only mapping can
    # overestimate length. If bounce-to-impact is very short, treat it as yorker-like.
    if bounce_t is None or impact_t is None:
        return length_category
    post_bounce_duration = impact_t - bounce_t
    if post_bounce_duration <= 0.0:
        return length_category
    pre_bounce = (bounce_t - release_t) if release_t is not None else None
    if length_category in {"short_ball", "bouncer"} and post_bounce_duration <= 0.48:
        if bounce_distance_from_batter is None or bounce_distance_from_batter <= 12.5:
            return "yorker"
    if length_category == "bouncer" and bounce_distance_from_batter is not None and bounce_distance_from_batter <= 10.5:
        if post_bounce_duration <= 0.55:
            return "yorker"
    if (
        pre_bounce is not None
        and pre_bounce >= 0.22
        and post_bounce_duration <= 0.42
        and (post_bounce_duration / pre_bounce) <= 0.72
        and (bounce_distance_from_batter is None or bounce_distance_from_batter < 9.0)
    ):
        return "yorker"
    return length_category


def swing_amount(solution: TrajectorySolution) -> float:
    points = solution.trajectory
    if len(points) < 3:
        return 0.0
    first = points[0]
    last = points[-1]
    denom = max(1e-4, math.hypot(last.x - first.x, last.z - first.z))
    max_distance = 0.0
    for point in points[1:-1]:
        distance = abs((last.z - first.z) * point.x - (last.x - first.x) * point.z + last.x * first.z - last.z * first.x) / denom
        max_distance = max(max_distance, distance)
    return max_distance


def _pitch_length_meters(metadata: DeliveryMetadata) -> float:
    if metadata.pitch is not None and metadata.pitch.lengthMeters > 0:
        return metadata.pitch.lengthMeters
    return metadata.pitchCalibration.pitchLengthMeters


def _calibration_confidence(metadata: DeliveryMetadata) -> float:
    if metadata.calibrationMode == CalibrationMode.video_only:
        base, cap = 0.20, 0.55
    elif metadata.calibrationMode == CalibrationMode.guided_boxes:
        base, cap = 0.50, 0.85
    else:
        base, cap = 0.65, 0.98
    score = base
    if metadata.pitch is not None:
        score += 0.10
        if metadata.pitch.batterEndStumpsWorld and metadata.pitch.bowlerEndStumpsWorld:
            score += 0.15
        if metadata.pitch.corridorWorld:
            score += 0.10
    if metadata.guidedBoxes is not None:
        score += 0.15
        if metadata.guidedBoxes.setupQuality is not None:
            score += max(0.0, min(0.20, metadata.guidedBoxes.setupQuality * 0.20))
    if metadata.camera is not None:
        if metadata.camera.intrinsics:
            score += 0.10
        if metadata.camera.staticPose or metadata.camera.extrinsicsByFrame:
            score += 0.10
    if len(metadata.corridorGeometry.pitchPolygon) >= 4:
        score += 0.20
    if metadata.pitchCalibration.pitchLengthMeters > 0:
        score += 0.10
    return max(base, min(cap, score))


def _pitch_coordinate(point: Point2D, metadata: DeliveryMetadata) -> tuple[float, float] | None:
    polygon = metadata.corridorGeometry.pitchPolygon
    if len(polygon) < 4:
        return (point.y, point.x)
    far_left, far_right, near_right, near_left = polygon[:4]
    best: tuple[float, float, float] | None = None
    for step in range(81):
        fraction = step / 80
        left_x = far_left.x + (near_left.x - far_left.x) * fraction
        left_y = far_left.y + (near_left.y - far_left.y) * fraction
        right_x = far_right.x + (near_right.x - far_right.x) * fraction
        right_y = far_right.y + (near_right.y - far_right.y) * fraction
        dx = right_x - left_x
        dy = right_y - left_y
        denom = max(1e-4, dx * dx + dy * dy)
        lateral = ((point.x - left_x) * dx + (point.y - left_y) * dy) / denom
        proj_x = left_x + dx * lateral
        proj_y = left_y + dy * lateral
        distance = math.hypot(point.x - proj_x, point.y - proj_y)
        if best is None or distance < best[0]:
            best = (distance, fraction, lateral)
    if best is None:
        return None
    return (max(0.0, min(1.0, best[1])), max(-1.5, min(2.5, best[2])))
