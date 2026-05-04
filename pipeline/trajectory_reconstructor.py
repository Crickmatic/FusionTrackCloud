from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas import Candidate, DeliveryMetadata, DeliveryResult, Point2D
from pipeline.bounce_model import apply_bounce_physics


@dataclass
class ReconstructedTrajectory:
    points: list[Point2D]
    final_points: list[dict[str, float | int | str]]
    release_sec: float | None
    bounce_sec: float | None
    impact_sec: float | None
    segments: dict[str, str]
    event_sources: dict[str, str]
    stats: dict[str, float | int | str]


class TrajectoryReconstructor:
    WEAK_CONF_MIN = 0.05
    STRONG_CONF_MIN = 0.25
    BOUNCE_WINDOW_SEC = 0.20
    POST_BOUNCE_WINDOW_SEC = 0.55

    def reconstruct(
        self,
        result: DeliveryResult,
        metadata: DeliveryMetadata,
        selected_tracklet: list[Candidate],
        merged_candidates: list[Candidate],
        kalman_tracked_points: list[Candidate] | None = None,
        ai_correction: dict | None = None,
        bounce_net: dict | None = None,
    ) -> ReconstructedTrajectory:
        fps = float(metadata.fps or metadata.extras.get("videoFps") or 30.0)
        release = result.releaseSec or result.releasePoint.t
        bounce = result.bounceSec or result.bouncePoint.t
        impact = result.stumpImpactSec or result.endPoint.t
        seed_candidates = list(selected_tracklet)
        if kalman_tracked_points:
            seed_candidates.extend(kalman_tracked_points)
        strong = sorted(
            [candidate for candidate in seed_candidates if self._candidate_time(candidate, metadata, fps) >= (release or 0.0)],
            key=lambda candidate: self._candidate_time(candidate, metadata, fps),
        )
        if not strong:
            raw_points = sorted(result.trajectory2D, key=lambda point: point.t or 0.0)
            strong = [
                Candidate(frameIndex=index, x=point.x, y=point.y, confidence=0.8, source="solver_path", timestampSec=point.t)
                for index, point in enumerate(raw_points)
                if point.t is not None
            ]
        strong = self._trim_in_hand_candidates(strong, metadata, fps)
        if strong:
            first_strong_time = self._candidate_time(strong[0], metadata, fps)
            if release is None or release < first_strong_time:
                release = first_strong_time

        release_src = "raw/highest-arm-path"
        bounce_src = "raw"
        impact_src = "projected_to_stumps" if result.endpointSource == "projected_to_stumps" else "raw"
        stats: dict[str, float | int | str] = {
            "totalRawBallCandidates": 0,
            "weakCandidatesSeen": 0,
            "weakCandidatesUsedInBestPath": 0,
            "bounceCandidatesConsidered": 0,
            "bounceCandidateChosen": "none",
            "postBounceCandidatesUsed": 0,
            "impactSource": "detected" if result.endpointSource == "ball_detector" else "projected",
            "rejectedCandidateReasons": "path_gating_or_temporal_inconsistency",
            "blueCandidatesNearBounce": 0,
            "acceptedCandidatesNearBounce": 0,
            "rejectedCandidatesNearBounce": 0,
            "actualBounceCandidateId": "none",
            "zoomCandidatesPerFrame": 0,
            "zoomCandidatesAccepted": 0,
            "rawCandidatesBetweenLastYellowAndBounce": 0,
            "promotedCandidatesBetweenLastYellowAndBounce": 0,
            "rejectedCandidatesBetweenLastYellowAndBounce": 0,
            "selectedPathFrameCoverage": 0,
            "framesMissingFromSelectedPath": 0,
            "zoomCandidatesAcceptedByFrame": "",
            "finalPreBounceEndFrame": 0,
            "postBounceAnchorFrame": 0,
            "lateralReversalCandidatesRejected": 0,
            "noBounceCandidate": "false",
            "aiCorrectionConfidence": 0.0,
            "postBounceHypothesisChosen": "straight",
            "bounceNetUsedInReconstruction": "false",
        }
        bounce_net = bounce_net or {}
        for key, value in bounce_net.items():
            stats[key] = value
        ai_correction = ai_correction or {}
        ai_bounce_offset = float(ai_correction.get("bounceOffsetY", 0.0) or 0.0)
        ai_angle_delta = float(ai_correction.get("postBounceAngleDelta", 0.0) or 0.0)
        ai_confidence = float(ai_correction.get("confidence", 0.0) or 0.0)
        stats["aiCorrectionConfidence"] = ai_confidence

        if release is None and strong:
            release = self._candidate_time(strong[0], metadata, fps)
            release_src = "tracklet_start"
        if impact is None and strong:
            impact = self._candidate_time(strong[-1], metadata, fps)
            impact_src = "tracklet_end"
        early_release_cand = self._earliest_pre_tracklet_release_candidate(
            strong_candidates=strong,
            merged_candidates=merged_candidates,
            metadata=metadata,
            fps=fps,
        )
        if early_release_cand is not None:
            early_t = self._candidate_time(early_release_cand, metadata, fps)
            if release is None or early_t < release - 0.012:
                release = early_t
                release_src = "earliest_weak_ball_before_tracklet"
                strong = self._merge_rescued_candidate(strong, early_release_cand, metadata, fps)
        recovered_release = self._recover_release_candidate(
            release_sec=release,
            strong_candidates=strong,
            merged_candidates=merged_candidates,
            metadata=metadata,
            fps=fps,
        )
        if recovered_release is not None:
            recovered_time = self._candidate_time(recovered_release, metadata, fps)
            if release is None or recovered_time < release:
                release = recovered_time
                release_src = "earliest_path_consistent_release"
                strong = self._merge_rescued_candidate(strong, recovered_release, metadata, fps)

        if release is not None and impact is not None:
            expected_bounce = release + (impact - release) * 0.58
            if bounce is None or abs(bounce - expected_bounce) > 0.08:
                bounce = expected_bounce
                bounce_src = "estimated_from_pitch_intersection"
        rescued_bounce = self._recover_bounce_candidate(
            estimate_bounce_sec=bounce,
            release_sec=release,
            strong_candidates=strong,
            merged_candidates=merged_candidates,
            metadata=metadata,
            fps=fps,
            stats=stats,
        )
        tentative_bounce_sec = self._candidate_time(rescued_bounce, metadata, fps) if rescued_bounce is not None else bounce
        detector_bounce_strong = self._detector_bounce_temporally_strong(
            rescued_bounce, merged_candidates, tentative_bounce_sec, metadata, fps
        )
        bn_conf = float(bounce_net.get("bounceNetConfidence") or 0.0)
        bn_frame = bounce_net.get("bounceNetBounceFrame")
        bn_pt = bounce_net.get("bounceNetBouncePoint")
        use_bounce_net = (
            not detector_bounce_strong
            and bounce_net.get("bounceSource") == "bounce_net"
            and bn_frame is not None
            and isinstance(bn_pt, dict)
            and bn_conf >= 0.42
        )
        if use_bounce_net:
            stats["bounceNetUsedInReconstruction"] = "true"
            t_bn = self._time_from_video_frame(int(bn_frame), metadata, fps)
            bounce = t_bn
            bounce_src = "bounce_net_visual"
            stats["bounceFallbackReason"] = str(bounce_net.get("bounceFallbackReason") or "")
            stats["weakCandidatesUsedInBestPath"] = int(stats["weakCandidatesUsedInBestPath"]) + 1
            stats["acceptedCandidatesNearBounce"] = int(stats["acceptedCandidatesNearBounce"]) + 1
            synth = Candidate(
                frameIndex=int(bn_frame),
                x=float(bn_pt["x"]),
                y=float(bn_pt["y"]),
                confidence=float(np.clip(bn_conf, 0.2, 0.92)),
                source="bounce_net",
                timestampSec=t_bn,
                modelRole="ball_detector",
            )
            strong = self._merge_rescued_candidate(strong, synth, metadata, fps)
            stats["bounceCandidateChosen"] = f"bounce_net frame={bn_frame},conf={bn_conf:.3f}"
            stats["actualBounceCandidateId"] = f"{bn_frame}:{bn_pt['x']:.3f}:{bn_pt['y']:.3f}"
            pd = bounce_net.get("bounceNetPostDirection") or {}
            if isinstance(pd, dict) and pd.get("dx") is not None and pd.get("dy") is not None:
                stats["bounceNetPostDirectionDx"] = float(pd["dx"])
                stats["bounceNetPostDirectionDy"] = float(pd["dy"])
        elif rescued_bounce is not None:
            bounce = self._candidate_time(rescued_bounce, metadata, fps)
            bounce_src = "rescued_low_conf_path_gated"
            strong = self._merge_rescued_candidate(strong, rescued_bounce, metadata, fps)
            stats["bounceCandidateChosen"] = f"frame={rescued_bounce.frameIndex},conf={rescued_bounce.confidence:.3f}"
            stats["weakCandidatesUsedInBestPath"] = int(stats["weakCandidatesUsedInBestPath"]) + 1
            stats["acceptedCandidatesNearBounce"] = int(stats["acceptedCandidatesNearBounce"]) + 1
            stats["actualBounceCandidateId"] = f"{rescued_bounce.frameIndex}:{rescued_bounce.x:.3f}:{rescued_bounce.y:.3f}"
            corrected_y = float(np.clip(rescued_bounce.y + ai_bounce_offset * max(0.3, ai_confidence), 0.0, 1.0))
            if abs(corrected_y - rescued_bounce.y) >= 1e-4:
                corrected_candidate = Candidate(
                    frameIndex=rescued_bounce.frameIndex,
                    x=rescued_bounce.x,
                    y=corrected_y,
                    confidence=max(float(rescued_bounce.confidence or 0.0), 0.3),
                    source="ai_bounce_corrected",
                    timestampSec=self._candidate_time(rescued_bounce, metadata, fps),
                    modelRole="ball_detector",
                )
                strong = self._merge_rescued_candidate(strong, corrected_candidate, metadata, fps)
        elif self._looks_like_no_bounce_delivery(strong, metadata, fps, release, impact):
            bounce = None
            bounce_src = "no_bounce_detected_full_toss_candidate"
            stats["noBounceCandidate"] = "true"
            reconstructed = self._dense_path_no_bounce(strong, metadata, fps, release, impact, result)
            final_points = self._renderable_path(
                reconstructed=reconstructed,
                raw_candidates=strong,
                metadata=metadata,
                fps=fps,
                release=release,
                bounce=None,
                impact=impact,
                impact_source="inferred",
            )
            return ReconstructedTrajectory(
                points=reconstructed,
                final_points=final_points,
                release_sec=release,
                bounce_sec=None,
                impact_sec=impact,
                segments={"flight": "raw+smoothed_no_bounce", "postBounce": "not_applicable"},
                event_sources={"release": release_src, "bounce": bounce_src, "impact": impact_src},
                stats=stats,
            )

        strong = self._promote_continuous_path_candidates(
            release_sec=release,
            bounce_sec=bounce,
            current_path=strong,
            merged_candidates=merged_candidates,
            metadata=metadata,
            fps=fps,
            stats=stats,
        )

        if bounce is not None and impact is not None and result.endpointSource == "projected_to_stumps":
            if impact - bounce < 0.40:
                impact = bounce + 0.42
                impact_src = "projected_to_stumps_duration_extended"
        if bounce is not None and impact is not None and impact - bounce < 0.30:
            impact = bounce + 0.36
            impact_src = "inferred_from_bounce_physics"

        post_bounce_rescued = self._recover_post_bounce_candidates(
            bounce_sec=bounce,
            impact_sec=impact,
            strong_candidates=strong,
            merged_candidates=merged_candidates,
            metadata=metadata,
            fps=fps,
            stump_roi=result.stumpRoi,
            stats=stats,
        )
        for rescued in post_bounce_rescued:
            strong = self._merge_rescued_candidate(strong, rescued, metadata, fps)
            stats["weakCandidatesUsedInBestPath"] = int(stats["weakCandidatesUsedInBestPath"]) + 1
            if rescued.source == "zoom_crop_tracker":
                stats["zoomCandidatesAccepted"] = int(stats["zoomCandidatesAccepted"]) + 1
        strong = self._filter_lateral_reversal_candidates(strong, metadata, fps, bounce, stats)

        if impact_src.startswith("projected"):
            stats["impactSource"] = "projected"
        elif impact_src.startswith("raw"):
            stats["impactSource"] = "detected"
        else:
            stats["impactSource"] = "inferred"
        reconstructed = self._dense_path_piecewise(strong, metadata, fps, release, bounce, impact, result, ai_angle_delta, stats)
        final_points = self._renderable_path(
            reconstructed=reconstructed,
            raw_candidates=strong,
            metadata=metadata,
            fps=fps,
            release=release,
            bounce=bounce,
            impact=impact,
            impact_source=str(stats["impactSource"]),
        )
        segments = {
            "preBounce": "raw+smoothed",
            "missingBounceWindow": "physics_predicted+path_gated_rescue",
            "postBounce": "bounce_aware_prediction+path_gated_rescue",
        }
        event_sources = {
            "release": release_src,
            "bounce": bounce_src,
            "impact": impact_src,
        }
        return ReconstructedTrajectory(
            points=reconstructed,
            final_points=final_points,
            release_sec=release,
            bounce_sec=bounce,
            impact_sec=impact,
            segments=segments,
            event_sources=event_sources,
            stats=stats,
        )

    def _trim_in_hand_candidates(self, candidates: list[Candidate], metadata: DeliveryMetadata, fps: float) -> list[Candidate]:
        if len(candidates) < 5:
            return candidates
        ordered = sorted(candidates, key=lambda candidate: self._candidate_time(candidate, metadata, fps))
        if len(ordered) >= 3:
            c0 = float(ordered[0].confidence or 0.0)
            c1 = float(ordered[1].confidence or 0.0)
            c2 = float(ordered[2].confidence or 0.0)
            if c0 < 0.65 * max(1e-6, (c1 + c2) * 0.5):
                ordered = ordered[1:]
        release_index = 0
        for idx in range(min(len(ordered) - 2, 8)):
            current = ordered[idx]
            nxt = ordered[idx + 1]
            nxt2 = ordered[idx + 2]
            dt1 = max(1e-6, self._candidate_time(nxt, metadata, fps) - self._candidate_time(current, metadata, fps))
            dt2 = max(1e-6, self._candidate_time(nxt2, metadata, fps) - self._candidate_time(nxt, metadata, fps))
            speed1 = float(np.hypot(nxt.x - current.x, nxt.y - current.y) / dt1)
            speed2 = float(np.hypot(nxt2.x - nxt.x, nxt2.y - nxt.y) / dt2)
            moving_forward = (nxt.x - current.x) > 0.006 and (nxt.y - current.y) > 0.002
            confident = float(current.confidence or 0.0) >= 0.32
            if moving_forward and confident and speed1 > 0.85 and speed2 > 0.85:
                release_index = idx
                break
            release_index = max(release_index, idx + 1)
        return ordered[release_index:]

    def _earliest_pre_tracklet_release_candidate(
        self,
        strong_candidates: list[Candidate],
        merged_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
    ) -> Candidate | None:
        if not strong_candidates:
            return None
        tracklet_start = self._candidate_time(
            min(strong_candidates, key=lambda c: self._candidate_time(c, metadata, fps)),
            metadata,
            fps,
        )
        best: Candidate | None = None
        best_t: float | None = None
        for candidate in merged_candidates:
            if candidate.modelRole != "ball_detector":
                continue
            conf = float(candidate.confidence or 0.0)
            if conf < 0.048:
                continue
            time_sec = self._candidate_time(candidate, metadata, fps)
            if time_sec >= tracklet_start - 0.03:
                continue
            if time_sec < tracklet_start - 0.62:
                continue
            if candidate.y > 0.52:
                continue
            if best_t is None or time_sec < best_t:
                best = candidate
                best_t = time_sec
        return best

    def _recover_release_candidate(
        self,
        release_sec: float | None,
        strong_candidates: list[Candidate],
        merged_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
    ) -> Candidate | None:
        if release_sec is None or len(strong_candidates) < 2:
            return None
        first = min(strong_candidates, key=lambda candidate: self._candidate_time(candidate, metadata, fps))
        second = sorted(strong_candidates, key=lambda candidate: self._candidate_time(candidate, metadata, fps))[1]
        direction = np.array([second.x - first.x, second.y - first.y], dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            return None
        direction = direction / direction_norm
        candidates: list[tuple[float, Candidate]] = []
        for candidate in merged_candidates:
            if candidate.modelRole != "ball_detector" or candidate.confidence < self.WEAK_CONF_MIN:
                continue
            if candidate.source == "zoom_crop_tracker" and float(candidate.confidence or 0.0) < 0.14:
                continue
            time_sec = self._candidate_time(candidate, metadata, fps)
            if time_sec > release_sec or time_sec < release_sec - 0.42:
                continue
            # Earliest release should be near/above the first accepted ball and continue along the same path.
            if candidate.y > first.y + 0.058:
                continue
            offset = np.array([first.x - candidate.x, first.y - candidate.y], dtype=float)
            distance = float(np.linalg.norm(offset))
            if distance > 0.17:
                continue
            alignment = float(np.dot(offset / max(1e-6, distance), direction)) if distance > 1e-6 else 1.0
            if alignment < 0.55:
                continue
            score = (release_sec - time_sec) * 10.0 + max(0.0, alignment) + min(1.0, candidate.confidence / self.STRONG_CONF_MIN)
            candidates.append((score, candidate))
        if not candidates:
            return None
        # Pick the highest/earliest valid candidate, using score only as a tie-breaker.
        return min((candidate for _score, candidate in candidates), key=lambda candidate: (candidate.frameIndex, candidate.y))

    def _dense_path_piecewise(
        self,
        raw: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        release: float | None,
        bounce: float | None,
        impact: float | None,
        result: DeliveryResult,
        ai_angle_delta: float,
        stats: dict[str, float | int | str],
    ) -> list[Point2D]:
        if release is None or bounce is None or impact is None or impact <= release or bounce <= release or impact <= bounce:
            return []
        points_by_time = sorted(
            [(self._candidate_time(candidate, metadata, fps), candidate.x, candidate.y) for candidate in raw],
            key=lambda item: item[0],
        )
        inv_fps = 1.0 / max(1.0, fps)
        pre = [point for point in points_by_time if point[0] <= bounce + inv_fps]
        post_pool = [point for point in points_by_time if point[0] >= bounce - inv_fps]
        pre_fit = [point for point in points_by_time if point[0] <= bounce]
        if len(pre_fit) < 2:
            pre_fit = pre
        pre_x_model, pre_y_model = self._fit_models(pre_fit, degree=2)
        bounce_anchor = self._bounce_anchor(points_by_time, bounce, fps, pre_x_model, pre_y_model)
        eps = 1.0 / max(30.0, fps)
        bounce_x = bounce_anchor[1]
        bounce_y = bounce_anchor[2]
        vx_in = (pre_x_model(bounce) - pre_x_model(max(release, bounce - eps * 3))) / max(eps * 3, 1e-6)
        vy_in = (pre_y_model(bounce) - pre_y_model(max(release, bounce - eps * 3))) / max(eps * 3, 1e-6)
        if len(pre_fit) >= 2:
            net_vx = (bounce_x - pre_fit[0][1]) / max(1e-6, bounce - pre_fit[0][0])
            if abs(net_vx) > 0.02 and (abs(vx_in) < 0.02 or np.sign(net_vx) != np.sign(vx_in)):
                vx_in = net_vx
        post = self._directionally_valid_post_points(post_pool, bounce, bounce_x, vx_in)
        post_fit_margin = max(0.85 * inv_fps, 0.02)
        post_for_poly = [point for point in post if point[0] >= bounce + post_fit_margin]
        post_x_model, post_y_model = self._fit_models(
            post_for_poly
            + [(impact, self._directional_impact_x(result.endPoint.x, bounce_x, vx_in), result.endPoint.y)],
            degree=2,
        )
        strict_post = [point for point in post if point[0] > bounce + inv_fps]
        post_obs_unreliable = len(strict_post) < 2 or self._post_bounce_fit_unreliable(
            strict_post, bounce, bounce_y, fps
        )
        use_reflected_projection = post_obs_unreliable
        angle_adj = float(ai_angle_delta)
        if stats.get("bounceNetPostDirectionDx") is not None and not post_obs_unreliable:
            angle_adj += float(stats["bounceNetPostDirectionDx"]) * 0.42
        duration = max(eps, impact - bounce)
        bounce_physics = apply_bounce_physics(vx_before=vx_in, vy_before=vy_in, restitution=0.56)
        vy_physics = bounce_physics.vy_after
        vx_physics = bounce_physics.vx_after
        chosen_hypothesis = self._choose_post_bounce_hypothesis(
            post_points=post,
            bounce=bounce,
            bounce_x=bounce_x,
            bounce_y=bounce_y,
            impact=impact,
            vx_after=vx_physics,
            vy_after=vy_physics,
            ai_angle_delta=angle_adj,
            result=result,
            trust_post_lateral_observations=not post_obs_unreliable,
        )
        stats["postBounceHypothesisChosen"] = chosen_hypothesis
        stats["postBounceUseReflected"] = 1 if use_reflected_projection else 0
        stats["postBounceStrictObservations"] = len(strict_post)
        stats["postBounceObservationsUnreliable"] = 1 if post_obs_unreliable else 0

        k_grav = 0.30
        direction_sign = {"slight_left": -1.0, "straight": 0.0, "slight_right": 1.0}.get(chosen_hypothesis, 0.0)
        vx_ai = vx_physics + direction_sign * min(0.12, abs(angle_adj) * 0.30)
        vy_ai = (0.7 * vy_physics) + (0.3 * (-abs(vy_in) * (0.52 + 0.28 * min(1.0, abs(angle_adj)))))
        dt_apex, y_apex_screen = self._screen_space_bounce_apex(bounce_y, vy_ai, k_grav, duration)
        if dt_apex is not None:
            stats["postBounceScreenApexSec"] = float(bounce + dt_apex)
            stats["postBounceScreenApexY"] = float(y_apex_screen)
        t_apex_poly: float | None = None
        y_apex_poly: float | None = None
        if not use_reflected_projection:
            nscan = max(24, int(min(96, 12 + 40 * duration * fps / 30.0)))
            scan = np.linspace(float(bounce), float(impact), nscan)
            y_vals = np.array([float(post_y_model(float(t))) for t in scan], dtype=float)
            ji = int(np.argmin(y_vals))
            y_apex_poly = float(y_vals[ji])
            t_apex_poly = float(scan[ji])
            stats["postBounceScreenApexSec"] = float(t_apex_poly)
            stats["postBounceScreenApexY"] = float(y_apex_poly)

        step = 1.0 / max(1.0, fps)
        output: list[Point2D] = []
        raw_times = [time_sec for time_sec, _, _ in points_by_time]
        impact_x_align = self._directional_impact_x(result.endPoint.x, bounce_x, vx_ai)
        for time_sec in np.arange(release, impact + step * 0.5, step):
            if time_sec <= bounce:
                x = pre_x_model(time_sec)
                y = pre_y_model(time_sec)
            else:
                if use_reflected_projection:
                    dt = float(time_sec - bounce)
                    alpha = min(1.0, dt / duration)
                    x_reflect = bounce_x + (vx_ai * dt)
                    y_reflect = bounce_y + (vy_ai * dt) + (k_grav * dt * dt)
                    should_converge = (not post_obs_unreliable) and self._should_converge_to_stumps(
                        post, bounce, result
                    )
                    if should_converge:
                        x = (1.0 - alpha) * x_reflect + alpha * impact_x_align
                        y = (1.0 - (alpha ** 1.8)) * y_reflect + (alpha ** 1.8) * result.endPoint.y
                    else:
                        x, y = x_reflect, y_reflect
                    if dt_apex is not None and dt + 1e-9 >= dt_apex:
                        x_at_apex = bounce_x + vx_ai * dt_apex
                        rem = max(1e-6, duration - dt_apex)
                        beta = min(1.0, max(0.0, (dt - dt_apex) / rem))
                        x = (1.0 - beta) * x_at_apex + beta * float(impact_x_align)
                        y = float(y_apex_screen)
                else:
                    x = post_x_model(time_sec)
                    y = post_y_model(time_sec)
                    y = self._enforce_rising_after_bounce(
                        y_value=y,
                        time_sec=float(time_sec),
                        bounce_sec=float(bounce),
                        post_model=post_y_model,
                        impact_sec=float(impact),
                        endpoint_y=float(result.endPoint.y),
                        bounce_y=float(bounce_y),
                        vy_after_phys=float(vy_physics),
                    )
                    if t_apex_poly is not None and float(time_sec) + 1e-9 >= t_apex_poly:
                        y = float(y_apex_poly)
            # Snap to raw detections only **before** bounce. After bounce the fitted/reflected
            # curve is authoritative — re-snapping replaces physics with misleading same-height pairs.
            valid_raw_times = [point[0] for point in pre + post]
            if valid_raw_times and float(time_sec) <= bounce:
                nearest = min(valid_raw_times, key=lambda value: abs(value - time_sec))
                if abs(nearest - time_sec) <= (0.6 * step):
                    source_points = pre + post
                    _, raw_x, raw_y = source_points[valid_raw_times.index(nearest)]
                    x = raw_x
                    y = raw_y
            if abs(float(time_sec) - float(bounce)) <= (0.55 * step):
                x = bounce_x
                y = bounce_y
            output.append(Point2D(x=float(np.clip(x, 0.0, 1.0)), y=float(np.clip(y, 0.0, 1.0)), t=float(time_sec)))
        return output

    def _dense_path_no_bounce(
        self,
        raw: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        release: float | None,
        impact: float | None,
        result: DeliveryResult,
    ) -> list[Point2D]:
        if release is None or impact is None or impact <= release:
            return []
        points_by_time = sorted(
            [(self._candidate_time(candidate, metadata, fps), candidate.x, candidate.y) for candidate in raw],
            key=lambda item: item[0],
        )
        x_model, y_model = self._fit_models(points_by_time + [(impact, result.endPoint.x, result.endPoint.y)], degree=2)
        step = 1.0 / max(1.0, fps)
        output: list[Point2D] = []
        raw_times = [point[0] for point in points_by_time]
        for time_sec in np.arange(release, impact + step * 0.5, step):
            x = x_model(time_sec)
            y = y_model(time_sec)
            if raw_times:
                nearest = min(raw_times, key=lambda value: abs(value - time_sec))
                if abs(nearest - time_sec) <= (0.6 * step):
                    _, raw_x, raw_y = points_by_time[raw_times.index(nearest)]
                    x = raw_x
                    y = raw_y
            output.append(Point2D(x=float(np.clip(x, 0.0, 1.0)), y=float(np.clip(y, 0.0, 1.0)), t=float(time_sec)))
        return output

    def _looks_like_no_bounce_delivery(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        release: float | None,
        impact: float | None,
    ) -> bool:
        if release is None or impact is None:
            return False
        ordered = [
            candidate
            for candidate in sorted(candidates, key=lambda value: self._candidate_time(value, metadata, fps))
            if release <= self._candidate_time(candidate, metadata, fps) <= impact
        ]
        if len(ordered) < 5:
            return False
        y_values = [candidate.y for candidate in ordered]
        deepest_index = int(np.argmax(y_values))
        deepest_y = y_values[deepest_index]
        if deepest_y < 0.62:
            return True
        if deepest_index >= len(ordered) - 2:
            return True
        post = y_values[deepest_index:]
        sustained_rise = sum(1 for idx in range(1, len(post)) if post[idx] < post[idx - 1] - 0.002) >= 2
        return not sustained_rise

    def _should_converge_to_stumps(self, post: list[tuple[float, float, float]], bounce: float, result: DeliveryResult) -> bool:
        if not post or not result.virtualWicketPlane:
            return False
        observed = [point for point in post if point[0] >= bounce + 0.01]
        if len(observed) < 2:
            return False
        observed = sorted(observed, key=lambda item: item[0])[:4]
        x_center = float(result.virtualWicketPlane.get("xCenter", result.endPoint.x))
        start_dist = abs(observed[0][1] - x_center)
        end_dist = abs(observed[-1][1] - x_center)
        # Converge only if real post-bounce evidence moves toward wicket center.
        return end_dist < start_dist - 0.01

    def _directional_impact_x(self, impact_x: float, bounce_x: float, vx_in: float) -> float:
        if abs(vx_in) < 0.02:
            return float(impact_x)
        if vx_in > 0 and impact_x < bounce_x - 0.05:
            return float(bounce_x + min(0.08, abs(vx_in) * 0.12))
        if vx_in < 0 and impact_x > bounce_x + 0.05:
            return float(bounce_x - min(0.08, abs(vx_in) * 0.12))
        return float(impact_x)

    def _choose_post_bounce_hypothesis(
        self,
        post_points: list[tuple[float, float, float]],
        bounce: float,
        bounce_x: float,
        bounce_y: float,
        impact: float,
        vx_after: float,
        vy_after: float,
        ai_angle_delta: float,
        result: DeliveryResult,
        trust_post_lateral_observations: bool = True,
    ) -> str:
        if not trust_post_lateral_observations:
            return "straight"
        options = {
            "straight": 0.0,
            "slight_left": -0.08,
            "slight_right": 0.08,
        }
        best_name = "straight"
        best_score = -1e9
        for name, delta in options.items():
            score = 0.0
            vx_h = vx_after + delta + (ai_angle_delta * 0.06)
            vy_h = vy_after
            for time_sec, obs_x, obs_y in post_points:
                if time_sec <= bounce:
                    continue
                dt = max(0.0, time_sec - bounce)
                pred_x = bounce_x + vx_h * dt
                pred_y = bounce_y + vy_h * dt + (0.30 * dt * dt)
                dist = float(np.hypot(obs_x - pred_x, obs_y - pred_y))
                score += max(0.0, 1.0 - dist / 0.12)
            # smoothness penalty for strong side movement
            score -= abs(delta) * 1.1
            if result.virtualWicketPlane:
                x_center = float(result.virtualWicketPlane.get("xCenter", result.endPoint.x))
                dt_end = max(0.0, impact - bounce)
                x_end = bounce_x + vx_h * dt_end
                score += max(0.0, 0.35 - abs(x_end - x_center)) * 0.25
            if score > best_score:
                best_score = score
                best_name = name
        return best_name

    def _screen_space_bounce_apex(
        self,
        bounce_y: float,
        vy_ai: float,
        k_gravity_screen: float,
        duration: float,
    ) -> tuple[float | None, float | None]:
        """Peak time dt after bounce and screen-space y at apex (minimum y = highest on image)."""
        if vy_ai >= 0 or k_gravity_screen <= 1e-9:
            return None, None
        dt_peak = -vy_ai / (2.0 * k_gravity_screen)
        if dt_peak <= 0 or dt_peak > duration + 1e-6:
            return None, None
        y_peak = bounce_y + vy_ai * dt_peak + k_gravity_screen * dt_peak * dt_peak
        return dt_peak, y_peak

    def _post_bounce_fit_unreliable(
        self,
        strict_post: list[tuple[float, float, float]],
        bounce: float,
        bounce_y: float,
        fps: float,
    ) -> bool:
        """True when post-bounce detections imply a sideways or downward tangent (same-height pair, etc.)."""
        if len(strict_post) < 2:
            return True
        ordered = sorted(strict_post, key=lambda item: item[0])
        t0, _x0, y0 = ordered[0]
        t1, _x1, y1 = ordered[1]
        dt = max(1e-6, t1 - t0)
        dy_dt = (y1 - y0) / dt
        if dy_dt > 0.012:
            return True
        ys = [point[2] for point in ordered]
        xs = [point[1] for point in ordered]
        y_span = float(max(ys) - min(ys))
        x_span = float(max(xs) - min(xs))
        if y_span < 0.011 and x_span > 0.024 and (t1 - t0) < 0.22:
            return True
        if len(ordered) >= 2:
            early_y = min(point[2] for point in ordered[:2])
            if early_y > bounce_y + 0.008:
                return True
        return False

    def _directionally_valid_post_points(
        self,
        post_points: list[tuple[float, float, float]],
        bounce: float,
        bounce_x: float,
        vx_in: float,
    ) -> list[tuple[float, float, float]]:
        if abs(vx_in) < 0.02:
            return post_points
        valid: list[tuple[float, float, float]] = []
        rejected: list[tuple[float, float, float]] = []
        direction = 1.0 if vx_in > 0 else -1.0
        for point in sorted(post_points, key=lambda item: item[0]):
            time_sec, x, _y = point
            if time_sec <= bounce + 0.5 / 60.0:
                valid.append(point)
                continue
            projected_x = bounce_x + (vx_in * 0.60 * max(0.0, time_sec - bounce))
            lane_tolerance = 0.065 + min(0.055, max(0.0, time_sec - bounce) * 0.10)
            reversed_laterally = direction * (x - bounce_x) < -0.035
            too_far_from_lane = abs(x - projected_x) > lane_tolerance
            if reversed_laterally or too_far_from_lane:
                rejected.append(point)
                continue
            valid.append(point)
        return valid

    def _filter_lateral_reversal_candidates(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        bounce: float | None,
        stats: dict[str, float | int | str],
    ) -> list[Candidate]:
        if bounce is None or len(candidates) < 4:
            return candidates
        ordered = sorted(candidates, key=lambda candidate: self._candidate_time(candidate, metadata, fps))
        pre = [candidate for candidate in ordered if self._candidate_time(candidate, metadata, fps) <= bounce]
        if len(pre) < 2:
            return candidates
        first_pre = pre[0]
        a, b = pre[-2], pre[-1]
        release_time = self._candidate_time(first_pre, metadata, fps)
        ta = self._candidate_time(a, metadata, fps)
        tb = self._candidate_time(b, metadata, fps)
        vx = (b.x - a.x) / max(1e-6, tb - ta)
        net_vx = (b.x - first_pre.x) / max(1e-6, tb - release_time)
        if abs(net_vx) > 0.02 and (abs(vx) < 0.02 or np.sign(net_vx) != np.sign(vx)):
            vx = net_vx
        if abs(vx) < 0.02:
            return candidates
        direction = 1.0 if vx > 0 else -1.0
        bounce_x = b.x
        filtered: list[Candidate] = []
        rejected = 0
        post_offlane: list[Candidate] = []
        for candidate in ordered:
            time_sec = self._candidate_time(candidate, metadata, fps)
            if time_sec <= bounce + 0.5 / max(1.0, fps):
                filtered.append(candidate)
                continue
            projected_x = bounce_x + (vx * 0.60 * max(0.0, time_sec - bounce))
            lane_tolerance = 0.065 + min(0.055, max(0.0, time_sec - bounce) * 0.10)
            reversed_laterally = direction * (candidate.x - bounce_x) < -0.035
            too_far_from_lane = abs(candidate.x - projected_x) > lane_tolerance
            if reversed_laterally or too_far_from_lane:
                post_offlane.append(candidate)
                rejected += 1
                continue
            filtered.append(candidate)
        stats["lateralReversalCandidatesRejected"] = rejected
        if rejected > 0:
            post_frames = [
                str(candidate.frameIndex)
                for candidate in filtered
                if self._candidate_time(candidate, metadata, fps) > bounce + (0.5 / max(1.0, fps))
            ]
            stats["postBounceCandidatesUsed"] = len(post_frames)
            stats["postBounceFramesUsed"] = ",".join(post_frames)
        return sorted(filtered, key=lambda candidate: self._candidate_time(candidate, metadata, fps))

    def _bounce_anchor(
        self,
        points_by_time: list[tuple[float, float, float]],
        bounce: float,
        fps: float,
        pre_x_model,
        pre_y_model,
    ) -> tuple[float, float, float]:
        if not points_by_time:
            return (bounce, 0.5, 0.5)
        window = 1.2 / max(1.0, fps)
        nearby = [point for point in points_by_time if abs(point[0] - bounce) <= window]
        if nearby:
            pred_x = float(pre_x_model(bounce))
            pred_y = float(pre_y_model(bounce))
            gated = [
                point
                for point in nearby
                if abs(point[1] - pred_x) <= 0.10 and abs(point[2] - pred_y) <= 0.14
            ]
            if gated:
                return max(gated, key=lambda point: point[2])
            return min(nearby, key=lambda point: np.hypot(point[1] - pred_x, point[2] - pred_y))
        return min(points_by_time, key=lambda point: abs(point[0] - bounce))

    def _promote_continuous_path_candidates(
        self,
        release_sec: float | None,
        bounce_sec: float | None,
        current_path: list[Candidate],
        merged_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        stats: dict[str, float | int | str],
    ) -> list[Candidate]:
        if release_sec is None or bounce_sec is None or not current_path:
            return current_path
        path_model = self._fit_path_from_candidates(current_path, metadata, fps)
        if path_model is None:
            return current_path
        selected_frames = {candidate.frameIndex for candidate in current_path}
        last_selected_pre_bounce = max(
            (candidate.frameIndex for candidate in current_path if self._candidate_time(candidate, metadata, fps) <= bounce_sec),
            default=None,
        )
        pool: list[Candidate] = []
        for candidate in merged_candidates:
            if candidate.modelRole != "ball_detector" or candidate.confidence < self.WEAK_CONF_MIN:
                continue
            time_sec = self._candidate_time(candidate, metadata, fps)
            if time_sec < release_sec or time_sec > bounce_sec + (1.0 / max(1.0, fps)):
                continue
            if candidate.frameIndex in selected_frames:
                continue
            score = self._path_gated_score(candidate, time_sec, path_model, metadata, current_path, fps, bounce_focus=True)
            if score >= 0.48:
                pool.append(candidate)
        stats["rawCandidatesBetweenLastYellowAndBounce"] = len(
            [
                candidate for candidate in merged_candidates
                if candidate.modelRole == "ball_detector"
                and candidate.confidence >= self.WEAK_CONF_MIN
                and last_selected_pre_bounce is not None
                and candidate.frameIndex > last_selected_pre_bounce
                and self._candidate_time(candidate, metadata, fps) <= bounce_sec + (1.0 / max(1.0, fps))
            ]
        )
        chain = self._best_temporal_chain(pool, metadata, fps, path_model, current_path, bounce_focus=True)
        promoted = [candidate for candidate in chain if self._candidate_time(candidate, metadata, fps) <= bounce_sec + (1.0 / max(1.0, fps))]
        merged = current_path
        zoom_frames: list[str] = []
        for candidate in promoted:
            merged = self._merge_rescued_candidate(merged, candidate, metadata, fps)
            if candidate.source == "zoom_crop_tracker":
                zoom_frames.append(str(candidate.frameIndex))
        stats["promotedCandidatesBetweenLastYellowAndBounce"] = len(promoted)
        stats["rejectedCandidatesBetweenLastYellowAndBounce"] = max(0, int(stats["rawCandidatesBetweenLastYellowAndBounce"]) - len(promoted))
        selected_frame_count = len({candidate.frameIndex for candidate in merged if release_sec <= self._candidate_time(candidate, metadata, fps) <= bounce_sec})
        expected_frame_count = max(1, int(round((bounce_sec - release_sec) * fps)) + 1)
        stats["selectedPathFrameCoverage"] = selected_frame_count
        stats["framesMissingFromSelectedPath"] = max(0, expected_frame_count - selected_frame_count)
        stats["zoomCandidatesAcceptedByFrame"] = ",".join(zoom_frames)
        pre_bounce = [candidate for candidate in merged if self._candidate_time(candidate, metadata, fps) <= bounce_sec]
        if pre_bounce:
            final_pre = max(pre_bounce, key=lambda candidate: self._candidate_time(candidate, metadata, fps))
            stats["finalPreBounceEndFrame"] = final_pre.frameIndex
            stats["postBounceAnchorFrame"] = final_pre.frameIndex
        return merged

    def _renderable_path(
        self,
        reconstructed: list[Point2D],
        raw_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        release: float | None,
        bounce: float | None,
        impact: float | None,
        impact_source: str,
    ) -> list[dict[str, float | int | str]]:
        raw_by_time = {
            self._candidate_time(candidate, metadata, fps): candidate
            for candidate in raw_candidates
        }
        raw_times = sorted(raw_by_time)
        last_raw_time = max(raw_times, default=None)
        step = 1.0 / max(1.0, fps)
        output: list[dict[str, float | int | str]] = []
        for point in reconstructed:
            if point.t is None:
                continue
            if bounce is not None and point.t < bounce - (1.0 / max(1.0, fps)):
                segment = "observed_pre_bounce"
                style = "solid_observed"
            elif bounce is not None and abs(point.t - bounce) <= (2.0 / max(1.0, fps)):
                segment = "reconstructed_bounce_window"
                style = "solid_reconstructed"
            elif bounce is not None and last_raw_time is not None and point.t <= last_raw_time + step:
                segment = "observed_post_bounce"
                style = "solid_observed"
            elif impact is not None and point.t >= impact - (2.0 / max(1.0, fps)):
                segment = "projected_to_virtual_stumps"
                style = "dashed_projected"
            else:
                segment = "projected_post_bounce"
                style = "dashed_projected"
            nearest_raw = min(raw_times, key=lambda value: abs(value - float(point.t))) if raw_times else None
            has_raw = nearest_raw is not None and abs(nearest_raw - float(point.t)) <= (0.55 / max(1.0, fps))
            raw_candidate = raw_by_time.get(nearest_raw) if nearest_raw is not None else None
            frame_index = raw_candidate.frameIndex if has_raw and raw_candidate is not None else int(round(float(point.t) * fps))
            source = "raw_detector" if has_raw and segment.startswith("observed") else ("weak_detector" if has_raw else "physics_projected")
            output.append(
                {
                    "frame": int(frame_index),
                    "x": float(point.x),
                    "y": float(point.y),
                    "t": float(point.t),
                    "confidence": 0.85 if has_raw else (0.62 if "reconstructed" in segment else 0.42),
                    "source": source,
                    "segmentType": segment,
                    "renderStyle": style,
                }
            )
        return output

    def _fit_models(self, points: list[tuple[float, float, float]], degree: int) -> tuple:
        if not points:
            return (lambda _t: 0.0, lambda _t: 0.0)
        dedup: dict[float, list[tuple[float, float]]] = {}
        for time_sec, x, y in points:
            dedup.setdefault(float(time_sec), []).append((float(x), float(y)))
        ordered = sorted(dedup.items(), key=lambda item: item[0])
        times = np.array([item[0] for item in ordered], dtype=float)
        xs = np.array([float(np.mean([value[0] for value in item[1]])) for item in ordered], dtype=float)
        ys = np.array([float(np.mean([value[1] for value in item[1]])) for item in ordered], dtype=float)
        if len(times) == 1:
            x_value = float(xs[0])
            y_value = float(ys[0])
            return (lambda _t: x_value, lambda _t: y_value)
        fit_degree = min(degree, max(1, len(times) - 1))
        x_coeff = np.polyfit(times, xs, deg=fit_degree)
        y_coeff = np.polyfit(times, ys, deg=fit_degree)
        return (
            lambda t: float(np.polyval(x_coeff, t)),
            lambda t: float(np.polyval(y_coeff, t)),
        )

    def _enforce_rising_after_bounce(
        self,
        y_value: float,
        time_sec: float,
        bounce_sec: float,
        post_model,
        impact_sec: float,
        endpoint_y: float,
        bounce_y: float | None = None,
        vy_after_phys: float | None = None,
    ) -> float:
        bounce_horizon = min(impact_sec, bounce_sec + 0.16)
        if time_sec <= bounce_horizon:
            first_step = min(impact_sec, bounce_sec + 0.03)
            rising_reference = post_model(first_step)
            y_value = min(y_value, rising_reference)
            if bounce_y is not None and vy_after_phys is not None:
                dt = max(0.0, float(time_sec) - bounce_sec)
                y_phys_cap = float(bounce_y) + float(vy_after_phys) * dt + (0.30 * dt * dt)
                y_value = min(y_value, y_phys_cap)
        if time_sec >= impact_sec - 0.03:
            y_value = (y_value * 0.6) + (endpoint_y * 0.4)
        return y_value

    def _recover_bounce_candidate(
        self,
        estimate_bounce_sec: float | None,
        release_sec: float | None,
        strong_candidates: list[Candidate],
        merged_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        stats: dict[str, float | int | str],
    ) -> Candidate | None:
        if estimate_bounce_sec is None:
            return None
        path_model = self._fit_path_from_candidates(strong_candidates, metadata, fps)
        if path_model is None:
            return None
        best_candidate: Candidate | None = None
        best_score = 0.0
        bounce_pool: list[Candidate] = []
        for candidate in merged_candidates:
            if candidate.modelRole != "ball_detector":
                continue
            if candidate.source == "zoom_crop_tracker":
                stats["zoomCandidatesPerFrame"] = int(stats["zoomCandidatesPerFrame"]) + 1
            stats["totalRawBallCandidates"] = int(stats["totalRawBallCandidates"]) + 1
            confidence = float(candidate.confidence or 0.0)
            if confidence < self.WEAK_CONF_MIN:
                continue
            if confidence < self.STRONG_CONF_MIN:
                stats["weakCandidatesSeen"] = int(stats["weakCandidatesSeen"]) + 1
            time_sec = self._candidate_time(candidate, metadata, fps)
            if release_sec is not None and time_sec < release_sec:
                continue
            # Bounce can be later than the timing prior; don't reject a continuing downward chain.
            if time_sec < estimate_bounce_sec or time_sec > estimate_bounce_sec + 0.35:
                continue
            bounce_pool.append(candidate)
        stats["blueCandidatesNearBounce"] = len(bounce_pool)
        stats["bounceCandidatesConsidered"] = len(bounce_pool)
        chain = self._best_temporal_chain(
            candidates=bounce_pool,
            metadata=metadata,
            fps=fps,
            path_model=path_model,
            history=strong_candidates,
            bounce_focus=True,
        )
        turn_candidate = self._turning_point_candidate(
            candidates=bounce_pool,
            metadata=metadata,
            fps=fps,
            path_model=path_model,
            estimate_bounce_sec=float(estimate_bounce_sec),
        )
        if turn_candidate is not None:
            return turn_candidate
        for candidate in chain:
            time_sec = self._candidate_time(candidate, metadata, fps)
            score = self._path_gated_score(candidate, time_sec, path_model, metadata, strong_candidates, fps, bounce_focus=True)
            # Prefer deeper points near bounce. A ball still descending means bounce has not occurred yet.
            score += min(0.30, candidate.y * 0.35)
            if score > 0.50 and score > best_score:
                best_score = score
                best_candidate = candidate
        if best_candidate is None:
            stats["rejectedCandidatesNearBounce"] = int(stats["blueCandidatesNearBounce"])
            return None
        rescued_time = self._candidate_time(best_candidate, metadata, fps)
        if rescued_time < estimate_bounce_sec - 0.06:
            stats["rejectedCandidatesNearBounce"] = int(stats["blueCandidatesNearBounce"])
            return None
        stats["rejectedCandidatesNearBounce"] = max(0, int(stats["blueCandidatesNearBounce"]) - 1)
        return best_candidate

    def _turning_point_candidate(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        path_model,
        estimate_bounce_sec: float,
    ) -> Candidate | None:
        if not candidates:
            return None
        detector_candidates = [candidate for candidate in candidates if candidate.source != "zoom_crop_tracker"]
        if detector_candidates:
            candidates = detector_candidates
        by_frame: dict[int, Candidate] = {}
        for candidate in candidates:
            existing = by_frame.get(candidate.frameIndex)
            if existing is None or candidate.y > existing.y:
                by_frame[candidate.frameIndex] = candidate
        ordered = sorted(by_frame.values(), key=lambda candidate: self._candidate_time(candidate, metadata, fps))
        if not ordered:
            return None
        scored: list[tuple[float, Candidate]] = []
        for candidate in ordered:
            time_sec = self._candidate_time(candidate, metadata, fps)
            pred_x, pred_y = path_model(time_sec)
            path_dist = float(np.hypot(candidate.x - pred_x, candidate.y - pred_y))
            if path_dist > 0.14:
                continue
            time_penalty = abs(time_sec - estimate_bounce_sec)
            score = (candidate.y * 2.0) + float(candidate.confidence or 0.0) - (path_dist * 4.0) - (time_penalty * 2.0)
            scored.append((score, candidate))
        if not scored:
            return None
        path_candidates = [candidate for _score, candidate in sorted(scored, key=lambda item: self._candidate_time(item[1], metadata, fps))]
        # Prefer the true turning point: last descending point before sustained upward motion.
        if len(path_candidates) >= 4:
            y_values = [candidate.y for candidate in path_candidates]
            for idx in range(1, len(y_values) - 2):
                dy1 = y_values[idx] - y_values[idx - 1]
                dy2 = y_values[idx + 1] - y_values[idx]
                dy3 = y_values[idx + 2] - y_values[idx + 1]
                if dy1 >= 0.0 and dy2 <= -0.001 and dy3 <= -0.001:
                    return path_candidates[idx]
        return max(path_candidates, key=lambda candidate: (candidate.y, self._candidate_time(candidate, metadata, fps)))

    def _recover_post_bounce_candidates(
        self,
        bounce_sec: float | None,
        impact_sec: float | None,
        strong_candidates: list[Candidate],
        merged_candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        stump_roi: dict[str, float] | None,
        stats: dict[str, float | int | str],
    ) -> list[Candidate]:
        if bounce_sec is None:
            return []
        path_model = self._fit_path_from_candidates(strong_candidates, metadata, fps)
        if path_model is None:
            return []
        accepted: list[Candidate] = []
        pool: list[Candidate] = []
        for candidate in merged_candidates:
            if candidate.modelRole != "ball_detector":
                continue
            confidence = float(candidate.confidence or 0.0)
            if confidence < self.WEAK_CONF_MIN:
                continue
            time_sec = self._candidate_time(candidate, metadata, fps)
            if time_sec <= bounce_sec + (0.25 / max(1.0, fps)) or time_sec > bounce_sec + self.POST_BOUNCE_WINDOW_SEC:
                continue
            if impact_sec is not None and time_sec >= impact_sec:
                continue
            pool.append(candidate)
        stats["postBounceCandidatesSeen"] = len(pool)
        chain = self._best_temporal_chain(
            candidates=pool,
            metadata=metadata,
            fps=fps,
            path_model=path_model,
            history=strong_candidates,
            bounce_focus=False,
        )
        for candidate in chain:
            time_sec = self._candidate_time(candidate, metadata, fps)
            score = self._path_gated_score(candidate, time_sec, path_model, metadata, strong_candidates + accepted, fps, bounce_focus=False)
            corridor_bonus = self._stump_corridor_score(candidate, stump_roi)
            if (score + 0.2 * corridor_bonus) > 0.60:
                accepted.append(candidate)
        stats["postBounceCandidatesUsed"] = len(accepted)
        stats["postBounceFramesUsed"] = ",".join(str(candidate.frameIndex) for candidate in accepted)
        return sorted(accepted, key=lambda value: self._candidate_time(value, metadata, fps))

    def _best_temporal_chain(
        self,
        candidates: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
        path_model,
        history: list[Candidate],
        bounce_focus: bool,
    ) -> list[Candidate]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda candidate: (self._candidate_time(candidate, metadata, fps), candidate.frameIndex))
        times = [self._candidate_time(candidate, metadata, fps) for candidate in ordered]
        dp = [-1e9 for _ in ordered]
        parent = [-1 for _ in ordered]
        for i, candidate in enumerate(ordered):
            t = times[i]
            base = self._path_gated_score(candidate, t, path_model, metadata, history, fps, bounce_focus)
            dp[i] = base
            for j in range(i):
                dt = t - times[j]
                if dt <= 0 or dt > 0.10:
                    continue
                dx = ordered[i].x - ordered[j].x
                dy = ordered[i].y - ordered[j].y
                if np.hypot(dx, dy) > 0.12:
                    continue
                continuity_bonus = max(0.0, 0.15 - np.hypot(dx, dy))
                score = dp[j] + base + continuity_bonus
                if score > dp[i]:
                    dp[i] = score
                    parent[i] = j
        best = int(np.argmax(dp))
        chain_indices: list[int] = []
        while best >= 0:
            chain_indices.append(best)
            best = parent[best]
        chain_indices.reverse()
        return [ordered[index] for index in chain_indices]

    def _path_gated_score(
        self,
        candidate: Candidate,
        time_sec: float,
        path_model,
        metadata: DeliveryMetadata,
        accepted: list[Candidate],
        fps: float,
        bounce_focus: bool,
    ) -> float:
        pred_x, pred_y = path_model(time_sec)
        path_dist = float(np.hypot(candidate.x - pred_x, candidate.y - pred_y))
        path_alignment = max(0.0, 1.0 - (path_dist / 0.09))
        corridor = self._corridor_score(candidate, metadata)
        velocity = self._velocity_continuity_score(candidate, time_sec, accepted, metadata, fps)
        confidence = float(candidate.confidence or 0.0)
        confidence_term = min(1.0, confidence / self.STRONG_CONF_MIN)
        if bounce_focus:
            ground_contact = max(0.0, min(1.0, candidate.y))
            return (confidence_term * 0.16) + (path_alignment * 0.34) + (corridor * 0.16) + (velocity * 0.14) + (ground_contact * 0.20)
        return (confidence_term * 0.20) + (path_alignment * 0.35) + (corridor * 0.20) + (velocity * 0.25)

    def _fit_path_from_candidates(self, candidates: list[Candidate], metadata: DeliveryMetadata, fps: float):
        if len(candidates) < 2:
            return None
        points = sorted(
            [(self._candidate_time(candidate, metadata, fps), candidate.x, candidate.y) for candidate in candidates],
            key=lambda item: item[0],
        )
        times = np.array([point[0] for point in points], dtype=float)
        xs = np.array([point[1] for point in points], dtype=float)
        ys = np.array([point[2] for point in points], dtype=float)
        deg = 2 if len(points) >= 3 else 1
        x_coeff = np.polyfit(times, xs, deg=deg)
        y_coeff = np.polyfit(times, ys, deg=deg)
        return lambda t: (float(np.polyval(x_coeff, t)), float(np.polyval(y_coeff, t)))

    def _corridor_score(self, candidate: Candidate, metadata: DeliveryMetadata) -> float:
        polygon = metadata.corridorGeometry.pitchPolygon
        if len(polygon) < 4:
            return 0.5
        xs = [point.x for point in polygon[:4]]
        ys = [point.y for point in polygon[:4]]
        in_bounds = (min(xs) <= candidate.x <= max(xs)) and (min(ys) <= candidate.y <= max(ys))
        return 1.0 if in_bounds else 0.0

    def _stump_corridor_score(self, candidate: Candidate, stump_roi: dict[str, float] | None) -> float:
        if not stump_roi:
            return 0.5
        dx = abs(candidate.x - float(stump_roi.get("centerX", candidate.x)))
        return max(0.0, 1.0 - (dx / 0.25))

    def _velocity_continuity_score(
        self,
        candidate: Candidate,
        candidate_time: float,
        accepted: list[Candidate],
        metadata: DeliveryMetadata,
        fps: float,
    ) -> float:
        if len(accepted) < 2:
            return 0.6
        history = sorted(accepted, key=lambda value: self._candidate_time(value, metadata, fps))
        a = history[-2]
        b = history[-1]
        ta = self._candidate_time(a, metadata, fps)
        tb = self._candidate_time(b, metadata, fps)
        dt1 = max(1e-6, tb - ta)
        dt2 = max(1e-6, candidate_time - tb)
        v1 = np.array([(b.x - a.x) / dt1, (b.y - a.y) / dt1], dtype=float)
        v2 = np.array([(candidate.x - b.x) / dt2, (candidate.y - b.y) / dt2], dtype=float)
        speed_jump = float(np.linalg.norm(v2 - v1))
        speed_limit = 6.0
        return max(0.0, 1.0 - (speed_jump / speed_limit))

    def _merge_rescued_candidate(
        self,
        candidates: list[Candidate],
        rescued: Candidate,
        metadata: DeliveryMetadata,
        fps: float,
    ) -> list[Candidate]:
        existing_keys = {(candidate.frameIndex, round(candidate.x, 4), round(candidate.y, 4)) for candidate in candidates}
        key = (rescued.frameIndex, round(rescued.x, 4), round(rescued.y, 4))
        if key not in existing_keys:
            candidates = candidates + [rescued]
        return sorted(candidates, key=lambda candidate: self._candidate_time(candidate, metadata, fps))

    def _candidate_time(self, candidate: Candidate, metadata: DeliveryMetadata, fps: float) -> float:
        if candidate.timestampSec is not None:
            return candidate.timestampSec
        if metadata.frameTimestamps and 0 <= candidate.frameIndex < len(metadata.frameTimestamps):
            return metadata.frameTimestamps[candidate.frameIndex]
        return candidate.frameIndex / max(1e-6, fps)

    def _time_from_video_frame(self, frame_idx: int, metadata: DeliveryMetadata, fps: float) -> float:
        if metadata.frameTimestamps and 0 <= frame_idx < len(metadata.frameTimestamps):
            return float(metadata.frameTimestamps[frame_idx])
        return float(frame_idx) / max(1e-6, fps)

    def _detector_bounce_temporally_strong(
        self,
        rescued: Candidate | None,
        merged: list[Candidate],
        bounce_sec: float | None,
        metadata: DeliveryMetadata,
        fps: float,
    ) -> bool:
        if rescued is None or bounce_sec is None:
            return False
        if float(rescued.confidence or 0.0) >= 0.55:
            return True
        if (rescued.source or "") == "cricket_ball_v2" and float(rescued.confidence or 0.0) >= 0.45:
            return True
        t_lo, t_hi = float(bounce_sec) - 0.14, float(bounce_sec) + 0.14
        count = 0
        for candidate in merged:
            if candidate.modelRole != "ball_detector":
                continue
            t = self._candidate_time(candidate, metadata, fps)
            if t_lo <= t <= t_hi and float(candidate.confidence or 0.0) >= 0.06:
                count += 1
        return count >= 14 and float(rescued.confidence or 0.0) >= 0.28
