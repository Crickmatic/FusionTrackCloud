from __future__ import annotations

import json
import html
import csv
import time
import zipfile
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.schemas import CalibrationMode, CameraResolution, DeliveryDebug, DeliveryMetadata, DeliveryResult, Point2D, Point3D, Resolution
from inference.cricket_models import CricketModelCandidateGenerator
from inference.motion_candidates import MotionCandidateGenerator
from inference.tracknet_adapter import TrackNetAdapter
from inference.yolo_ultralytics import YoloUltralyticsCandidateGenerator
from pipeline.metrics import (
    CREASE_TO_CREASE_M,
    STUMP_TO_STUMP_PITCH_M,
    classify_length_from_distance,
    compute_physical_metrics,
    refine_physical_metrics_after_reconstruction,
    swing_amount,
)
from pipeline.trajectory_renderer import TrajectoryRenderer
from pipeline.tracklet_builder import CandidateTrackletBuilder, TrackletBuildResult, candidate_key
from pipeline.zoom_ball_tracker import ZoomBallTracker
from pipeline.trajectory_reconstructor import TrajectoryReconstructor
from pipeline.trajectory_solver import TrajectorySolver
from pipeline.kalman_ball_tracker import KalmanBallTracker
from pipeline.trajectory_ai_refiner import TrajectoryAiRefiner
from pipeline.bounce_net import BounceNetEstimator


@dataclass
class ProcessingOptions:
    use_ground_truth_window: bool = False
    ground_truth: dict | None = None
    enable_motion: bool = False
    max_frames: int | None = None
    tracklet_max_frame_gap: int = 3
    tracklet_max_pixel_displacement: float = 110.0
    allow_motion_only_tracklet_start_debug: bool = False
    render_annotated_video: bool = False
    render_mode: str = "product"


class DeliveryProcessingPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.yolo = YoloUltralyticsCandidateGenerator(settings)
        self.cricket_models = CricketModelCandidateGenerator(settings)
        self.motion = MotionCandidateGenerator()
        self.tracknet = TrackNetAdapter(enabled=settings.enable_tracknet_adapter)
        self.solver = TrajectorySolver()
        self.reconstructor = TrajectoryReconstructor()
        self.zoom_tracker = ZoomBallTracker()
        self.kalman_tracker = KalmanBallTracker()
        self.ai_refiner = TrajectoryAiRefiner()
        self.bounce_net = BounceNetEstimator()
        self.renderer = TrajectoryRenderer()
        self.loaded_models: list[str] = []
        self.missing_model_files: list[str] = []
        self.gpu_available = False
        self.torch_cuda_available = False
        self.gpu_name: str | None = None
        self.ultralytics_version: str | None = None
        self.opencv_version: str | None = cv2.__version__
        self.warmup_status: str = "not_started"
        self.warmup_device: str | None = None
        self.warmup_notes: list[str] = []
        self.logger = logging.getLogger("fusiontrack-cloud.pipeline")

    def load_models(self) -> None:
        self._detect_gpu()
        self.yolo.load()
        self.cricket_models.load()
        self.tracknet.load()
        self.loaded_models = (
            self.yolo.loaded_model_names
            + self.cricket_models.loaded_model_names
            + self.tracknet.loaded_model_names
            + ["opencv_farneback_optical_flow"]
        )
        self.missing_model_files = self.yolo.missing_model_files + self.cricket_models.missing_model_files
        self.missing_model_files.extend(self.cricket_models.load_errors)
        self.ultralytics_version = self.yolo.ultralytics_version

    def warmup_models(self) -> None:
        self.warmup_device = self.settings.device
        if not self.has_required_models:
            self.warmup_status = "models_missing"
            self.warmup_notes = ["Warmup skipped: required models are missing."]
            return
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.yolo.generate([frame], [0])
            self.cricket_models.generate([frame], [0], DeliveryMetadata(sessionId="warmup", deliveryId="warmup"))
            self.warmup_status = "ready"
            self.warmup_notes = ["Warmup inference completed for loaded detectors."]
        except Exception as exc:
            self.warmup_status = "failed"
            self.warmup_notes = [f"Warmup failed: {exc}"]

    def model_registry(self) -> tuple[dict[str, str], dict[str, str]]:
        roles: dict[str, str] = {}
        paths: dict[str, str] = {}
        for name in self.settings.model_names:
            alias = name.removesuffix(".pt")
            if alias.startswith("cricket_ball"):
                roles[alias] = "ball_detector"
            elif alias.startswith("cricket_stumps"):
                roles[alias] = "stump_detector"
            elif "yolo" in alias:
                roles[alias] = "context_detector"
            else:
                roles[alias] = "unknown"
            paths[alias] = str(self.settings._resolve_model_path(self.settings._normalize_model_name(name)))
        return roles, paths

    @property
    def has_required_models(self) -> bool:
        return bool(self.yolo.loaded_model_names or self.cricket_models.loaded_model_names or self.tracknet.loaded_model_names)

    def process(
        self,
        upload_path: Path,
        metadata: DeliveryMetadata,
        artifact_dir: Path | None = None,
        options: ProcessingOptions | None = None,
    ) -> DeliveryResult:
        options = options or ProcessingOptions()
        if not self.has_required_models:
            missing = ", ".join(self.missing_model_files) if self.missing_model_files else "no YOLO models loaded"
            raise RuntimeError(f"FusionTrack Cloud cannot process deliveries: required YOLO model files are missing ({missing})")

        started = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        mark = started
        frames, frame_indices = self._decode_upload(upload_path, metadata)
        now = time.perf_counter()
        stage_timings_ms["decode"] = (now - mark) * 1000
        mark = now
        frames, frame_indices = self._sample_frames(frames, frame_indices, metadata, max_frames=options.max_frames)
        now = time.perf_counter()
        stage_timings_ms["frame_extraction"] = (now - mark) * 1000
        mark = now
        if not frames:
            raise ValueError("No frames decoded from uploaded delivery clip")
        corridor_confidence = self._apply_calibration_priors(metadata)

        yolo_candidates = self.yolo.generate(frames, frame_indices)
        now = time.perf_counter()
        stage_timings_ms["yolo_inference"] = (now - mark) * 1000
        mark = now
        cricket_candidates = self.cricket_models.generate(frames, frame_indices, metadata)
        now = time.perf_counter()
        stage_timings_ms["ball_stump_inference"] = (now - mark) * 1000
        mark = now
        stump_detections = self.cricket_models.last_stump_detections
        if metadata.calibrationMode == CalibrationMode.video_only:
            corridor_confidence = max(corridor_confidence, self._infer_video_only_corridor_from_stumps(metadata, stump_detections))
        motion_candidates = self.motion.generate(frames, frame_indices, metadata) if options.enable_motion else []
        now = time.perf_counter()
        stage_timings_ms["motion_inference"] = (now - mark) * 1000
        mark = now
        tracknet_candidates = self.tracknet.generate(frames, frame_indices)
        now = time.perf_counter()
        stage_timings_ms["tracknet_inference"] = (now - mark) * 1000
        mark = now
        ball_like_candidates = [
            candidate
            for candidate in (yolo_candidates + cricket_candidates + tracknet_candidates)
            if candidate.modelRole == "ball_detector"
        ]
        release_hint = metadata.watchEvents.releaseDetected or metadata.watchEvents.releaseWindowLikely
        zoom_candidates = self.zoom_tracker.generate(
            frames=frames,
            frame_indices=frame_indices,
            metadata=metadata,
            ball_candidates=ball_like_candidates,
            release_hint_sec=release_hint,
            bounce_hint_sec=None,
        )
        now = time.perf_counter()
        stage_timings_ms["zoom_tracking"] = (now - mark) * 1000
        mark = now
        all_candidates = self._filter_contextual_ball_candidates(
            yolo_candidates + cricket_candidates + motion_candidates + tracknet_candidates + zoom_candidates,
            metadata=metadata,
        )
        best_stump = max(stump_detections, key=lambda candidate: candidate.confidence, default=None)

        active_window = self._infer_active_window(
            metadata=metadata,
            candidates=all_candidates,
            options=options,
        )
        if artifact_dir is not None:
            self._write_candidate_timeline_csv(
                artifact_dir=artifact_dir,
                all_candidates=all_candidates,
                active_window=active_window,
                metadata=metadata,
            )
        now = time.perf_counter()
        stage_timings_ms["timeline_artifacts"] = (now - mark) * 1000
        mark = now
        windowed_candidates = self._filter_candidates_to_active_window(all_candidates, metadata, active_window)
        if not windowed_candidates:
            if artifact_dir is not None:
                empty_tracklets = TrackletBuildResult(tracklets=[], best_tracklet=None, selected_candidates=[], rejected_candidate_reasons={})
                self._write_tracklet_artifacts(artifact_dir=artifact_dir, tracklet_result=empty_tracklets, metadata=metadata)
                self._write_debug_artifacts_failed_processing(
                    artifact_dir=artifact_dir,
                    frames=frames,
                    frame_indices=frame_indices,
                    candidates_by_source={
                        "yolo": yolo_candidates,
                        "cricket": cricket_candidates,
                        "stumps": stump_detections,
                        "motion": motion_candidates,
                        "tracknet": tracknet_candidates,
                        "zoom": zoom_candidates,
                        "merged": all_candidates,
                        "active_window": [],
                        "selected_tracklet": [],
                    },
                    rejection_reason="no candidates found inside active window",
                    active_window=active_window,
                    tracklet_result=empty_tracklets,
                    options=options,
                    metadata=metadata,
                )
            raise ValueError("no candidates found inside active window")
        kalman_result = self.kalman_tracker.track(
            candidates=windowed_candidates,
            metadata=metadata,
            fps=float(metadata.fps or metadata.extras.get("videoFps") or 30.0),
            release_hint_sec=release_hint,
        )
        ai_correction = self.ai_refiner.refine(
            tracked_points=kalman_result.tracked_points,
            metadata=metadata,
            estimated_bounce_sec=metadata.watchEvents.releaseDetected,
            fps=float(metadata.fps or metadata.extras.get("videoFps") or 30.0),
        )
        metadata.extras["kalmanTrackedPoints"] = [candidate.model_dump() for candidate in kalman_result.tracked_points]
        metadata.extras["kalmanPredictions"] = [candidate.model_dump() for candidate in kalman_result.predictions]
        metadata.extras["trajectoryAiCorrection"] = {
            "bounceOffsetY": ai_correction.bounce_offset_y,
            "postBounceAngleDelta": ai_correction.post_bounce_angle_delta,
            "confidence": ai_correction.confidence,
        }
        windowed_candidates = windowed_candidates + kalman_result.tracked_points
        now = time.perf_counter()
        stage_timings_ms["kalman_plus_ai"] = (now - mark) * 1000
        mark = now
        frame_height, frame_width = frames[0].shape[:2]
        tracklet_builder = CandidateTrackletBuilder(
            max_frame_gap=options.tracklet_max_frame_gap,
            max_pixel_displacement=options.tracklet_max_pixel_displacement,
            allow_motion_only_start=options.allow_motion_only_tracklet_start_debug and self.settings.debug,
        )
        tracklet_result = tracklet_builder.build(
            candidates=windowed_candidates,
            metadata=metadata,
            frame_width=frame_width,
            frame_height=frame_height,
            active_window=active_window,
        )
        now = time.perf_counter()
        stage_timings_ms["tracklet_build"] = (now - mark) * 1000
        mark = now
        if artifact_dir is not None:
            self._write_tracklet_artifacts(
                artifact_dir=artifact_dir,
                tracklet_result=tracklet_result,
                metadata=metadata,
            )
        if not tracklet_result.selected_candidates:
            if artifact_dir is not None:
                self._write_debug_artifacts_failed_processing(
                    artifact_dir=artifact_dir,
                    frames=frames,
                    frame_indices=frame_indices,
                    candidates_by_source={
                        "yolo": yolo_candidates,
                        "cricket": cricket_candidates,
                        "stumps": stump_detections,
                        "motion": motion_candidates,
                        "tracknet": tracknet_candidates,
                        "zoom": zoom_candidates,
                        "merged": all_candidates,
                        "active_window": windowed_candidates,
                        "selected_tracklet": [],
                    },
                    rejection_reason="no valid detector-supported tracklet",
                    active_window=active_window,
                    tracklet_result=tracklet_result,
                    options=options,
                    metadata=metadata,
                )
            raise ValueError("no valid detector-supported tracklet")
        relaxed_for_window = self._relaxed_solver_candidates(windowed_candidates, metadata)
        if len(relaxed_for_window) >= max(8, len(tracklet_result.selected_candidates) + 4):
            tracklet_result.selected_candidates = relaxed_for_window

        try:
            solution = self.solver.solve(tracklet_result.selected_candidates, metadata)
            self._validate_solution_window(solution, active_window, debug=self.settings.debug)
        except Exception as exc:
            fallback_used = False
            if "not enough temporally consistent ball candidates" in str(exc).lower():
                relaxed_candidates = self._relaxed_solver_candidates(windowed_candidates, metadata)
                if len(relaxed_candidates) >= 6:
                    solution = self.solver.solve(relaxed_candidates, metadata)
                    self._validate_solution_window(solution, active_window, debug=self.settings.debug)
                    tracklet_result.selected_candidates = relaxed_candidates
                    fallback_used = True
                    self.logger.info("solver fallback used relaxed candidate chain (%d points)", len(relaxed_candidates))
            if fallback_used:
                now = time.perf_counter()
                stage_timings_ms["solver"] = (now - mark) * 1000
                mark = now
            else:
                if artifact_dir is not None:
                    self._write_debug_artifacts_failed_processing(
                        artifact_dir=artifact_dir,
                        frames=frames,
                        frame_indices=frame_indices,
                        candidates_by_source={
                            "yolo": yolo_candidates,
                            "cricket": cricket_candidates,
                            "stumps": stump_detections,
                            "motion": motion_candidates,
                            "tracknet": tracknet_candidates,
                            "zoom": zoom_candidates,
                            "merged": all_candidates,
                            "active_window": windowed_candidates,
                            "selected_tracklet": tracklet_result.selected_candidates,
                        },
                        rejection_reason=str(exc),
                        active_window=active_window,
                        tracklet_result=tracklet_result,
                        options=options,
                        metadata=metadata,
                    )
                raise
        now = time.perf_counter()
        stage_timings_ms["solver"] = (now - mark) * 1000
        mark = now
        _fps = float(metadata.fps or metadata.extras.get("videoFps") or 30.0)
        predicted_bounce_video_frame = int(round(float(solution.bounce_point.t or 0.0) * _fps))
        bounce_net_dict = self.bounce_net.estimate(
            frames=frames,
            frame_indices=frame_indices,
            predicted_bounce_frame_num=predicted_bounce_video_frame,
            metadata=metadata,
            candidates=windowed_candidates,
            kalman_points=kalman_result.tracked_points + kalman_result.predictions,
            frame_width=frame_width,
            frame_height=frame_height,
            artifact_dir=artifact_dir,
        ).to_dict()
        metadata.extras["bounceNet"] = bounce_net_dict
        now = time.perf_counter()
        stage_timings_ms["bounce_net"] = (now - mark) * 1000
        mark = now
        endpoint_refinement = self._refine_endpoint_with_stumps(
            solution=solution,
            selected_candidates=tracklet_result.selected_candidates,
            all_candidates=windowed_candidates,
            stump_detections=stump_detections,
            metadata=metadata,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        physical_metrics = compute_physical_metrics(solution, metadata)
        now = time.perf_counter()
        stage_timings_ms["physical_metrics"] = (now - mark) * 1000
        mark = now
        speed = physical_metrics.speed_kph
        swing = swing_amount(solution)
        elapsed_ms = (time.perf_counter() - started) * 1000

        result = DeliveryResult(
            modelsUsed=self.loaded_models,
            calibrationMode=metadata.calibrationMode,
            processingTimeMs=elapsed_ms,
            activeStartSec=active_window["activeStartSec"],
            activeEndSec=active_window["activeEndSec"],
            activeWindowDurationSec=active_window["activeWindowDurationSec"],
            windowSource=active_window["windowSource"],
            releasePoint=solution.release_point,
            bouncePoint=solution.bounce_point,
            endPoint=solution.end_point,
            trajectory=solution.trajectory,
            trajectory2D=[Point2D(x=point.x, y=point.z, t=point.t) for point in solution.trajectory],
            trajectoryPitchCoords=self._trajectory_pitch_coords(solution.trajectory, metadata),
            releaseSec=solution.release_point.t,
            bounceSec=solution.bounce_point.t,
            stumpImpactSec=solution.end_point.t,
            speedKph=speed,
            speedMph=physical_metrics.speed_mph,
            speedConfidence=physical_metrics.speed_confidence,
            bounceDistanceFromBatterStumpsMeters=physical_metrics.bounce_distance_from_batter_stumps_m,
            bounceDistanceFromBowlerStumpsMeters=physical_metrics.bounce_distance_from_bowler_stumps_m,
            lineMetersFromCenter=physical_metrics.line_meters_from_center,
            lineMetersFromOffStump=physical_metrics.line_meters_from_off_stump,
            lengthCategory=physical_metrics.length_category,
            trajectoryConfidence=solution.confidence,
            calibrationConfidence=physical_metrics.calibration_confidence,
            corridorConfidence=corridor_confidence,
            line=physical_metrics.line_category,
            length=physical_metrics.length_category,
            swingAmount=swing,
            seamDeviation=None,
            spinEffectScore=min(1.0, swing * 3.0),
            confidence=solution.confidence,
            fps=float(metadata.extras.get("videoFps")) if metadata.extras.get("videoFps") is not None else None,
            stumpDetections=stump_detections,
            bestStumpDetection=best_stump,
            stumpRoi=endpoint_refinement.get("stumpRoi"),
            stumpRois=endpoint_refinement.get("stumpRois") or {},
            visualStumpBoxes=endpoint_refinement.get("visualStumpBoxes") or {},
            virtualWicketPlane=endpoint_refinement.get("virtualWicketPlane"),
            stumpImpactPredictedSec=endpoint_refinement.get("stumpImpactPredictedSec"),
            stumpImpactConfidence=endpoint_refinement.get("stumpImpactConfidence"),
            distanceToStumpRoiPx=endpoint_refinement.get("distanceToStumpRoiPx"),
            endpointSource=endpoint_refinement.get("endpointSource"),
            debug=DeliveryDebug(
                modelsUsed=self.loaded_models,
                candidateCounts={
                    "yolo": len(yolo_candidates),
                    "yolo_primary": self.yolo.last_stats.get("yolo_primary", 0),
                    "yolo_fallback": self.yolo.last_stats.get("yolo_fallback", 0),
                    "yolo_crop": self.yolo.last_stats.get("yolo_crop", 0),
                    "cricket": len(cricket_candidates),
                    "stumps": len(stump_detections),
                    "motion": len(motion_candidates),
                    "motion_before_gating": self.motion.last_stats.get("motion_before_gating", 0),
                    "motion_after_corridor": self.motion.last_stats.get("motion_after_corridor", 0),
                    "motion_after_cluster": self.motion.last_stats.get("motion_after_cluster", 0),
                    "tracknet": len(tracknet_candidates),
                    "zoom": len(zoom_candidates),
                    "merged": len(all_candidates),
                    "active_window_candidates": len(windowed_candidates),
                    "selected_tracklet_candidates": len(tracklet_result.selected_candidates),
                    "tracklets_total": len(tracklet_result.tracklets),
                    "robust_fit_used_points": solution.robust_fit_used_points,
                    "robust_fit_outliers_removed": solution.robust_fit_outliers_removed,
                    "yolo_points_used": solution.yolo_points_used,
                    "ball_points_used": solution.ball_points_used,
                    "motion_points_used": solution.motion_points_used,
                    "frames_processed": len(frames),
                },
                processingTimeMs=elapsed_ms,
                fps=float(metadata.extras.get("videoFps")) if metadata.extras.get("videoFps") is not None else None,
                stageTimingsMs=stage_timings_ms,
                notes=[
                    "Models propose candidates; trajectory solver selects the cricket-valid path.",
                    f"Motion-only path: {solution.motion_only_path}",
                    f"Active window {active_window['activeStartSec']:.3f}s..{active_window['activeEndSec']:.3f}s source={active_window['windowSource']}",
                    f"Tracklets built={len(tracklet_result.tracklets)} selected_points={len(tracklet_result.selected_candidates)}",
                    f"Endpoint source={endpoint_refinement.get('endpointSource')} stumpImpact={endpoint_refinement.get('stumpImpactPredictedSec')}",
                    f"Calibration confidence={physical_metrics.calibration_confidence:.2f} speed confidence={physical_metrics.speed_confidence:.2f}",
                    f"Device={self.settings.device}",
                ],
            ),
        )
        reconstructed = self.reconstructor.reconstruct(
            result=result,
            metadata=metadata,
            selected_tracklet=tracklet_result.selected_candidates,
            merged_candidates=windowed_candidates,
            kalman_tracked_points=kalman_result.tracked_points,
            ai_correction=metadata.extras.get("trajectoryAiCorrection", {}),
            bounce_net=bounce_net_dict,
        )
        result.reconstructedTrajectory = reconstructed.points
        result.finalTrajectory = reconstructed.final_points
        result.trajectorySegments = reconstructed.segments
        result.eventSources = reconstructed.event_sources
        if reconstructed.release_sec is not None:
            result.releaseSec = reconstructed.release_sec
            result.releasePoint.t = reconstructed.release_sec
            if result.reconstructedTrajectory:
                nearest_rel = min(
                    result.reconstructedTrajectory,
                    key=lambda point: abs(float(point.t or 0.0) - float(reconstructed.release_sec)),
                )
                result.releasePoint.x = nearest_rel.x
                result.releasePoint.y = nearest_rel.y
        if reconstructed.bounce_sec is not None:
            result.bounceSec = reconstructed.bounce_sec
            result.bouncePoint.t = reconstructed.bounce_sec
            nearest_bounce = min(
                result.reconstructedTrajectory,
                key=lambda point: abs(float(point.t or 0.0) - float(reconstructed.bounce_sec)),
                default=None,
            )
            if nearest_bounce is not None:
                result.bouncePoint.x = nearest_bounce.x
                result.bouncePoint.y = nearest_bounce.y
        elif (reconstructed.event_sources or {}).get("bounce", "").startswith("no_bounce"):
            result.bounceSec = None
            result.bouncePoint.t = None
        if reconstructed.impact_sec is not None:
            result.stumpImpactSec = reconstructed.impact_sec
            result.endPoint.t = reconstructed.impact_sec
        result.debug.reconstructionStats = dict(reconstructed.stats)
        result.debug.reconstructionStats["kalmanTrackedPoints"] = len(kalman_result.tracked_points)
        result.debug.reconstructionStats["kalmanSyntheticPoints"] = kalman_result.synthetic_points
        result.debug.reconstructionStats["kalmanWeakAccepted"] = kalman_result.accepted_weak_candidates
        for key, value in bounce_net_dict.items():
            result.debug.reconstructionStats[key] = value
        refined_metrics = refine_physical_metrics_after_reconstruction(result, metadata, physical_metrics)
        result.speedKph = refined_metrics.speed_kph
        result.speedMph = refined_metrics.speed_mph
        result.speedConfidence = refined_metrics.speed_confidence
        result.bounceDistanceFromBatterStumpsMeters = refined_metrics.bounce_distance_from_batter_stumps_m
        result.bounceDistanceFromBowlerStumpsMeters = refined_metrics.bounce_distance_from_bowler_stumps_m
        result.lineMetersFromCenter = refined_metrics.line_meters_from_center
        result.lineMetersFromOffStump = refined_metrics.line_meters_from_off_stump
        result.lengthCategory = refined_metrics.length_category
        result.length = refined_metrics.length_category
        result.line = refined_metrics.line_category
        result.deliverySpeedKph = result.speedKph
        result.deliverySpeedMph = result.speedMph
        result.speedSource = "radar_release_window"
        detected_frames = [
            int(point["frame"])
            for point in result.finalTrajectory
            if isinstance(point, dict) and point.get("source") in {"raw_detector", "weak_detector"} and point.get("frame") is not None
        ]
        result.lastDetectedBallFrame = max(detected_frames, default=max((candidate.frameIndex for candidate in tracklet_result.selected_candidates), default=None))
        result.releaseFrame = self._time_to_frame(result.releaseSec, metadata)
        bounce_source = (result.eventSources or {}).get("bounce", "")
        impact_source = str(reconstructed.stats.get("impactSource", "inferred"))
        result.impactSource = impact_source
        if bounce_source.startswith("no_bounce"):
            result.actualBounceFrame = None
            result.projectedBounceFrame = None
        elif bounce_source.startswith("bounce_net"):
            result.actualBounceFrame = bounce_net_dict.get("bounceNetBounceFrame")
            result.projectedBounceFrame = None
        elif bounce_source.startswith("rescued") or bounce_source.startswith("raw"):
            result.actualBounceFrame = self._time_to_frame(result.bounceSec, metadata)
            result.projectedBounceFrame = None
        else:
            result.actualBounceFrame = None
            result.projectedBounceFrame = self._time_to_frame(result.bounceSec, metadata)
        if impact_source == "detected":
            result.stumpImpactFrame = self._time_to_frame(result.stumpImpactSec, metadata)
            result.projectedImpactFrame = None
        else:
            result.stumpImpactFrame = None
            result.projectedImpactFrame = self._time_to_frame(result.stumpImpactSec, metadata)
        stumps_decision = self._stumps_hitting_decision(result)
        result.stumpsHitting = stumps_decision["hitting"]
        result.stumpsHittingConfidence = stumps_decision["confidence"]
        result.virtualPitchCorridor = self._virtual_pitch_corridor(result)
        line_decision = self._pitched_in_line_decision(result)
        result.pitchedInLine = line_decision["pitchedInLine"]
        result.pitchedInLineConfidence = line_decision["confidence"]
        result.debug.reconstructionStats["pitchedInLine"] = str(result.pitchedInLine)
        result.debug.reconstructionStats["pitchedInLineConfidence"] = float(result.pitchedInLineConfidence or 0.0)
        result.trajectory3D = self._pseudo_trajectory_3d(result)
        drs_payload = self._build_drs_payload(result, metadata)
        result.drsDecision = drs_payload["decision"]
        result.drsAnalytics = drs_payload["analytics"]
        if artifact_dir is not None:
            self._write_debug_artifacts(
                artifact_dir=artifact_dir,
                result=result,
                frames=frames,
                frame_indices=frame_indices,
                candidates_by_source={
                    "yolo": yolo_candidates,
                    "cricket": cricket_candidates,
                    "stumps": stump_detections,
                    "motion": motion_candidates,
                    "tracknet": tracknet_candidates,
                    "zoom": zoom_candidates,
                    "merged": all_candidates,
                    "active_window": windowed_candidates,
                    "selected_tracklet": tracklet_result.selected_candidates,
                },
                active_window=active_window,
                tracklet_result=tracklet_result,
                options=options,
                metadata=metadata,
            )
        render_elapsed_ms = 0.0
        if options.render_annotated_video and artifact_dir is not None:
            render_started = time.perf_counter()
            render_dir = artifact_dir / "visualization"
            render_name = "debug_overlay.mp4" if options.render_mode == "debug" else "trajectory_overlay.mp4"
            render_path = render_dir / render_name
            artifacts = self.renderer.render(
                clip_path=upload_path,
                result=result,
                metadata=metadata,
                selected_tracklet=tracklet_result.selected_candidates,
                merged_candidates=windowed_candidates,
                output_path=render_path,
                mode=options.render_mode,
            )
            result.annotatedVideoPath = str(artifacts.annotated_video_path)
            result.renderFramesCount = artifacts.frames_rendered
            result.outputVideoSizeMb = artifacts.output_video_size_mb
            result.debug.notes.append(f"Rendered overlay video: {artifacts.annotated_video_path}")
            consumer_path = render_dir / "consumer_overlay.mp4"
            consumer_artifacts = self.renderer.render(
                clip_path=upload_path,
                result=result,
                metadata=metadata,
                selected_tracklet=tracklet_result.selected_candidates,
                merged_candidates=windowed_candidates,
                output_path=consumer_path,
                mode="consumer",
            )
            result.consumerAnnotatedVideoPath = str(consumer_artifacts.annotated_video_path)
            result.consumerRenderFramesCount = consumer_artifacts.frames_rendered
            result.consumerOutputVideoSizeMb = consumer_artifacts.output_video_size_mb
            result.debug.notes.append(f"Rendered consumer overlay video: {consumer_artifacts.annotated_video_path}")
            consumer_sync_path = render_dir / "consumer_overlay_sync.mp4"
            sync_artifacts = self.renderer.render(
                clip_path=upload_path,
                result=result,
                metadata=metadata,
                selected_tracklet=tracklet_result.selected_candidates,
                merged_candidates=windowed_candidates,
                output_path=consumer_sync_path,
                mode="consumer_sync",
            )
            result.consumerSyncAnnotatedVideoPath = str(sync_artifacts.annotated_video_path)
            result.consumerSyncRenderFramesCount = sync_artifacts.frames_rendered
            result.consumerSyncOutputVideoSizeMb = sync_artifacts.output_video_size_mb
            result.debug.notes.append(f"Rendered consumer sync-only overlay: {sync_artifacts.annotated_video_path}")
            render_elapsed_ms = (time.perf_counter() - render_started) * 1000.0
            result_path = artifact_dir / "result.json"
            if result_path.exists():
                result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        now = time.perf_counter()
        stage_timings_ms["debug_artifacts"] = (now - mark) * 1000
        if render_elapsed_ms > 0:
            stage_timings_ms["render_video"] = render_elapsed_ms
        stage_timings_ms["total"] = elapsed_ms
        self.logger.info(
            "delivery timing ms decode=%.1f extract=%.1f ball_stump=%.1f solver=%.1f total=%.1f",
            stage_timings_ms.get("decode", 0.0),
            stage_timings_ms.get("frame_extraction", 0.0),
            stage_timings_ms.get("ball_stump_inference", 0.0),
            stage_timings_ms.get("solver", 0.0),
            stage_timings_ms.get("total", 0.0),
        )
        return result

    def write_failure_artifacts(
        self,
        artifact_dir: Path,
        reason: str,
        frames: list[np.ndarray] | None = None,
        frame_indices: list[int] | None = None,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "failed",
            "reason": reason,
            "sampledFrameCount": len(frames or []),
            "sampledFrameIndices": frame_indices or [],
        }
        (artifact_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (artifact_dir / "candidates.json").write_text(
            json.dumps(
                {
                    "sampledFrameCount": len(frames or []),
                    "sampledFrameIndices": frame_indices or [],
                    "candidateCounts": {},
                    "candidates": {},
                    "rejectionReasons": [reason],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_debug_report(
            artifact_dir=artifact_dir,
            result=None,
            frame_indices=frame_indices or [],
            candidates_by_source={},
            rejection_reasons=[reason],
            active_window={"activeStartSec": 0.0, "activeEndSec": 0.0, "activeWindowDurationSec": 0.0, "windowSource": "none"},
            metadata=None,
        )

    def _decode_upload(self, upload_path: Path, metadata: DeliveryMetadata) -> tuple[list[np.ndarray], list[int]]:
        if upload_path.suffix.lower() == ".zip":
            return self._decode_frame_zip(upload_path)
        capture = cv2.VideoCapture(str(upload_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = float(metadata.fps or metadata.extras.get("videoFps") or 0.0)
        if fps <= 0:
            capture.release()
            raise ValueError("Unable to decode valid FPS from upload and metadata.")
        metadata.fps = fps
        metadata.extras["videoFps"] = fps
        frames: list[np.ndarray] = []
        frame_indices: list[int] = []
        frame_timestamps: list[float] = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            timestamp = (pos_msec / 1000.0) if pos_msec > 0 else (index / fps)
            frames.append(frame)
            frame_indices.append(index)
            frame_timestamps.append(timestamp)
            index += 1
        capture.release()
        if frames:
            height, width = frames[0].shape[:2]
            if metadata.resolution is None:
                metadata.resolution = Resolution(width=width, height=height)
            if metadata.cameraResolution is None:
                metadata.cameraResolution = CameraResolution(width=metadata.resolution.width, height=metadata.resolution.height)
            metadata.frameTimestamps.clear()
            metadata.frameTimestamps.extend(frame_timestamps)
        duration_sec = len(frames) / fps if fps > 0 else 0.0
        if duration_sec <= 0:
            raise ValueError("Video decode failed: no valid frames available.")
        if duration_sec > self.settings.max_clip_duration_sec:
            raise ValueError(f"Clip too long ({duration_sec:.2f}s). Max allowed: {self.settings.max_clip_duration_sec:.2f}s.")
        return frames, frame_indices

    def _decode_frame_zip(self, upload_path: Path) -> tuple[list[np.ndarray], list[int]]:
        frames: list[np.ndarray] = []
        frame_indices: list[int] = []
        with zipfile.ZipFile(upload_path) as archive:
            image_names = sorted(
                name for name in archive.namelist()
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            for index, name in enumerate(image_names):
                data = np.frombuffer(archive.read(name), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frames.append(frame)
                frame_indices.append(index)
        if frames and not metadata.frameTimestamps:
            fps = float(metadata.extras.get("videoFps") or 30.0)
            metadata.extras["videoFps"] = fps
            metadata.frameTimestamps.extend([i / fps for i in range(len(frames))])
        return frames, frame_indices

    def _sample_frames(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        metadata: DeliveryMetadata,
        max_frames: int | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        if not frames:
            return [], []
        release_time = metadata.watchEvents.releaseDetected or metadata.watchEvents.releaseWindowLikely
        if release_time is None or not metadata.frameTimestamps:
            return self._limit_frames(frames, frame_indices, max_frames)

        sampled_frames: list[np.ndarray] = []
        sampled_indices: list[int] = []
        half_window = self.settings.dense_release_window_seconds / 2
        for frame, frame_index in zip(frames, frame_indices):
            timestamp = metadata.frameTimestamps[min(frame_index, len(metadata.frameTimestamps) - 1)]
            dense = abs(timestamp - release_time) <= half_window
            if dense or frame_index % max(1, self.settings.moderate_frame_stride) == 0:
                sampled_frames.append(frame)
                sampled_indices.append(frame_index)
        return self._limit_frames(sampled_frames, sampled_indices, max_frames)

    def _limit_frames(
        self,
        frames: list[np.ndarray],
        frame_indices: list[int],
        max_frames: int | None,
    ) -> tuple[list[np.ndarray], list[int]]:
        if not max_frames or len(frames) <= max_frames:
            return frames, frame_indices
        stride = max(1, int(np.ceil(len(frames) / max_frames)))
        return frames[::stride][:max_frames], frame_indices[::stride][:max_frames]

    def _infer_active_window(
        self,
        metadata: DeliveryMetadata,
        candidates: list,
        options: ProcessingOptions,
    ) -> dict[str, float | str]:
        gt = options.ground_truth or {}
        if options.use_ground_truth_window and gt:
            start = float(gt.get("releaseSecRange", [0, 0])[0]) - 0.07
            end = float(gt.get("stumpImpactSec", start + 1.0)) + 0.05
            return {
                "activeStartSec": max(0.0, start),
                "activeEndSec": max(start + 0.1, end),
                "activeWindowDurationSec": max(0.1, end - start),
                "windowSource": "ground_truth_debug",
            }

        watch_release = metadata.watchEvents.releaseDetected or metadata.watchEvents.releaseWindowLikely
        if watch_release is not None:
            start = max(0.0, watch_release - 0.1)
            end = start + 1.2
            return {
                "activeStartSec": start,
                "activeEndSec": end,
                "activeWindowDurationSec": end - start,
                "windowSource": "watch",
            }

        candidate_times = self._candidate_times(candidates, metadata)
        if not candidate_times:
            return {"activeStartSec": 0.0, "activeEndSec": 2.0, "activeWindowDurationSec": 2.0, "windowSource": "inferred"}
        timed_candidates = sorted(
            [
                (
                    candidate_time,
                    float(getattr(candidate, "confidence", 0.0) or 0.0),
                    str(getattr(candidate, "source", "")),
                    float(getattr(candidate, "x", 0.0)),
                    float(getattr(candidate, "y", 0.0)),
                )
                for candidate, candidate_time in zip(candidates, candidate_times)
            ],
            key=lambda item: item[0],
        )
        clusters: list[list[tuple[float, float, str, float, float]]] = [[]]
        for value in timed_candidates:
            if not clusters[-1] or value[0] - clusters[-1][-1][0] <= 0.35:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        best_cluster = max(clusters, key=self._cluster_window_score)
        cluster_times = [item[0] for item in best_cluster]
        start = max(0.0, min(cluster_times) - 0.08)
        end = max(start + 0.2, max(cluster_times) + 0.30)
        if end - start > 2.5 and not self.settings.debug:
            end = start + 2.5
        return {
            "activeStartSec": start,
            "activeEndSec": end,
            "activeWindowDurationSec": end - start,
            "windowSource": "inferred",
        }

    def _cluster_window_score(self, cluster: list[tuple[float, float, str, float, float]]) -> float:
        if not cluster:
            return float("-inf")
        times = [item[0] for item in cluster]
        confidences = [item[1] for item in cluster]
        sources = [item[2] for item in cluster]
        xs = [item[3] for item in cluster]
        ys = [item[4] for item in cluster]
        duration = max(0.01, max(times) - min(times))
        yolo_supported = sum(1 for source in sources if "yolo" in source or "cricket_ball_v2" in source)
        motion_points = sum(1 for source in sources if "optical_flow" in source)
        early_start = min(times)
        confidence_mean = float(np.mean(confidences)) if confidences else 0.0
        spatial_travel = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) if len(xs) > 1 else 0.0
        spatial_span = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) if xs and ys else 0.0
        y_span = float(max(ys) - min(ys)) if ys else 0.0
        x_span = float(max(xs) - min(xs)) if xs else 0.0
        if max(spatial_travel, spatial_span) < 0.08:
            return float("-inf")
        duration_penalty = 0.0 if 0.30 <= duration <= 1.80 else 2.0
        early_penalty = max(0.0, (0.8 - early_start) * 2.5)
        source_penalty = 2.5 if yolo_supported == 0 and motion_points > 0 else 0.0
        static_penalty = 2.0 if max(spatial_travel, spatial_span) < 0.12 else 0.0
        travel_bonus = min(2.0, max(spatial_travel, spatial_span) * 8.0) + min(1.2, y_span * 4.0) + min(0.8, x_span * 2.0)
        return (len(cluster) * 1.0) + (yolo_supported * 2.0) + confidence_mean + travel_bonus - duration_penalty - early_penalty - source_penalty - static_penalty

    def _candidate_times(self, candidates: list, metadata: DeliveryMetadata) -> list[float]:
        times: list[float] = []
        max_index = len(metadata.frameTimestamps) - 1
        for candidate in candidates:
            if metadata.frameTimestamps and max_index >= 0:
                idx = min(max(0, candidate.frameIndex), max_index)
                times.append(float(metadata.frameTimestamps[idx]))
            else:
                fps = float(metadata.extras.get("videoFps") or 30.0)
                times.append(candidate.frameIndex / fps)
        return times

    def _candidate_time(self, candidate, metadata: DeliveryMetadata) -> float:
        if metadata.frameTimestamps:
            max_index = len(metadata.frameTimestamps) - 1
            idx = min(max(0, candidate.frameIndex), max_index)
            return float(metadata.frameTimestamps[idx])
        fps = float(metadata.extras.get("videoFps") or 30.0)
        return candidate.frameIndex / fps

    def _filter_contextual_ball_candidates(self, candidates: list, metadata: DeliveryMetadata) -> list:
        filtered: list = []
        for candidate in candidates:
            if getattr(candidate, "modelRole", "") != "ball_detector":
                filtered.append(candidate)
                continue
            y = float(getattr(candidate, "y", 0.0) or 0.0)
            x = float(getattr(candidate, "x", 0.0) or 0.0)
            # Outdoor clips can contain birds/airborne debris. A valid cricket ball
            # should not seed tracking from the sky band far above the pitch/stumps.
            if y < self._minimum_playable_ball_y(metadata):
                continue
            if x < -0.05 or x > 1.05:
                continue
            filtered.append(candidate)
        return filtered

    def _minimum_playable_ball_y(self, metadata: DeliveryMetadata) -> float:
        polygon = metadata.corridorGeometry.pitchPolygon
        if len(polygon) >= 4:
            top = min(point.y for point in polygon[:4])
            return max(0.14, top - 0.18)
        if metadata.guidedBoxes and metadata.guidedBoxes.pitchCorridorNorm:
            top = min(point.y for point in metadata.guidedBoxes.pitchCorridorNorm[:4])
            return max(0.14, top - 0.18)
        return 0.16

    def _relaxed_solver_candidates(self, candidates: list, metadata: DeliveryMetadata) -> list:
        ball_candidates = [
            candidate
            for candidate in candidates
            if getattr(candidate, "modelRole", "") == "ball_detector" and float(getattr(candidate, "confidence", 0.0) or 0.0) >= 0.10
        ]
        by_frame: dict[int, object] = {}
        for candidate in ball_candidates:
            existing = by_frame.get(candidate.frameIndex)
            if existing is None or float(candidate.confidence or 0.0) > float(existing.confidence or 0.0):
                by_frame[candidate.frameIndex] = candidate
        ordered = sorted(by_frame.values(), key=lambda item: self._candidate_time(item, metadata))
        if len(ordered) <= 2:
            return ordered
        pruned: list = [ordered[0]]
        for candidate in ordered[1:]:
            prev = pruned[-1]
            dt = self._candidate_time(candidate, metadata) - self._candidate_time(prev, metadata)
            if dt <= 0 or dt > 0.35:
                continue
            distance = float(np.hypot(candidate.x - prev.x, candidate.y - prev.y))
            if distance > 0.16:
                continue
            pruned.append(candidate)
        minimum_useful_coverage = max(8, int(len(ordered) * 0.5))
        return pruned if len(pruned) >= minimum_useful_coverage else ordered

    def _time_to_frame(self, time_sec: float | None, metadata: DeliveryMetadata) -> int | None:
        if time_sec is None:
            return None
        if metadata.frameTimestamps:
            best_idx = min(
                range(len(metadata.frameTimestamps)),
                key=lambda index: abs(float(metadata.frameTimestamps[index]) - float(time_sec)),
            )
            return int(best_idx)
        fps = float(metadata.fps or metadata.extras.get("videoFps") or 30.0)
        return int(round(float(time_sec) * fps))

    def _stumps_hitting_decision(self, result: DeliveryResult) -> dict[str, float | bool | None]:
        plane = result.virtualWicketPlane
        if not plane:
            return {"hitting": None, "confidence": 0.0}
        x_center = float(plane.get("xCenter", 0.5))
        half_width = float(plane.get("halfWidth", 0.02))
        y_top = float(plane.get("yTop", 0.0))
        y_bottom = float(plane.get("yBottom", 1.0))
        impact_x = float(result.endPoint.x)
        impact_y = float(result.endPoint.y)
        observed_post = sorted(
            [
                point
                for point in (result.finalTrajectory or [])
                if isinstance(point, dict)
                and str(point.get("segmentType", "")).startswith("observed_post_bounce")
                and str(point.get("source", "")) != "physics_projected"
            ],
            key=lambda item: float(item.get("t", 0.0) or 0.0),
        )
        if len(observed_post) >= 2:
            p1 = observed_post[-2]
            p2 = observed_post[-1]
            y_target = float((y_top + y_bottom) * 0.5)
            y1 = float(p1.get("y", impact_y))
            y2 = float(p2.get("y", impact_y))
            x1 = float(p1.get("x", impact_x))
            x2 = float(p2.get("x", impact_x))
            dy = y2 - y1
            if abs(dy) > 1e-6:
                scale = (y_target - y2) / dy
                if 0.0 <= scale <= 6.0:
                    impact_x = float(x2 + (x2 - x1) * scale)
                    impact_y = y_target
        inside_x = abs(impact_x - x_center) <= half_width
        inside_y = y_top <= impact_y <= y_bottom
        distance_x = max(0.0, abs(impact_x - x_center) - half_width)
        distance_y = 0.0 if inside_y else min(abs(impact_y - y_top), abs(impact_y - y_bottom))
        miss_distance = float(np.hypot(distance_x, distance_y))
        confidence = max(0.0, min(0.98, 0.92 - miss_distance * 10.0))
        if result.impactSource == "projected":
            confidence *= 0.86
        return {"hitting": bool(inside_x and inside_y), "confidence": float(confidence)}

    def _virtual_pitch_corridor(self, result: DeliveryResult) -> list[dict[str, float]]:
        boxes = result.visualStumpBoxes or result.stumpRois or {}
        far = boxes.get("far") or result.stumpRoi
        if not far:
            return []
        near = boxes.get("near")
        far_left = float(far.get("x1", far.get("centerX", 0.5)))
        far_right = float(far.get("x2", far.get("centerX", 0.5)))
        far_center = float(far.get("centerX", (far_left + far_right) * 0.5))
        far_y = float(far.get("y2", far.get("centerY", 0.55)))
        far_width = max(0.01, far_right - far_left)
        if near:
            near_center = float(near.get("centerX", far_center))
            near_width = max(far_width * 1.8, float(near.get("x2", near_center) - near.get("x1", near_center)))
            near_y = float(near.get("y2", min(0.96, far_y + 0.35)))
        else:
            near_center = far_center + (far_center - 0.5) * 0.35
            near_width = far_width * 4.0
            near_y = min(0.96, far_y + 0.34)
        near_width = min(0.32, max(far_width * 1.8, near_width))
        points = [
            {"x": float(np.clip(far_center - far_width * 0.5, 0.0, 1.0)), "y": float(np.clip(far_y, 0.0, 1.0))},
            {"x": float(np.clip(far_center + far_width * 0.5, 0.0, 1.0)), "y": float(np.clip(far_y, 0.0, 1.0))},
            {"x": float(np.clip(near_center + near_width * 0.5, 0.0, 1.0)), "y": float(np.clip(near_y, 0.0, 1.0))},
            {"x": float(np.clip(near_center - near_width * 0.5, 0.0, 1.0)), "y": float(np.clip(near_y, 0.0, 1.0))},
        ]
        return points

    def _pitched_in_line_decision(self, result: DeliveryResult) -> dict[str, float | bool | None]:
        corridor = result.virtualPitchCorridor
        if len(corridor) < 4:
            return {"pitchedInLine": None, "confidence": 0.0}
        x = float(result.bouncePoint.x)
        y = float(result.bouncePoint.y)
        left_far, right_far, right_near, left_near = corridor[:4]
        y_far = float((left_far["y"] + right_far["y"]) * 0.5)
        y_near = float((left_near["y"] + right_near["y"]) * 0.5)
        if y_near <= y_far:
            return {"pitchedInLine": None, "confidence": 0.0}
        alpha = float(np.clip((y - y_far) / max(1e-6, y_near - y_far), 0.0, 1.0))
        left_x = float(left_far["x"] * (1.0 - alpha) + left_near["x"] * alpha)
        right_x = float(right_far["x"] * (1.0 - alpha) + right_near["x"] * alpha)
        center = (left_x + right_x) * 0.5
        half_width = max(1e-6, abs(right_x - left_x) * 0.5)
        distance = max(0.0, abs(x - center) - half_width)
        tolerance = max(0.012, half_width * 0.35)
        in_line = distance <= tolerance and y >= y_far - 0.04 and y <= y_near + 0.04
        confidence = max(0.0, min(0.96, 0.90 - (distance / max(tolerance, 1e-6)) * 0.22))
        return {"pitchedInLine": bool(in_line), "confidence": float(confidence)}

    def _build_drs_payload(self, result: DeliveryResult, metadata: DeliveryMetadata) -> dict[str, dict[str, object]]:
        pitch_zone = self._wicket_line_zone(result, result.bouncePoint.x)
        impact_zone = "not_detected"
        pitch_valid = pitch_zone != "outside_leg"
        wicket_zone = "hitting" if result.stumpsHitting else "missing"
        if result.stumpsHitting and (result.stumpsHittingConfidence or 0.0) < 0.55:
            wicket_zone = "umpires_call"
        final_decision = "Out" if pitch_valid and wicket_zone == "hitting" else ("Umpire's Call" if wicket_zone == "umpires_call" else "Not Out")
        reconstruction = result.debug.reconstructionStats or {}
        post_seen = int(reconstruction.get("postBounceCandidatesSeen", 0) or 0)
        post_used = int(reconstruction.get("postBounceCandidatesUsed", 0) or 0)
        if post_seen > 0 and post_used == 0 and (result.stumpsHittingConfidence or 0.0) < 0.35:
            final_decision = "Insufficient Evidence"
        if impact_zone == "not_detected":
            final_basis = "wicket_projection_only_pad_impact_not_detected"
        else:
            final_basis = "lbw_three_checkpoint"
        pitch_m = self._pitch_coordinate_from_y(result, result.bouncePoint.y)
        length_label = classify_length_from_distance(float(result.bounceDistanceFromBatterStumpsMeters or pitch_m), result.bouncePoint)
        post_bounce_duration = None
        if result.bounceSec is not None and result.stumpImpactSec is not None:
            post_bounce_duration = float(result.stumpImpactSec - result.bounceSec)
        if (
            length_label in {"short_ball", "bouncer"}
            and post_bounce_duration is not None
            and 0.0 < post_bounce_duration <= 0.40
            and (result.bounceDistanceFromBatterStumpsMeters is None or float(result.bounceDistanceFromBatterStumpsMeters) <= 11.5)
        ):
            length_label = "yorker"
        decision = {
            "pitching": {
                "zone": pitch_zone,
                "status": "red_not_out" if pitch_zone == "outside_leg" else "green_valid",
                "point": {"x": float(result.bouncePoint.x), "y": float(result.bouncePoint.y), "t": float(result.bouncePoint.t or result.bounceSec or 0.0)},
                "confidence": float(result.pitchedInLineConfidence or result.trajectoryConfidence or 0.0),
            },
            "impact": {
                "zone": impact_zone,
                "status": "conditional_pad_detector_required",
                "shotOffered": metadata.extras.get("shotOffered"),
                "confidence": 0.0,
            },
            "wicketHitting": {
                "zone": wicket_zone,
                "status": "green_hitting" if wicket_zone == "hitting" else ("yellow_umpires_call" if wicket_zone == "umpires_call" else "red_missing"),
                "point": {"x": float(result.endPoint.x), "y": float(result.endPoint.y), "t": float(result.endPoint.t or result.stumpImpactSec or 0.0)},
                "confidence": float(result.stumpsHittingConfidence or 0.0),
            },
            "finalDecision": final_decision,
            "finalBasis": final_basis,
            "dimensions": {
                "stumpWidthMeters": 0.2286,
                "stumpHeightMeters": 0.711,
                "ballDiameterMeters": 0.072,
                "pitchLengthMeters": STUMP_TO_STUMP_PITCH_M,
                "creaseToCreaseMeters": CREASE_TO_CREASE_M,
            },
        }
        analytics = {
            "pitchMap": {"xNorm": float(result.bouncePoint.x), "yNorm": float(result.bouncePoint.y), "distanceFromBatterStumpsMeters": float(result.bounceDistanceFromBatterStumpsMeters or pitch_m)},
            "beehive": {"impactXNorm": float(result.endPoint.x), "impactYNorm": float(result.endPoint.y), "stumpsHitting": bool(result.stumpsHitting)},
            "lengthClassification": length_label,
            "lineClassification": pitch_zone,
            "releaseSpeedKph": result.deliverySpeedKph or result.speedKph,
            "postBounceSpeedKph": self._segment_speed_kph(result, after_bounce=True),
            "averageSpeedKph": result.deliverySpeedKph or result.speedKph,
            "viewsAvailable": ["broadcast_overlay", "top_view_data", "side_view_data"],
        }
        return {"decision": decision, "analytics": analytics}

    def _wicket_line_zone(self, result: DeliveryResult, x: float) -> str:
        plane = result.virtualWicketPlane or {}
        center = float(plane.get("xCenter", 0.5))
        half_width = float(plane.get("halfWidth", 0.018))
        rel = (float(x) - center) / max(half_width, 1e-6)
        if rel < -1.35:
            return "outside_off"
        if rel < -0.35:
            return "off_stump_line"
        if rel <= 0.35:
            return "middle_stump_line"
        if rel <= 1.35:
            return "leg_stump_line"
        return "outside_leg"

    def _pitch_coordinate_from_y(self, result: DeliveryResult, y: float) -> float:
        corridor = result.virtualPitchCorridor
        if len(corridor) < 4:
            return 10.0
        y_far = (float(corridor[0]["y"]) + float(corridor[1]["y"])) * 0.5
        y_near = (float(corridor[2]["y"]) + float(corridor[3]["y"])) * 0.5
        alpha = float(np.clip((float(y) - y_far) / max(1e-6, y_near - y_far), 0.0, 1.0))
        return (1.0 - alpha) * 20.1168

    def _segment_speed_kph(self, result: DeliveryResult, after_bounce: bool) -> float | None:
        points = sorted(result.reconstructedTrajectory, key=lambda point: point.t or 0.0)
        if len(points) < 2 or result.bounceSec is None:
            return None
        segment = [point for point in points if (point.t or 0.0) >= result.bounceSec] if after_bounce else [point for point in points if (point.t or 0.0) <= result.bounceSec]
        if len(segment) < 2:
            return None
        t0 = float(segment[0].t or 0.0)
        t1 = float(segment[-1].t or 0.0)
        if t1 <= t0:
            return None
        pixel_distance = sum(
            float(np.hypot(segment[idx].x - segment[idx - 1].x, segment[idx].y - segment[idx - 1].y))
            for idx in range(1, len(segment))
        )
        scale_m = 20.1168 / max(0.35, self._corridor_pixel_length(result))
        return float((pixel_distance * scale_m / (t1 - t0)) * 3.6)

    def _corridor_pixel_length(self, result: DeliveryResult) -> float:
        corridor = result.virtualPitchCorridor
        if len(corridor) < 4:
            return 1.0
        far_x = (float(corridor[0]["x"]) + float(corridor[1]["x"])) * 0.5
        far_y = (float(corridor[0]["y"]) + float(corridor[1]["y"])) * 0.5
        near_x = (float(corridor[2]["x"]) + float(corridor[3]["x"])) * 0.5
        near_y = (float(corridor[2]["y"]) + float(corridor[3]["y"])) * 0.5
        return float(np.hypot(near_x - far_x, near_y - far_y))

    def _pseudo_trajectory_3d(self, result: DeliveryResult) -> list[dict[str, float]]:
        points = sorted(result.reconstructedTrajectory or result.trajectory2D, key=lambda point: point.t or 0.0)
        if not points:
            return []
        release_t = float(result.releaseSec or points[0].t or 0.0)
        bounce_t = float(result.bounceSec or result.bouncePoint.t or release_t)
        impact_t = float(result.stumpImpactSec or result.endPoint.t or points[-1].t or bounce_t)
        output: list[dict[str, float]] = []
        for point in points:
            t = float(point.t or 0.0)
            pitch_depth_m = self._pitch_coordinate_from_y(result, point.y)
            line_m = (point.x - float((result.virtualWicketPlane or {}).get("xCenter", 0.5))) * 4.0
            if t <= bounce_t:
                phase = (t - release_t) / max(1e-6, bounce_t - release_t)
                height_m = max(0.0, 2.05 * (1.0 - phase ** 1.15))
            else:
                phase = (t - bounce_t) / max(1e-6, impact_t - bounce_t)
                height_m = max(0.0, 0.45 * (1.0 - phase) + 0.12 * np.sin(np.pi * min(1.0, phase)))
            output.append({"xMeters": float(line_m), "yMeters": float(height_m), "zMetersFromBatter": float(pitch_depth_m), "t": t})
        return output

    def _write_candidate_timeline_csv(
        self,
        artifact_dir: Path,
        all_candidates: list,
        active_window: dict[str, float | str],
        metadata: DeliveryMetadata,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        start = float(active_window["activeStartSec"])
        end = float(active_window["activeEndSec"])
        path = artifact_dir / "candidate_timeline.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frameIndex",
                    "timeSec",
                    "x",
                    "y",
                    "source",
                    "modelAlias",
                    "modelRole",
                    "classId",
                    "className",
                    "confidence",
                    "bbox",
                    "bboxWidthNorm",
                    "bboxHeightNorm",
                    "modelName",
                    "insideActiveWindow",
                ],
            )
            writer.writeheader()
            for candidate in sorted(all_candidates, key=lambda value: (value.frameIndex, value.source, -value.confidence)):
                time_sec = self._candidate_time(candidate, metadata)
                diagnostics = getattr(candidate, "diagnostics", {}) or {}
                source = str(candidate.source)
                model_name = source.split(":")[0] if ":" in source else source
                writer.writerow(
                    {
                        "frameIndex": candidate.frameIndex,
                        "timeSec": f"{time_sec:.6f}",
                        "x": f"{candidate.x:.6f}",
                        "y": f"{candidate.y:.6f}",
                        "source": source,
                        "modelAlias": candidate.modelAlias,
                        "modelRole": candidate.modelRole,
                        "classId": candidate.classId,
                        "className": candidate.className,
                        "confidence": f"{candidate.confidence:.6f}",
                        "bbox": json.dumps(candidate.bbox),
                        "bboxWidthNorm": diagnostics.get("bboxWidthNorm"),
                        "bboxHeightNorm": diagnostics.get("bboxHeightNorm"),
                        "modelName": model_name,
                        "insideActiveWindow": start <= time_sec <= end,
                    }
                )

    def _write_tracklet_artifacts(
        self,
        artifact_dir: Path,
        tracklet_result: TrackletBuildResult,
        metadata: DeliveryMetadata,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "trackletCount": len(tracklet_result.tracklets),
            "selectedTrackletId": tracklet_result.best_tracklet.id if tracklet_result.best_tracklet else None,
            "tracklets": [],
        }
        lines = ["# Tracklet Debug", "", f"- Tracklets built: `{len(tracklet_result.tracklets)}`"]
        if tracklet_result.best_tracklet is not None:
            lines.append(f"- Selected tracklet: `{tracklet_result.best_tracklet.id}`")
            lines.append(f"- Selected points: `{len(tracklet_result.best_tracklet.candidates)}`")
        lines.append("")
        for tracklet in tracklet_result.tracklets:
            times = [self._candidate_time(candidate, metadata) for candidate in tracklet.candidates]
            payload["tracklets"].append(
                {
                    "id": tracklet.id,
                    "score": tracklet.score,
                    "rejected": tracklet.rejected,
                    "rejectionReasons": tracklet.rejection_reasons,
                    "metrics": tracklet.metrics,
                    "points": [
                        {
                            "frameIndex": candidate.frameIndex,
                            "timeSec": self._candidate_time(candidate, metadata),
                            "x": candidate.x,
                            "y": candidate.y,
                            "source": candidate.source,
                            "confidence": candidate.confidence,
                        }
                        for candidate in tracklet.candidates
                    ],
                    "startSec": min(times) if times else None,
                    "endSec": max(times) if times else None,
                }
            )
            lines.append(f"## Tracklet {tracklet.id}")
            lines.append(f"- Score: `{tracklet.score:.3f}`")
            lines.append(f"- Rejected: `{tracklet.rejected}`")
            lines.append(f"- Reasons: `{'; '.join(tracklet.rejection_reasons) if tracklet.rejection_reasons else 'none'}`")
            lines.append(f"- Metrics: `{json.dumps(tracklet.metrics, sort_keys=True)}`")
            lines.append("")
        (artifact_dir / "tracklets.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (artifact_dir / "tracklet_debug.md").write_text("\n".join(lines), encoding="utf-8")

    def _filter_candidates_to_active_window(self, candidates: list, metadata: DeliveryMetadata, window: dict[str, float | str]) -> list:
        start = float(window["activeStartSec"])
        end = float(window["activeEndSec"])
        times = self._candidate_times(candidates, metadata)
        return [candidate for candidate, candidate_time in zip(candidates, times) if start <= candidate_time <= end]

    def _validate_solution_window(self, solution, window: dict[str, float | str], debug: bool) -> None:
        if solution.bounce_point.t is not None and solution.release_point.t is not None:
            rb = solution.bounce_point.t - solution.release_point.t
            if rb > 2.0:
                raise ValueError(f"release_to_bounce too long: {rb:.3f}s")
        if solution.end_point.t is not None and solution.release_point.t is not None:
            re = solution.end_point.t - solution.release_point.t
            if re > 2.5 and not debug:
                raise ValueError(f"release_to_end too long: {re:.3f}s")
        if float(window["activeWindowDurationSec"]) > 2.5 and not debug:
            raise ValueError("active window duration exceeds limit")

    def _refine_endpoint_with_stumps(
        self,
        solution,
        selected_candidates: list,
        all_candidates: list,
        stump_detections: list,
        metadata: DeliveryMetadata,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, float | str | dict | None]:
        stump_geometry_rois, visual_boxes = self._stable_stump_rois(stump_detections, frame_width, frame_height)
        stump_roi = stump_geometry_rois.get("far") or stump_geometry_rois.get("near")
        virtual_plane = self._virtual_wicket_plane(stump_geometry_rois)
        if not stump_roi or len(selected_candidates) < 3:
            return {
                "stumpRoi": stump_roi,
                "stumpRois": stump_geometry_rois,
                "visualStumpBoxes": visual_boxes,
                "virtualWicketPlane": virtual_plane,
                "stumpImpactPredictedSec": solution.end_point.t,
                "stumpImpactConfidence": 0.0,
                "distanceToStumpRoiPx": None,
                "endpointSource": "ball_detector",
            }

        selected_sorted = sorted(selected_candidates, key=lambda candidate: self._candidate_time(candidate, metadata))
        tail = selected_sorted[-min(5, len(selected_sorted)):]
        times = np.array([self._candidate_time(candidate, metadata) for candidate in tail], dtype=float)
        xs = np.array([candidate.x for candidate in tail], dtype=float)
        ys = np.array([candidate.y for candidate in tail], dtype=float)
        if len(np.unique(times)) < 2:
            return {
                "stumpRoi": stump_roi,
                "stumpRois": stump_geometry_rois,
                "visualStumpBoxes": visual_boxes,
                "virtualWicketPlane": virtual_plane,
                "stumpImpactPredictedSec": solution.end_point.t,
                "stumpImpactConfidence": 0.0,
                "distanceToStumpRoiPx": self._distance_to_roi_px(solution.end_point.x, solution.end_point.y, stump_roi, frame_width, frame_height),
                "endpointSource": "ball_detector",
            }
        vx, x0 = np.polyfit(times, xs, deg=1)
        vy, y0 = np.polyfit(times, ys, deg=1)
        last_time = self._candidate_time(selected_sorted[-1], metadata)
        last_x = selected_sorted[-1].x
        last_y = selected_sorted[-1].y
        current_distance = self._distance_to_roi_px(last_x, last_y, stump_roi, frame_width, frame_height)

        best_late = self._late_endpoint_candidate(
            all_candidates=all_candidates,
            selected_keys={candidate_key(candidate) for candidate in selected_candidates},
            last_time=last_time,
            last_x=last_x,
            last_y=last_y,
            vx=float(vx),
            vy=float(vy),
            stump_roi=stump_roi,
            metadata=metadata,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        projected = self._project_endpoint_to_stumps(
            last_time=last_time,
            last_x=last_x,
            last_y=last_y,
            vx=float(vx),
            vy=float(vy),
            stump_roi=stump_roi,
            current_distance=current_distance,
            frame_width=frame_width,
            frame_height=frame_height,
            min_time=solution.bounce_point.t or last_time,
        )

        chosen = projected
        if best_late and (not projected or best_late["distanceToStumpRoiPx"] <= projected["distanceToStumpRoiPx"] + 35):
            chosen = best_late
        if not chosen:
            return {
                "stumpRoi": stump_roi,
                "stumpRois": stump_geometry_rois,
                "visualStumpBoxes": visual_boxes,
                "virtualWicketPlane": virtual_plane,
                "stumpImpactPredictedSec": solution.end_point.t,
                "stumpImpactConfidence": 0.0,
                "distanceToStumpRoiPx": current_distance,
                "endpointSource": "ball_detector",
            }

        endpoint = Point2D(x=float(chosen["x"]), y=float(chosen["y"]), t=float(chosen["t"]))
        if endpoint.t is not None and solution.end_point.t is not None and endpoint.t > solution.end_point.t:
            solution.end_point = endpoint
            if not solution.trajectory or endpoint.t > solution.trajectory[-1].t:
                solution.trajectory.append(Point3D(x=endpoint.x, y=0.0, z=endpoint.y, t=endpoint.t))
        return {
            "stumpRoi": stump_roi,
            "stumpRois": stump_geometry_rois,
            "visualStumpBoxes": visual_boxes,
            "virtualWicketPlane": virtual_plane,
            "stumpImpactPredictedSec": endpoint.t,
            "stumpImpactConfidence": float(chosen["confidence"]),
            "distanceToStumpRoiPx": float(chosen["distanceToStumpRoiPx"]),
            "endpointSource": str(chosen["endpointSource"]),
        }

    def _stable_stump_rois(self, stump_detections: list, frame_width: int, frame_height: int) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        usable = self._temporal_stump_fusion(stump_detections, conf_floor=0.10)
        if not usable:
            return {}, {}
        centers_y = [((candidate.bbox["y1"] + candidate.bbox["y2"]) * 0.5) for candidate in usable]
        split = float(np.median(centers_y))
        far_group = [candidate for candidate, cy in zip(usable, centers_y) if cy < split]
        near_group = [candidate for candidate, cy in zip(usable, centers_y) if cy >= split]

        def build_roi(group: list, visual: bool) -> dict[str, float] | None:
            if not group:
                return None
            x1 = float(np.median([candidate.bbox["x1"] / frame_width for candidate in group]))
            y1 = float(np.median([candidate.bbox["y1"] / frame_height for candidate in group]))
            x2 = float(np.median([candidate.bbox["x2"] / frame_width for candidate in group]))
            y2 = float(np.median([candidate.bbox["y2"] / frame_height for candidate in group]))
            pad_ratio = 0.004 if visual else 0.12
            pad_x = 0.0 if visual else max(0.015, (x2 - x1) * pad_ratio)
            pad_y = 0.0 if visual else max(0.015, (y2 - y1) * 0.08)
            x1 = max(0.0, x1 - pad_x)
            y1 = max(0.0, y1 - pad_y)
            x2 = min(1.0, x2 + pad_x)
            y2 = min(1.0, y2 + pad_y)
            return {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "centerX": (x1 + x2) / 2,
                "centerY": (y1 + y2) / 2,
                "confidence": float(np.median([candidate.confidence for candidate in group])),
                "detectionsUsed": float(len(group)),
            }

        rois: dict[str, dict[str, float]] = {}
        visual_boxes: dict[str, dict[str, float]] = {}
        far_roi = build_roi(far_group, visual=False)
        near_roi = build_roi(near_group, visual=False)
        far_visual = build_roi(far_group, visual=True)
        near_visual = build_roi(near_group, visual=True)
        if far_roi is not None:
            rois["far"] = far_roi
        if far_visual is not None:
            visual_boxes["far"] = far_visual
        if near_roi is not None:
            rois["near"] = near_roi
        if near_visual is not None:
            visual_boxes["near"] = near_visual
        if not rois:
            fallback = build_roi(usable, visual=False)
            fallback_visual = build_roi(usable, visual=True)
            if fallback is not None:
                rois["near"] = fallback
            if fallback_visual is not None:
                visual_boxes["near"] = fallback_visual
        return rois, visual_boxes

    def _virtual_wicket_plane(self, stump_geometry_rois: dict[str, dict[str, float]]) -> dict[str, float] | None:
        far = stump_geometry_rois.get("far")
        near = stump_geometry_rois.get("near")
        target = far or near
        if not target:
            return None
        width = max(0.01, target["x2"] - target["x1"])
        return {
            "xCenter": float(target["centerX"]),
            "yTop": float(target["y1"]),
            "yBottom": float(target["y2"]),
            "halfWidth": float(width * 0.45),
        }

    def _temporal_stump_fusion(self, stump_detections: list, conf_floor: float = 0.10) -> list:
        candidates = [candidate for candidate in stump_detections if candidate.bbox and candidate.confidence >= conf_floor]
        if not candidates:
            return []
        centers = np.array(
            [
                (
                    (candidate.bbox["x1"] + candidate.bbox["x2"]) * 0.5,
                    (candidate.bbox["y1"] + candidate.bbox["y2"]) * 0.5,
                )
                for candidate in candidates
            ],
            dtype=float,
        )
        reference = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - reference, axis=1)
        tight = np.percentile(distances, 70) if len(distances) > 4 else max(20.0, float(np.max(distances)))
        stable = [candidate for candidate, distance in zip(candidates, distances) if distance <= tight]
        if len(stable) >= 4:
            return stable
        return candidates

    def _late_endpoint_candidate(
        self,
        all_candidates: list,
        selected_keys: set[str],
        last_time: float,
        last_x: float,
        last_y: float,
        vx: float,
        vy: float,
        stump_roi: dict[str, float],
        metadata: DeliveryMetadata,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, float | str] | None:
        best: dict[str, float | str] | None = None
        current_distance = self._distance_to_roi_px(last_x, last_y, stump_roi, frame_width, frame_height)
        for candidate in all_candidates:
            if candidate_key(candidate) in selected_keys:
                continue
            candidate_time = self._candidate_time(candidate, metadata)
            dt = candidate_time - last_time
            if dt <= 0.0 or dt > 1.2:
                continue
            is_ball = candidate.modelRole == "ball_detector" or candidate.source == "cricket_ball_v2"
            is_motion = "optical_flow" in candidate.source
            if not (is_ball or is_motion):
                continue
            dx = candidate.x - last_x
            dy = candidate.y - last_y
            direction_dot = (dx * vx) + (dy * vy)
            if direction_dot <= 0:
                continue
            distance = self._distance_to_roi_px(candidate.x, candidate.y, stump_roi, frame_width, frame_height)
            if distance >= current_distance:
                continue
            source = "ball_detector" if is_ball else "motion_extended_to_stumps"
            score = (current_distance - distance) + candidate.confidence * 50.0 + dt * 6.0
            if best is None or score > float(best["score"]):
                best = {
                    "x": float(candidate.x),
                    "y": float(candidate.y),
                    "t": float(candidate_time),
                    "confidence": float(min(0.95, 0.35 + candidate.confidence * 0.45 + max(0.0, (current_distance - distance) / 160.0))),
                    "distanceToStumpRoiPx": float(distance),
                    "endpointSource": source,
                    "score": float(score),
                }
        return best

    def _project_endpoint_to_stumps(
        self,
        last_time: float,
        last_x: float,
        last_y: float,
        vx: float,
        vy: float,
        stump_roi: dict[str, float],
        current_distance: float,
        frame_width: int,
        frame_height: int,
        min_time: float,
    ) -> dict[str, float | str] | None:
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return None
        candidates: list[tuple[float, float, float]] = []
        for x_edge in [stump_roi["x1"], stump_roi["x2"], stump_roi["centerX"]]:
            if abs(vx) > 1e-6:
                dt = (x_edge - last_x) / vx
                y = last_y + vy * dt
                if dt > 0 and stump_roi["y1"] <= y <= stump_roi["y2"]:
                    candidates.append((dt, x_edge, y))
        for y_edge in [stump_roi["y1"], stump_roi["y2"], stump_roi["centerY"]]:
            if abs(vy) > 1e-6:
                dt = (y_edge - last_y) / vy
                x = last_x + vx * dt
                if dt > 0 and stump_roi["x1"] <= x <= stump_roi["x2"]:
                    candidates.append((dt, x, y_edge))
        if not candidates:
            return None
        dt, x, y = min(candidates, key=lambda item: item[0])
        predicted_time = last_time + dt
        if predicted_time < min_time or dt > 1.2:
            return None
        distance = self._distance_to_roi_px(x, y, stump_roi, frame_width, frame_height)
        if distance > max(8.0, current_distance * 0.25):
            return None
        confidence = min(0.88, max(0.25, 0.55 + min(0.25, current_distance / 300.0) - max(0.0, dt - 0.6) * 0.25))
        return {
            "x": float(x),
            "y": float(y),
            "t": float(predicted_time),
            "confidence": float(confidence),
            "distanceToStumpRoiPx": float(distance),
            "endpointSource": "projected_to_stumps",
            "score": float(100.0 - distance - dt * 10.0),
        }

    def _distance_to_roi_px(
        self,
        x: float,
        y: float,
        roi: dict[str, float],
        frame_width: int,
        frame_height: int,
    ) -> float:
        clamped_x = min(max(x, roi["x1"]), roi["x2"])
        clamped_y = min(max(y, roi["y1"]), roi["y2"])
        return float(np.hypot((x - clamped_x) * frame_width, (y - clamped_y) * frame_height))

    def _apply_calibration_priors(self, metadata: DeliveryMetadata) -> float:
        if metadata.calibrationMode == CalibrationMode.guided_boxes and metadata.guidedBoxes is not None:
            polygon = metadata.guidedBoxes.pitchCorridorNorm
            if len(polygon) >= 4:
                metadata.corridorGeometry.pitchPolygon = [Point2D(x=point.x, y=point.y) for point in polygon[:4]]
            elif metadata.guidedBoxes.nearStumpsBoxNorm and metadata.guidedBoxes.farStumpsBoxNorm:
                near = metadata.guidedBoxes.nearStumpsBoxNorm
                far = metadata.guidedBoxes.farStumpsBoxNorm
                near_left = max(0.0, near.x - 0.22 * near.w)
                near_right = min(1.0, near.x + near.w + 0.22 * near.w)
                far_left = max(0.0, far.x - 0.30 * far.w)
                far_right = min(1.0, far.x + far.w + 0.30 * far.w)
                metadata.corridorGeometry.pitchPolygon = [
                    Point2D(x=far_left, y=far.y),
                    Point2D(x=far_right, y=far.y),
                    Point2D(x=near_right, y=min(1.0, near.y + near.h)),
                    Point2D(x=near_left, y=min(1.0, near.y + near.h)),
                ]
            setup_quality = float(metadata.guidedBoxes.setupQuality or 0.65)
            return max(0.45, min(0.92, 0.45 + setup_quality * 0.45))
        if metadata.calibrationMode == CalibrationMode.ar_world:
            return 0.90
        return 0.35 if len(metadata.corridorGeometry.pitchPolygon) >= 4 else 0.20

    def _infer_video_only_corridor_from_stumps(self, metadata: DeliveryMetadata, stump_detections: list) -> float:
        if not stump_detections:
            return 0.20
        cluster = self._temporal_stump_fusion(stump_detections, conf_floor=0.12)
        if len(cluster) < 3:
            return 0.25
        centers_y = [((candidate.bbox["y1"] + candidate.bbox["y2"]) * 0.5) for candidate in cluster]
        split = float(np.median(centers_y))
        near = [candidate for candidate, cy in zip(cluster, centers_y) if cy >= split]
        far = [candidate for candidate, cy in zip(cluster, centers_y) if cy < split]
        if not far:
            far = sorted(cluster, key=lambda candidate: candidate.bbox["y1"])[: max(1, len(cluster) // 3)]
        if not near:
            near = sorted(cluster, key=lambda candidate: candidate.bbox["y2"], reverse=True)[: max(1, len(cluster) // 2)]
        near_center_x = float(np.median([((candidate.bbox["x1"] + candidate.bbox["x2"]) * 0.5) for candidate in near]))
        far_center_x = float(np.median([((candidate.bbox["x1"] + candidate.bbox["x2"]) * 0.5) for candidate in far]))
        near_y = float(np.median([candidate.bbox["y2"] for candidate in near]))
        far_y = float(np.median([candidate.bbox["y1"] for candidate in far]))
        near_width = float(np.median([candidate.bbox["x2"] - candidate.bbox["x1"] for candidate in near]))
        far_width = float(np.median([candidate.bbox["x2"] - candidate.bbox["x1"] for candidate in far]))
        width = float(metadata.cameraResolution.width if metadata.cameraResolution else metadata.resolution.width if metadata.resolution else 1080)
        height = float(metadata.cameraResolution.height if metadata.cameraResolution else metadata.resolution.height if metadata.resolution else 1920)
        near_center_x /= width
        far_center_x /= width
        near_y = min(0.98, max(0.40, near_y / height))
        far_y = max(0.03, min(near_y - 0.12, far_y / height))
        near_half = max(0.07, min(0.18, (near_width / width) * 1.8))
        far_half = max(0.03, min(0.10, (far_width / width) * 1.5))
        metadata.corridorGeometry.pitchPolygon = [
            Point2D(x=max(0.0, far_center_x - far_half), y=far_y),
            Point2D(x=min(1.0, far_center_x + far_half), y=far_y),
            Point2D(x=min(1.0, near_center_x + near_half), y=near_y),
            Point2D(x=max(0.0, near_center_x - near_half), y=near_y),
        ]
        metadata.corridorGeometry.spine = [Point2D(x=far_center_x, y=far_y), Point2D(x=near_center_x, y=near_y)]
        return 0.42

    def _trajectory_pitch_coords(self, trajectory: list[Point3D], metadata: DeliveryMetadata) -> list[Point3D]:
        polygon = metadata.corridorGeometry.pitchPolygon
        if len(polygon) < 4:
            return []
        far_left, far_right, near_right, near_left = polygon[:4]
        points: list[Point3D] = []
        for point in trajectory:
            best: tuple[float, float, float] | None = None
            for step in range(81):
                fraction = step / 80
                left_x = far_left.x + (near_left.x - far_left.x) * fraction
                left_y = far_left.y + (near_left.y - far_left.y) * fraction
                right_x = far_right.x + (near_right.x - far_right.x) * fraction
                right_y = far_right.y + (near_right.y - far_right.y) * fraction
                dx = right_x - left_x
                dy = right_y - left_y
                denom = max(1e-6, (dx * dx + dy * dy))
                lateral = ((point.x - left_x) * dx + (point.z - left_y) * dy) / denom
                proj_x = left_x + dx * lateral
                proj_y = left_y + dy * lateral
                distance = float(np.hypot(point.x - proj_x, point.z - proj_y))
                if best is None or distance < best[0]:
                    best = (distance, fraction, lateral)
            if best is None:
                continue
            points.append(Point3D(x=float(best[2]), y=0.0, z=float(best[1]), t=point.t))
        return points

    def _detect_gpu(self) -> None:
        try:
            import torch

            self.torch_cuda_available = bool(torch.cuda.is_available())
            self.gpu_available = self.torch_cuda_available
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available else None
        except Exception:
            self.torch_cuda_available = False
            self.gpu_available = False
            self.gpu_name = None

    def _write_debug_artifacts(
        self,
        artifact_dir: Path,
        result: DeliveryResult,
        frames: list[np.ndarray],
        frame_indices: list[int],
        candidates_by_source: dict[str, list],
        active_window: dict[str, float | str],
        tracklet_result: TrackletBuildResult | None = None,
        options: ProcessingOptions | None = None,
        metadata: DeliveryMetadata | None = None,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        payload = {
            "sampledFrameCount": len(frames),
            "sampledFrameIndices": frame_indices,
            "candidateCounts": {source: len(candidates) for source, candidates in candidates_by_source.items()},
            "candidates": {
                source: [candidate.model_dump() for candidate in candidates]
                for source, candidates in candidates_by_source.items()
            },
            "activeWindow": active_window,
        }
        (artifact_dir / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.settings.debug:
            self._write_annotated_frames(
                artifact_dir / "annotated_frames",
                frames,
                frame_indices,
                candidates_by_source,
                result,
                active_window,
                tracklet_result,
                options,
                metadata or DeliveryMetadata(sessionId="debug", deliveryId="debug"),
            )
        self._write_debug_report(
            artifact_dir=artifact_dir,
            result=result,
            frame_indices=frame_indices,
            candidates_by_source=candidates_by_source,
            rejection_reasons=[],
            active_window=active_window,
            metadata=metadata,
        )

    def _write_debug_artifacts_failed_processing(
        self,
        artifact_dir: Path,
        frames: list[np.ndarray],
        frame_indices: list[int],
        candidates_by_source: dict[str, list],
        rejection_reason: str,
        active_window: dict[str, float | str],
        tracklet_result: TrackletBuildResult | None = None,
        options: ProcessingOptions | None = None,
        metadata: DeliveryMetadata | None = None,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "failed",
            "sampledFrameCount": len(frames),
            "sampledFrameIndices": frame_indices,
            "candidateCounts": {source: len(candidates) for source, candidates in candidates_by_source.items()},
            "candidates": {
                source: [candidate.model_dump() for candidate in candidates]
                for source, candidates in candidates_by_source.items()
            },
            "rejectionReasons": [rejection_reason],
            "activeWindow": active_window,
        }
        (artifact_dir / "candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (artifact_dir / "result.json").write_text(
            json.dumps({"status": "failed", "reason": rejection_reason}, indent=2),
            encoding="utf-8",
        )
        if self.settings.debug:
            self._write_annotated_frames(
                artifact_dir / "annotated_frames",
                frames,
                frame_indices,
                candidates_by_source,
                None,
                active_window,
                tracklet_result,
                options,
                metadata or DeliveryMetadata(sessionId="debug", deliveryId="debug"),
            )
        self._write_debug_report(
            artifact_dir=artifact_dir,
            result=None,
            frame_indices=frame_indices,
            candidates_by_source=candidates_by_source,
            rejection_reasons=[rejection_reason],
            active_window=active_window,
            metadata=metadata,
        )

    def _write_annotated_frames(
        self,
        target_dir: Path,
        frames: list[np.ndarray],
        frame_indices: list[int],
        candidates_by_source: dict[str, list],
        result: DeliveryResult | None,
        active_window: dict[str, float | str],
        tracklet_result: TrackletBuildResult | None,
        options: ProcessingOptions | None,
        metadata: DeliveryMetadata,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        merged_candidates = candidates_by_source.get("merged", [])
        accepted_candidates = set(candidate_key(candidate) for candidate in (tracklet_result.selected_candidates if tracklet_result else []))
        rejected_reasons = tracklet_result.rejected_candidate_reasons if tracklet_result else {}
        candidates_by_frame: dict[int, list] = {}
        for candidate in merged_candidates:
            candidates_by_frame.setdefault(candidate.frameIndex, []).append(candidate)
        trajectory_by_frame = self._trajectory_points_by_frame(result, frame_indices) if result else {}
        frame_window = list(zip(frames, frame_indices))
        if options and options.use_ground_truth_window:
            start = float(active_window.get("activeStartSec", 0.0))
            end = float(active_window.get("activeEndSec", 0.0))
            filtered: list[tuple[np.ndarray, int]] = []
            for frame, frame_index in frame_window:
                ts = metadata.frameTimestamps[min(max(0, frame_index), len(metadata.frameTimestamps) - 1)] if metadata.frameTimestamps else frame_index / float(metadata.extras.get("videoFps") or 30.0)
                if start <= ts <= end:
                    filtered.append((frame, frame_index))
            frame_window = filtered
        else:
            frame_window = frame_window[:40]
        for frame, frame_index in frame_window:
            annotated = frame.copy()
            height, width = annotated.shape[:2]
            for candidate in candidates_by_frame.get(frame_index, []):
                center = (int(candidate.x * width), int(candidate.y * height))
                key = candidate_key(candidate)
                is_accepted = key in accepted_candidates
                is_rejected = key in rejected_reasons
                if "yolo" in candidate.source:
                    box_half = 10
                    cv2.rectangle(
                        annotated,
                        (max(0, center[0] - box_half), max(0, center[1] - box_half)),
                        (min(width - 1, center[0] + box_half), min(height - 1, center[1] + box_half)),
                        (0, 255, 0) if is_accepted else (255, 255, 0),
                        2,
                    )
                else:
                    color = (0, 200, 255) if is_accepted else (0, 0, 255)
                    cv2.circle(annotated, center, 8, color, 2)
                label = "accepted" if is_accepted else ("rejected" if is_rejected else "candidate")
                text = f"{candidate.source} {label}"
                cv2.putText(annotated, text, (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            for point in trajectory_by_frame.get(frame_index, []):
                cv2.circle(annotated, (int(point[0] * width), int(point[1] * height)), 5, (255, 0, 0), -1)
            cv2.imwrite(str(target_dir / f"frame_{frame_index:05d}.jpg"), annotated)

    def _trajectory_points_by_frame(
        self,
        result: DeliveryResult,
        frame_indices: list[int],
    ) -> dict[int, list[tuple[float, float]]]:
        if not result.trajectory:
            return {}
        start_time = result.trajectory[0].t
        end_time = result.trajectory[-1].t
        duration = max(1e-6, end_time - start_time)
        max_frame = max(frame_indices) if frame_indices else 0
        mapping: dict[int, list[tuple[float, float]]] = {}
        for point in result.trajectory:
            normalized = (point.t - start_time) / duration
            frame_index = int(round(normalized * max_frame))
            mapping.setdefault(frame_index, []).append((point.x, point.z))
        return mapping

    def _write_debug_report(
        self,
        artifact_dir: Path,
        result: DeliveryResult | None,
        frame_indices: list[int],
        candidates_by_source: dict[str, list],
        rejection_reasons: list[str],
        active_window: dict[str, float | str],
        metadata: DeliveryMetadata | None = None,
    ) -> None:
        report_path = artifact_dir / "debug_report.html"
        annotated_dir = artifact_dir / "annotated_frames"
        annotated_files = sorted(annotated_dir.glob("frame_*.jpg")) if annotated_dir.exists() else []
        source_rows = []
        for source, candidates in candidates_by_source.items():
            source_rows.append(f"<tr><td>{html.escape(source)}</td><td>{len(candidates)}</td></tr>")
        source_table = "\n".join(source_rows) if source_rows else "<tr><td colspan='2'>No candidates</td></tr>"
        gallery = "\n".join(
            [
                f"<div class='thumb'><img src='annotated_frames/{path.name}' alt='{path.name}'/><p>{path.name}</p></div>"
                for path in annotated_files[:80]
            ]
        ) if annotated_files else "<p>No annotated frames generated (enable FUSIONTRACK_DEBUG=true).</p>"

        trajectory_svg = self._trajectory_svg(result)
        landmarks_html = self._landmark_details(result)
        rejection_html = (
            "<ul>" + "".join([f"<li>{html.escape(reason)}</li>" for reason in rejection_reasons]) + "</ul>"
            if rejection_reasons
            else "<p>None</p>"
        )
        timing_histogram = self._timing_histogram_svg(candidates_by_source, active_window, metadata)
        report_path.write_text(
            f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>FusionTrack Debug Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #0f1115; color: #f3f5f7; }}
    h1,h2 {{ margin: 0 0 12px; }}
    .card {{ background: #181c23; border: 1px solid #2b313d; border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td,th {{ border: 1px solid #30384a; padding: 6px; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }}
    .thumb img {{ width: 100%; border-radius: 6px; border: 1px solid #39445a; }}
    .muted {{ color: #93a2bd; }}
  </style>
</head>
<body>
  <h1>FusionTrack Debug Report</h1>
  <div class="card">
    <h2>Job Summary</h2>
    <p class="muted">Sampled frames: {len(frame_indices)}</p>
    <p class="muted">Frame indices: {frame_indices[:24]}{"..." if len(frame_indices) > 24 else ""}</p>
  </div>
  <div class="card">
    <h2>Candidate Sources</h2>
    <table>
      <thead><tr><th>Source</th><th>Count</th></tr></thead>
      <tbody>{source_table}</tbody>
    </table>
  </div>
  <div class="card">
    <h2>Final Trajectory Plot</h2>
    {trajectory_svg}
    {landmarks_html}
    <p class="muted">Active window: {active_window.get("activeStartSec")}s..{active_window.get("activeEndSec")}s ({active_window.get("windowSource")})</p>
  </div>
  <div class="card">
    <h2>Candidate Timing Histogram</h2>
    {timing_histogram}
  </div>
  <div class="card">
    <h2>Rejection Reasons</h2>
    {rejection_html}
  </div>
  <div class="card">
    <h2>Sampled Frames (Annotated)</h2>
    <div class="grid">{gallery}</div>
  </div>
</body>
</html>
""",
            encoding="utf-8",
        )

    def _trajectory_svg(self, result: DeliveryResult | None) -> str:
        if result is None or not result.trajectory:
            return "<p>No trajectory available.</p>"
        points = " ".join([f"{point.x * 600:.1f},{point.z * 320:.1f}" for point in result.trajectory])
        release = f"{result.releasePoint.x * 600:.1f},{result.releasePoint.y * 320:.1f}"
        bounce = f"{result.bouncePoint.x * 600:.1f},{result.bouncePoint.y * 320:.1f}"
        end = f"{result.endPoint.x * 600:.1f},{result.endPoint.y * 320:.1f}"
        return f"""
<svg width="600" height="320" viewBox="0 0 600 320" style="background:#0f141e;border:1px solid #2d3544;border-radius:6px">
  <polyline fill="none" stroke="#ff4d4f" stroke-width="3" points="{points}" />
  <circle cx="{release.split(',')[0]}" cy="{release.split(',')[1]}" r="5" fill="#00d46a" />
  <circle cx="{bounce.split(',')[0]}" cy="{bounce.split(',')[1]}" r="6" fill="#ffb703" />
  <circle cx="{end.split(',')[0]}" cy="{end.split(',')[1]}" r="5" fill="#8ecae6" />
</svg>
"""

    def _landmark_details(self, result: DeliveryResult | None) -> str:
        if result is None:
            return "<p>No landmarks available.</p>"
        return (
            f"<p>Release: ({result.releasePoint.x:.3f}, {result.releasePoint.y:.3f}, t={result.releasePoint.t})</p>"
            f"<p>Bounce: ({result.bouncePoint.x:.3f}, {result.bouncePoint.y:.3f}, t={result.bouncePoint.t})</p>"
            f"<p>End: ({result.endPoint.x:.3f}, {result.endPoint.y:.3f}, t={result.endPoint.t})</p>"
            f"<p>Speed: {result.speedKph} kph | Confidence: {result.confidence:.3f}</p>"
        )

    def _timing_histogram_svg(
        self,
        candidates_by_source: dict[str, list],
        active_window: dict[str, float | str],
        metadata: DeliveryMetadata | None = None,
    ) -> str:
        merged = candidates_by_source.get("merged", [])
        if not merged:
            return "<p>No candidate timing data available.</p>"
        fps = float((metadata.extras or {}).get("videoFps") or 30.0) if metadata else 30.0
        time_values = sorted(
            [
                float(candidate.timestampSec)
                if candidate.timestampSec is not None
                else (
                    float(metadata.frameTimestamps[min(max(0, candidate.frameIndex), len(metadata.frameTimestamps) - 1)])
                    if metadata and metadata.frameTimestamps
                    else candidate.frameIndex / fps
                )
                for candidate in merged
            ]
        )
        min_t = min(time_values)
        max_t = max(time_values)
        span = max(1e-6, max_t - min_t)
        bins = 32
        counts = [0 for _ in range(bins)]
        yolo_counts = [0 for _ in range(bins)]
        motion_counts = [0 for _ in range(bins)]
        for candidate in merged:
            candidate_time = (
                float(candidate.timestampSec)
                if candidate.timestampSec is not None
                else (
                    float(metadata.frameTimestamps[min(max(0, candidate.frameIndex), len(metadata.frameTimestamps) - 1)])
                    if metadata and metadata.frameTimestamps
                    else candidate.frameIndex / fps
                )
            )
            index = int(((candidate_time - min_t) / span) * (bins - 1))
            counts[index] += 1
            if "yolo" in candidate.source:
                yolo_counts[index] += 1
            if "optical_flow" in candidate.source:
                motion_counts[index] += 1
        max_count = max(1, max(counts))
        bars = []
        bars_yolo = []
        bars_motion = []
        width = 600 / bins
        for i in range(bins):
            h = (counts[i] / max_count) * 120
            hy = (yolo_counts[i] / max_count) * 120
            hm = (motion_counts[i] / max_count) * 120
            x = i * width
            bars.append(f"<rect x='{x:.1f}' y='{140-h:.1f}' width='{width-2:.1f}' height='{h:.1f}' fill='#6c7a96' opacity='0.45'/>")
            bars_yolo.append(f"<rect x='{x:.1f}' y='{140-hy:.1f}' width='{(width-3)/2:.1f}' height='{hy:.1f}' fill='#4cc9f0' opacity='0.9'/>")
            bars_motion.append(f"<rect x='{x + (width-3)/2:.1f}' y='{140-hm:.1f}' width='{(width-3)/2:.1f}' height='{hm:.1f}' fill='#f72585' opacity='0.8'/>")
        active_start = float(active_window.get("activeStartSec", 0))
        active_end = float(active_window.get("activeEndSec", 0))
        start_x = max(0.0, min(600.0, ((active_start - min_t) / span) * 600))
        end_x = max(0.0, min(600.0, ((active_end - min_t) / span) * 600))
        overlay = f"<rect x='{start_x:.1f}' y='0' width='{max(1.0, end_x-start_x):.1f}' height='150' fill='#00d46a' opacity='0.08'/>"
        return (
            "<svg width='600' height='150' viewBox='0 0 600 150' style='background:#0f141e;border:1px solid #2d3544;border-radius:6px'>"
            + overlay
            + "".join(bars)
            + "".join(bars_yolo)
            + "".join(bars_motion)
            + "<line x1='0' y1='140' x2='600' y2='140' stroke='#50607f' stroke-width='1'/>"
            + "</svg>"
            + "<p class='muted'>Grey=all candidates, Cyan=YOLO clusters, Pink=motion clusters, Green overlay=active window.</p>"
        )
