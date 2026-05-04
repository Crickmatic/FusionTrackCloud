from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    complete = "complete"
    failed = "failed"


class CalibrationMode(str, Enum):
    video_only = "video_only"
    guided_boxes = "guided_boxes"
    ar_world = "ar_world"


class Point2D(BaseModel):
    x: float
    y: float
    t: float | None = None


class Point3D(BaseModel):
    x: float
    y: float
    z: float
    t: float


class Resolution(BaseModel):
    width: int
    height: int


class PitchCaptureMetadata(BaseModel):
    lengthMeters: float = 20.12
    batterEndStumpsWorld: list[float] | None = None
    bowlerEndStumpsWorld: list[float] | None = None
    creasePositionsMeters: dict[str, Any] | None = None
    corridorWorld: list[list[float]] | None = None


class CameraCaptureMetadata(BaseModel):
    intrinsics: list[list[float]] | None = None
    extrinsicsByFrame: dict[str, list[list[float]]] | None = None
    staticPose: list[list[float]] | None = None


class WatchCaptureMetadata(BaseModel):
    releaseTimestampSec: float | None = None
    confidence: float | None = None


class CaptureTimingMetadata(BaseModel):
    triggerTimestampSec: float | None = None
    preRollSec: float | None = None
    clipStartWallTime: str | None = None


class NormBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class GuidedBoxesMetadata(BaseModel):
    nearStumpsBoxNorm: NormBox | None = None
    farStumpsBoxNorm: NormBox | None = None
    pitchCorridorNorm: list[Point2D] = Field(default_factory=list)
    setupQuality: float | None = None
    nearStumpsDetected: bool | None = None
    farStumpsDetected: bool | None = None


class Candidate(BaseModel):
    frameIndex: int
    x: float
    y: float
    confidence: float
    source: str
    score: float | None = None
    timestampSec: float | None = None
    modelAlias: str | None = None
    modelRole: str | None = None
    classId: int | None = None
    className: str | None = None
    bbox: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class WatchEvents(BaseModel):
    runUpStarted: float | None = None
    armActionStarted: float | None = None
    releaseWindowLikely: float | None = None
    releaseDetected: float | None = None
    deliveryEnded: float | None = None


class CameraResolution(BaseModel):
    width: int
    height: int


class PitchCalibration(BaseModel):
    pitchLengthMeters: float = 20.12
    pitchWidthMeters: float = 3.05
    anchors: dict[str, Point2D] = Field(default_factory=dict)
    worldTransform: list[list[float]] | None = None


class CorridorGeometry(BaseModel):
    pitchPolygon: list[Point2D] = Field(default_factory=list)
    spine: list[Point2D] = Field(default_factory=list)
    releaseZone: dict[str, float] | None = None


class BowlingProfile(BaseModel):
    bowlerId: str | None = None
    hand: Literal["right", "left", "unknown"] = "unknown"
    style: str | None = None


class DeliveryMetadata(BaseModel):
    schemaVersion: str = "fusiontrack.captureMetadata.v1"
    sessionId: str
    deliveryId: str
    calibrationMode: CalibrationMode = CalibrationMode.video_only
    confidence: float = 0.0
    fps: float | None = None
    resolution: Resolution | None = None
    guidedBoxes: GuidedBoxesMetadata | None = None
    pitch: PitchCaptureMetadata | None = None
    camera: CameraCaptureMetadata | None = None
    watch: WatchCaptureMetadata | None = None
    capture: CaptureTimingMetadata | None = None
    frameTimestamps: list[float] = Field(default_factory=list)
    pitchCalibration: PitchCalibration = Field(default_factory=PitchCalibration)
    corridorGeometry: CorridorGeometry = Field(default_factory=CorridorGeometry)
    cameraResolution: CameraResolution | None = None
    watchEvents: WatchEvents = Field(default_factory=WatchEvents)
    bowlingProfile: BowlingProfile = Field(default_factory=BowlingProfile)
    localYoloCandidates: list[Candidate] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


class DeliveryDebug(BaseModel):
    modelsUsed: list[str] = Field(default_factory=list)
    candidateCounts: dict[str, int] = Field(default_factory=dict)
    processingTimeMs: float = 0
    fps: float | None = None
    stageTimingsMs: dict[str, float] = Field(default_factory=dict)
    reconstructionStats: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DeliveryResult(BaseModel):
    schemaVersion: str = "fusiontrack.deliveryResult.v1"
    processingMode: str = "cloud"
    calibrationMode: CalibrationMode = CalibrationMode.video_only
    modelsUsed: list[str] = Field(default_factory=list)
    processingTimeMs: float = 0
    activeStartSec: float | None = None
    activeEndSec: float | None = None
    activeWindowDurationSec: float | None = None
    windowSource: str | None = None
    releasePoint: Point2D
    bouncePoint: Point2D
    endPoint: Point2D
    trajectory: list[Point3D]
    trajectory2D: list[Point2D] = Field(default_factory=list)
    trajectoryPitchCoords: list[Point3D] = Field(default_factory=list)
    reconstructedTrajectory: list[Point2D] = Field(default_factory=list)
    finalTrajectory: list[dict[str, Any]] = Field(default_factory=list)
    trajectorySegments: dict[str, str] = Field(default_factory=dict)
    eventSources: dict[str, str] = Field(default_factory=dict)
    releaseSec: float | None = None
    bounceSec: float | None = None
    stumpImpactSec: float | None = None
    releaseFrame: int | None = None
    actualBounceFrame: int | None = None
    projectedBounceFrame: int | None = None
    lastDetectedBallFrame: int | None = None
    stumpImpactFrame: int | None = None
    projectedImpactFrame: int | None = None
    impactSource: str | None = None
    stumpsHitting: bool | None = None
    stumpsHittingConfidence: float | None = None
    pitchedInLine: bool | None = None
    pitchedInLineConfidence: float | None = None
    virtualPitchCorridor: list[dict[str, float]] = Field(default_factory=list)
    drsDecision: dict[str, Any] = Field(default_factory=dict)
    drsAnalytics: dict[str, Any] = Field(default_factory=dict)
    trajectory3D: list[dict[str, float]] = Field(default_factory=list)
    speedKph: float | None = None
    speedMph: float | None = None
    deliverySpeedKph: float | None = None
    deliverySpeedMph: float | None = None
    speedSource: str | None = None
    speedConfidence: float | None = None
    bounceDistanceFromBatterStumpsMeters: float | None = None
    bounceDistanceFromBowlerStumpsMeters: float | None = None
    lineMetersFromCenter: float | None = None
    lineMetersFromOffStump: float | None = None
    lengthCategory: str | None = None
    trajectoryConfidence: float | None = None
    calibrationConfidence: float | None = None
    corridorConfidence: float | None = None
    line: str
    length: str
    swingAmount: float = 0
    seamDeviation: float | None = None
    spinEffectScore: float | None = None
    confidence: float
    fps: float | None = None
    stumpDetections: list[Candidate] = Field(default_factory=list)
    bestStumpDetection: Candidate | None = None
    stumpRoi: dict[str, float] | None = None
    stumpRois: dict[str, dict[str, float]] = Field(default_factory=dict)
    visualStumpBoxes: dict[str, dict[str, float]] = Field(default_factory=dict)
    virtualWicketPlane: dict[str, float] | None = None
    stumpImpactPredictedSec: float | None = None
    stumpImpactConfidence: float | None = None
    distanceToStumpRoiPx: float | None = None
    endpointSource: str | None = None
    annotatedVideoPath: str | None = None
    consumerAnnotatedVideoPath: str | None = None
    consumerSyncAnnotatedVideoPath: str | None = None
    renderFramesCount: int | None = None
    consumerRenderFramesCount: int | None = None
    consumerSyncRenderFramesCount: int | None = None
    outputVideoSizeMb: float | None = None
    consumerOutputVideoSizeMb: float | None = None
    consumerSyncOutputVideoSizeMb: float | None = None
    debug: DeliveryDebug


class DeliveryJobCreated(BaseModel):
    jobId: str
    status: JobStatus


class DeliveryJobStatus(BaseModel):
    jobId: str
    status: JobStatus
    result: DeliveryResult | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    loadedModels: list[str]
    missingModelFiles: list[str]
    gpuAvailable: bool
    torchCudaAvailable: bool
    gpuName: str | None = None
    ultralyticsVersion: str | None = None
    opencvVersion: str | None = None
    device: str
    debug: bool
    warmupStatus: str = "unknown"
    warmupDevice: str | None = None
    warmupNotes: list[str] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    defaultProductionConfig: str
    modelsUsedByDefault: list[str]
    loadedModels: list[str]
    modelRoles: dict[str, str] = Field(default_factory=dict)
    modelPaths: dict[str, str] = Field(default_factory=dict)
    modelLoaded: dict[str, bool] = Field(default_factory=dict)
    warmupStatus: str = "unknown"
    warmupDevice: str | None = None
    warmupNotes: list[str] = Field(default_factory=list)


class DebugProcessResponse(BaseModel):
    result: DeliveryResult
    debugFolder: str
    debugManifest: list[str] = Field(default_factory=list)
    annotatedVideoPath: str | None = None
    consumerAnnotatedVideoPath: str | None = None
    annotatedFramesPath: str | None = None
    summaryPath: str | None = None
