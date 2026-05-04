# FusionTrack Cloud

Standalone GPU inference service for FusionTrack (this repo is **not** the iOS app; pair it with Crickmatic or your own client).

FastAPI cloud-enhanced delivery reconstruction for FusionTrack.

This service does **not** replace the iOS local reconstruction engine. It is a GPU-backed enhancement path:

1. iPhone detects that a delivery is starting and captures a clean short clip.
2. iPhone keeps live work lightweight: AR pitch calibration, person/run-up trigger, Watch prior, and rolling buffer only.
3. iPhone uploads the clip + calibrated pitch/camera metadata to this service.
4. Cloud runs stronger cricket-ball/stump reconstruction.
5. Cloud uses iPhone pitch geometry for physical speed, line, and length instead of guessing real-world distance from pixels alone.

## Architecture

```text
POST /v1/deliveries
  clip/video or zipped frame package
  metadata JSON

Pipeline:
  decode frames
  sample densely around watch release cues
  cricket_ball_v2 ball candidates
  cricket_stumps_v1 stump endpoint geometry
  optical-flow motion candidates
  optional future TrackNet adapter
  merge/filter timeline
  trajectory solver
  calibrated physical metrics + confidence result
```

Core rule:

```text
Models propose candidates. The solver decides the final cricket-valid trajectory.
```

## Endpoints

### `GET /health`

Returns service status, loaded models, missing model files, Torch CUDA status, GPU name, Ultralytics version, OpenCV version, selected device, and debug mode.

### `POST /process-delivery`

Synchronous production endpoint.

Multipart form:

- `upload`: video file or zipped frame package
- `metadata`: JSON string matching `DeliveryMetadata`
- `debug`: optional bool (default `false`)

Returns compact `DeliveryResult` JSON.

### `POST /process-delivery-debug`

Synchronous debug endpoint. Same as `/process-delivery` with debug artifacts always written.

### `POST /v1/deliveries`

Multipart form:

- `upload`: video file or zipped frame package
- `metadata`: JSON string matching `DeliveryMetadata`

Returns:

```json
{ "jobId": "...", "status": "queued" }
```

### `GET /v1/deliveries/{jobId}`

Returns `queued`, `processing`, `complete`, or `failed`. Complete responses include release/bounce/end, trajectory, speed, line/length, confidence, and debug metadata.

### `GET /models`

Returns loaded models and the default production config (`ball_v2_plus_stumps`).

### `POST /v1/runsync`

Synchronous JSON for **Speed Studio** (and similar clients). Body shape:

```json
{ "input": { "metadata": { ... }, "videoBase64": "..." } }
```

Response: `{ "status": "COMPLETED", "id": "...", "output": { ... } }`. When `renderConsumerOverlay` is true (default), `output.consumerOverlayVideoBase64` is the **time-synced** consumer overlay MP4 (same style as `consumer_overlay_sync.mp4` from local runs). Optional `output.consumerOverlayKind` is `"sync"` or `"full"`.

## Local Run

```bash
cd .  # repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

For local model loading, place weights in `models/` or keep them at the repo root.

## Configuration

Environment variables:

```bash
FUSIONTRACK_MODELS=yolo26n,yolo26s,yolov8s
FUSIONTRACK_CONF=0.20
FUSIONTRACK_DEVICE=cuda:0
FUSIONTRACK_DEBUG=true
```

Notes:

- `FUSIONTRACK_MODELS` accepts comma-separated names with or without `.pt`.
- If no requested YOLO model files can be loaded, `/health` returns `status=models_missing` and delivery processing returns `503`.
- `FUSIONTRACK_DEBUG=true` writes annotated candidate frames under each job/output artifact directory.

## Local Smoke Test

Add a small `.mp4`, `.mov`, or frame `.zip` under `samples/`, then run:

```bash
python scripts/test_process_local.py
```

The script calls the orchestrator directly without FastAPI, prints candidate counts/trajectory/speed/confidence, and writes debug artifacts under `outputs/`.

## Docker

```bash
cd .  # repo root
docker build -t fusiontrack-cloud .
docker run --gpus all -p 8000:8000 fusiontrack-cloud
```

Validate:

```bash
curl http://localhost:8000/health
```

Upload a sample:

```bash
curl -X POST http://localhost:8000/v1/deliveries \
  -F "upload=@samples/sample.mp4" \
  -F 'metadata={
    "sessionId":"demo-session",
    "deliveryId":"demo-delivery",
    "frameTimestamps":[],
    "pitchCalibration":{"pitchLengthMeters":20.12,"pitchWidthMeters":3.05},
    "corridorGeometry":{"pitchPolygon":[{"x":0.44,"y":0.18},{"x":0.56,"y":0.18},{"x":0.62,"y":0.88},{"x":0.38,"y":0.88}],"spine":[{"x":0.5,"y":0.18},{"x":0.5,"y":0.88}]},
    "cameraResolution":{"width":1280,"height":720},
    "watchEvents":{},
    "bowlingProfile":{"hand":"unknown"},
    "localYoloCandidates":[]
  }'
```

The Docker image uses a CUDA runtime base (CUDA 12.4). On a rented GPU pod (for example RTX 4090 / A40), 24 GB VRAM is comfortable; H100 is not required.

## FusionTrack engine pod (RunPod GPU pod)

This repo targets a **dedicated pod** (for example `fusiontrack_engine_pod`) with GPU + PyTorch (2.4.x is fine), not RunPod Serverless. The pod runs the FastAPI app and returns analysis plus the sync consumer overlay as base64 from `POST /v1/runsync`.

### One-time setup on the pod (SSH or web terminal)

1. **Open a shell** on the pod (RunPod SSH, TCP SSH, or web terminal).

2. **Clone the repo** (after you push to GitHub):

   ```bash
   cd ~
   git clone https://github.com/<your-org>/FusionTrackCloud.git
   cd FusionTrackCloud
   ```

3. **Python environment** (use the pod’s Python 3 if already good; otherwise create a venv):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Model weights** — copy your YOLO checkpoints into `models/` as described in `models/README.md`.

5. **Configuration** (optional):

   ```bash
   export FUSIONTRACK_DEVICE=cuda:0
   # export FUSIONTRACK_MODELS=...   # see README “Configuration”
   ```

6. **Start the API** (bind all interfaces so RunPod’s HTTP proxy can reach you):

   ```bash
   cd ~/FusionTrackCloud   # or your clone path
   source .venv/bin/activate
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   - RunPod **HTTP services** often expose a URL like `https://<pod-id>-8000.proxy.runpod.net` → your process must listen on **port 8000** inside the pod (or change the exposed port in the RunPod UI to match your `--port`).
   - **Jupyter on 8888** does not run FusionTrack; keep Jupyter if you like, but Speed Studio should call **8000** (or whichever port you map to `uvicorn`).

7. **Smoke test** — on the pod (second terminal) or from your laptop:

   ```bash
   curl -sS "http://127.0.0.1:8000/health"
   # or: curl -sS "https://<pod-id>-8000.proxy.runpod.net/health"
   ```

### Notes

- **No `handler.py` / serverless**: inference is only via **FastAPI** (`uvicorn app.main:app`).
- **Artifacts**: `run_speed_studio_job` writes temp job dirs then deletes them; the client receives metrics and `consumerOverlayVideoBase64`, not files on disk.
- **PyTorch 2.4.0 template pods**: you do not have to use the repo `Dockerfile` if the template already has CUDA + Python; `pip install -r requirements.txt` on top is enough as long as versions resolve.

## iOS Integration Notes

Keep local processing as the fallback:

- Local iPhone engine handles AR calibration, Watch timing, clip buffering, and quick local estimate.
- Upload delivery clip after local capture.
- Show local estimate first.
- Show state: `Enhanced cloud processing`.
- Replace/upgrade UI when cloud result returns.
- If cloud fails, show `Cloud unavailable, local estimate used`.

Cloud payload should include:

- compressed video or zipped frames
- frame timestamps
- pitch calibration metadata and stump-to-stump world geometry
- AR geometry/corridor data
- camera intrinsics/pose when available
- Watch release/action timestamps
- bowling profile
- capture trigger/pre-roll timing
- local trigger candidates when available

### Exact iOS Metadata Contract

The iPhone should send `metadata` as JSON in the multipart form:

```json
{
  "schemaVersion": "fusiontrack.captureMetadata.v1",
  "sessionId": "ios-session-uuid",
  "deliveryId": "delivery-uuid",
  "fps": 56.73,
  "resolution": { "width": 1080, "height": 1920 },
  "pitch": {
    "lengthMeters": 20.12,
    "batterEndStumpsWorld": [0.0, 0.0, 0.0],
    "bowlerEndStumpsWorld": [0.0, 0.0, 20.12],
    "creasePositionsMeters": {
      "battingCreaseFromBatterStumps": 1.22,
      "bowlingCreaseFromBowlerStumps": 1.22
    },
    "corridorWorld": [
      [-1.525, 0.0, 0.0],
      [1.525, 0.0, 0.0],
      [1.525, 0.0, 20.12],
      [-1.525, 0.0, 20.12]
    ]
  },
  "camera": {
    "intrinsics": [[1200.0, 0.0, 540.0], [0.0, 1200.0, 960.0], [0.0, 0.0, 1.0]],
    "extrinsicsByFrame": null,
    "staticPose": null
  },
  "watch": {
    "releaseTimestampSec": null,
    "confidence": null
  },
  "capture": {
    "triggerTimestampSec": null,
    "preRollSec": 1.0,
    "clipStartWallTime": null
  },
  "frameTimestamps": [0.0, 0.033, 0.066],
  "pitchCalibration": {
    "pitchLengthMeters": 20.12,
    "pitchWidthMeters": 3.05,
    "anchors": {
      "nearBottomCenter": { "x": 0.50, "y": 0.86 },
      "farBottomCenter": { "x": 0.50, "y": 0.18 }
    },
    "worldTransform": null
  },
  "corridorGeometry": {
    "pitchPolygon": [
      { "x": 0.44, "y": 0.18 },
      { "x": 0.56, "y": 0.18 },
      { "x": 0.62, "y": 0.88 },
      { "x": 0.38, "y": 0.88 }
    ],
    "spine": [
      { "x": 0.50, "y": 0.18 },
      { "x": 0.50, "y": 0.88 }
    ],
    "releaseZone": { "x": 0.35, "y": 0.08, "width": 0.30, "height": 0.24 }
  },
  "cameraResolution": { "width": 1280, "height": 720 },
  "watchEvents": {
    "runUpStarted": null,
    "armActionStarted": null,
    "releaseWindowLikely": null,
    "releaseDetected": null,
    "deliveryEnded": null
  },
  "bowlingProfile": {
    "bowlerId": null,
    "hand": "unknown",
    "style": null
  },
  "localYoloCandidates": [
    {
      "frameIndex": 12,
      "x": 0.51,
      "y": 0.42,
      "confidence": 0.64,
      "source": "ios_yolo26n"
    }
  ],
  "extras": {}
}
```

Cloud result maps directly to a reconstructed delivery:

- release point and `releaseSec`
- bounce point and `bounceSec`
- end/stump-impact point and `stumpImpactSec`
- trajectory points
- calibrated `speedKph`, `speedMph`, and `speedConfidence`
- `bounceDistanceFromBatterStumpsMeters`
- `bounceDistanceFromBowlerStumpsMeters`
- `lineMetersFromCenter` / `lineMetersFromOffStump`
- calibrated `line`, `length`, and `lengthCategory`
- `trajectoryConfidence` and `calibrationConfidence`
- swing/seam/spin proxies
- confidence
- debug metadata

Every result includes:

- `schemaVersion: "fusiontrack.deliveryResult.v1"`
- `processingMode: "cloud"`
- `modelsUsed`
- `processingTimeMs`

## Phases

## Calibration Modes

Every delivery includes:

```json
{
  "calibrationMode": "video_only | guided_boxes | ar_world",
  "confidence": 0.0
}
```

- `video_only`: baseline cloud reconstruction with inferred corridor and lower physical confidence.
- `guided_boxes`: guided stump/corridor priors from setup UI, medium-high confidence.
- `ar_world`: AR world anchors/intrinsics metadata, highest physical confidence.

### iPhone Capture Trigger

The iPhone should only decide when to capture. It should not run live cricket-ball tracking as a production dependency.

- Use Apple Vision person detection or a lightweight local person model to detect a bowler-side human entering the run-up zone.
- Ignore batter-side person detections near the batter-end stump ROI.
- Arm the rolling buffer when bowler-end run-up motion/person presence is persistent.
- Capture a configurable clip, default 5 seconds, with pre-roll from `DeliveryClipBuffer`.
- Upload the clip and `fusiontrack.captureMetadata.v1` immediately after capture while leaving the camera UI active.

### Training Roles

- `cricket_ball_v2`: ball-only cloud detector. Label release, post-release, bounce, after-bounce, and near-stump frames, including blur, low light, nets, different balls, and hard negatives such as shoes, cones, markings, hands, and stump tops.
- `cricket_stumps_v1`: wicket/stump geometry detector for endpoint projection and iPhone setup validation.
- Local person detector: capture trigger only. Do not use it as a ball tracker.

### Benchmark Modes

- `latest_video_only`: clip plus video/corridor metadata only.
- `latest_calibrated`: clip plus manual or AR pitch/camera metadata.
- `latest_oracle`: evaluation mode with GT active window; GT timings remain evaluation-only and are not solver inputs.

Phase 1:

- YOLO26n/YOLO26s via Ultralytics.
- COCO sports ball class filter (`32`).
- Optical-flow motion candidates.
- Solver-based trajectory reconstruction.

Phase 2:

- Add V1/V2 custom cricket models as additional candidate sources.
- Compare per-clip candidate contribution.

Phase 3:

- Add TrackNetV3 adapter as a temporal heatmap candidate source.
- Do not make TrackNet required until the pipeline is stable.
