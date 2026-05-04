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
# Production: require this on /v1/runsync, /process-delivery*, /v1/deliveries* (see below)
FUSIONTRACK_API_KEY=<long-random-secret>
```

Notes:

- `FUSIONTRACK_MODELS` accepts comma-separated names with or without `.pt`.
- If no requested YOLO model files can be loaded, `/health` returns `status=models_missing` and delivery processing returns `503`.
- `FUSIONTRACK_DEBUG=true` writes annotated candidate frames under each job/output artifact directory.
- **`FUSIONTRACK_API_KEY`**: when set, clients must send **`Authorization: Bearer <key>`** or **`X-Api-Key: <key>`** on protected routes. **`/health`** and **`/models`** stay open for probes.

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

## FusionTrack engine (GPU cloud: Vast.ai, RunPod, etc.)

The service is **FastAPI + PyTorch** on a GPU VM. Same code path everywhere; only **networking and disk** change between providers.

### Vast.ai (PyTorch template, RTX 4090, etc.)

Vast maps **random public ports** to fixed **container ports** (see **IP & Port Info** on the instance). Typical PyTorch/Jupyter templates expose **8080** for Jupyter in the browser, **not** 8000 for FusionTrack unless you add a mapping.

#### Disk when creating the instance

Use **`--disk` ≥ 64** (GB) for a comfortable install (`torch` + `ultralytics` + venv + models + temp). **`--disk 16` is too small** and installs will fail or fill the root filesystem.

#### One-time setup (SSH into the instance)

Use **Connect → SSH** (or direct TCP). Example: public **`213.181.123.59`**, SSH **`40899` → 22**:

```bash
ssh -p 40899 root@213.181.123.59 -i ~/.ssh/<your_key>
```

If SSH **asks for a password**, key auth is not wired: add your **public** SSH key in **Vast account → SSH keys** (and/or the instance template), then use **`-i`** to the matching private key. Password login is insecure for automation; prefer keys only.

On the instance (paths often under `/workspace`):

```bash
cd /workspace   # or $DATA_DIRECTORY if set
git clone https://github.com/<your-org>/FusionTrackCloud.git
cd FusionTrackCloud

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Put YOLO weights in `models/` per `models/README.md`, then:

```bash
export FUSIONTRACK_DEVICE=cuda:0
chmod +x scripts/run_engine.sh   # once
./scripts/run_engine.sh
```

`run_engine.sh` binds **`0.0.0.0:8000`** (override with `FUSIONTRACK_PORT`) and sets **`YOLO_CONFIG_DIR`** under the repo so Ultralytics is writable.

#### Expose port **8000** so clients can call the API (no SSH required)

**Recommended — Vast “open port” / port mapping**

1. Ensure **`./scripts/run_engine.sh`** (or `uvicorn ... --port 8000`) is running **inside** the instance.
2. In the Vast UI for that instance, open **Edit** / **Connect** / **Ports** (wording varies) and **add a published port** for **container TCP `8000`**.
3. After it appears under **IP & Port Info**, you will see e.g. **`213.181.123.59:40555 -> 8000/tcp`**.

Then any client on the internet can reach:

- `http://213.181.123.59:40555/health`
- `http://213.181.123.59:40555/v1/runsync`

Use **HTTPS** in production (reverse proxy, Cloudflare Tunnel, or TLS-terminating load balancer in front of that TCP port). The FastAPI app itself speaks plain HTTP.

**Jupyter** (e.g. **`40196 -> 8080`**) is separate from FusionTrack on **8000**; both can run at once.

**Optional — SSH local port forward (dev only)**

If you cannot publish 8000 yet, from a machine with working key-based SSH:

```bash
ssh -p 40899 root@213.181.123.59 -i ~/.ssh/<your_key> -L 8000:127.0.0.1:8000 -N
```

Then `curl http://127.0.0.1:8000/health` on that machine forwards to the instance.

#### Securing who can call the engine

1. **Set a secret on the GPU box** (long random string):

   ```bash
   export FUSIONTRACK_API_KEY='paste-a-long-random-secret-here'
   ./scripts/run_engine.sh
   ```

2. **Clients** send either header on each request to **`/v1/runsync`**, **`/process-delivery`**, **`/process-delivery-debug`**, **`/v1/deliveries`**:

   - `Authorization: Bearer <same-secret>`  
   - or `X-Api-Key: <same-secret>`

   Example:

   ```bash
   curl -sS http://PUBLIC_IP:MAPPED_PORT/health
   curl -sS -X POST http://PUBLIC_IP:MAPPED_PORT/v1/runsync \
     -H "Authorization: Bearer $FUSIONTRACK_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"input":{"metadata":{...},"videoBase64":"..."}}'
   ```

3. **Best practice for iOS + webapp**: do **not** put the GPU secret inside the iPhone app if you can avoid it. Run a small **web backend** (your server) that authenticates users (session, JWT, Sign in with Apple, etc.), then **only the backend** calls Vast with `FUSIONTRACK_API_KEY`. The iOS app calls **your** HTTPS API; your server forwards to the GPU URL. That way a leaked build does not leak GPU access.

4. **Defense in depth**: rate-limit at your web proxy, cap upload size (`FUSIONTRACK_MAX_UPLOAD_MB`), rotate `FUSIONTRACK_API_KEY` when staff change, and restrict Vast firewall / allowlist IPs if Vast offers it for your tier.

### RunPod (optional)

Same clone → venv → `models/` → `./scripts/run_engine.sh` or `uvicorn app.main:app --host 0.0.0.0 --port 8000`. You must expose **HTTP port 8000** in the pod UI (or use SSH **`-L 8000:127.0.0.1:8000`**). URL pattern: `https://<pod-id>-8000.proxy.runpod.net`.

### Notes (all providers)

- **No serverless handler**: only **FastAPI** (`uvicorn app.main:app`).
- **`POST /v1/runsync`**: response includes metrics and **`consumerOverlayVideoBase64`** (sync overlay when available); see `speed_studio_job.py`.
- **Pre-built PyTorch images**: you usually do **not** need this repo’s `Dockerfile`; `pip install -r requirements.txt` is enough if versions resolve.
- **`FUSIONTRACK_API_KEY`**: optional; when unset, protected routes accept any caller (fine on localhost only). Set it for any instance reachable from the public internet.

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
