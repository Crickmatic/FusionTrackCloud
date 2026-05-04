# iPhone Capture Trigger Plan

FusionTrack iPhone capture should stay lightweight and trigger-centric. The phone decides **when to capture**, not final ball tracking.

## Capture Flow

1. Keep a rolling buffer (pre-roll + short post-roll).
2. Detect bowler-side person/run-up entry in a run-up zone.
3. Ignore batter-side person presence near batter-end stumps.
4. Trigger capture when bowler-side motion/person persistence crosses threshold.
5. Save a 4-6 second clip (default 5s) including pre-roll.
6. Upload clip + metadata immediately while camera UI remains active.
7. Render cloud result overlay on the same camera screen.

## Trigger Inputs

- Primary: Apple Vision person detection in bowler/run-up zone.
- Optional: lightweight local YOLO person model if Vision is insufficient.
- Additive: Watch release/run-up prior.

## Metadata to Send

- `calibrationMode`: `video_only`, `guided_boxes`, or `ar_world`
- `fps`, `resolution`
- `capture.triggerTimestampSec`, `capture.preRollSec`
- `watch.releaseTimestampSec` (if available)
- `guidedBoxes` when guided setup is used
- AR pitch/world metadata when `ar_world` is used

## Product Modes

- **video_only**: no setup friction, lowest physical confidence.
- **guided_boxes**: best default for grassroots/stumps-present scenarios.
- **ar_world**: premium accuracy for physically meaningful metrics.

## Non-Goals on iPhone Live Path

- No full live ball tracking dependency.
- No heavy trajectory solving in the capture loop.
- No UI handoff away from camera during processing.
