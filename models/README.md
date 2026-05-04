# FusionTrack Cloud Models

Place inference weights here for deployment:

- `yolo26n.pt` - default Phase 1 detector.
- `yolo26s.pt` - optional stronger cloud detector.
- `yolov8s.pt` - optional comparison detector.

The service also resolves model paths from the current working directory and parent directory, so local development can reuse the repository-root `.pt` files without duplicating them.

Do not train models inside API requests. Training and export scripts should live separately under a future `training/` directory.
