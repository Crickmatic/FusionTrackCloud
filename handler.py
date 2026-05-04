"""
RunPod Serverless: set handler to ``handler.handler`` (this file, function ``handler``).

Dedicated pod + Speed Studio: run ``uvicorn`` (Dockerfile CMD) and set URL to::
  https://<pod-id>-8000.proxy.runpod.net/v1/runsync
"""
from runpod_speed_studio import handler
