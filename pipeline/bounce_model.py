from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BouncePhysicsResult:
    vx_after: float
    vy_after: float


def apply_bounce_physics(vx_before: float, vy_before: float, restitution: float = 0.58) -> BouncePhysicsResult:
    e = max(0.35, min(0.75, float(restitution)))
    return BouncePhysicsResult(vx_after=float(vx_before), vy_after=float(-e * vy_before))

