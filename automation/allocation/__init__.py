"""
automation.allocation — AllocationSource boundary layer.

The executor consumes a verified target-allocation vector through this package.
The alpha signal (from HAARP via RemoteSignalSource) arrives in B5; this package
ships an equal_weight example for offline / testing use.

Exports
-------
- TargetAllocation      — pydantic model matching the signed-signal payload (spec §5.2)
- AllocationSource      — Protocol for any allocation source
- AllocationRejected    — Raised (never silently clamped) on validation failure
- validate_target_allocation — local safety validation, called before every trade
"""

from automation.allocation.base import (
    AllocationRejected,
    AllocationSource,
    TargetAllocation,
    validate_target_allocation,
)

__all__ = [
    "AllocationRejected",
    "AllocationSource",
    "TargetAllocation",
    "validate_target_allocation",
]
