"""automation.monitoring — read-only position-health monitoring.

Pure health checks over a :class:`HealthSnapshot` plus a thin read-only adapter
(:func:`gather_health_snapshot`) that assembles the snapshot from an HLClient.
All IO lives in the adapter; the checks are pure functions over data.
"""
