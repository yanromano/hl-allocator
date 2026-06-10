"""automation.reporting — persistent executor state."""

from automation.reporting.state import State, StateError, load, save

__all__ = ["State", "StateError", "load", "save"]
