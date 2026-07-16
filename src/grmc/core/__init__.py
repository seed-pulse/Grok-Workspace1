from .memory_manager import MemoryManager

__all__ = ["MemoryManager", "ReflectionEngine"]


def __getattr__(name: str):
    """Lazy re-export so `from grmc.core import ReflectionEngine` works."""
    if name == "ReflectionEngine":
        from ..reflection.reflection_engine import ReflectionEngine

        return ReflectionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
