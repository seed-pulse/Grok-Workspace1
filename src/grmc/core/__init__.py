from .memory_manager import MemoryManager

__all__ = ["MemoryManager", "ReflectionEngine", "ApprovalQueue"]


def __getattr__(name: str):
    if name == "ReflectionEngine":
        from ..reflection.reflection_engine import ReflectionEngine

        return ReflectionEngine
    if name == "ApprovalQueue":
        from .approval_queue import ApprovalQueue

        return ApprovalQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
