"""Dual-Grok bridge: reliable file channel + optional public page fetch."""

from .channel import BridgeChannel
from .protocol import BridgeMessage, Party

__all__ = ["BridgeChannel", "BridgeMessage", "Party"]
