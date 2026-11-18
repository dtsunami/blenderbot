"""Stdlib-only localhost TCP JSON-RPC bridge for Blender 5.x (GUI & Headless)."""

from .protocol import (
    DEFAULT_BRIDGE_HOST,
    DEFAULT_BRIDGE_PORT,
    BridgeDiscovery,
    bridge_discovery_path,
    decode_frame,
    encode_frame,
    generate_bridge_token,
    read_discovery,
    remove_discovery_if_owned,
    verify_bridge_token,
    write_discovery,
)

__all__ = [
    "DEFAULT_BRIDGE_HOST",
    "DEFAULT_BRIDGE_PORT",
    "BridgeDiscovery",
    "bridge_discovery_path",
    "decode_frame",
    "encode_frame",
    "generate_bridge_token",
    "read_discovery",
    "remove_discovery_if_owned",
    "verify_bridge_token",
    "write_discovery",
]