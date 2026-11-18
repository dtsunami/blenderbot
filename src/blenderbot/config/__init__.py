"""Configuration, `.env` key harvesting, and cross-platform Blender 5.x discovery."""

from .blender_paths import BlenderInstallation, discover_blender, resolve_home
from .env import EFFORTS, PROVIDERS, TRANSPORTS, Config, key_var_for, load_config
from .models import DEFAULT_MODELS, MODELS, ModelSpec, resolve_model
from .settings import load_settings, save_settings, settings_to_defaults

__all__ = [
    "DEFAULT_MODELS",
    "EFFORTS",
    "MODELS",
    "PROVIDERS",
    "TRANSPORTS",
    "BlenderInstallation",
    "Config",
    "ModelSpec",
    "discover_blender",
    "key_var_for",
    "load_config",
    "load_settings",
    "resolve_home",
    "resolve_model",
    "save_settings",
    "settings_to_defaults",
]