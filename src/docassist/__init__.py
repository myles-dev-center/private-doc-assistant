"""Private document assistant: cited answers, structured extraction and accuracy evals."""

from .config import Settings, get_settings
from .pipeline import Answer, Assistant

__all__ = ["Answer", "Assistant", "Settings", "get_settings"]
