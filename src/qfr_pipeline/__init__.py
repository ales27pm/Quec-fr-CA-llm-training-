"""qfr_pipeline package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qfr-pipeline")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
