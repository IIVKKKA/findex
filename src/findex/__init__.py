"""findex - a small search engine built lab by lab."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("findex")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
