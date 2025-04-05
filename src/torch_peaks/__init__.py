"""Peak finding and fitting using torch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-peaks")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"
