from .metrics import *
from .view import *
from .preprocessing import *
from .datasets import *
from .classifier import *
from .test import * 
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version(__name__)
except PackageNotFoundError:
    pass
