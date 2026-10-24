import os
import sys

# The vendored Hi3D inference package imports itself as top-level ``sparc3d``.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from . import trellis2_encode
from .hi3d_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **trellis2_encode.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **trellis2_encode.NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
