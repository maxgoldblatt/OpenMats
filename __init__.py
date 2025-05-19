bl_info = {
    "name": "OpenMats",
    "author": "Max Goldblatt",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > OpenMats",
    "description": "An addon that integrates OpenAI prompt-based image generation directly into Blender, with support for DeepBump-based normal map processing.",
    "category": "Material"
}

import bpy
from bpy.props import StringProperty
from bpy.types import (Panel, Operator, PropertyGroup)
from bpy.props import (EnumProperty, BoolProperty, PointerProperty)
import os
import subprocess
import sys
import importlib
from collections import namedtuple
import addon_utils
import time
import threading
import subprocess
import base64


# Get addon path
addon_dir = os.path.dirname(__file__)

# Finding Modules
libs_path = os.path.join(addon_dir, "libs")
if libs_path not in sys.path:
    sys.path.append(libs_path)

import pydantic
import openai
import requests
import numpy as np
import onnxruntime as ort


from . import ui
from . import openai_integration
from . import retrieve
from . import normal_to_height
from . import utils
from . import module_color_to_normals

# Preferences for User's API Key
class AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    print(f"__package__: {__package__}")

    user_input: StringProperty(
        name="OpenAI API Key",
        subtype='FILE_PATH',
        default='',
    )
    def draw(self, context):
        layout = self.layout
        layout.label(text="Enter your OpenAI API Key:")
        layout.prop(self, "user_input")

# Reg/unreg
def register():
    bpy.utils.register_class(AddonPreferences)
    ui.register()

def unregister():
    bpy.utils.unregister_class(AddonPreferences)
    ui.unregister()

if __name__ == "__main__":
    register()