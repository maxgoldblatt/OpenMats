import os
import bpy
import threading
import time
import numpy as np
from . import addon_dir # imports addon dir for images
# Importing functions
from . import openai_integration 
from . import retrieve
from . import normal_to_height
from . import utils
from . import module_color_to_normals
from . import utils_inference

"""
OpenMats Main Module - Blender Addon Core

This is the primary module for the OpenMats addon, responsible for:
- Registering all UI panels, operators, and properties
- Handling prompt submission to OpenAI (DALL·E 3, DALL·E 2, Images API)
- Creating materials from AI-generated textures
- Generating normal maps using DeepBump (ONNX) inference
- Reconstructing height maps from normal maps (Frankot-Chellappa)

Features:
- User prompt input and model selection UI
- Material presets (Metal, Diffuse, Specular)
- Real-time feedback through Blender's reporting system
- Seamless vs. non-seamless normal-to-height support

Modules used:
- openai_integration: Handles OpenAI API calls
- retrieve: Downloads images from DALL·E URLs
- normal_to_height: Converts normal maps to height maps
- utils: Blender ↔ NumPy image conversion
- module_color_to_normals: DeepBump inference interface
- utils_inference: Tiling, inference, and merging support

Originally inspired by work from HugoTini (GitHub). Integrated and extended
for direct use inside Blender through the OpenMats addon interface.

Version: 1.0
Primary Author: Max Goldblatt
"""

# Global Vars
material = None
generated_image_node = None
generated_normal_node = None


def create_mat(context, tex_path):
    """Creates Material on Selected Object"""
    global material
    global generated_image_node

    # Access Booleans for Material Presets
    metal_mat = context.scene.enable_metal_mat
    diffuse_mat = context.scene.enable_diffuse_mat
    spec_mat = context.scene.enable_spec_mat
    
    material = bpy.data.materials.new(name= context.scene.prompt_prop.split()[0] + " Material" ) # Creates new Material with first word of given prompt
    material.use_nodes = True
    
    # Principled BSDF Reference
    bsdf_node = material.node_tree.nodes["Principled BSDF"]
    
    # Material Presets
    if metal_mat:
        bsdf_node.inputs["Metallic"].default_value = 1.0
        bsdf_node.inputs["Roughness"].default_value = 0.25
        bsdf_node.inputs[13].default_value = 0.875
    if diffuse_mat:
        bsdf_node.inputs["Metallic"].default_value = 0
        bsdf_node.inputs["Roughness"].default_value = 0.85
        bsdf_node.inputs[13].default_value = 0.50
    if (diffuse_mat and metal_mat) or (spec_mat and metal_mat):
        bsdf_node.inputs["Metallic"].default_value = 5
        bsdf_node.inputs["Roughness"].default_value = 0.65
        bsdf_node.inputs[13].default_value = 0.7
    if spec_mat:
        bsdf_node.inputs["Metallic"].default_value = 0.45
        bsdf_node.inputs["Roughness"].default_value = 0.1
        bsdf_node.inputs[13].default_value = 1
    if spec_mat and diffuse_mat:
        bsdf_node.inputs["Metallic"].default_value = 0
        bsdf_node.inputs["Roughness"].default_value = 0.45
        bsdf_node.inputs[13].default_value = 1
    if spec_mat and diffuse_mat and metal_mat:
        bsdf_node.inputs["Metallic"].default_value = 1
        bsdf_node.inputs["Roughness"].default_value = 0.65
        bsdf_node.inputs[13].default_value = 1
    
    # Load Image, modified code from https://www.youtube.com/watch?v=xz9Tn6rUzzg, & ChatGPT
    try:
        image_obj = bpy.data.images.load(tex_path)
    except RuntimeError as e:
        print(f"ERROR: Failed to load image from {tex_path}: {e}")
        return None
    
    image_node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
    image_node.image = image_obj
    image_node.location.x = -400
    image_node.location.y = 300
    material.node_tree.links.new(image_node.outputs[0],
                                bsdf_node.inputs["Base Color"])
    #Apply Material
    obj = bpy.context.active_object
    # Ensure the object has material slots
    if obj.material_slots:
        # Assign the new material to the active material slot
        obj.material_slots[obj.active_material_index].material = material
    else:
        # Add a new material slot and assign the new material if no slots exist
        obj.data.materials.append(material)
    
    generated_image_node = image_node
    
    try:
        bpy.utils.register_class(Normal_PT_UI_)
    except ValueError:
        print("INFO: Normal Class Already Registered")


def register_height_ui(context):
    """Registers Height Panel UI after Normal Map Generation"""
    bpy.utils.register_class(HeightMap_UI_Panel_PT)
    return


class UIPanel (bpy.types.Panel):
    """Main UI Panel to host addon"""
    bl_label = "OpenMats"
    bl_idname = "UIPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OpenMats"  # Tab Name
    
    def draw(self, context):
        layout = self.layout

        obj = context.object

        row = layout.row()
        row = layout.row()
        row.alignment = 'LEFT'
        row.scale_y = 0.1
        row.label(text=f"GPT Version: {bpy.types.Scene.bl_rna.properties['model_choice'].enum_items[context.scene.model_choice].name}") # Show Active Gen Model
        
        row = layout.row()
        row.scale_y = 1.5
        row.label(text="v1.00 Mar 2025")
        
        # Prompt Input
        col = self.layout.column(align = True)
        col.scale_y = 1.5
        col.prop(context.scene, "prompt_prop")
        
        # Boolean Toggle for enabling/disabling metal material
        col = self.layout.column (align = True)
        col.prop(context.scene, "enable_metal_mat", text="Metal Material")
        col.prop(context.scene, "enable_diffuse_mat", text="Diffuse Material")
        
        col.prop(context.scene, "enable_spec_mat", text="Specular Material")
        #row = self.layout.row (align = True)

        col = layout.column()
        col.label(text="Image Generation Model")
        col.prop(context.scene, "model_choice", text="")

        #Create Texture button
        button = layout.row()
        button.scale_y = 2
        button.operator("object.send_prompt", icon='IMAGE_DATA')
        
        
class send_prompt (bpy.types.Operator):
    """Generate texture & material based on current prompt"""
    
    bl_idname = "object.send_prompt"
    bl_label = "Create Texture"

    def execute(self, context):
        global generated_image_node
        prompt_value = bpy.context.scene.prompt_prop
        
        # Ensures Prompt is valid. Keeps going with function if it is
        if (prompt_value != "") and prompt_value != "Enter Prompt":
            
            # Logging
            print("Given Prompt: " + prompt_value)

            # Sending & Getting Response from OpenAI - debugged accessing my addon prefs with ChatGPT
            addon_prefs = bpy.context.preferences.addons[__package__].preferences
            key_dir = addon_prefs.user_input

            if key_dir is not None:
                
                with open(str(key_dir), 'r') as file: # Reads text file w/ API Key
                    key = file.read()
                
                print("SUBMITTING PROMPT")
                self.report({'INFO'}, 'SUBMITTING PROMPT.')
                
                # Access OpenAI Model Choice
                model = context.scene.model_choice
                if model == 'DALLE3' or model == 'DALLE2':

                    response = openai_integration.generate_response(prompt_value, key, model)
                    
                    print("STARTING DOWNLOAD")
                    self.report({'INFO'}, 'STARTING DOWNLOAD')

                    # Download Image
                    tex_path = retrieve.download_files(response)
                else:
                    tex_path = openai_integration.generate_response_images(prompt_value, key)

                # Create Material in Blender
                create_mat(context, tex_path)
                
                self.report({'INFO'}, "Completed Generation.")

                return {'FINISHED'}
            else:
                self.report({'ERROR'}, 'No Key Given.')
        
        else:
            self.report({'ERROR'}, "Invalid Prompt. Please change prompt.")
        return {'FINISHED'}


class Normal_PT_UI_(bpy.types.Panel):
    """UI Panel to host Normal Map adjustments"""
    bl_label = "Normal Map Adjustments"
    bl_idname = "NormalUIPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OpenMats"  # Tab Name

    def draw(self, context):
        layout = self.layout
        obj = context.object

        button = layout.row()
        button.scale_y = 2
        button.operator("object.create_normal", icon='IMAGE_DATA')

# Modified from https://github.com/HugoTini/DeepBump/blob/master/__init__.py
class create_normal_map_OT(bpy.types.Operator):
    """Creates Normal Map"""
    bl_idname = "object.create_normal"
    bl_label = "Generate Normal Map (CPU)"

    progress_started = False

    global material
    global generated_image_node
    global generated_normal_node
    
    def progress_print(self, current, total):
        wm = bpy.context.window_manager
        if self.progress_started:
            wm.progress_update(current)
            print(f'DeepBump Color → Normals : {current}/{total}')
        else:
            wm.progress_begin(0, total)
            self.progress_started = True

    def execute(self, context):
         # Get input image from selected node
        input_node = generated_image_node
        input_bl_img = input_node.image
        if input_bl_img is None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}

        # Convert image to numpy C,H,W array
        input_img = utils.bl_image_to_np(input_bl_img)

        # Compute normals
        OVERLAP = 'LARGE'
        self.progress_started = False
        output_img = module_color_to_normals.apply(input_img, OVERLAP, self.progress_print)

        # Create new image datablock
        input_img_name = os.path.splitext(input_bl_img.name)
        output_img_name = input_img_name[0] + '_normals' + input_img_name[1]
        output_bl_img = bpy.data.images.new(
            output_img_name, width=input_bl_img.size[0], height=input_bl_img.size[1])
        output_bl_img.colorspace_settings.name = 'Non-Color'

        # Convert numpy C,H,W array back to blender image pixels
        output_bl_img.pixels = utils.np_to_bl_pixels(output_img)

        # Create new node for normal map
        output_node = material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        output_node.location = input_node.location
        output_node.location[1] -= input_node.width*1.2
        output_node.image = output_bl_img

        # Create normal vector node & link nodes
        normal_vec_node = material.node_tree.nodes.new(
            type='ShaderNodeNormalMap')
        normal_vec_node.location = output_node.location
        normal_vec_node.location[0] += output_node.width*1.1
        links = material.node_tree.links
        links.new(output_node.outputs['Color'],
                  normal_vec_node.inputs['Color'])

        # If input image was linked to a BSDF, link to BSDF normal slot
        if input_node.outputs['Color'].is_linked:
            if len(input_node.outputs['Color'].links) == 1:
                to_node = input_node.outputs['Color'].links[0].to_node
                if to_node.bl_idname == 'ShaderNodeBsdfPrincipled':
                    links.new(
                        normal_vec_node.outputs['Normal'], to_node.inputs['Normal'])

        try:
            register_height_ui(context)
        except Value:
             print('Color → Normals : Done')
        return {'FINISHED'}


class create_height_map_OT(bpy.types.Operator):
    """Creates height map"""
    bl_idname = "object.create_height"
    bl_label = "Generate Height Map (CPU)"

    # Modified & extended from https://github.com/HugoTini/NormalHeight/blob/master/__init__.py
    def execute(self, context):
        global generated_normal_node
        global material
        global generated_image_node
        generated_normal_node = generated_image_node
        
        # Get Input from selected node
        input_img = generated_normal_node.image
        if input_img == None:
            self.report(
                {'WARNING'}, 'Selected image node must have an image assigned to it.')
            return {'CANCELLED'}
        
        # convert to C,H,W numpy array
        width = input_img.size[0]
        height = input_img.size[1]
        channels = input_img.channels
        img = np.array(input_img.pixels)
        img = np.reshape(img, (channels, width, height), order='F')
        img = np.transpose(img, (0, 2, 1))

        # get gradients from normal map
        grad_x, grad_y = normal_to_height.normal_to_grad(img)
        grad_x = np.flip(grad_x, axis=0)
        grad_y = np.flip(grad_y, axis=0)

        # if non-seamless chosen, expand gradients
        HEIGHT_TYPE = context.scene.seamless_height_prop
        if HEIGHT_TYPE == True:
            HEIGHT_TYPE = 'SEAMLESS'
        else:
            HEIGHT_TYPE = 'NON_SEAMLESS'

        if HEIGHT_TYPE == 'NON_SEAMLESS':
            grad_x, grad_y = normal_to_height.copy_flip(grad_x, grad_y)


        # compute height map
        pred_img = normal_to_height.frankot_chellappa(
            -grad_x, grad_y)
        if HEIGHT_TYPE != 'SEAMLESS':
            # cut to valid part if gradients were expanded
            pred_img = pred_img[:height, :width]
        pred_img = np.stack([pred_img, pred_img, pred_img])

         # create new image datablock
        img_name = os.path.splitext(input_img.name)
        height_name = img_name[0] + '_height' + img_name[1]
        height_img = bpy.data.images.new(
            height_name, width=width, height=height)
        height_img.colorspace_settings.name = 'Non-Color'

        # flip height
        pred_img = np.flip(pred_img, axis=1)
        # add alpha channel
        pred_img = np.concatenate(
            [pred_img, np.ones((1, height, width))], axis=0)
        # flatten to array
        pred_img = np.transpose(pred_img, (0, 2, 1)).flatten('F')
        # write to image block
        height_img.pixels = pred_img

        # create new node for height map
        height_node = material.node_tree.nodes.new(
            type='ShaderNodeTexImage')
        height_node.location.x = -500
        height_node.location.y = -200
        height_node.image = height_img

        # Create Displacement
        material.displacement_method = 'BOTH'
        displacement_node = material.node_tree.nodes.new(type='ShaderNodeDisplacement')
        displacement_node.location.x = 100
        displacement_node.location.y = -300

        # Get Active Material Output node & make links
        output_node = None
        nodes = material.node_tree.nodes
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                break
        material.node_tree.links.new(height_node.outputs[0],
                                displacement_node.inputs[0])
        material.node_tree.links.new(displacement_node.outputs[0],
                                output_node.inputs[2])
        
        return {'FINISHED'}


class HeightMap_UI_Panel_PT(bpy.types.Panel):
    """UI Panel to host Normal Map adjustments"""
    bl_label = "Material Adjustments"
    bl_idname = "HeightUIPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OpenMats"  # Tab Name

    def draw(self, context):
        layout = self.layout
        obj = context.object

        row = layout.row()
        row.prop(context.scene, "seamless_height_prop", text="Seamless Normal Map?")

        button = layout.row()
        button.scale_y = 2
        button.operator("object.create_height", icon='IMAGE_DATA')


class_list = [UIPanel, send_prompt, create_normal_map_OT, create_height_map_OT]


def register():
    
    # Register Classes
    for cls in class_list:
        bpy.utils.register_class(cls)
        
    # Register Input for Prompt
    bpy.types.Scene.prompt_prop = bpy.props.StringProperty(
        name="Prompt Input",
        description="Text Field to store prompt to send to DALL-E",
        default="Enter Prompt"
    )
    
    # Register Metal Boolean Property
    bpy.types.Scene.enable_metal_mat = bpy.props.BoolProperty(
        name="Metal Material Boolean",
        description="Enable or disable metallic preset in generated material",
        default=False
    )
    
    # Register Diffuse Boolean Property
    bpy.types.Scene.enable_diffuse_mat = bpy.props.BoolProperty(
        name="Diffuse Material Boolean",
        description="Enable or disable diffuse preset in generated material",
        default=False
    )
    
    # Register Specular Boolean Property
    bpy.types.Scene.enable_spec_mat = bpy.props.BoolProperty(
        name="Specular Material Boolean",
        description="Enable or disable specular preset in generated material",
        default=False
    )

    # Register Seamless Boolean Property
    bpy.types.Scene.seamless_height_prop = bpy.props.BoolProperty(
        name="Seamless Normal Texture Boolean",
        description="Enable if Normal Map is seamless. If not, disable.",
        default=False
    )

    # Register OpenAI Model Choice Property
    bpy.types.Scene.model_choice = bpy.props.EnumProperty(
        name="Model",
        description="Choose the image generation model",
        items=[
            ('DALLE3', "DALL·E 3", "Use DALL·E 3 for image generation"),
            ('DALLE2', "DALL·E 2", "Use DALL·E 2 for image generation"),
            ('IMGAPI', "Images API", "Use custom Images API"),
        ],
        default='DALLE3',
    )



def unregister():
    # Unregister Classes
    for cls in class_list:
        bpy.utils.unregister_class(cls)
    
    bpy.utils.unregister_class(Normal_PT_UI_)
    bpy.utils.unregister_class(HeightMap_UI_Panel_PT)


    # Unregister Props 
    del bpy.types.Scene.prompt_prop
    del bpy.types.Scene.enable_metal_mat
    del bpy.types.Scene.enable_diffuse_mat
    del bpy.types.Scene.seamless_height_prop
    del bpy.types.Scene.model_choice