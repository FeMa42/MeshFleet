import os
import pandas as pd
from typing import Dict, List
import shutil
from main import render_objects
from setting import g_blender_excutable_path, g_base_path, g_use_fbx_processing
from oxl_embedder.status_logging import failed_processing_folder, log_object_to_csv
if g_use_fbx_processing:
    from fbx_processing import generate_obj_from_fbx
else:
    generate_obj_from_fbx = None

class LoadedObjectHandler:
    """
    This class is used to handle downloaded objects.
    It is used to process the object and render it using blender.
    """
    def __init__(self, sha256:str, file_identifier:str):
        self.sha256 = sha256
        self.file_identifier = file_identifier
        # create the render path
        render_path = os.path.join(g_base_path, "renders")
        os.makedirs(render_path, exist_ok=True)
        self.render_file_path = os.path.join(render_path, self.sha256)
        os.makedirs(self.render_file_path, exist_ok=True)
        self.rendered_object = None

    def process_blend_file(self, new_path:str):
        new_file_path = os.path.join(new_path, f"{self.sha256}.blend")
        # Create .glb file path
        glb_file_path = os.path.join(new_path, f"{self.sha256}.glb")
        # Call Blender to export the file
        call_string = f"{g_blender_excutable_path} --background {new_file_path} --python export_glb.py -- {glb_file_path}"
        # Run the command and capture the exit code
        ret = os.system(call_string)
        if ret != 0 or not os.path.exists(glb_file_path):
            # Remove the original .blend file
            os.remove(new_file_path)
            return None
        # Remove the original .blend file
        os.remove(new_file_path)
        return glb_file_path

    def remove_object_files(self):
        new_path = os.path.join(os.path.join(g_base_path, "shape"), self.sha256)
        if os.path.exists(new_path):
            shutil.rmtree(new_path)

    def process_object(self):
        """Process a downloaded 3D object file for rendering.

        This method handles processing of downloaded 3D object files in various formats:
        - For .blend files: Converts to .glb using Blender
        - For .fbx files: Converts to .obj if conversion function available
        - For other formats: Uses file directly

        The processed file is stored in a directory structure based on the object's SHA256 hash.
        """
        new_path = os.path.join(os.path.join(g_base_path, "shape"), self.sha256)
        filename = None
        # check if the file exists
        if os.path.exists(new_path):
            # get the filename by checking what files are in the new_path
            filename = next((file for file in os.listdir(new_path)
                            if os.path.isfile(os.path.join(new_path, file))), None)
        # if the file does not exist return None
        if filename is None:
            new_file_path = None
        # Check if it is a .blend file
        elif filename.endswith(".blend"):
            new_file_path = self.process_blend_file(
                new_path=new_path)
        elif filename.endswith(".fbx"):
            # convert fbx to obj
            if generate_obj_from_fbx is not None:
                try:
                    new_file_path = generate_obj_from_fbx(os.path.join(new_path, filename))
                except Exception as e:
                    print(f"Error converting fbx to obj: {e}")
                    new_file_path = generate_obj_from_fbx(
                        os.path.join(new_path, filename))
            else:
                print("No fbx to obj conversion function provided, using fbx file directly")
                new_file_path = os.path.join(new_path, filename)   
        else:
            new_file_path = os.path.join(new_path, filename)
        
        # if the file does not exist, write to failed_processing.csv
        if new_file_path is None or not os.path.exists(new_file_path):
            sha256_string = f"{self.sha256}.csv"
            failed_processing_file = os.path.join(failed_processing_folder, sha256_string)
            log_object_to_csv(failed_processing_file, self.sha256, self.file_identifier)
            return None
        return new_file_path

    def render_object_wrapper(self, new_file_path: str):
        # render the object
        try:
            render_objects(mesh_path=new_file_path,
                        render_path=self.render_file_path)
        except Exception as e:
            print(f"Error rendering object: {e}")
        finally:
            # Remove the shape file
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
            self.remove_object_files()

    def render_object(self):
        """Processes and renders a 3D object, preparing it for classification.

        This method handles the full pipeline of processing and rendering a 3D object:
        1. Processes the object file to get a renderable format
        2. Renders the processed object
        3. Checks for successful rendering and prepares metadata for classification
        """
        new_file_path = self.process_object()
        if new_file_path is None:
            return None

        self.render_object_wrapper(new_file_path)

        # if the render folder exist
        rendered_object = None
        if os.path.exists(self.render_file_path):
            # check if the render file exists, create a dict for the classification actor
            all_zip_files = [os.path.join(self.render_file_path, file_name) for file_name in os.listdir(
                self.render_file_path) if file_name.endswith(".zip")]
            if len(all_zip_files) > 0:
                # send to classification actor
                rendered_object = {
                    "render_path": self.render_file_path,
                    "sha256": self.sha256,
                    "fileIdentifier": self.file_identifier
                }
            else:
                # remove the empty render folder
                shutil.rmtree(self.render_file_path)
        return rendered_object

