# checkout the website from autodesk to install the python package (downlad the installer and build from install location)
import fbx
import os

def generate_obj_from_fbx(fbx_path):
    '''
    Convert fbx file to obj file. 
    This methods needs fbx sdk to be installed. 
    Checkout the website from autodesk here:
    https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html
    To install the python package you should downlad the 
    installer, run it and build the python package from the install location with pip. 
    '''
    # init fbx sdk manager and scene
    manager = fbx.FbxManager.Create()
    scene = fbx.FbxScene.Create(manager, "")
    # Import
    importer = fbx.FbxImporter.Create(manager, "")
    importstat = importer.Initialize(fbx_path, -1)
    importstat = importer.Import(scene)
    # obj name
    base_dir = os.path.dirname(fbx_path)
    file_name = os.path.basename(fbx_path)
    file_name = file_name.split(".")[0]
    save_path = os.path.join(base_dir, file_name + ".obj")
    # Export
    exporter = fbx.FbxExporter.Create(manager, "")
    exportstat = exporter.Initialize(save_path, -1)
    exportstat = exporter.Export(scene)
    return save_path
