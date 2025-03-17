"""settings.py contains all configuration parameters the blender needs


reference: https://github.com/weiaicunzai/blender_shapenet_render
"""

DISPLAY_ID = 1 #required for BMW Workstation 
g_base_path = "/mnt/damian/.objaverse/new_oxl_objects/"
g_objaverse_base_path = "/mnt/damian/.objaverse/"
g_device = "cpu"
g_max_batch_threshold = 40
g_keep_rendered_objects = True
g_use_fbx_processing = False
g_yolo_model_path = '../../Diffus3D/00_model_checkpoints/yolo11x.pt'

g_object_input_path = '~/.objaverse/car_meshes_trellis_aesthetics_lower65'
g_render_output_path = '~/.objaverse/car_renders_tellis_lower65'
g_blender_excutable_path = '../../Diffus3D/blender-3.2.2-linux-x64/blender'

#viewpoint setting
#for random camera, the last param would take effect
g_random_camera = False
g_random_lights = False
g_num_renders = 4
g_azimuths_deg = None #or list of floats
g_elevations_deg = 80.0  # MEASURED FROM TOP!! a float or list of floats
g_distances = 1.5 # None or a float or list of floats

g_only_northern_hemisphere = False # does not matter if g_random_camera is False 

g_render_timeout = 5000
g_skip_on_found_rendering = False #TODO: verify before running script
#if you have multiple viewpoint files, add to the dict
#files contains azimuth,elevation,tilt angles and distance for each row

g_num_workers = 1 # num of multiprocessing workers
g_use_multiprocessing = False

g_script_model = "sv3d"
g_sv3d_script_path = ""