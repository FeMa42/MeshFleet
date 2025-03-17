import glob
import multiprocessing
import os
import platform
import random
import math
import subprocess
import tempfile
import time
import zipfile
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union
from functools import partial

import pandas as pd


import fire
import fsspec
import GPUtil
from loguru import logger
import tqdm

from setting import *
# from setting import *
from typing import List, Optional
import numpy as np
import json
from multiprocessing import Pool

# set logging level
import sys
logger.remove()
logger.add(sys.stderr, level="INFO")

# import objaverse.xl as oxl
# from objaverse.utils import get_uid_from_str

FILE_EXTs = ["obj", "glb", "gltf", "usd", "fbx", "stl", "dae", "ply", "abc", "blend"]

def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"~/objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")

def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.

    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # this ensures the structure inside the zip starts at folder/
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)

def render_obj(
    local_path: str,
    num_renders: int,
    render_dir: str,
    random_camera: bool,
    azimuths_deg: Optional[List[float]],
    elevations_deg: Optional[float | List[float]],
    distances: Optional[float | List[float]],
    # viewpoint_path: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    successful_log_file: Optional[str] = "render-successful.csv",
    failed_log_file: Optional[str] = "render-failed.csv",
    random_lights: bool = False,
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """

    os.makedirs(render_dir, exist_ok=True)
    file_name = local_path.split("/")[-1]
    save_uid = file_name.split(".")[0]
    args = f"--object_path {local_path}"

    # get the GPU to use for rendering
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        # get the target directory for the rendering job
        target_directory = os.path.join(temp_dir, save_uid)
        os.makedirs(target_directory, exist_ok=True)
        args += f" --output_dir {target_directory}"

        # check for Linux / Ubuntu or MacOS
        if platform.system() == "Linux" and using_gpu:
            args += " --engine CYCLES"
            # args += " --engine BLENDER_EEVEE"
        elif platform.system() == "Darwin" or (
            platform.system() == "Linux" and not using_gpu
        ):
            # As far as I know, MacOS does not support BLENER_EEVEE, which uses GPU
            # rendering. Generally, I'd only recommend using MacOS for debugging and
            # small rendering jobs, since CYCLES is much slower than BLENDER_EEVEE.
            args += " --engine CYCLES"
            # args += " --engine BLENDER_EEVEE"
        else:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

        # check if we should only render the northern hemisphere
        if random_camera:
            args += " --random_camera"
            if only_northern_hemisphere:
                args += " --only_northern_hemisphere"
            args += f" --num_renders {num_renders}"
        else:
            if azimuths_deg is None or len(azimuths_deg) == 0:
                azimuths_deg = np.linspace(0, 360, num_renders + 1)[1:] % 360
                # azimuths_deg = np.linspace(0, math.pi * 2, num_renders + 1)[1:] % (math.pi * 2)
            if elevations_deg is None:
                elevations_deg = [60.0] * len(azimuths_deg)
            if isinstance(elevations_deg, float):
                elevations_deg = [elevations_deg] * len(azimuths_deg)
            if distances is None:
                distances = [3.0] * len(azimuths_deg)
            if isinstance(distances, float):
                distances = [distances] * len(azimuths_deg)
        
            assert len(azimuths_deg) == len(elevations_deg) == len(distances) 

            num_renders = len(azimuths_deg)
            azimuth = ' '.join(map(str, azimuths_deg))
            elevation = ' '.join(map(str, elevations_deg))
            distances = ' '.join(map(str, distances))
            args += f" --azimuths_deg {azimuth} --elevations_deg {elevation} --distances {distances} --num_renders {num_renders}"

        if random_lights:
            args += " --random_lights"

        # get the command to run
        command = f"{g_blender_excutable_path} --background --python blender_script.py -- {args}"
        if using_gpu:
            command = f"export DISPLAY=:{DISPLAY_ID} && {command}"

        logger.debug(command)
        #print(command)
        # render the object (put in dev null)
        subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # check that the renders were saved successfully
        png_files = glob.glob(os.path.join(target_directory, "*.png"))
        aed_files = glob.glob(os.path.join(target_directory, "*.json"))
        npy_files = glob.glob(os.path.join(target_directory, "*.npy"))

        if (
            (len(png_files) != num_renders)
            or (len(npy_files) != num_renders)
            or (len(aed_files) != num_renders)
        ):
            logger.error(
                f"Found object {file_name} was not rendered successfully!"
            )
            if failed_log_file is not None:
                log_processed_object(
                    failed_log_file,
                    file_name
                )
            return False

        if g_script_model == "sv3d":
            # save sv3d command line
            image_one_pick = random.choice(png_files)
            pat = os.path.join(image_one_pick, os.sep.join(image_one_pick.rsplit(r"/")[-2:]))
            sv3d_u_command = f"python {g_sv3d_script_path} --input_path {pat} --version sv3d_u \n"
            sv3d_p_command = f"python {g_sv3d_script_path} --input_path {pat} --version sv3d_p"

            if random_camera:
                azimuths_deg = []
                elevations_deg = []
                def aed_key(e):
                    return e.split("/")[-1].split(".")[0][:3]
                aed_files.sort(key=aed_key)
                for aed in aed_files:
                    with open(aed) as f:
                        ad = json.load(f)
                        azimuths_deg.append(ad['azimuth'])
                        elevations_deg.append(ad['elevation'])

            if len(azimuths_deg) == 0 and len(elevations_deg) == 1:
                sv3d_p_command += f" --elevations_deg {elevations_deg[0]}"
            
            if len(azimuths_deg) >= 21 and len(elevations_deg) >= 21:
                azi= ' '.join(map(str, azimuths_deg[:21]))
                ele = ' '.join(map(str, elevations_deg[:21]))
                sv3d_p_command += f" --elevations_deg {ele} --azimuths_deg {azi}"

            with open(os.path.join(target_directory, "sv3d_commands.txt"), "w+") as outfile: 
                outfile.write(sv3d_u_command)
                outfile.write(sv3d_p_command)

        # Make a zip of the target_directory.
        # Keeps the {save_uid} directory structure when unzipped
        with zipfile.ZipFile(
            f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
        ) as ziph:
            zipdir(target_directory, ziph)

        # move the zip to the render_dir
        fs, path = fsspec.core.url_to_fs(render_dir)

        # move the zip to the render_dir
        fs.put(
            os.path.join(f"{target_directory}.zip"),
            os.path.join(path, f"{save_uid}.zip"),
        )

        # log that this object was rendered successfully
        if successful_log_file is not None:
            log_processed_object(successful_log_file, file_name) #, file_identifier, sha256)

        return True

def render_objects(
    mesh_path: str = g_object_input_path,
    render_path: str = g_render_output_path,
    random_camera: bool = g_random_camera,
    azimuths_deg: Optional[List[float]] = g_azimuths_deg,
    elevations_deg: Optional[float | List[float]] = g_elevations_deg, 
    distances: Optional[float | List[float]] = g_distances,
    num_renders: int = g_num_renders,
    only_northern_hemisphere: bool = g_only_northern_hemisphere,
    render_timeout: int = g_render_timeout,
    gpu_devices: Optional[Union[int, List[int]]] = None,
    skip_on_found_rendering: bool = g_skip_on_found_rendering, 
    random_lights: bool = g_random_lights
) -> None:
    """
    Adapted from Objaverse
    Renders objects in the Objaverse-XL dataset with Blender

    Args:
        render_dir (str, optional): Directory where the objects will be rendered.
        num_renders (int, optional): Number of renders to save of the object. Defaults
            to 21.
        only_northern_hemisphere (bool, optional): Only render the northern hemisphere
            of the object. Useful for rendering objects that are obtained from
            photogrammetry, since the southern hemisphere is often has holes. Defaults
            to False.
        render_timeout (int, optional): Number of seconds to wait for the rendering job
            to complete. Defaults to 300.
        gpu_devices (Optional[Union[int, List[int]]], optional): GPU device(s) to use
            for rendering. If an int, the GPU device will be randomly selected from 0 to
            gpu_devices - 1. If a list, the GPU device will be randomly selected from
            the list. If 0, the CPU will be used for rendering. If None, all available
            GPUs will be used. Defaults to None.

    Returns:
        None
    """
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )

    # get the gpu devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")


    # get the objects to render
    if os.path.isdir(mesh_path):
        #objects =  [os.path.join(mesh_path, obj) for obj in os.listdir(mesh_path) if os.path.isfile(os.path.join(mesh_path, obj)) and obj.split(".")[1] in FILE_EXTs]
        objects = []
        for ext in FILE_EXTs:
            objects.extend(
                glob.glob(os.path.join(mesh_path,f"**/*.{ext}"), recursive=True)
            )
    elif os.path.isfile(mesh_path):
        obj = mesh_path
        objects = [obj] if obj.split(".")[-1] in FILE_EXTs else []
    else:
        raise ValueError("Cannot find objects to render in the given path.")
    logger.info(f"Provided {len(objects)} objects to render.")

    if skip_on_found_rendering:
        # get the already rendered objects
        fs, path = fsspec.core.url_to_fs(render_path)
        try:
            zip_files = fs.glob(os.path.join(path, "*.zip"), refresh=True)
        except TypeError:
            # s3fs may not support refresh depending on the version
            zip_files = fs.glob(os.path.join(path, "*.zip"))
    
        saved_ids = set(zip_file.split("/")[-1].split(".")[0] for zip_file in zip_files)
        logger.info(f"Found {len(saved_ids)} objects already rendered.")
        # logger.info(saved_ids)

        # # logger.info(objects)
        # # filter out the already rendered objects
        # df = pd.read_csv("/home/q655474/SV3D/Diffus3D/02_multi-view-rendering/uid_lists/objaverse_car_quality_votes_damian.csv")
        # uid_list = df[df.vote >= 0].uid.values

        # objects = [obj for obj in objects if obj.split("/")[-1].split(".")[0] not in saved_ids and obj.split("/")[-1].split(".")[0] in uid_list]
        objects = [obj for obj in objects if obj.split("/")[-1].split(".")[0] not in saved_ids]
        logger.info(f"Rendering {len(objects)} new objects.")

    if g_use_multiprocessing:
        # rendering multiple objects in parallel using multiprocessing workers
        num_workers = min(g_num_workers, multiprocessing.cpu_count())
        logger.info(f"Using multiprocessing with {num_workers} workers.")
        with Pool(num_workers) as pool:
            with tqdm.tqdm(total=len(objects)) as pbar:
                for r in pool.imap_unordered(
                        partial(
                            render_obj,
                            render_dir=render_path,
                            random_camera=random_camera,
                            num_renders=num_renders,
                            azimuths_deg=azimuths_deg,
                            elevations_deg=elevations_deg,
                            distances=distances,
                            only_northern_hemisphere=only_northern_hemisphere,
                            gpu_devices=parsed_gpu_devices,
                            render_timeout=render_timeout,
                            random_lights=random_lights
                            ), objects
                        ):
                    pbar.update()
    else:
        with tqdm.tqdm(total=len(objects)) as pbar:
            for obj in objects:
                render_obj(
                    render_dir=render_path,
                    local_path=obj,
                    random_camera=random_camera,
                    num_renders=num_renders,
                    azimuths_deg=azimuths_deg,
                    elevations_deg=elevations_deg,
                    distances=distances,
                    only_northern_hemisphere=only_northern_hemisphere,
                    gpu_devices=parsed_gpu_devices,
                    render_timeout=render_timeout,
                    random_lights=random_lights)
                pbar.update()

if __name__ == "__main__":
    fire.Fire(render_objects)
