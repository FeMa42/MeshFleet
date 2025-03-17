import os
import sys
import csv
import time
from typing import List
from zipfile import ZipFile
import PIL.Image
import numpy as np
import shutil
import h5py
import cv2
import gc
import torch
from yolo_labeling import YOLO_helper
from oxl_embedder.status_logging import log_object_to_csv 
sys.path.append("../quality_classifier")
from quality_classifier.embedding_models import generate_new_dino_embedding_model, generate_siglip_embedding_model

def quantize_data(all_embeddings_uncompressed):
    """Quantizes floating point embedding data to uint8 format for compression.

    This function performs linear quantization of floating point embeddings to uint8 (0-255) range.
    It computes the scale and zero point parameters needed for dequantization and saves both the
    quantized data and parameters to disk.

    Args:
        all_embeddings_uncompressed: numpy array of floating point embeddings to quantize

    Returns:
        tuple containing:
            - quantized_data: numpy array of quantized embeddings as uint8
            - scale: float scaling factor used in quantization
            - zero_point: float zero point offset used in quantization

    The function also saves:
        - quantized_data.npy: The quantized embeddings array
        - quant_params.npy: Dictionary with scale and zero_point parameters
    """
    # Determine the minimum and maximum of your data
    data_min = all_embeddings_uncompressed.min()
    data_max = all_embeddings_uncompressed.max()

    # Target quantized data type range
    qmin = 0
    qmax = 255  # For uint8

    # Compute the scale and zero-point
    scale = (data_max - data_min) / (qmax - qmin)
    if scale == 0:
        scale = 1e-8  # Prevent division by zero
    zero_point = qmin - data_min / scale

    # Quantize the data
    quantized_data = ((all_embeddings_uncompressed / scale) +
                    zero_point).round().astype(np.uint8)

    # Save the quantized data
    np.save('quantized_data.npy', quantized_data)

    # Save the scale and zero-point for dequantization
    quant_params = {'scale': scale, 'zero_point': zero_point}
    np.save('quant_params.npy', quant_params)
    return quantized_data, scale, zero_point


class ClassificationQueueHandler:
    """
    This class is used to handle the classification queue.
    It is used to add objects that have been rendered to the queue 
    The queue uses the RenderedImageProcessor to process (e.g. embed and classify) the renders.
    """
    def __init__(self, base_folder, class_estimation_csv, object_detection_csv,
                 expected_sequence_length, device, run_dir: str,
                 max_batch_threshold: int = 100, 
                 keep_rendered_objects: bool = False):
        self.pending_classifications = []
        self.run_dir = run_dir
        self.max_batch_threshold = max_batch_threshold
        self.keep_rendered_objects = keep_rendered_objects
        self.render_processor = RenderedImageProcessor(
            base_folder=base_folder,
            class_estimation_csv=class_estimation_csv,
            object_detection_csv=object_detection_csv,
            expected_sequence_length=expected_sequence_length,
            device=device
        )

    def add_classification(self, classification_data):
        self.pending_classifications.append(classification_data)
        return len(self.pending_classifications)

    def get_pending_classifications(self):
        return len(self.pending_classifications)

    def process_pending_classifications(self):
        print(f"Found {len(self.pending_classifications)} renders to process")
        max_number_of_renders = min(len(self.pending_classifications), self.max_batch_threshold)
        objects_to_process = [self.pending_classifications.pop()
                              for _ in range(max_number_of_renders)]
        print(f"Processing {len(objects_to_process)} renders")
        self.process_renders(objects_to_process)
    
    def empty_pending_classifications(self):
        pending_class_loop_counter = 0
        while len(self.pending_classifications) > 0:
            self.process_pending_classifications()
            pending_class_loop_counter += 1
            if pending_class_loop_counter > 10:
                raise Exception("Iterated more than 10 times while waiting for pending classifications")

    def process_renders(self, render_file_paths: List[str]):
        try:
            not_processed_objects = self.render_processor.process_render_batch(
                render_file_paths)
            # Handle failed processing
            for obj_sha256 in not_processed_objects:
                log_object_to_csv(os.path.join(self.run_dir, "failed_embedding_classification.csv"), obj_sha256, "unknown")
        except Exception as e:
            print(f"Error processing renders: {e}")
            error_file_name = f"embedding_error_{time.time()}.txt"
            with open(os.path.join(self.run_dir, error_file_name), "w") as f:
                f.write(f"Error while processing renders: {e}")
            # Handle all as failed in case of error
            for render_file_path in render_file_paths:
                sha256 = os.path.basename(render_file_path)
                log_object_to_csv(os.path.join(self.run_dir, "failed_embedding_classification.csv"), sha256, "unknown")
        finally:
            if not self.keep_rendered_objects:
                # Clean up renders
                for render_path in render_file_paths:
                    if os.path.exists(render_path):
                        shutil.rmtree(render_path)

class RenderedImageProcessor:
    """
    This class is used to process the renders of the objects.
    It is used to embed them and classify the rendered objects.
    """
    def __init__(self, base_folder: str, class_estimation_csv: str, object_detection_csv: str, 
                 expected_sequence_length: int = 4, device: str = "cuda") -> bool:
        self.device = device
        self.base_folder = base_folder
        self.class_estimation_csv = class_estimation_csv
        self.object_detection_csv = object_detection_csv

        self.yolo_labeling_model = YOLO_helper(device=device)

        self.dino_embedding_model = generate_new_dino_embedding_model(pca_file_name='../car_quality_models/pca_model_DINOv2.pkl', 
                                                           device=device, expected_sequence_length=expected_sequence_length)
        self.siglip_embedding_model = generate_siglip_embedding_model(device=device)

    def get_images_from_zip(self, zip_file_path: str) -> List[PIL.Image.Image]:
        with ZipFile(zip_file_path) as zf:
            zip_items = [item for item in sorted(zf.namelist()) if item.endswith(".png")]
            pil_imgs = [PIL.Image.open(zf.open(zip_item)).convert("RGB") for zip_item in zip_items]
        return pil_imgs
    
    def save_quantized_embeddings(self, object_uid: str, dino_embeddings: np.ndarray, siglip_embeddings: np.ndarray) -> None:
        quantized_dino_data, scale_dino, zero_point_dino = quantize_data(dino_embeddings)
        quantized_siglip_data, scale_siglip, zero_point_siglip = quantize_data(siglip_embeddings)
        embedding_filename = f"{self.base_folder}/embeddings_{object_uid}.h5"
        with h5py.File(embedding_filename, 'w') as hf:
            hf.create_dataset('dino', data=quantized_dino_data, compression='gzip')
            hf.create_dataset('siglip', data=quantized_siglip_data, compression='gzip')
            hf.create_dataset('dino_scale', data=scale_dino)
            hf.create_dataset('dino_zero_point', data=zero_point_dino)
            hf.create_dataset('siglip_scale', data=scale_siglip)
            hf.create_dataset('siglip_zero_point', data=zero_point_siglip)
    
    def process_render_batch(self, render_file_paths: List[str]) -> List[str]:
        """Process a batch of rendered 3D object images for embedding and classification.
        This method handles the full pipeline of processing rendered images:
        1. Loads images from zip files for each rendered object
        2. Generates YOLO object detection labels
        3. Generates DINOv2 and SigLIP embeddings
        4. Groups results by object and saves to disk
        5. Cleans up GPU memory
        """
        all_images: List[PIL.Image.Image] = []
        sha256_mappings: List[str] = []
        not_processed_objects = []
        for render_file_path in render_file_paths:
            # get the sha256 from the render_file_path
            sha256 = os.path.basename(render_file_path)
            all_zip_files = [os.path.join(render_file_path, file_name) for file_name in os.listdir(render_file_path) if file_name.endswith(".zip")]
            images_to_process = False
            zip_file = all_zip_files[0]
            images = self.get_images_from_zip(zip_file)
            amount_of_images = len(images)
            if amount_of_images > 0:
                images_to_process = True
                all_images.extend(images)
                sha256_mappings.extend([sha256] * amount_of_images)
            if not images_to_process:
                not_processed_objects.append(sha256)
        
        if len(all_images) == 0:
            return not_processed_objects

        # Generate YOLO detections
        labels = self.yolo_labeling_model.label_batch_images(all_images)
        # Generate DINOv2 and SigLIP embeddings
        dino_embeddings = self.dino_embedding_model(all_images)
        siglip_embeddings = self.siglip_embedding_model(all_images)

        # using the sha256 mappings to chunk the labels back to the original objects, packing all labels with the same sha256 together 
        object_label_dict = {}
        object_dino_embeddings_dict = {}
        object_siglip_embeddings_dict = {}
        for i, sha256 in enumerate(sha256_mappings):
            if sha256 not in object_label_dict:
                object_label_dict[sha256] = []
            object_label_dict[sha256].append(labels[i])

        # get the number of images per object to chunk the embeddings back to the original objects
        n_images = dino_embeddings.shape[1]
        for i, sha256 in enumerate(sha256_mappings[::n_images]):
            if sha256 not in object_dino_embeddings_dict:
                object_dino_embeddings_dict[sha256] = []
            object_dino_embeddings_dict[sha256].append(dino_embeddings[i].detach().cpu().numpy())
            if sha256 not in object_siglip_embeddings_dict:
                object_siglip_embeddings_dict[sha256] = []
            object_siglip_embeddings_dict[sha256].append(siglip_embeddings[i].detach().cpu().numpy())

        for sha256 in object_label_dict:
            # add labels to the csv file 
            with open(self.object_detection_csv, "a") as f:
                object_label_str = "; ".join(object_label_dict[sha256])
                f.write(f"{sha256},{object_label_str}\n")

            # save the quantized embeddings
            # make array from object_dino_embeddings_dict[sha256]
            dino_embeddings_array = np.array(object_dino_embeddings_dict[sha256])
            siglip_embeddings_array = np.array(object_siglip_embeddings_dict[sha256])
            self.save_quantized_embeddings(sha256, dino_embeddings_array, siglip_embeddings_array)

        # clear gpu memory
        del dino_embeddings_array, siglip_embeddings_array, dino_embeddings, siglip_embeddings
        torch.cuda.empty_cache()
        gc.collect()
        return not_processed_objects
