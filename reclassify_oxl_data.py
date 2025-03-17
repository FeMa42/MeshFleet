import argparse
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import PIL.Image
from typing import List
from quality_classifier.quality_classifier import generate_siglip_embedding_estimator, generate_combined_embedding_estimator
from quality_classifier.embedding_models import generate_new_dino_embedding_model
from quality_classifier.uncertrainty_estimation import prepare_model_for_uncertainty_estimation, create_uncertainty_estimates
from quality_classifier.utils import get_images_from_zip

def process_files_in_parallel(file_paths):
    with Pool() as pool:
        results = list(pool.imap(get_images_from_zip, [file_path for file_path in file_paths]))
    return results

def process_render_batch(combined_render_df_batch):
    """
    Process a batch of rendered images from a DataFrame containing file paths and object IDs.

    Args:
        combined_render_df_batch (pd.DataFrame): DataFrame batch containing 'img_path' and 'sha256' columns,
            where 'img_path' points to zip files containing rendered PNG images and 'sha256' contains object IDs

    Returns:
        Tuple[List[PIL.Image.Image], np.ndarray, np.ndarray]: Tuple containing:
            - List of PIL Image objects extracted from the zip files
            - Array of placeholder labels (all zeros) to support format with labels
            - Array of object IDs corresponding to the images
    """
    file_paths = combined_render_df_batch['img_path'].values
    object_ids = combined_render_df_batch['sha256'].values
    all_images: List[PIL.Image.Image] = []

    imgs_tmp = process_files_in_parallel(file_paths)
    for imgs in imgs_tmp:
        all_images.extend(imgs)
    
    val_labels = np.array([0] * len(all_images))
    val_uids = object_ids
    return all_images, val_labels, val_uids

def estimate_uncertainty_batch(embedding_classifier, embeddings, val_labels, val_uids, device, batch_size):
    """
    Estimate uncertainty for a batch of embeddings using a given embedding classifier.
    Simply calls the create_uncertainty_estimates function with passing the model in the embedding classifier.
    
    Args:
        embedding_classifier: The classifier to use for uncertainty estimation
        embeddings: The embeddings to estimate uncertainty for
        val_labels: The labels for the validation set
        val_uids: The object IDs for the validation set
        device: The device to run the model on
        batch_size: The batch size to use for the uncertainty estimation

    Returns:
        object_uncertainty_df: A DataFrame containing the uncertainty estimates
    """
    return create_uncertainty_estimates(embedding_classifier.mlp_model, embeddings, val_labels, val_uids, device, batch_size)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_sequence_length = 4

    combined_render_df = pd.read_csv(args.render_df_path)

    siglip_embedding_classifier = generate_siglip_embedding_estimator(
        mlp_model_name=f"./car_quality_models/car_quality_model_siglip.pkl",
        device=device,
        expected_sequence_length=expected_sequence_length
    )
    siglip_embedding_classifier.mlp_model = prepare_model_for_uncertainty_estimation(siglip_embedding_classifier.mlp_model)

    if args.use_combined_embeddings:
        embedding_dino = generate_new_dino_embedding_model(pca_file_name='./car_quality_models/pca_model_DINOv2.pkl', 
                                                           device=device, expected_sequence_length=expected_sequence_length)
        combined_embedding_classifier = generate_combined_embedding_estimator(
            mlp_model_name=f"./car_quality_models/car_quality_model_combined.pkl",
            device=device,
            expected_sequence_length=expected_sequence_length*2
        )
        combined_embedding_classifier.mlp_model = prepare_model_for_uncertainty_estimation(combined_embedding_classifier.mlp_model)
    
    start_index = args.objects_start
    end_index = args.objects_start + args.num_objects
    # check if the end index is greater than the length of the dataframe
    if end_index > len(combined_render_df):
        end_index = len(combined_render_df)
    combined_render_df_part = combined_render_df.iloc[start_index:end_index]
    amount_of_images = len(combined_render_df_part)
    gpu_batch_size = args.gpu_batch_size
    num_processing_batches = int(amount_of_images / gpu_batch_size) 

    # Split DataFrame into batches
    df_batches = np.array_split(combined_render_df_part, num_processing_batches)
    siglip_uncertainty_df = pd.DataFrame(columns=["sha256", "predicted_label", "output_score", "original_label",
                                                "uncertainty_entropy", "uncertainty_mutual_info", 
                                                "uncertainty_variation_ratio"])
    if args.use_combined_embeddings:
        combined_uncertainty_df = pd.DataFrame(columns=["sha256", "predicted_label", "output_score", "original_label",
                                                    "uncertainty_entropy", "uncertainty_mutual_info", 
                                                    "uncertainty_variation_ratio"])
    for df_batch in tqdm(df_batches, desc="Processing batches"):
        all_images, val_labels, val_uids = process_render_batch(df_batch)
        with torch.no_grad():
            siglip_embeddings = siglip_embedding_classifier.embed_image(all_images)
        batch_siglip_uncertainty_df = estimate_uncertainty_batch(siglip_embedding_classifier, siglip_embeddings, 
                                                                 val_labels, val_uids, device, gpu_batch_size)
        siglip_uncertainty_df = pd.concat([siglip_uncertainty_df, batch_siglip_uncertainty_df], ignore_index=True)

        if args.use_combined_embeddings:
            with torch.no_grad():
                dino_embeddings = embedding_dino(all_images)
            combined_embeddings = torch.cat((siglip_embeddings, dino_embeddings), dim=1)
            batch_combined_uncertainty_df = estimate_uncertainty_batch(combined_embedding_classifier, combined_embeddings, 
                                                                       val_labels, val_uids, device, gpu_batch_size)
            combined_uncertainty_df = pd.concat([combined_uncertainty_df, batch_combined_uncertainty_df], ignore_index=True)

    siglip_uncertainty_df.drop(columns=['original_label'], inplace=True)
    siglip_uncertainty_df.to_csv(f'./siglip_uncertainty_df_{start_index}_{end_index}.csv', index=False)
    
    if args.use_combined_embeddings:
        combined_uncertainty_df.drop(columns=['original_label'], inplace=True)
        combined_uncertainty_df.to_csv(f'./combined_uncertainty_df_{start_index}_{end_index}.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects_start", default=0, type=int)
    parser.add_argument("--num_objects", default=500000, type=int)
    parser.add_argument("--gpu_batch_size", default=500, type=int)
    parser.add_argument("--use_combined_embeddings", action="store_true")
    parser.add_argument("--render_df_path", default='./data/combined_oxl_renders_df.csv', type=str)
    args = parser.parse_args()

    main(args)