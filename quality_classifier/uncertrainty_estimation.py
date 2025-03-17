import torch 
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm


def enable_dropout(model):
    """
    Go through all modules and activate dropout modules
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def prepare_model_for_uncertainty_estimation(model): 
    """
    Prepare the model for uncertainty estimation by enabling dropout with evaluation mode
    """
    # Ensure the model is in evaluation mode
    model.eval() 
    enable_dropout(model)
    return model


def predictive_entropy(output):
    """
    Calculates the predictive entropy for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the predictive entropy for each 
        sample in the batch.
    """
    eps = 1e-8  # For numerical stability, avoid log(0)
    mean_probs = output.mean(dim=0)  # Average probabilities over samples
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    return entropy


def mutual_information(output):
    """
    Calculates the mutual information for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the mutual information for each 
        sample in the batch.
    """
    eps = 1e-8
    mean_probs = output.mean(dim=0)
    entropy_of_mean = - \
        torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
    expected_entropy = - \
        torch.mean(torch.sum(output * torch.log(output + eps), dim=-1), dim=0)
    mutual_info = entropy_of_mean - expected_entropy
    return mutual_info

def variation_ratio(output):
    """
    Calculates the variation ratio for a batch of samples.
    Args:
        output: Tensor of shape (num_samples, batch_size, num_classes) containing 
                probabilities for each class.
    Returns:
        Tensor of shape (batch_size,) containing the variation ratio for each 
        sample in the batch.
    """
    output = output.cpu()
    predicted_classes = torch.argmax(output, dim=-1)  # [num_samples, batch_size]
    most_frequent_class = torch.mode(
        predicted_classes, dim=0).values  # [batch_size]
    est_variation_ratio = 1 - (torch.sum(predicted_classes ==
                               most_frequent_class, dim=0) / output.shape[0])  # [batch_size]
    return est_variation_ratio

def create_uncertainty_estimates(model, val_embeddings, val_labels, val_uids, device="cpu", batch_size=200):
    """
    Creates uncertainty estimates for model predictions using Monte Carlo sampling.

    This function performs multiple forward passes through the model with dropout enabled
    to estimate prediction uncertainty using various metrics including predictive entropy,
    mutual information, and variation ratio.

    Args:
        model (nn.Module): The neural network model to evaluate
        val_embeddings (torch.Tensor or numpy.ndarray): Validation embeddings of shape [N, seq_len, embed_dim]
        val_labels (list or numpy.ndarray): Ground truth labels for validation data
        val_uids (list): Unique identifiers for each validation sample
        device (str, optional): Device to run computations on. Defaults to "cpu"
        batch_size (int, optional): Batch size for processing. Defaults to 200

    Returns:
        pandas.DataFrame: DataFrame containing per-sample predictions and uncertainty metrics with columns:
            - sha256: Sample identifier
            - predicted_label: Model's predicted class
            - output_score: Prediction confidence score
            - original_label: Ground truth label
            - uncertainty_entropy: Predictive entropy
            - uncertainty_mutual_info: Mutual information
            - uncertainty_variation_ratio: Variation ratio
    """
    object_uncertainty_df = pd.DataFrame(columns=["sha256", "predicted_label", "output_score", "original_label",
                                                  "uncertainty_entropy", "uncertainty_mutual_info", 
                                                  "uncertainty_variation_ratio"])
    for i in range(len(val_uids) // batch_size):
        embeddings = val_embeddings[i*batch_size:(i+1)*batch_size]
        # check if the batch embedding is torch tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = np.array(embeddings)
            embeddings = torch.tensor(embeddings)
        embeddings = embeddings.to(device)
        uids_of_embeddings = val_uids[i*batch_size:(i+1)*batch_size]
        output_list = []
        for internal_iteration in range(500):
            with torch.no_grad():
                output = model(embeddings)
                output = output.detach().cpu()
            output_list.append(output)
        output = torch.stack(output_list, dim=0)
        output_avg = output.mean(dim=0)
        predictions = torch.argmax(output_avg, dim=1)
        list_of_predictions = predictions.cpu().numpy()
        list_of_predictions = list_of_predictions.tolist()
        labels = np.array(val_labels[i*batch_size:(i+1)*batch_size])
        labels = labels.tolist()
        output_average_list = output_avg[:, 1].cpu().numpy().tolist()
        output_entropy = predictive_entropy(output)
        output_mutual_info = mutual_information(output)
        output_variation_ratio = variation_ratio(output)
        new_row = pd.DataFrame({
            "sha256": uids_of_embeddings,
            "predicted_label": list_of_predictions,
            "output_score": output_average_list,
            "original_label": labels,
            "uncertainty_entropy": output_entropy.detach().cpu().numpy(),
            "uncertainty_mutual_info": output_mutual_info.detach().cpu().numpy(),
            "uncertainty_variation_ratio": output_variation_ratio.detach().cpu().numpy()
        })
        object_uncertainty_df = pd.concat([object_uncertainty_df, new_row], ignore_index=True)
    return object_uncertainty_df