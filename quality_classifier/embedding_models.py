from transformers import AutoImageProcessor, AutoModel, AutoProcessor, SiglipModel
import pickle
import torch


def generate_siglip_embedding_model(device="cpu"):
    """
    Creates and returns a function that generates SigLIP embeddings for images.

    This function initializes a SigLIP model and processor, and returns a callable
    that processes images through the model to generate embeddings. The embedding
    function handles batches of images and runs without gradient computation.

    Args:
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Returns:
        callable: A function that takes an image or batch of images and returns
                 their SigLIP embeddings as a flattened tensor.

    Example:
        embed_fn = generate_siglip_embedding_model(device="cuda")
        embeddings = embed_fn(images)  # Shape: [batch_size, embedding_dim]
    """
    embed_model_siglip = SiglipModel.from_pretrained(
        "nielsr/siglip-base-patch16-224").to(device)
    processor_siglip = AutoProcessor.from_pretrained(
        "nielsr/siglip-base-patch16-224")

    def embed_siglip(image):
        with torch.no_grad():
            inputs = processor_siglip(images=image,
                                      return_tensors="pt").to(device)
            image_features = embed_model_siglip.get_image_features(**inputs)
            image_features = image_features.view(image_features.size(0), -1)
        return image_features

    return embed_siglip


def generate_new_dino_embedding_model(pca_file_name=None, device="cpu", expected_sequence_length=4):
    """
    Creates and returns a function that generates DINOv2 embeddings for images with PCA dimensionality reduction.

    This function initializes a DINOv2 model and processor, loads a PCA model, and returns a callable
    that processes images through both models to generate reduced-dimension embeddings. The embedding
    function handles batches of images, applies PCA transformation, and runs without gradient computation.

    Args:
        pca_file_name (str, optional): Path to the PCA model file. If None, uses default path.
            Defaults to None.
        device (str, optional): Device to run the model on. Defaults to "cpu".
        expected_sequence_length (int, optional): Expected sequence length for reshaping embeddings.
            Defaults to 4.

    Returns:
        callable: A function that takes an image or batch of images and returns their
                 PCA-reduced DINOv2 embeddings as a tensor with shape 
                 [batch_size, sequence_length, reduced_dim].

    Example:
        embed_fn = generate_new_dino_embedding_model(device="cuda")
        embeddings = embed_fn(images)  # Shape: [batch_size, sequence_length, reduced_dim]
    """
    processor_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    embed_model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    # load the pca model
    if pca_file_name is None:
        pca_file = '../car_quality_models/pca_model_DINOv2.pkl'
    else:
        pca_file = pca_file_name
    with open(pca_file, 'rb') as f:
        pca = pickle.load(f)

    def embedding_dino(image):
        with torch.no_grad():
            inputs = processor_dino(image, return_tensors="pt").to(device) # padding=True
            image_features = embed_model_dino(**inputs)
            image_features = image_features.last_hidden_state 
            original_shape = image_features.shape
            # flatten the features using the original batch size by separating the sequence length
            original_batch_size = original_shape[0] // expected_sequence_length
            image_features = image_features.view(original_batch_size, -1)
            # reduce the dimensionality of the features
            image_features = pca.transform(image_features.detach().cpu().numpy())
            # reshape the features back to the original shape
            image_features = image_features.reshape(original_batch_size, expected_sequence_length, -1)
            # make torch tensor
            image_features = torch.tensor(image_features).to(device)
        return image_features

    return embedding_dino
