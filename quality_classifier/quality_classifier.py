import math 
import pickle
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from quality_classifier.embedding_models import generate_siglip_embedding_model, generate_new_dino_embedding_model

class SequenceEmbeddingEstimator(nn.Module):
    """A PyTorch module that combines image embedding generation with sequence-based classification.

    This class takes images, generates embeddings (e.g. SigLIP or DINOv2) using a provided embedding function, reshapes them
    into sequences, and passes them through an MLP model for classification. It supports both direct
    prediction and intermediate embedding access.

    Args:
        mlp_model (nn.Module): The MLP model that processes sequence embeddings for classification.
        embedding_fnc (callable): Function that generates embeddings from input images.
        expected_sequence_length (int): Expected length of sequences for reshaping embeddings.

    Methods:
        reshape_embedding(embedding): Reshapes flat embeddings into sequence format.
        forward(image): Forward pass from image to classification output.
        forward_embeddings(embedding): Forward pass starting from pre-computed embeddings.
        embed_image(image): Generates and reshapes embeddings from images.
        forward_with_embedding(image): Returns both classification output and embeddings.
        predict(image): Returns predicted class indices for input images.
    """
    def __init__(self, mlp_model, embedding_fnc, expected_sequence_length):
        super(SequenceEmbeddingEstimator, self).__init__()
        self.mlp_model = mlp_model
        self.embedding_fnc = embedding_fnc
        self.expected_sequence_length = expected_sequence_length

    def reshape_embedding(self, embedding):
        img_batch_size, embed_dim = embedding.shape
        # reshape to [original_batch_size, expected_sequence_length, embed_dim]
        original_batch_size = img_batch_size // self.expected_sequence_length
        return embedding.view(original_batch_size, self.expected_sequence_length, embed_dim)

    def forward(self, image):
        embedding = self.embed_image(image)
        with torch.no_grad():
            outputs = self.mlp_model(embedding)
            return outputs
    
    def forward_embeddings(self, embedding):
        with torch.no_grad():
            outputs = self.mlp_model(embedding)
            return outputs
    
    def embed_image(self, image):
        with torch.no_grad():
            embedding = self.embedding_fnc(image)
            embedding = self.reshape_embedding(embedding)
        return embedding

    def forward_with_embedding(self, image):
        embedding = self.embed_image(image)
        with torch.no_grad():
            outputs = self.mlp_model(embedding)
            return outputs, embedding

    def predict(self, image):
        outputs = self.forward(image)
        return outputs.argmax(dim=1)

class CombinedSequenceEmbeddingEstimator(nn.Module):
    """
    A neural network module that combines two sequence embeddings and performs classification.

    This class takes two sequence embeddings, concatenates them along the embedding dimension,
    and passes them through an MLP model for classification. It handles reshaping of embeddings
    to match the expected sequence format.

    Args:
        mlp_model (nn.Module): The MLP model that processes the combined embeddings.
        expected_sequence_length (int): Expected length of sequences for reshaping embeddings.

    Methods:
        reshape_embedding(embedding): Reshapes flat embeddings into sequence format.
        forward(embedding_1, embedding_2): Forward pass combining two embeddings for classification.
        predict(embedding): Returns predicted class indices for input embeddings.
    """
    def __init__(self, mlp_model, expected_sequence_length):
        super(CombinedSequenceEmbeddingEstimator, self).__init__()
        self.mlp_model = mlp_model
        self.expected_sequence_length = expected_sequence_length

    def reshape_embedding(self, embedding):
        img_batch_size, embed_dim = embedding.shape
        original_batch_size = img_batch_size // self.expected_sequence_length
        return embedding.view(original_batch_size, self.expected_sequence_length, embed_dim)

    def forward(self, embedding_1, embedding_2):
        embedding = torch.cat((embedding_1, embedding_2), dim=-1)
        with torch.no_grad():
            outputs = self.mlp_model(embedding)
            return outputs

    def predict(self, embedding):
        outputs = self.forward(embedding)
        return outputs.argmax(dim=1)

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to sequence embeddings.

    This module implements the sinusoidal positional encoding described in 
    "Attention Is All You Need" (Vaswani et al., 2017). It creates position-dependent
    patterns that help the model understand the sequential order of elements.

    Args:
        embed_dim (int): Dimension of the embeddings
        max_len (int, optional): Maximum sequence length. Defaults to 5000.

    Shape:
        - Input: (batch_size, seq_len, embed_dim)
        - Output: (batch_size, seq_len, embed_dim)

    Example:
        >>> pos_encoder = PositionalEncoding(512)
        >>> x = torch.randn(32, 100, 512)  # [batch_size, seq_len, embed_dim]
        >>> output = pos_encoder(x)  # Adds positional encoding
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)  # [1, max_len, embed_dim]
        
        # Create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1), :]  # pe will broadcast across batch dimension

class CustomTransformer(nn.Module):
    """
    A custom transformer model for sequence classification.

    This module implements a transformer encoder architecture with positional encoding,
    followed by sequence pooling and classification layers. It processes sequential data
    through self-attention mechanisms and outputs class probabilities.

    Args:
        input_dim (int): Dimension of input embeddings
        dim_feedforward (int): Dimension of feedforward network in transformer layers
        output_dim (int): Number of output classes
        nhead (int): Number of attention heads in transformer layers
        num_layers (int): Number of transformer encoder layers
        dropout (float, optional): Dropout probability. Defaults to 0.1

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, output_dim)

    Example:
        >>> model = CustomTransformer(512, 2048, 2, 8, 6)
        >>> x = torch.randn(32, 10, 512)  # [batch_size, seq_len, input_dim]
        >>> output = model(x)  # Shape: [32, 2]
    """
    def __init__(self, input_dim, dim_feedforward, output_dim, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embed_dim = input_dim 
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, batch_first=True, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(self.embed_dim))
        self.norm_out = nn.LayerNorm(self.embed_dim)
        self.output_layer = nn.Linear(self.embed_dim, output_dim)  

    def forward(self, x):
        # encode the sequence
        x = self.positional_encoding(x)
        x = self.encoder(x)
        # take the mean over the sequence length
        x = torch.mean(x, dim=1)
        x = self.norm_out(x)
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x

class MLPMixer(nn.Module):
    """
    MLP-Mixer architecture for sequence classification.

    This module implements the MLP-Mixer architecture which processes sequences using
    alternating token-mixing and channel-mixing MLPs. It applies layer normalization
    and residual connections throughout, followed by sequence pooling and classification.

    Args:
        input_dim (int): Dimension of input embeddings
        sequence_dim (int): Length of input sequences
        hidden_dim (int): Dimension of hidden layers in mixing MLPs
        output_dim (int): Number of output classes
        num_layers (int): Number of mixer layers
        dropout (float, optional): Dropout probability. Defaults to 0.1

    Shape:
        - Input: (batch_size, seq_len, input_dim)
        - Output: (batch_size, output_dim)

    Example:
        >>> model = MLPMixer(512, 4, 2048, 2, 6)
        >>> x = torch.randn(32, 4, 512)  # [batch_size, seq_len, input_dim]
        >>> output = model(x)  # Shape: [32, 2]
    """
    def __init__(self, input_dim, sequence_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super(MLPMixer, self).__init__()
        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, sequence_dim),
            nn.Dropout(dropout)
        )
        self.token_layers = nn.ModuleList(
            [self._create_token_layer(input_dim) for _ in range(num_layers)]
        )
        self.channel_layers = nn.ModuleList(
            [self._create_channel_layer(sequence_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(input_dim, output_dim)

    def _create_token_layer(self, input_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            self.token_mixing,
        )
    
    def _create_channel_layer(self, sequence_dim):
        return nn.Sequential(
            nn.LayerNorm(sequence_dim),
            self.channel_mixing,
        )

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        for i in range(len(self.token_layers)):
            # first process all sequence pieces in parallel
            x_token_mixed_input = self.input_layer_norm(x)
            x_token_mixed = x_token_mixed_input.view(batch_size * seq_len, embed_dim)
            x_token_mixed = self.token_layers[i](x_token_mixed)
            x_token_mixed = x_token_mixed.view(batch_size, seq_len, embed_dim)
            # now process all embedding_pieces in parallel, by reshaping to [batch_size*embed_dim, seq_len]
            x_channel_mixed_input = x + self.input_layer_norm(x_token_mixed)
            x_channel_mixed = x_channel_mixed_input.view(batch_size*embed_dim, seq_len)
            x_channel_mixed = self.channel_layers[i](x_channel_mixed)
            x_channel_mixed = x_channel_mixed.view(batch_size, seq_len, embed_dim)
            x = x + x_channel_mixed
        # get average over sequence 
        x = torch.mean(x, dim=1)
        x_out = self.output_layer(x)
        return torch.softmax(x_out, dim=1)

class SequenceMLP(nn.Module):
    """
    A multi-layer perceptron model for processing sequences of embeddings.

    This model applies multiple layers of transformations to each element in the input sequence
    independently, then averages across the sequence dimension before final classification.
    Each layer consists of layer normalization, dimensionality expansion through a hidden layer,
    non-linearity (GELU), and dimension reduction back to input size, with dropout for regularization.

    Args:
        input_dim (int): Dimension of input embeddings
        hidden_dim (int): Dimension of hidden layer in each transformation block
        output_dim (int): Number of output classes
        num_layers (int): Number of transformation layers to apply
        dropout (float, optional): Dropout probability. Defaults to 0.25

    Shape:
        - Input: (batch_size, sequence_length, input_dim)
        - Output: (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.25):
        super(SequenceMLP, self).__init__()
        self.embedding_layers = nn.ModuleList(
            [self._create_layer(input_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def _create_layer(self, input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [batch, sequence, embed_dim]
        batch_size, seq_len, embed_dim = x.shape
        
        # Reshape to process each sequence element independently
        x_reshaped = x.view(-1, embed_dim)  # [batch * sequence, embed_dim]
        for embedding_layer in self.embedding_layers:
            x_reshaped = embedding_layer(x_reshaped)  # [batch * sequence, embed_dim]
        # reshape back 
        processed = x_reshaped.view(batch_size, seq_len, embed_dim)  # [batch, sequence, embed_dim]
        # get average over sequence 
        x_pooled = torch.mean(processed, dim=1)  # [batch, embed_dim]
        
        # Final classification
        x_out = self.output_layer(x_pooled)
        x_out = self.softmax(x_out)
        return x_out


def generate_combined_embedding_estimator(mlp_model_name, expected_sequence_length=4, device="cpu"):
    """
    Creates a combined sequence embedding estimator by loading a pre-trained model.

    Args:
        mlp_model_name (str): Path to the pickled model file
        expected_sequence_length (int, optional): Expected length of input sequences. Defaults to 4
        device (str, optional): Device to run model on. Defaults to "cpu"

    Returns:
        CombinedSequenceEmbeddingEstimator: Loaded model wrapped in a combined sequence estimator
    """
    with open(mlp_model_name, 'rb') as f:
        model_combined = pickle.load(f)
    model_combined.to(device)
    model_combined.eval()
    combined_embedding_estimator = CombinedSequenceEmbeddingEstimator(
        model_combined, expected_sequence_length=expected_sequence_length)
    return combined_embedding_estimator


def generate_siglip_embedding_estimator(mlp_model_name, expected_sequence_length=4, device="cpu"):
    """
    Creates a SigLIP-based sequence embedding estimator by loading a pre-trained model.

    Args:
        mlp_model_name (str): Path to the pickled model file
        expected_sequence_length (int, optional): Expected length of input sequences. Defaults to 4
        device (str, optional): Device to run model on. Defaults to "cpu"

    Returns:
        SequenceEmbeddingEstimator: Loaded model wrapped in a sequence estimator with SigLIP embeddings
    """
    with open(mlp_model_name, 'rb') as f:
        model_siglip = pickle.load(f)
    model_siglip.to(device)
    model_siglip.eval()
    embed_siglip = generate_siglip_embedding_model(device=device)
    siglip_embedding_estimator = SequenceEmbeddingEstimator(
        model_siglip, embed_siglip, expected_sequence_length=expected_sequence_length)
    return siglip_embedding_estimator


def generate_dino_embedding_estimator(mlp_model_name, load_pca=False, pca_file_name=None, expected_sequence_length=4, device="cpu"):
    """
    Creates a DINOv2-based sequence embedding estimator by loading a pre-trained model.

    Args:
        mlp_model_name (str): Path to the pickled model file
        load_pca (bool, optional): Whether to load PCA model for dimensionality reduction. Defaults to False
        pca_file_name (str, optional): Path to PCA model file. Required if load_pca is True. Defaults to None
        expected_sequence_length (int, optional): Expected length of input sequences. Defaults to 4
        device (str, optional): Device to run model on. Defaults to "cpu"

    Returns:
        SequenceEmbeddingEstimator: Loaded model wrapped in a sequence estimator with DINOv2 embeddings
    """
    with open(mlp_model_name, 'rb') as f:
        model_dino = pickle.load(f)
    model_dino.to(device)
    model_dino.eval()
    embedding_dino = generate_new_dino_embedding_model(load_pca=load_pca, pca_file_name=pca_file_name, device=device)
    dino_embedding_estimator = SequenceEmbeddingEstimator(
        model_dino, embedding_dino, expected_sequence_length=expected_sequence_length)
    return dino_embedding_estimator
