import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class CarDataset(Dataset):
    """
    A PyTorch Dataset for car image embeddings and their quality labels.

    This dataset holds pre-computed embeddings of car images along with their corresponding
    quality labels and unique identifiers. It implements the basic functionality required
    for a PyTorch Dataset, allowing it to be used with DataLoaders for batch processing.

    Args:
        embeddings (list or numpy.ndarray): Pre-computed embeddings of car images
        labels (list or numpy.ndarray): Quality labels corresponding to the embeddings
        uids (list): Unique identifiers for each car image sample (sha256 for Objaverse XL objects)

    Example:
        >>> dataset = CarDataset(embeddings, labels, uids)
        >>> embedding, label = dataset[0]  # Get first sample
    """
    def __init__(self, embeddings, labels, uids):
        self.embeddings = embeddings
        self.labels = labels
        self.uids = uids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    A PyTorch Sampler that generates balanced batches with equal samples from each class.

    This sampler ensures that each batch contains an equal number of samples from each class,
    which can help prevent class imbalance issues during training. It shuffles samples within
    each class and generates batches by taking an equal number of samples from each class. It
    did not imporve the performance in our case. 

    Args:
        dataset (Dataset): Dataset to sample from. Must have a 'labels' attribute containing
                          the class labels for each sample.
        batch_size (int): Size of each batch. Must be divisible by the number of unique classes.

    Example:
        >>> sampler = BalancedBatchSampler(dataset, batch_size=32)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)

    Note:
        The total number of batches is limited by the size of the smallest class, as the sampler
        needs to maintain equal representation from all classes.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get labels from dataset
        self.labels = list(dataset.labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                for label in self.labels_set}
        
        # Ensure batch size is divisible by number of classes
        assert batch_size % len(self.labels_set) == 0, \
            f'Batch size must be divisible by number of classes ({len(self.labels_set)})'
        
        # Calculate samples per class in each batch
        self.samples_per_class = batch_size // len(self.labels_set)
        
        # Calculate number of batches
        self.min_class_size = min([len(indices) for indices in self.label_to_indices.values()])
        self.num_batches = self.min_class_size // self.samples_per_class
        
    def __iter__(self):
        # Create a copy of indices for each class that we can modify
        indices_per_class = {label: indices.copy() 
                           for label, indices in self.label_to_indices.items()}
        
        # Shuffle indices for each class
        for indices in indices_per_class.values():
            np.random.shuffle(indices)
        
        # Generate batches
        for _ in range(self.num_batches):
            batch_indices = []
            # Sample equally from each class
            for label in self.labels_set:
                batch_indices.extend(indices_per_class[label][:self.samples_per_class])
                indices_per_class[label] = indices_per_class[label][self.samples_per_class:]
            
            # Shuffle indices within the batch
            np.random.shuffle(batch_indices)
            yield batch_indices
            
    def __len__(self):
        return self.num_batches


class CarImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path_dict, quality_labels_dict, transform=None, color_transform=None):
        """
        Args:
            image_path_dict (dict): Dictionary with uid as key and list of image paths as value
            quality_labels_dict (dict): Dictionary with uid as key and quality label as value
            transform (callable, optional): Optional transform to be applied on images
        """
        self.uids = list(image_path_dict.keys())
        self.image_path_dict = image_path_dict
        self.quality_labels_dict = quality_labels_dict
        self.transform = transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        image_paths = self.image_path_dict[uid]

        # Load and concatenate images horizontally
        images = [Image.open(path).convert("RGB") for path in image_paths]

        image_size = images[0].size
        concatenated_image = Image.new(
            'RGB', (image_size[0] * 2, image_size[1] * 2))
        for i, img in enumerate(images):
            x = i % 2
            y = i // 2
            if self.transform:
                img = self.transform(img)
            concatenated_image.paste(img, (x * image_size[0], y * image_size[1]))

        if self.color_transform:
            concatenated_image = self.color_transform(concatenated_image)

        # convert to tensor
        concatenated_image = torch.tensor(np.array(concatenated_image))
        

        # Get label
        label = self.quality_labels_dict[uid]

        return concatenated_image, label