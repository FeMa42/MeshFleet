import os
import argparse
import pickle
import numpy as np
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import accuracy_score

from quality_classifier.quality_trainer import CarDataset, BalancedBatchSampler
from quality_classifier.quality_classifier import CustomTransformer, MLPMixer, SequenceMLP


def main(args):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create output path if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project="car-quality-classifier",
        config={
            "model_type": args.model_type,
            "embedding_model": args.embedding_model,
            "hidden_dim": args.hidden_dim,
            "inmtransformer_mlp_ratio": args.inmtransformer_mlp_ratio,
            "output_dim": args.n_classes,
            "dropout": args.dropout,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "n_epochs": args.n_epochs,
            "use_balanced_sampling": args.use_balanced_sampling,
            "amount_of_embedding_splits_to_keep": args.amount_of_embedding_splits_to_keep
        }
    )

    # load embeddings from an HDF5 file 
    base_folder = './data/'
    ending = '_seq_4'
    if args.embedding_model == "combined":
        dino_filename = base_folder + 'car_model_embedding_' + "DINOv2" + ending + '_reduced.h5'
        with h5py.File(dino_filename, 'r') as f:
            all_dino_embeddings = f['embedding_dataset'][:]
        siglip_filename =  base_folder +  'car_model_embedding_' + "siglip" + ending + '.h5'
        with h5py.File(siglip_filename, 'r') as f:
            all_siglip_embeddings = f['embedding_dataset'][:]
        all_embeddings = np.concatenate((all_dino_embeddings, all_siglip_embeddings), axis=1)
        print(all_embeddings.shape)
    else:
        if args.embedding_model == "DINOv2":
            filename =  base_folder + 'car_model_embedding_' + args.embedding_model + ending + '_reduced.h5'
        else:
            filename =  base_folder + 'car_model_embedding_' + args.embedding_model + ending + '.h5'
        with h5py.File(filename, 'r') as f:
            all_embeddings = f['embedding_dataset'][:]
            print(all_embeddings.shape)
    # load the votes from an HDF5 file
    filename =  base_folder + 'car_model_votes.h5'
    with h5py.File(filename, 'r') as f:
        votes = f['vote_dataset'][:]
        votes = votes - 1
        print(votes.shape)
    # load the uids from an HDF5 file
    filename =  base_folder + 'car_model_uids.h5'
    with h5py.File(filename, 'r') as f:
        uids = f['uid_dataset'][:]
        print(uids.shape)
    
    # remap the votes from 5 and 4 to 2, from 3 and 2 to 1 and from 1 to 0 - 3 is the best quality
    n_classes = args.n_classes
    if n_classes == 2:
        new_votes = [1 if vote == 4 or vote == 3 else 0 for vote in votes]
    elif n_classes == 3:
        new_votes = [2 if vote == 4 or vote == 3 else 1 if vote == 2 or vote == 1 else 0 for vote in votes]
    elif n_classes == 4:
        new_votes = [3 if (vote == 4 or vote == 3) else 2 if (vote == 2) else 1 if (vote == 1) else 0 for vote in votes]
    elif n_classes == 5:
        new_votes = [4 if (vote == 4) else 3 if (vote == 3) else 2 if (vote == 2) else 1 if (vote == 1) else 0 for vote in votes]
    else:
        raise ValueError("Output dimension must be 3, 4 or 5")

    # get all the embeddings and labels and split them into training and validation sets
    all_labels = np.array(new_votes)
    n_samples = len(all_embeddings)
    amount_of_embedding_splits = 1
    amount_of_embedding_splits_to_keep = args.amount_of_embedding_splits_to_keep
    if amount_of_embedding_splits_to_keep > amount_of_embedding_splits:
        amount_of_embedding_splits_to_keep = amount_of_embedding_splits
    
    #  sample train and test set while making sure that we don't have augmented test samples in train set
    n_split_samples = n_samples // amount_of_embedding_splits
    indices = np.random.permutation(n_split_samples)
    first_train_indices = indices[:int(0.8 * n_split_samples)]
    first_val_indices = indices[int(0.8 * n_split_samples):]

    # apply the same split to all N times the same data
    train_indices = []
    val_indices = []
    for i in range(amount_of_embedding_splits_to_keep):
        train_indices_tmp = first_train_indices + i * n_split_samples
        val_indices_tmp = first_val_indices + i * n_split_samples
        train_indices.extend(train_indices_tmp)
        val_indices.extend(val_indices_tmp)

    print(len(train_indices), len(val_indices))

    train_embeddings = np.array([all_embeddings[i] for i in train_indices])
    train_labels = np.array([all_labels[i] for i in train_indices])
    train_uids = np.array([uids[i] for i in train_indices])
    print(f"shape of train_embeddings: {train_embeddings.shape}")

    val_embeddings = [all_embeddings[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    val_uids = [uids[i] for i in val_indices]
    train_dataset = CarDataset(train_embeddings, train_labels, train_uids)
    val_dataset = CarDataset(val_embeddings, val_labels, val_uids)

    # create dataloaders for the training and validation sets
    batch_size = args.batch_size  # Must be divisible by number of classes (3)

    if args.use_balanced_sampling:
        if batch_size % n_classes != 0:
            new_batch_size = batch_size - batch_size % n_classes
            print(f"Batch size {batch_size} is not divisible by {n_classes}, setting new batch size to {new_batch_size}")
            batch_size = new_batch_size
        sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define the model, loss function and optimizer
    input_dim = all_embeddings.shape[-1]
    if args.model_type == "transformer":
        model = CustomTransformer(input_dim=input_dim, dim_feedforward=args.hidden_dim, output_dim=n_classes, 
                                nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout).to(device)
        model_config = {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": n_classes,
            "dropout": args.dropout,
            "nhead": args.nhead,
            "num_layers": args.num_layers
        }
        with open(os.path.join(args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_seq_do_jit_sn.json"), 'w') as f:
            json.dump(model_config, f)
    elif args.model_type == "mixer":
        sequence_dim = all_embeddings.shape[1]
        model = MLPMixer(input_dim=input_dim, sequence_dim=sequence_dim, hidden_dim=args.hidden_dim, output_dim=n_classes, 
                        num_layers=args.num_layers, dropout=args.dropout).to(device)
        model_config = {
            "input_dim": input_dim,
            "sequence_dim": sequence_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": n_classes,
            "dropout": args.dropout,
            "num_layers": args.num_layers
        }
        with open(os.path.join(args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_seq_do_jit_sn.json"), 'w') as f:
            json.dump(model_config, f)
    elif args.model_type == "sequence":
        model = SequenceMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=n_classes, num_layers=args.num_layers, 
                            dropout=args.dropout).to(device)
        model_config = {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": n_classes,
            "dropout": args.dropout,
            "num_layers": args.num_layers
        }
        with open(os.path.join(args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_seq_do_jit_sn.json"), 'w') as f:
            json.dump(model_config, f)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Amount of parameters: {sum(p.numel() for p in model.parameters())}")

    # train the model
    n_epochs = args.n_epochs
    train_losses = []
    val_losses = []
    best_val_accuracy = 0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            embeddings, labels = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels) # + 0.001 * torch.norm(model.fc1.weight, p=1)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        val_accuracy_tmp = []
        for i, data in enumerate(val_loader):
            embeddings, labels = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_accuracy_tmp.append(accuracy_score(labels.cpu(), predicted.detach().cpu()))

        val_loss = running_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = np.mean(val_accuracy_tmp)
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}", f"Val Accuracy: {val_accuracy}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_name = os.path.join(args.output_path, f"best_model.pkl")
            with open(best_model_name, 'wb') as f:
                pickle.dump(model, f)
    
    # save the model as a pickle file
    with open(os.path.join(args.output_path, f"car_quality_model_{args.embedding_model}_{args.model_type}_seq.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    wandb.finish()

if __name__ == "__main__":
    # load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="mixer", type=str, choices=["transformer", "mixer", "sequence"])
    parser.add_argument("--output_path", default="./Experiments", type=str)
    parser.add_argument("--embedding_model", default="combined", type=str)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--inmtransformer_mlp_ratio", default=4, type=int)
    parser.add_argument("--n_classes", default=2, type=int, choices=[2, 3, 4, 5])
    parser.add_argument("--dropout", default=0.369, type=float)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--learning_rate", default=0.00046, type=float)
    parser.add_argument("--weight_decay", default=0.00029, type=float)
    parser.add_argument("--use_balanced_sampling", default=False, action="store_true")
    parser.add_argument("--amount_of_embedding_splits_to_keep", default=1, type=int)
    args = parser.parse_args()
    
    main(args)