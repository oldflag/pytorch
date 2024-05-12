
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Set the device for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(dataset_name, batch_size, train_ratio=0.8, transform_train=None, transform_test=None):
    """Loads data for a given dataset with specified transforms, splits it into train, validation and test sets."""
    # Define default transformations if none provided
    if transform_train is None:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    # Load the appropriate dataset
    dataset_classes = {
        "cifar10": datasets.CIFAR10,
        "mnist": datasets.MNIST,
        "stl10": datasets.STL10
    }
    
    if dataset_name.lower() not in dataset_classes:
        raise ValueError("Unsupported dataset. Please add the dataset in the load_data function.")
    
    DatasetClass = dataset_classes[dataset_name.lower()]
    root_dir = f'./data/{dataset_name.upper()}'
    
    # Download and load the datasets
    train_and_valid = DatasetClass(root=root_dir, train=True, download=True, transform=transform_train)
    test_dataset = DatasetClass(root=root_dir, train=False, download=True, transform=transform_test)
    
    # Split train dataset into train and validation
    train_size = int(len(train_and_valid) * train_ratio)
    valid_size = len(train_and_valid) - train_size
    train_dataset, valid_dataset = random_split(train_and_valid, [train_size, valid_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader, test_loader

def Train(model, train_DL, val_DL, criterion, optimizer, EPOCH, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path, **kwargs):
    """Trains a model using the specified data loaders, optimizer, and loss criterion, also handles optional LR scheduling."""
    scheduler = StepLR(optimizer, step_size=kwargs.get("LR_STEP", 30), gamma=kwargs.get("LR_GAMMA", 0.7)) if "LR_STEP" in kwargs else None

    loss_history = {"train": [], "val": []}
    acc_history = {"train": [], "val": []}
    model_history = []

    best_loss = float('inf')
    for ep in range(EPOCH):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {ep+1}, current_LR = {current_lr}")

        model.train()
        train_loss, train_acc, _, model_changes = loss_epoch(model, train_DL, criterion, optimizer)
        loss_history["train"].append(train_loss)
        acc_history["train"].append(train_acc)
        model_history.append(model_changes)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, _, model_changes = loss_epoch(model, val_DL, criterion)
            loss_history["val"].append(val_loss)
            acc_history["val"].append(val_acc)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "model": model,
                    "ep": ep,
                    "optimizer": optimizer,
                    "scheduler": scheduler
                }, save_model_path)

        if scheduler:
            scheduler.step()

        print(f"train loss: {train_loss:.5f}, val loss: {val_loss:.5f}, "
              f"train acc: {train_acc:.1f} %, val acc: {val_acc:.1f} %, "
              f"time: {time.time() - epoch_start:.0f} s")
        print("-" * 20)

    torch.save({
        "loss_history": loss_history,
        "acc_history": acc_history,
        "model_history": model_history,
        "EPOCH": EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAIN_RATIO": TRAIN_RATIO
    }, save_history_path)

def loss_epoch(model, DL, criterion, optimizer=None):
    """Calculates loss and accuracy for an epoch, updates model if optimizer is provided."""
    total_samples = len(DL.dataset)
    running_loss = 0.0
    correct_predictions = 0
    
    for inputs, labels in tqdm(DL, leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
    
    average_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples * 100
    model_updates = analyze_model(model) if optimizer else None
    return average_loss, accuracy, correct_predictions, model_updates

def analyze_model(model):
    results = {}  # Dictionary to store results
    for name, module in model.named_modules():
        if any(isinstance(module, cls) for cls in [nn.Conv2d, nn.Linear, nn.BatchNorm2d]):
            total_params = sum(p.numel() for p in module.parameters())
            mean_weights = torch.mean(torch.tensor([p.data.abs().mean().item() for p in module.parameters() if p.requires_grad]))
            if any(p.grad is not None for p in module.parameters()):
                mean_grads = torch.mean(torch.tensor([p.grad.abs().mean().item() for p in module.parameters() if p.grad is not None]))
            else:
                mean_grads = None
            results[name] = {
                'Total Params': total_params,
                'Mean Weights': mean_weights,
                'Mean Gradients': mean_grads
            }
    return results

def Test(model, test_DL, criterion):
    """Evaluates the model on the test set and prints the loss and accuracy."""
    model.eval()
    test_loss, test_acc, correct_count, _ = loss_epoch(model, test_DL, criterion)
    print(f"Test loss: {test_loss:.5f}")
    print(f"Test accuracy: {correct_count}/{len(test_DL.dataset)} ({test_acc:.1f} %)")
    return test_acc

def Test_plot(model, test_DL):
    """Plots first six test images with predicted and true labels."""
    model.eval()
    inputs, labels = next(iter(test_DL))
    inputs = inputs.to(DEVICE)
    outputs = model(inputs)
    predictions = outputs.argmax(dim=1)
    inputs = inputs.to("cpu")  # Move inputs back to CPU for plotting
    
    plt.figure(figsize=(8, 4))
    for idx in range(6):
        ax = plt.subplot(2, 3, idx + 1, xticks=[], yticks=[])
        plt.imshow(inputs[idx].permute(1, 2, 0).squeeze(), cmap="gray")
        pred_class = test_DL.dataset.classes[predictions[idx]]
        true_class = test_DL.dataset.classes[labels[idx]]
        plt.title(f"{pred_class} ({true_class})", color=("green" if pred_class == true_class else "red"))
    plt.show()

def count_params(model):
    """Returns the count of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_conf(model, test_DL):
    """Generates a confusion matrix for the model predictions on the test dataset."""
    model.eval()
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in test_DL:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            for pred, true in zip(predictions, labels):
                confusion_matrix[pred, true] += 1

    return confusion_matrix.numpy()

def plot_confusion_matrix(confusion):
    """Plots a confusion matrix."""
    accuracy = np.trace(confusion) / np.sum(confusion) * 100
    confusion = confusion / np.sum(confusion, axis=1)[:, np.newaxis]  # Normalize confusion matrix
    plt.figure(figsize=(10, 7))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(False)
    for i in range(10):
        for j in range(10):
            plt.text(j, i, f"{confusion[i, j]:.2f}", horizontalalignment="center", color="white" if confusion[i, j] > 0.5 else "black")
    plt.show()