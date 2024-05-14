import torch
import json
from torch import optim, nn
import matplotlib.pyplot as plt
from modeling_functions import *
from torchvision import transforms
from models import CNN_STL10
from plot_functions import plot_saved_training_history, plot_model_analysis

# Load configuration from JSON file
with open('config_STL10.json', 'r') as f:
    config = json.load(f)

# Extract parameters from the config
DEVICE = config["DEVICE"] if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config["BATCH_SIZE"]
LR = config["LR"]
EPOCH = config["EPOCH"]
TRAIN_RATIO = config["TRAIN_RATIO"]
criterion = getattr(nn, config["criterion"])()
model_type = config["model_type"]
dataset = config["dataset"]
new_model_train = config["new_model_train"]
save_model_path = config["save_model_path"]
save_history_path = config["save_history_path"]
transform_mean = config["transform_mean"]
transform_std = config["transform_std"]

# Dictionary mapping model type names to model classes
model_classes = {
    # 'MLP_STL10': MLP_STL10,
    'CNN_STL10': CNN_STL10,
    # 'CNN_deep_STL10': CNN_deep_STL10
}

def main():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])
    
    # Load data
    train_DL, val_DL, test_DL = load_data(dataset, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO, transform_train=transform_train, transform_test=transform_test)
    
    # Create the model instance
    if model_type in model_classes:
        model = model_classes[model_type]().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if new_model_train:
        optimizer = optim.Adam(model.parameters(), lr=LR)
        Train(model, train_DL, val_DL, criterion, optimizer, EPOCH, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path)
        torch.save(model.state_dict(), save_model_path)
    
    # Load the model for testing and analysis
    model.load_state_dict(torch.load(save_model_path))
    Test(model, test_DL, criterion)


    # Plot
    plot_saved_training_history(save_history_path)
    plot_model_analysis(save_history_path)
    plt.show()

if __name__ == "__main__":
    main()
