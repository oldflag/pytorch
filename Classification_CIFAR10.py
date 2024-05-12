import torch
from torch import optim
import matplotlib.pyplot as plt
from modeling_functions import Train, Test, Test_plot, get_conf, plot_confusion_matrix, count_params,load_data
from torchvision import transforms
from models import MLP, CNN, CNN_deep  # Import model classes
from plot_functions import plot_saved_training_history, plot_model_analysis

# global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 2e-3
EPOCH = 10
TRAIN_RATIO = 0.8
criterion = torch.nn.CrossEntropyLoss()
model_type = "MLP"
dataset = "CIFAR10"
new_model_train= True
save_model_path = f"./results/{model_type}_{dataset}.pt"
save_history_path = f"./results/{model_type}_history_{dataset}.pt"


    # Dictionary mapping model type names to model classes
model_classes = {
    'MLP': MLP,
    'CNN': CNN,
    'CNN_deep': CNN_deep
}

def main():


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load data here (not shown, assuming functions are in modeling_functions.py)
    train_DL, val_DL, test_DL = load_data(dataset, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO, transform_train=transform_train, transform_test=transform_test)
    
    # Create the model instance using the dictionary
    if model_type in model_classes:
        model = model_classes[model_type]().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if new_model_train:
        optimizer = optim.Adam(model.parameters(), lr=LR)
        Train(model, train_DL, val_DL, criterion, optimizer, EPOCH, BATCH_SIZE, TRAIN_RATIO, save_model_path, save_history_path)
        torch.save(model, save_model_path)
    
    # Load training history for plotting
    plot_saved_training_history(save_history_path, DEVICE, EPOCH)

    plot_model_analysis(save_history_path, DEVICE, EPOCH)
    
    # Test the model
    load_model = torch.load(save_model_path, map_location=DEVICE)
    Test(load_model, test_DL, criterion)
    print(count_params(load_model))
    # Test_plot(load_model, test_DL)
    
    # confusion_matrix = get_conf(model, test_DL)
    # plot_confusion_matrix(confusion_matrix)
    plt.show() # Show the plots at the end
if __name__ == "__main__":
    main()
