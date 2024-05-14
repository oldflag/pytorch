# import torch
# import matplotlib.pyplot as plt

# def plot_saved_training_history(history_path):
#     """Plot the saved training history."""
#     history = torch.load(history_path)
#     plt.figure()
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# def plot_model_analysis(model, test_loader):
#     """Analyze the model's performance and plot confusion matrix."""
#     y_true = []
#     y_pred = []
#     model.eval()
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = model(inputs)
#             pred = outputs.argmax(dim=1, keepdim=True)
#             y_true.extend(targets.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())
    
#     cm = get_conf(y_true, y_pred)
#     plot_confusion_matrix(cm)


# def plot_saved_training_history(save_history_path, device, epochs):
#     # Load the training history from the specified path
#     train_history = torch.load(save_history_path, map_location=device)
#     loss_history = train_history['loss_history']

#     # Plotting the training and validation loss
#     plt.plot(range(1, epochs+1), loss_history['train'], label='Train Loss')
#     plt.plot(range(1, epochs+1), loss_history['val'], '--', label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title("Train & Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.show(block=False)

# def plot_model_analysis(file_path, device, epochs):
#     # Load the model analysis data
#     train_history = torch.load(file_path, map_location=device)
#     model_analysis_data = train_history["model_history"]

#     # Prepare to collect mean weight and gradient values for each layer
#     weight_history = {}
#     gradient_history = {}

#     # Extract data
#     for epoch_data in model_analysis_data:
#         for layer, metrics in epoch_data.items():
#             if 'Mean Weights' in metrics and metrics['Mean Weights'] is not None:
#                 if layer not in weight_history:
#                     weight_history[layer] = []
#                 weight_history[layer].append(metrics['Mean Weights'])
                
#             if 'Mean Gradients' in metrics and metrics['Mean Gradients'] is not None:
#                 if layer not in gradient_history:
#                     gradient_history[layer] = []
#                 gradient_history[layer].append(metrics['Mean Gradients'])

#     # Plotting
#     plt.figure(figsize=(12, 5))

#     # Plot mean weight history
#     plt.subplot(1, 2, 1)
#     for layer, weights in weight_history.items():
#         plt.plot(range(1, epochs+1), weights, label=f'{layer} weights')
#     plt.title('Mean Weight History by Layer')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Weight Value')
#     plt.legend()
#     plt.grid(True)

#     # Plot mean gradient history
#     plt.subplot(1, 2, 2)
#     for layer, gradients in gradient_history.items():
#         plt.plot(range(1, epochs+1), gradients, label=f'{layer} gradients')
#     plt.title('Mean Gradient History by Layer')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean Gradient Value')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     # plt.show()

import torch
import matplotlib.pyplot as plt

def plot_saved_training_history(save_history_path):
    """
    Plots the saved training and validation loss history.

    Args:
    save_history_path (str): Path to the saved training history file.

    Returns:
    None
    """
    # Load the training history from the specified path
    train_history = torch.load(save_history_path)
    loss_history = train_history['loss_history']

    # Infer the number of epochs from the length of the loss history
    epochs = len(loss_history['train'])

    # Plotting the training and validation loss
    plt.plot(range(1, epochs + 1), loss_history['train'], label='Train Loss')
    plt.plot(range(1, epochs + 1), loss_history['val'], '--', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_model_analysis(file_path):
    """
    Plots the model analysis data, including mean weights and gradients for each layer.

    Args:
    file_path (str): Path to the file containing model analysis data.

    Returns:
    None
    """
    # Load the model analysis data
    train_history = torch.load(file_path)
    model_analysis_data = train_history["model_history"]

    # Prepare to collect mean weight and gradient values for each layer
    weight_history = {}
    gradient_history = {}

    # Extract data
    for epoch_data in model_analysis_data:
        for layer, metrics in epoch_data.items():
            if 'Mean Weights' in metrics and metrics['Mean Weights'] is not None:
                if layer not in weight_history:
                    weight_history[layer] = []
                weight_history[layer].append(metrics['Mean Weights'])
                
            if 'Mean Gradients' in metrics and metrics['Mean Gradients'] is not None:
                if layer not in gradient_history:
                    gradient_history[layer] = []
                gradient_history[layer].append(metrics['Mean Gradients'])

    # Infer the number of epochs from the length of the weight history of the first layer
    first_layer = next(iter(weight_history))
    epochs = len(weight_history[first_layer])

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot mean weight history
    plt.subplot(1, 2, 1)
    for layer, weights in weight_history.items():
        plt.plot(range(1, epochs + 1), weights, label=f'{layer} weights')
    plt.title('Mean Weight History by Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Weight Value')
    plt.legend()
    plt.grid(True)

    # Plot mean gradient history
    plt.subplot(1, 2, 2)
    for layer, gradients in gradient_history.items():
        plt.plot(range(1, epochs + 1), gradients, label=f'{layer} gradients')
    plt.title('Mean Gradient History by Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Gradient Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)
