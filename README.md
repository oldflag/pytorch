# PyTorch Classification Models Project

This project showcases a collection of neural network models implemented in PyTorch, aimed at tackling classification tasks on popular datasets like CIFAR-10, MNIST, and STL10. The project includes models such as Multilayer Perceptron (MLP), Convolutional Neural Networks (CNN), and deeper CNN architectures. Additionally, it includes various plots such as loss and accuracy curves, as well as absolute mean weight and gradient curves over epochs. These plots help in evaluating not only the performance of the models but also in analyzing issues like vanishing gradients.


## Features

- **Multiple Dataset Support:** Supports CIFAR-10, MNIST, and STL10 datasets.
- **Model Variants:** Includes implementations of MLP, CNN, and deeper CNN models.
- **Training and Evaluation:** Functions to train models, evaluate them on test data, and visualize their performance.
- **Utility Functions:** Includes functions for plotting training history, model analysis (weights and gradients), and displaying confusion matrices.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with PyTorch and torchvision. This project uses CUDA-enabled devices for computation; ensure you have a compatible GPU and CUDA installed to use GPU capabilities.

### Installation

1. **Clone the repository**:   
   git clone https://github.com/oldflag/pytorch   

2. **Install dependencies**:   
   pip install torch torchvision matplotlib tqdm   

### Running the Project

To run the project, execute the main script for the desired dataset from the command line:

For CIFAR-10:   
python Classification_CIFAR10.py   

For MNIST:   
python Classification_MNIST.py   

For STL10:   
python Classification_STL10.py   

### Configuration

Each dataset has its own configuration file (e.g., \`config_CIFAR10.json\`, \`config_MNIST.json\`, \`config_STL10.json\`). You can modify these files to switch between different models and datasets. Set the \`model_type\` and \`dataset\` variables in the configuration file to one of the supported models and datasets, respectively.

### Example Configuration (\`config_CIFAR10.json\`)

json
{
  "DEVICE": "cuda",
  "BATCH_SIZE": 64,
  "LR": 0.002,
  "EPOCH": 10,
  "TRAIN_RATIO": 0.8,
  "criterion": "CrossEntropyLoss",
  "model_type": "CNN_CIFAR10",
  "dataset": "CIFAR10",
  "new_model_train": false,
  "save_model_path": "./results/CNN_CIFAR10_CIFAR10.pt",
  "save_history_path": "./results/CNN_CIFAR10_history_CIFAR10.pt",
  "transform_mean": [0.485, 0.456, 0.406],
  "transform_std": [0.229, 0.224, 0.225]
}   

## Project Structure   

- **Classification_CIFAR10.py**: Script to train and evaluate models on the CIFAR-10 dataset.
- **Classification_MNIST.py**: Script to train and evaluate models on the MNIST dataset.
- **Classification_STL10.py**: Script to train and evaluate models on the STL10 dataset.
- **models.py**: Contains definitions for the MLP, CNN, and deeper CNN classes.
- **modeling_functions.py**: Includes the training loop, data loading, and evaluation functions.
- **plot_functions.py**: Contains functions to plot training history and model analyses.
- **config_CIFAR10.json**: Configuration file for the CIFAR-10 dataset.
- **config_MNIST.json**: Configuration file for the MNIST dataset.
- **config_STL10.json**: Configuration file for the STL10 dataset.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests if you have suggestions for improvements or have added support for additional datasets or models.
