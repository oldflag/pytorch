
# PyTorch Classification Models Project

This project showcases a collection of neural network models implemented in PyTorch, aimed at tackling classification tasks on popular datasets like CIFAR-10, MNIST, and STL10. The project includes models such as Multilayer Perceptron (MLP), Convolutional Neural Networks (CNN), and deeper CNN architectures.
Also it includes some plots such as loss & accouracy curve, absolute mean weight and gradient curve over epochs. With these plots, I wan to check not only performance of models but also vanishing gradients.

## Features

- **Multiple Dataset Support:** Currently supports CIFAR-10, with planned extensions for MNIST and STL10 datasets.
- **Model Variants:** Includes implementations of MLP, CNN, and CNN_deep models.
- **Training and Evaluation:** Functions to train models, evaluate them on test data, and visualize their performance.
- **Utility Functions:** Includes functions for plotting training history, model analysis (weights and gradients), and displaying confusion matrices.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with PyTorch and torchvision. This project uses CUDA-enabled devices for computation; ensure you have a compatible GPU and CUDA installed to use GPU capabilities.

### Installation

1. **Clone the repository**:
   \`\`\`bash
   git clone https://github.com/oldflag/pytorch   
   cd your-project-directory
   \`\`\`

2. **Install dependencies**:
   \`\`\`bash
   pip install torch torchvision matplotlib tqdm
   \`\`\`

### Running the Project

To run the project, execute the main script from the command line:

\`\`\`bash
python main.py
\`\`\`

### Configuration

Modify the \`main.py\` script to switch between different models and datasets. Set the \`model_type\` and \`dataset\` variables at the top of the script to one of the supported models and datasets, respectively.

## Structure

- **models.py**: Contains definitions for the MLP, CNN, and CNN_deep classes.
- **modeling_functions.py**: Includes the training loop, data loading, and evaluation functions.
- **plot_functions.py**: Contains functions to plot training history and model analyses.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests if you have suggestions for improvements or have added support for additional datasets or models.
