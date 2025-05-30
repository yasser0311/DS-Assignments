# Neural Network for Image Recognition of Letters A, B, and C

## Project Overview
This project implements a basic feedforward neural network from scratch using NumPy to classify binary patterns representing the letters A, B, and C. Each letter is represented as a 5×6 (30-pixel) grid flattened into a 1D array. The neural network is trained using backpropagation with a sigmoid activation function.

## Project Structure
1. **Dataset Creation**: Binary patterns for letters A, B, and C are defined
2. **Neural Network Implementation**: A two-layer neural network class with:
   - Weight initialization
   - Forward propagation
   - Backpropagation
   - Training loop
3. **Training**: The network is trained on the letter patterns
4. **Testing**: The trained network is tested on noisy versions of the letters
5. **Analysis**: Network weights are examined and random patterns are tested

## Key Components
- **NeuralNetwork Class**: Contains all neural network functionality
  - `__init__`: Initializes weights and biases
  - `sigmoid`: Activation function
  - `forward`: Performs forward propagation
  - `compute_loss`: Calculates binary cross-entropy loss
  - `backward`: Performs backpropagation
  - `train`: Runs the training loop
  - `predict`: Makes predictions on new data

## How to Use
1. Run all cells in order to:
   - Create the dataset
   - Initialize and train the neural network
   - View training progress (loss and accuracy plots)
   - Test the network on noisy patterns
   - Analyze the network's performance

## Parameters
- **Network Architecture**:
  - Input layer: 30 neurons (5×6 pixels)
  - Hidden layer: 10 neurons
  - Output layer: 3 neurons (one for each letter)
- **Training Parameters**:
  - Epochs: 10,000
  - Learning rate: 0.1

## Results
- The network achieves 100% accuracy on the training data
- It can correctly classify slightly noisy versions of the letters
- The training loss decreases steadily while accuracy increases

## Dependencies
- Python 3.x
- NumPy
- Matplotlib

## Note
The implementation uses only NumPy, with no external machine learning libraries, to demonstrate fundamental neural network concepts.

## Files
- `neural_network_classifier.ipynb`: Jupyter Notebook containing the complete implementation
- `README.md`: This file with project documentation

## Author
M Mohammed Yasser
+91 9360074131 
