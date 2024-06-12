# Interactive Neural Network for Logic Gates

This repository contains a Streamlit-based interactive interface for training and evaluating a neural network model on various logic gates (XOR, AND, OR, NOR, NAND).

## Overview

This interface allows users to:
1. Select a logic gate (XOR, AND, OR, NOR, NAND).
2. Adjust hyperparameters (learning rate, number of hidden units, number of epochs).
3. Train and evaluate the neural network model.
4. Visualize the decision boundary of the trained model.
5. Input custom logical conditions to predict the output using the trained model.

## Requirements
- Python 3.7 or higher
- PyTorch
- Streamlit
- Matplotlib
- NumPy

## Running the Application
To run the Streamlit app, follow these steps:
1. Open a terminal and navigate to the project directory if you haven't already:
   ```sh
   cd neural-networks-logic-gate
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
3. A new tab will open in your default web browser displaying the Streamlit interface.

This interactive tool provides an intuitive way to understand how neural networks can learn and predict the behavior of different logic gates. By adjusting the hyperparameters and visualizing the decision boundaries, users can gain insights into the model's learning process and performance.
