import streamlit as st
import torch
from model import get_dataset, LogicGateNet, train_model, evaluate_model, plot_decision_boundary
import matplotlib.pyplot as plt

"""
Designing the Streamlit UI for user interaction with the Neural Network Model
"""

# Title for the Streamlit app
st.title("Interactive Neural Network for Logic Gates")

# Dropdown for selecting the logic gate
gate = st.selectbox("Select Logic Gate", ['XOR', 'AND', 'OR', 'NOR', 'NAND'])

# Sliders for adjusting hyperparameters: learning_rate, hidden_units, and epochs
learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)  # Slider for learning rate
hidden_units = st.slider("Number of Hidden Units", 1, 10, 2)  # Slider for number of hidden units in the hidden layer
epochs = st.slider("Number of Epochs", 1000, 10000, 1000)  # Slider for number of epochs

# Button to start training and evaluating the model
if st.button("Train and Evaluate"):
    model, loss = train_model(gate, learning_rate, hidden_units, epochs)  # Train the model
    accuracy, predicted = evaluate_model(model, gate)  # Evaluate the model
    
    st.session_state.model = model  # Store the model in session state for later use
    st.write(f"Loss: {loss:.4f}")  # Display the final loss
    st.write(f"Accuracy: {accuracy * 100:.2f}%")  # Display the accuracy of the model
    
    st.write("Predicted Outputs:")  # Display the predicted outputs
    st.write(predicted)
    
    st.write("Actual Outputs:")  # Display the actual outputs from the dataset
    X, y = get_dataset(gate)
    st.write(y.numpy())
    
    # Plot and display the decision boundary
    st.write("Decision Boundary:")
    fig, ax = plt.subplots()
    plot_decision_boundary(model, gate)
    st.pyplot(fig)

# Section for user to input logical conditions
st.write("Input Logical Conditions:")
input_1 = st.number_input("Input 1", min_value=0, max_value=1, value=0, step=1)  # Input 1
input_2 = st.number_input("Input 2", min_value=0, max_value=1, value=0, step=1)  # Input 2

# Button to predict the output for given logical inputs
if st.button("Predict"):
    if 'model' in st.session_state:
        model = st.session_state.model  # Retrieve the trained model from session state
        input_tensor = torch.tensor([[input_1, input_2]], dtype=torch.float32)  # Create a tensor for the inputs
        with torch.no_grad():  # Ensure no gradient is computed during prediction
            prediction = model(input_tensor)  # Make a prediction
            predicted_value = (prediction > 0.5).float().item()  # Convert the output to binary (0 or 1)
        st.write(f"Prediction for inputs ({input_1}, {input_2}): {int(predicted_value)}")  # Display the prediction
    else:
        st.write("Please train the model first.")  # Prompt the user to train the model if not done already

