"""
This setup allows your reinforcement learning agent to learn from its actions and improve over time by updating its
neural network model, which predicts the best action to take based on the current game state.
"""

import torch
import torch.nn as nn  # Import neural network module
import torch.optim as optim  # Optimizers for updating model parameters
import torch.nn.functional as F  # Functional module for common operations (e.g., activation functions)
import numpy as np
import os  # For file operations like saving the model


# Define the neural network model for Q-learning
class Linear_QNet(nn.Module):
    """
    This class defines the neural network architecture used by the agent to predict Q-values for each action in a given
    state.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Sets up the structure of the network with two linear layers:
            - The first layer takes the state as input and passes it through a hidden layer of neurons.
            - The second layer produces the output, representing the Q-values for each possible action (e.g., move
              straight, turn left, turn right).
        :param input_size: This represents the number of features in the state. For example, in your Snake
        game, it includes information like the direction of the snake, danger detection, and food location.
        :param hidden_size: A fully connected layer with `hidden_size` neurons, activated by ReLU
        (Rectified Linear Unit).
        :param output_size: This represents the number of possible actions the agent can take. In your case,
         the snake can go straight, turn left, or turn right.
        """

        # Initialize the parent class (nn.Module) to set up the neural network
        super().__init__()
        # First linear layer: input_size (state space) to hidden_size neurons
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Second linear layer: hidden_size neurons to output_size (action space)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass through the network. The input (state) is passed through the first linear layer,
        activated using the ReLU function, and then through the second linear layer to generate the output (Q-values
        for actions).
        :param x:
        :return:
        """

        # Forward pass through the network
        # Apply ReLU activation after the first layer
        x = F.relu(self.linear1(x))
        # Output layer (linear activation)
        x = self.linear2(x)
        return x  # Return the final output (Q-values for each action)

    def save(self, file_name='model.pth'):
        """
        This function saves the model's parameters (weights and biases) to a file. The model is saved when the agent
        achieves a new high score, allowing for future reuse without retraining.
        :param file_name:
        :return:
        """

        # Save the model's state to a file (used when reaching a high score)
        model_folder_path = './model'  # Define the folder path for saving the model
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # Create the folder if it doesn't exist

        file_name = os.path.join(model_folder_path, file_name)  # Full path to the model file
        torch.save(self.state_dict(), file_name)  # Save the model's state dict (weights and biases)


# Define a class to handle training the neural network (Q-learning)
class QTrainer:
    """
    This class handles the training process for the neural network, applying Q-learning to update the model's weights
    """

    def __init__(self, model, lr, gamma):
        """
        Initializes the trainer with the neural network model, learning rate (lr), and discount factor (gamma). The
        optimizer updates the model's weights, and the criterion defines the loss function (Mean Squared Error) used to
        calculate the error in predicted Q-values.
        :param model:
        :param lr: (Learning Rate) Controls how much to update the model weights at each step.
        :param gamma: (Gamma) The discount factor that determines how much future rewards are considered.
        """

        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor (for future rewards)
        self.model = model  # The neural network model (Linear_QNet)
        # Adam optimizer to update model parameters based on the loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Mean Squared Error (MSE) loss function for Q-value prediction
        self.criterion = nn.MSELoss()

    # Method to train the network for a single step
    def train_step(self, state, action, reward, next_state, done):
        """
        This function implements the key part of the Q-learning algorithm (Performs a single step of training):
            * It computes the predicted Q-values for the current state and then updates them using the Q-learning
             formula: Q_new = reward + gamma * max(next Q-value). This formula encourages the agent to prioritize
             actions that maximize future rewards.

            * The loss between predicted and target Q-values is calculated and backpropagated through the network to
             adjust its weights, improving the agent's future predictions.

            1- Predicted Q-values: The model predicts the Q-values for each action based on the current state.
            2- Target Q-values: The target Q-values are computed based on the reward and the maximum predicted Q-value
             for the next state. If the episode ends (`done` is `True`), the Q-value is just the immediate reward.
            3- Backpropagation: The difference (loss) between predicted and target Q-values is calculated using Mean
             Squared Error (MSE). The gradients are computed, and the optimizer updates the model's weights to reduce
             this loss.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """

        # Convert the state and next_state from a list of numpy arrays to a single numpy array
        state = np.array(state)
        next_state = np.array(next_state)

        # Convert the numpy arrays to PyTorch tensors for model training
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # Handle the case where we have only one sample (unsqueeze to match batch size)
        if len(state.shape) == 1:
            # Add an extra dimension for batch processing (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)  # Convert done to a tuple

        # 1. Predicted Q-values for the current state
        pred = self.model(state)  # Pass the current state through the model

        # Clone the predicted Q-values to modify them
        target = pred.clone()
        for idx in range(len(done)):  # Iterate through each sample in the batch
            Q_new = reward[idx]  # Initialize Q_new with the reward
            if not done[idx]:  # If the episode isn't done, add the future reward
                # Q_new = reward(r) + gamma(y) * max(next predicted Q-value) -> only do this if not done
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the Q-value for the taken action (torch.argmax gets the action index)
            target[idx][torch.argmax(action).item()] = Q_new

        # 2. Perform the backpropagation and optimization steps
        # Zero the gradients from the previous step
        self.optimizer.zero_grad()
        # Calculate the loss between the predicted and target Q-values
        loss = self.criterion(target, pred)
        # Backpropagate the loss to compute the gradients
        loss.backward()
        # Update the model parameters based on the gradients
        self.optimizer.step()
