# Snake Game AI using Reinforcement Learning (Linear Q-Network)

This project implements the classic Snake game with a reinforcement learning agent that learns how to play using a **Replay Q-Network (RQN)** and **Linear Q-Learning**. The agent is trained through trial and error by receiving rewards for beneficial actions (like eating food) and penalties for harmful actions (like hitting a wall).

## Table of Contents
- [Overview](#overview)
- [Reinforcement Learning](#reinforcement-learning)
- [Linear Q-Network (Q-Learning)](#linear-q-network-q-learning)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Training the Agent](#training-the-agent)
- [Dependencies](#dependencies)
- [Integration and Modifications](#integration-and-modifications)

## Overview

The Snake game consists of an AI agent that learns to play autonomously through **reinforcement learning**. The game logic is written using the Pygame library, while the agent uses a neural network for decision-making. The agent is trained using Q-Learning, where it learns to predict the best possible action based on the current state of the game.

### Key Components:
- **Game Environment**: Manages the Snake game logic (movement, food placement, collision detection).
- **Reinforcement Learning Agent**: Learns to play the game using a neural network and Q-learning.
- **Linear Q-Network**: A fully connected neural network that predicts Q-values (expected future rewards) for each possible action.
- **Q-Trainer**: Handles the training process, updating the neural network's weights based on the agent’s experiences.

## Reinforcement Learning

Reinforcement learning (RL) is a machine learning paradigm where an agent interacts with an environment to maximize cumulative rewards. The agent makes decisions at each time step based on the current state of the environment, receives rewards, and updates its strategy to improve future actions.

In the context of Snake:
- **State**: Represents the current situation of the game (snake position, food location, etc.).
- **Action**: The moves the snake can make (turn left, turn right, or continue straight).
- **Reward**: A positive reward is given for eating food, while a negative reward is given for hitting the walls or its own body.
- **Goal**: The agent's goal is to learn an optimal policy that maximizes its cumulative reward by playing the game efficiently.

### Q-Learning
Q-Learning is a reinforcement learning technique that allows the agent to learn by maximizing the expected reward for each action. It uses a **Q-Value function**, which predicts the future reward for taking a particular action in a particular state. The agent updates these Q-values over time, using the Bellman equation:

```
Q_new = reward + gamma * max(Q(next state))
```

Where:
- `reward` is the immediate reward received after taking an action.
- `gamma` is the discount factor (how much future rewards are valued compared to immediate rewards).
- `Q(next state)` is the estimated value of the next state.

## Linear Q-Network (Q-Learning)

The **Linear Q-Network** is a simple feed-forward neural network that approximates the Q-value function. It takes the game’s current state as input and outputs the Q-values for all possible actions (turn left, go straight, or turn right).

### Network Structure:
- **Input Layer**: The game’s state, which includes information about the snake’s position, food location, and any potential danger.
- **Hidden Layer**: A fully connected layer with ReLU activation that processes the state information.
- **Output Layer**: Three outputs corresponding to the three possible actions. The Q-values indicate how beneficial each action is in the current state.

The network is trained using **backpropagation** to minimize the difference between predicted Q-values and target Q-values, based on the agent’s experiences during gameplay.

---

## Project Structure

```
.
├── game.py              # Snake game environment (Pygame implementation)
├── agent.py             # Reinforcement Learning agent using Linear Q-Network
├── model.py             # Neural network (Linear Q-Net) and Q-learning trainer
├── helper.py            # Helper functions (for plotting training progress)
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
└── model/               # Folder for saving the trained model
```

### Files:
- **`game.py`**: Contains the Snake game logic (handling movement, placing food, detecting collisions).
- **`agent.py`**: Defines the reinforcement learning agent, which interacts with the game environment, learns from rewards, and makes decisions.
- **`model.py`**: Defines the neural network model (Linear Q-Net) and the training process (Q-learning).
- **`helper.py`**: Provides a utility function for plotting the scores and training progress.
- **`model/`**: This folder stores the trained model (saved as `model.pth`).

---

## How to Use

### 1. Clone the Repository:
```
git clone https://github.com/imankarimi/snake-game-ai.git
cd snake_rl
```

### 2. Install Dependencies:
Install the required dependencies using pip:
```
pip install -r requirements.txt
```

### 3. Run the Game with Reinforcement Learning:
Run the training process:
```
python agent.py
```

This will start the game and train the reinforcement learning agent. The agent will initially make random moves but will improve its performance over time by learning from its actions.

---

## Training the Agent

The agent interacts with the game environment (Snake) by making decisions and receiving rewards. It uses **experience replay**, storing past experiences in memory and using a batch of them to train the neural network at each step. This helps prevent the agent from overfitting to the most recent experiences.

### Training Process:
1. **Get State**: The agent observes the current state of the game (snake's position, direction, food location, and potential dangers).
2. **Get Action**: The agent predicts the Q-values for each possible action using the neural network, then chooses the best action (or explores randomly based on `epsilon`).
3. **Take Action**: The agent performs the chosen action and moves the snake.
4. **Get Reward**: The agent receives a reward for the action (positive for eating food, negative for collisions).
5. **Train**: The agent uses this experience to update the neural network, improving future predictions.

### Visualization:
You can monitor the agent's learning progress using the real-time plot generated by the `helper.py` script, which tracks the score and the mean score over time.

---

## Dependencies

Make sure to have the following dependencies installed. You can install them using the `requirements.txt` file:

- `pygame`: For rendering the Snake game.
- `torch`: For building and training the neural network (PyTorch).
- `matplotlib`: For visualizing training progress.

Install all dependencies with:
```
pip install -r requirements.txt
```

---

## Integration and Modifications

If you want to integrate this project into your own system or modify it, here are a few pointers:

### Modifying the Neural Network:
You can easily change the architecture of the neural network in `model.py`. For example, you can add more hidden layers or change the size of the existing ones.

### Customizing the Game:
In `game.py`, you can modify the game mechanics (e.g., changing the grid size, snake speed, or how the food is placed) to experiment with different environments.

### Saving and Loading Models:
The agent automatically saves the best model (based on the highest score) to the `model/` directory. You can modify this behavior by adjusting the `save` method in the `Linear_QNet` class. To load a previously saved model, use `torch.load()` in the `train()` function.

### Hyperparameters:
The learning rate, discount factor (`gamma`), and other hyperparameters are defined in `agent.py`. You can tweak these parameters to experiment with the agent's learning behavior.

---

## Conclusion

This project demonstrates the use of reinforcement learning to train an AI agent to play Snake autonomously. The agent uses a simple **Linear Q-Network** to approximate the Q-value function, and the training is based on Q-learning. By interacting with the game environment, the agent improves its performance over time, making smarter decisions and achieving higher scores.

Feel free to explore and modify the code to enhance the agent’s performance or experiment with different game environments. Happy coding!

---

## References

1. GitHub Repository for the original Python Snake Game: [patrickloeber/python-fun](https://github.com/patrickloeber/python-fun)
2. YouTube Tutorial on Q-Learning for Snake: [Code Basics: Snake AI](https://www.youtube.com/watch?v=L8ypSXwyBds)

---