import torch
import random
import numpy as np
from collections import deque  # Double-ended queue for replay memory

from environment import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from utils import plot

# Define constants for replay memory and training
MAX_MEMORY = 100_000  # Maximum size of memory buffer
BATCH_SIZE = 1000  # Batch size for training
LR = 0.001  # Learning rate for the optimizer


class Agent:
    """
    This class implements the agent responsible for learning to play the Snake game using a Replay Q-Network (RQN).
    """

    # Initialize the agent
    def __init__(self):
        """
        Initializes the agent's core components:
            - n_games: Tracks the number of games played.
            - epsilon: Controls exploration (random actions) vs. exploitation (choosing the best-known action).
            - gamma: The discount factor that prioritizes immediate rewards over long-term rewards.
            - memory: Stores experiences (state, action, reward, next state) in a deque for experience replay.
            - The model is the neural network that predicts Q-values for actions, and the trainer handles updating the
              model's weights using Q-learning.
        """
        self.n_games = 0  # Counter for number of games played
        self.epsilon = 0  # Exploration rate (starts with high exploration)
        self.gamma = 0.9  # Discount rate (between 0 and 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # Memory buffer for experience replay
        self.model = Linear_QNet(11, 256, 3)  # Neural network with 11 inputs, 256 hidden units, 3 outputs (actions)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        This method extracts the current state of the game (such as the position of the snake, food, and potential
        dangers) and encodes it into a numerical format (a list of features). The agent uses this information to decide
        the next action.
        :param game:
        :return:
        """

        # Get the state of the game, including positions of the snake and food
        head = game.snake[0]  # Snake's head position
        point_l = Point(head.x - 20, head.y)  # Left point
        point_r = Point(head.x + 20, head.y)  # Right point
        point_u = Point(head.x, head.y - 20)  # Up point
        point_d = Point(head.x, head.y + 20)  # Down point

        # Get the direction the snake is moving in
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Create the state array: 11 features (danger, direction, food location)
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Current direction of the snake
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Location of the food relative to the snake's head
            game.food.x < game.head.x,  # Food is left
            game.food.x > game.head.x,  # Food is right
            game.food.y < game.head.y,  # Food is above
            game.food.y > game.head.y  # Food is below
        ]

        return np.array(state, dtype=int)  # Return the state as a numpy array

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the experiences (state, action, reward, next state) in memory for replay. This helps the agent learn
        from past-experiences.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """

        # Store the experience in memory (state, action, reward, next state, done flag)
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        """
        This method trains the agent using experience replay. If enough experiences have been stored in memory, it
        samples a batch and trains the model on it. Otherwise, it trains on all the stored experiences.
        :return:
        """

        # Train on a random batch of experience from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Random sample from memory
        else:
            mini_sample = self.memory  # Use all memory if not enough samples

        # Extract state, action, reward, next state, and done flag from samples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        This method trains the agent immediately after each step, using the most recent experience. It updates the model
        based on the current state, action taken, reward received, and the next state.
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """

        # Train on the most recent experience
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        This method determines the action the agent will take. It either chooses a random action (exploration) or
        selects the best action based on the current state (exploitation) by predicting Q-values using the neural
        network. As the agent plays more games, it explores less and exploits more.
        :param state:
        :return:
        """

        # Choose an action (exploration vs. exploitation)
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games  # Decrease exploration as more games are played
        final_move = [0, 0, 0]  # Action representation: [straight, right, left]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # Random action (exploration)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # Convert state to tensor
            prediction = self.model(state0)  # Predict action using the neural network
            move = torch.argmax(prediction).item()  # Choose action with the highest Q-value
            final_move[move] = 1

        return final_move


if __name__ == '__main__':
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old game state
        state_old = agent.get_state(game)

        # Get an action from the agent
        final_move = agent.get_action(state_old)

        # Perform the action and get the new state and reward
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train the agent with short-term memory (single step)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Store the experience in memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:  # If game over, train the long memory, and plot the result
            # Reset the game and train on long-term memory (batch training)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save the model if a new high score is achieved
            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Record:', record)

            # Update the score plots
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
