import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame and set up fonts for rendering text
pygame.init()
font = pygame.font.Font('assets/font/arial.ttf', 25)  # Set up a font for score display

# Enum for snake direction
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a named tuple to store points (x, y) on the grid
Point = namedtuple('Point', 'x, y')

# Define RGB color values for drawing the snake and food
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Define constants for block size and game speed
BLOCK_SIZE = 20
SPEED = 100

class SnakeGameAI:
    """
    This class handles the core mechanics of the Snake game, such as rendering the snake, handling user input, placing
    food, and detecting collisions.
    """

    # Constructor method to initialize the game screen and snake state
    def __init__(self, w=640, h=480, speed=SPEED):
        """
        This initializes the game environment. It sets up the game display, dimensions, and initializes a reset function
        that defines the snake's starting position and direction.
        :param w:
        :param h:
        :param speed:
        """

        self.w = w  # Width of game screen
        self.h = h  # Height of game screen
        self.speed = speed  # Speed of game
        # Initialize the display and set window title
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()  # Game clock to control the speed
        self.reset()  # Reset game state

    def reset(self):
        """
        Resets the game to its initial state. The snake starts in the middle of the screen, moving to the right, and is
        given a length of three blocks. It also resets the score and places the food randomly.
        :return:
        """

        # Initialize game state, such as snake position and direction
        self.direction = Direction.RIGHT
        # Starting position of the snake (head at the center)
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]  # Snake body
        self.score = 0  # Initial score
        self.food = None  # Placeholder for food
        self._place_food()  # Place food at a random position
        self.frame_iteration = 0  # Count frames to avoid infinite loops

    def _place_food(self):
        """
        This function places food at a random position on the grid, ensuring it doesn't appear on the snake's body.
        :return:
        """

        # Place food at a random position on the grid, avoiding the snake's body
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()  # If food spawns on the snake, re-place it

    def play_step(self, action):
        """
        Advances the game by one step. It handles the snake’s movement, checks for collisions, updates the score, and
        controls the frame rate. It also detects whether the snake has collided with the walls or itself, determining
        if the game is over. The method returns a reward (positive for eating food, negative for collisions) and updates
        the game state.
        :param action:
        :return:
        """

        # Advance the game by one step based on the action provided
        self.frame_iteration += 1  # Increment frame counter
        # Collect user input, mainly to allow quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake's head in the specified direction
        self._move(action)
        self.snake.insert(0, self.head)  # Add new head position to the snake's body

        # Check if the game is over due to collision or exceeding frame limit
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Penalize if the game ends
            return reward, game_over, self.score

        # Check if snake has eaten food
        if self.head == self.food:
            self.score += 1  # Increase score
            reward = 10  # Reward for eating food
            self._place_food()  # Place new food
        else:
            self.snake.pop()  # Remove tail if no food is eaten

        # Update the game UI and control the frame rate
        self._update_ui()
        self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """
        Checks whether the snake has collided with the walls or itself. If the snake's head crosses the boundaries of
        the game window or intersects with its body, the game is over.
        :param pt:
        :return:
        """

        # Check if the snake collides with walls or itself
        if pt is None:
            pt = self.head
        # Check boundary collisions
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Handles the graphical update of the game, including drawing the snake, food, and score on the screen.
        :return:
        """

        # Clear the screen by filling it with black color
        self.display.fill(BLACK)

        # Draw the snake's body and head
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))  # Inner rectangle for design

        # Draw the food as a red square
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Render and display the score on the top-left corner
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Update the display

    def _move(self, action):
        """
        Moves the snake's head based on the action (direction) provided. The direction is adjusted based on input,
        allowing the snake to go straight, turn left, or turn right. The method updates the snake's head position
        accordingly.
        :param action:
        :return:
        """

        # Determine the new direction based on action (straight, right, or left)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]  # Order of directions
        idx = clock_wise.index(self.direction)  # Get current direction's index

        # Update direction based on action: straight, right, or left
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Continue in the same direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Turn right
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4  # Turn left
            new_dir = clock_wise[next_idx]

        self.direction = new_dir  # Set the new direction

        # Update the position of the head based on direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)  # Update the head position
