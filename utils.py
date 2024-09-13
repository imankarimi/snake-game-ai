"""
The helper function plot is responsible for visualizing the training progress over time. It plots the scores and mean
scores after each game to monitor the agent's learning performance.
"""

import matplotlib.pyplot as plt
from IPython import display

plt.ion()  # Turn on interactive mode

def plot(scores, mean_scores):
    """
     This uses matplotlib to dynamically update a graph that shows the agent's scores over time. It also displays the
     average score, helping track improvements as the agent trains.
    :param scores:
    :param mean_scores:
    :return:
    """
    display.clear_output(wait=True)  # Clear previous output
    plt.clf()  # Clear the current figure
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot the scores and mean scores
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')

    # Set the y-axis limits to ensure it doesn't go below 0
    plt.ylim(ymin=0)

    # Add annotations for the latest score and mean score
    if len(scores) > 0:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]), ha='center', va='bottom')
    if len(mean_scores) > 0:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]), ha='center', va='bottom')

    # Show legend
    plt.legend()

    plt.draw()  # Draw the updated figure
    plt.pause(0.1)  # Pause to allow for the update
