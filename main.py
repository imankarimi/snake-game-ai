from agent import Agent
from environment import SnakeGameAI
from utils import plot

SPEED = 100


if __name__ == '__main__':
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(speed=SPEED)

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
