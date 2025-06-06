import pygame
import numpy as np
import random

import assets
import configs
from objects.background import Background
from objects.bird import Bird
from objects.column import Column
from objects.floor import Floor
from objects.gameover_message import GameOverMessage
from objects.gamestart_message import GameStartMessage
from objects.score import Score


class FlappyBird:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (configs.SCREEN_WIDTH, configs.SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird Game v1.0.2")
        try:
            img = pygame.image.load('assets/icons/red_bird.png')
            pygame.display.set_icon(img)
        except pygame.error as e:
            print(
                f"Warning: Could not load icon 'assets/icons/red_bird.png': {e}")

        self.clock = pygame.time.Clock()
        self.column_create_event = pygame.USEREVENT
        self.running = True
        self.gameover = False
        self.gamestarted = False  # Will be set true by game.start() or game.reset()

        assets.load_sprites()
        assets.load_audios()

        self.sprites = pygame.sprite.LayeredUpdates()
        # Initialize bird, message, score. GameStartMessage will be killed by start/reset.
        self.bird, self.game_start_message, self.score = self.create_sprites()

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)
        # GameStartMessage is created here and shown until game.start() or game.reset() kills it
        return Bird(self.sprites), GameStartMessage(self.sprites), Score(self.sprites)

    def start(self):
        # This method is called by the RL script once at the beginning.
        if not self.gamestarted and not self.gameover:
            self.gamestarted = True
            if self.game_start_message.alive():  # Kill message if it exists
                self.game_start_message.kill()
            pygame.time.set_timer(self.column_create_event, 1500)

    def reset(self):
        # This method is called by the RL script when game over.
        self.gameover = False
        self.gamestarted = True  # Game is considered active immediately for the agent
        self.sprites.empty()  # Clear all old sprites

        # Re-create essential sprites for a new game
        self.bird, self.game_start_message, self.score = self.create_sprites()

        if self.game_start_message.alive():  # Kill the new GameStartMessage immediately
            self.game_start_message.kill()

        pygame.time.set_timer(self.column_create_event,
                              1500)  # Restart column spawning

    def quit(self):
        self.running = False
        pygame.quit()

    def step(self, action):
        reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False  # Signal main loop to exit
            if event.type == self.column_create_event:
                Column(self.sprites)

        if not self.running:  # If QUIT event was handled
            return None, None, 0  # Return neutral values, main loop should stop

        # Bird jumping action
        if action == 1:  # Jump
            if not self.gameover:
                # The RL agent controls game start via game.start() and game.reset()
                # So self.gamestarted should be True.
                # If not self.gamestarted: # This block might be redundant for RL
                #    self.start()
                self.bird.handle_event(pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_SPACE))
        elif action == 0:  # No jump
            pass

        self.screen.fill(0)  # Black background
        self.sprites.draw(self.screen)

        if self.gamestarted and not self.gameover:
            self.sprites.update()

        # Collision and game over logic
        if self.bird.check_collision(self.sprites) and not self.gameover:
            self.gameover = True
            # self.gamestarted = False # Keep True, reset manages this
            GameOverMessage(self.sprites)
            pygame.time.set_timer(self.column_create_event, 0)  # Stop columns
            assets.play_audio("hit")
            reward = -10  # Penalty for dying
        else:  # If no collision this step
            if not self.gameover:  # Only give survival reward if game is active
                reward = 1  # Default reward for surviving a step

        # Scoring points
        columns = [
            sprite for sprite in self.sprites if isinstance(sprite, Column)]
        for sprite in columns:
            if sprite.is_passed():  # is_passed should ensure one-time trigger
                self.score.value += 1
                assets.play_audio("point")
                reward = 10  # Higher reward for passing a column

        pygame.display.flip()
        self.clock.tick(configs.FPS)

        # Calculate state features for the RL agent
        # These are distances from the bird to the next relevant column/gap
        delta_x_to_column = None
        delta_y_to_gap = None

        # Original logic for finding the next column and distances:
        upcoming_columns = [
            column for column in columns if column.rect.x > self.bird.rect.x]
        if upcoming_columns:
            nearest_column = min(
                upcoming_columns, key=lambda col: col.rect.x - self.bird.rect.x)
            delta_x_to_column = nearest_column.rect.x - self.bird.rect.x
            gap_center_y = nearest_column.get_gap_center_y()
            # Bird's top Y to gap center Y
            delta_y_to_gap = abs(gap_center_y - self.bird.rect.y)
            # print(f"gap center y: {gap_center_y}") # Original debug print
        else:
            delta_y_to_gap = None  # No upcoming columns implies no specific gap

        # print(f"Distance from bird to column: {delta_x_to_column if delta_x_to_column is not None else 'No columns'}") # Original debug print
        # print(f"Distance from bird to gap center: {delta_y_to_gap if delta_y_to_gap is not None else 'No columns'}") # Original debug print

        return delta_x_to_column, delta_y_to_gap, reward


game = FlappyBird()

# Hyperparameters
# Learning rate (original was 0.9, typically smaller like 0.1)
alpha = 0.1
gamma = 0.95         # Discount factor
# Total training episodes (not strictly used by current loop structure)
num_episodes = 10000
# Max steps per episode (not strictly used by current loop structure)
max_steps = 100
epsilon = 0.3        # Exploration rate
epsilon_decay = 0.9995
epsilon_min = 0.01
action_size = 2      # 0: no jump, 1: jump
state_size = 2       # (delta_x_to_column, delta_y_to_gap)

# Discretization bins
bin_size_y = 25
# configs.SCREEN_HEIGHT is likely 512. num_y_bins = int(512 / 25) = 20. Max index 19.
num_y_bins = int(configs.SCREEN_HEIGHT / bin_size_y)

bin_size_x = 20
# configs.SCREEN_WIDTH is likely 288. num_x_bins = int(288 / 20) = 14. Max index 13.
num_x_bins = int(configs.SCREEN_WIDTH / bin_size_x)

# Q-table: num_x_bins possible values for discrete_x, num_y_bins for discrete_y
q_table = np.zeros((num_x_bins, num_y_bins, action_size))


def choose_action(current_state_tuple):  # state_tuple is (discrete_x, discrete_y)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)  # Explore: random action
    else:
        # Exploit: best action from Q-table for current_state_tuple
        return np.argmax(q_table[current_state_tuple[0], current_state_tuple[1], :])


def get_discrete(dx, dy):
    # Discretize delta_x_to_column
    if dx is not None:
        # Ensure dx is non-negative (it should be by definition of "upcoming")
        current_delta_x = max(0, dx)
        # Convert to bin index
        discrete_x = int(current_delta_x / bin_size_x)
        # Clip to ensure it's within bounds [0, num_x_bins - 1]
        discrete_x = min(discrete_x, num_x_bins - 1)
    else:
        # Default state if no column is upcoming (e.g., very start or after passing all)
        discrete_x = num_x_bins - 1  # Use the "farthest" bin

    # Discretize delta_y_to_gap
    if dy is not None:
        # dy is already abs(gap_center_y - bird.rect.y), so non-negative
        current_delta_y = dy
        discrete_y = int(current_delta_y / bin_size_y)
        # Clip to ensure it's within bounds [0, num_y_bins - 1]
        discrete_y = min(discrete_y, num_y_bins - 1)
    else:
        # Default state if no specific gap information
        # Use the "farthest" bin (or a middle one like num_y_bins // 2)
        discrete_y = num_y_bins - 1

    return (discrete_x, discrete_y)


def update_q_table(st, act, rew, next_st):  # st and next_st are (dx,dy) tuples
    # Q-value of best action from next_state
    best_next_action_q_value = np.max(q_table[next_st[0], next_st[1], :])
    td_target = rew + gamma * best_next_action_q_value
    td_error = td_target - q_table[st[0], st[1], act]
    q_table[st[0], st[1], act] += alpha * td_error


# --- Main RL Loop ---
game.start()  # Initial call to set up the game (starts column timer, kills start message)

# Get initial state by taking a dummy step (action 0: do nothing)
# The reward from this step is usually ignored.
observed_dx, observed_dy, _ = game.step(0)
current_discrete_state = get_discrete(observed_dx, observed_dy)

episode_count = 0  # For tracking episodes, if needed for printouts or epsilon decay schedule

while game.running:
    action = choose_action(current_discrete_state)
    # print(f"Action: {action}, State: {current_discrete_state}") # For debugging

    # Take action, get next state components and reward
    # These observed_dx, observed_dy are for the *next* state (S')
    next_observed_dx, next_observed_dy, reward = game.step(action)

    if not game.running:  # If game.step() detected a quit event
        break

    next_discrete_state = get_discrete(next_observed_dx, next_observed_dy)

    # Update Q-table
    update_q_table(current_discrete_state, action, reward, next_discrete_state)

    current_discrete_state = next_discrete_state  # Move to the next state

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if game.gameover:
        episode_count += 1
        # print(f"Episode {episode_count} finished. Score: {game.score.value}. Epsilon: {epsilon:.4f}")
        game.reset()  # Reset game environment for a new "episode"

        # Get the state after reset (again, by a dummy step)
        observed_dx_after_reset, observed_dy_after_reset, _ = game.step(0)
        current_discrete_state = get_discrete(
            observed_dx_after_reset, observed_dy_after_reset)
        if not game.running:  # Check again if quit event happened during reset/step
            break

game.quit()
