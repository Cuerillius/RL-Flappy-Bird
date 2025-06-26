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

        img = pygame.image.load('assets/icons/red_bird.png')
        pygame.display.set_icon(img)

        self.clock = pygame.time.Clock()
        self.column_create_event = pygame.USEREVENT
        self.running = True
        self.gameover = False
        self.gamestarted = False

        assets.load_sprites()

        self.sprites = pygame.sprite.LayeredUpdates()

        self.bird, self.game_start_message, self.score = self.create_sprites()

    def create_sprites(self):
        Background(0, self.sprites)
        Background(1, self.sprites)
        Floor(0, self.sprites)
        Floor(1, self.sprites)

        return Bird(self.sprites), GameStartMessage(self.sprites), Score(self.sprites)

    def start(self):
        if not self.gamestarted and not self.gameover:
            self.gamestarted = True
            self.game_start_message.kill()

    def reset(self):
        self.gameover = False
        self.gamestarted = True
        self.sprites.empty()
        self.bird, self.game_start_message, self.score = self.create_sprites()
        self.game_start_message.kill()

    def quit(self):
        self.running = False
        pygame.quit()

    def step(self, action, count):
        reward = 0

        if count % 100 == 0:
            Column(self.sprites)

        for event in pygame.event.get():
            if event.type == self.column_create_event:
                Column(self.sprites)
        # jump
        if action == 1:
            if not self.gameover:
                self.bird.handle_event(pygame.event.Event(
                    pygame.KEYDOWN, key=pygame.K_SPACE))
        # no jump
        elif action == 0:
            pass

        self.screen.fill(0)
        self.sprites.draw(self.screen)
        pygame.display.flip()

        if self.gamestarted and not self.gameover:
            self.sprites.update()

        if self.bird.check_collision(self.sprites) and not self.gameover:
            self.gameover = True
            GameOverMessage(self.sprites)
            pygame.time.set_timer(self.column_create_event, 0)
            reward += -100
        elif not self.gameover:
            reward += 0.1

        if self.bird.check_near_floor_or_top() and not self.gameover:
            reward += -10

        columns = [
            sprite for sprite in self.sprites if isinstance(sprite, Column)]
        for sprite in columns:
            if sprite.is_passed():
                self.score.value += 1
                distance_from_center = sprite.get_distance_from_center(
                    self.bird.rect.y)
                reward_scale = 1 - (distance_from_center / configs.COLUMN_GAP)
                reward += 1.5 * reward_scale

        self.clock.tick(configs.FPS)

        delta_x_to_column = None
        delta_y_to_gap = None

        upcoming_columns = [
            column for column in columns if column.rect.x > self.bird.rect.x]
        if upcoming_columns:
            nearest_column = min(
                upcoming_columns, key=lambda col: col.rect.x - self.bird.rect.x)
            delta_x_to_column = nearest_column.rect.x - self.bird.rect.x
            gap_center_y = nearest_column.get_gap_center_y()
            delta_y_to_gap = abs(gap_center_y - self.bird.rect.y)
        else:
            delta_y_to_gap = None

        return delta_x_to_column, delta_y_to_gap, self.bird.rect.y, reward


filename = 'q_table.npy'

game = FlappyBird()

# what extent newly acquired information overrides old information.
alpha = 0.1
# Measures the importance of future rewards.
gamma = 0.95
# Exploration rate
epsilon = 0.7
epsilon_decay = 0.8
epsilon_min = 0.01
action_size = 2


# limit the range of delta_x_to_column, delta_y_to_gap
bin_size_y = 25
num_y_bins = int(configs.SCREEN_HEIGHT / bin_size_y)

bin_size_x = 20
num_x_bins = int(configs.SCREEN_WIDTH / bin_size_x)

bin_size_abs_y = 25
num_abs_y_bins = int(configs.SCREEN_HEIGHT / bin_size_abs_y)

try:
    q_table = np.fromfile(filename, dtype=float, sep=" ").reshape(
        (num_x_bins, num_y_bins, num_abs_y_bins, action_size))
except:
    q_table = np.random.uniform(
        low=-1, high=1, size=(num_x_bins, num_y_bins, num_abs_y_bins, action_size))


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 1)
    else:
        return np.argmax(q_table[state[0], state[1], state[2], :])


def get_discrete(delta_x_to_column, delta_y_to_gap, delta_y_abs):
    if delta_x_to_column is not None:
        dx = max(0, delta_x_to_column)
        discrete_x = int(dx / bin_size_x)
        discrete_x = min(discrete_x, num_x_bins - 1)
    else:
        discrete_x = num_x_bins - 1

    if delta_y_abs is not None:
        dy = max(0, delta_y_abs)
        discrete_y_abs = int(dy / bin_size_abs_y)
        discrete_y_abs = min(discrete_y_abs, num_abs_y_bins - 1)
    else:
        discrete_y_abs = num_abs_y_bins - 1

    if delta_y_to_gap is not None:
        dy = delta_y_to_gap
        discrete_y = int(dy / bin_size_y)
        discrete_y = min(discrete_y, num_y_bins - 1)
    else:
        discrete_y = num_y_bins - 1

    return (discrete_x, discrete_y, discrete_y_abs)


def update_q_table(state, action, reward, next_state):
    best_next_q_value = np.max(
        q_table[next_state[0], next_state[1], next_state[2], :])
    td_target = reward + gamma * best_next_q_value
    td_error = td_target - q_table[state[0], state[1], state[2], action]
    q_table[state[0], state[1], state[2], action] += alpha * td_error


game.start()
count = 0


delta_x_to_column, delta_y_to_gap, delta_y_abs, _ = game.step(0, count)
state = get_discrete(delta_x_to_column, delta_y_to_gap, delta_y_abs)

while game.running:
    count += 1

    print(game.score.value, end="\r")

    action = choose_action(state)
    delta_x_to_column, delta_y_to_gap, delta_y_abs, reward = game.step(
        action, count)

    next_state = get_discrete(delta_x_to_column, delta_y_to_gap, delta_y_abs)
    update_q_table(state, action, reward, next_state)
    state = next_state

    if count % 100 == 0:
        q_table.tofile(filename, sep=" ", format="%f")

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if game.gameover:
        game.reset()

game.quit()
