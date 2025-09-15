import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
FROG_SIZE = 20
CAR_WIDTH = 40
CAR_HEIGHT = 20
LANES = [100, 150, 200, 250]
STEP_SIZE = 5  # Tốc độ ếch

class FroggerEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.closed = False

        self.max_cars = 20  # Số xe tối đa để cố định chiều dài obs
        self.action_space = spaces.Discrete(4)

        obs_high = np.array(
            [SCREEN_WIDTH, SCREEN_HEIGHT] + [SCREEN_WIDTH, SCREEN_HEIGHT] * self.max_cars,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0, high=obs_high, dtype=np.float32)

        self.frog_img = None
        self.car_img = None
        self.level = 1
        self.score = 0

        # Giới hạn bước
        self.max_steps = 2000
        self.current_step = 0

        self._init_game()

    def _init_game(self):
        if self.closed: 
            return
        self.frog_x = SCREEN_WIDTH // 2
        self.frog_y = SCREEN_HEIGHT - FROG_SIZE
        self.cars = []
        self.passed_lanes = set()

        num_cars = min(len(LANES) + self.level - 1, self.max_cars)
        for _ in range(num_cars):
            lane = random.choice(LANES)
            car_x = random.randint(0, SCREEN_WIDTH - CAR_WIDTH)
            direction = random.choice([-1, 1])
            speed = random.randint(2 + self.level, 4 + self.level)
            self.cars.append([car_x, lane, speed * direction])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.closed:  
            return np.zeros(2 + self.max_cars * 2, dtype=np.float32), {}
        self.level = 1
        self.score = 0
        self.current_step = 0
        self._init_game()
        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self):
        obs = [self.frog_x, self.frog_y]
        for car in self.cars:
            obs.extend([car[0], car[1]])
        while len(obs) < 2 + self.max_cars * 2:
            obs.extend([0, 0])
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        if self.closed:  
            return self._get_obs(), 0.0, True, False, {}

        old_y = self.frog_y

        # Di chuyển ếch
        if action == 0:  # lên
            self.frog_y -= STEP_SIZE
        elif action == 1:  # xuống
            self.frog_y += STEP_SIZE
        elif action == 2:  # trái
            self.frog_x -= STEP_SIZE
        elif action == 3:  # phải
            self.frog_x += STEP_SIZE

        self.frog_x = np.clip(self.frog_x, 0, SCREEN_WIDTH - FROG_SIZE)
        self.frog_y = np.clip(self.frog_y, 0, SCREEN_HEIGHT - FROG_SIZE)

        # Update xe
        for car in self.cars:
            car[0] += car[2]
            if car[0] < -CAR_WIDTH:
                car[0] = SCREEN_WIDTH
            elif car[0] > SCREEN_WIDTH:
                car[0] = -CAR_WIDTH

        # Reward shaping
        reward = -0.01
        if self.frog_y < old_y:
            reward += 0.1

        done = False

        # Kiểm tra va chạm
        for car in self.cars:
            if (
                self.frog_x < car[0] + CAR_WIDTH and
                self.frog_x + FROG_SIZE > car[0] and
                self.frog_y < car[1] + CAR_HEIGHT and
                self.frog_y + FROG_SIZE > car[1]
            ):
                reward = -1.0
                done = True
                return self._get_obs(), reward, done, False, {}

        # Đi qua lane thành công
        for lane_y in LANES:
            if self.frog_y < lane_y and lane_y not in self.passed_lanes:
                self.score += 10
                reward += 10
                self.passed_lanes.add(lane_y)

        # Hoàn thành màn
        if self.frog_y <= 0:
            reward += 50
            self.score += 50
            self.level += 1
            self._init_game()
            return self._get_obs(), reward, False, False, {}

        # Giới hạn số bước
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}

    def render(self):
        if self.render_mode != "human" or self.closed:
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Frogger AI")
            self.clock = pygame.time.Clock()
            self.frog_img = pygame.image.load("./umairyokaimon-froakie-the-bubble-frog-pokemon.png").convert_alpha()
            self.frog_img = pygame.transform.scale(self.frog_img, (FROG_SIZE, FROG_SIZE))
            self.car_img = pygame.image.load("./images.jpg").convert_alpha()
            self.car_img = pygame.transform.scale(self.car_img, (CAR_WIDTH, CAR_HEIGHT))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill((0, 0, 0))
        self.window.blit(self.frog_img, (self.frog_x, self.frog_y))

        for car in self.cars:
            self.window.blit(self.car_img, (car[0], car[1]))

        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        level_text = font.render(f"Level: {self.level}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        self.window.blit(level_text, (10, 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
        self.closed = True
