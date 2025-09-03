from frogger_env import FroggerEnv
import time

# Khởi tạo môi trường
env = FroggerEnv()

obs = env.reset()
done = False

# Vòng lặp mô phỏng trò chơi với hành động ngẫu nhiên
while not done:
    env.render()
    action = env.action_space.sample()  # Hành động ngẫu nhiên
    obs, reward, done, info = env.step(action)
    time.sleep(0.1)

env.close()
