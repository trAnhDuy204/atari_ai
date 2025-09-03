from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from frogger_env import FroggerEnv

# Tạo môi trường render
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])

# Load mô hình DQN đã train
model = DQN.load("frogger_dqn_model")

# Chạy thử nghiệm với AI
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)  # deterministic=True để ít random
    obs, reward, done, info = env.step(action)
    env.envs[0].render()

    if done:
        obs = env.reset()
