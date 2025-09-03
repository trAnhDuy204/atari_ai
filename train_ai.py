from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from frogger_env import FroggerEnv

# Tạo môi trường
env = DummyVecEnv([lambda: FroggerEnv()])

# Huấn luyện lại mô hình mới
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Lưu lại mô hình mới
model.save("frogger_ppo_model")
env.close()
