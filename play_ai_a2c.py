from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from frogger_env import FroggerEnv

# Tạo môi trường render
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])

# Load mô hình A2C đã train
model = A2C.load("frogger_a2c_model")

# Chạy thử nghiệm với AI
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.envs[0].render()

    if done:
        obs = env.reset()
