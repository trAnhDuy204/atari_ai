import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from frogger_env import FroggerEnv

# Tạo môi trường render
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])

# Load mô hình (model tốt nhất trong ./logs/ hoặc model cuối cùng)
model = PPO.load("frogger_ppo_model_final", env=env)

# Reset env -> lấy obs
obs = env.reset()

while True:
    # Dự đoán hành động
    action, _ = model.predict(obs, deterministic=True)

    # Bước tiếp
    obs, reward, done, info = env.step(action)

    # Render
    env.render(mode="human")

    # Thêm sleep nhỏ cho mượt
    time.sleep(0.05)

    # Nếu episode kết thúc -> reset lại
    if done.any():  # vì DummyVecEnv trả về mảng done
        obs = env.reset()
