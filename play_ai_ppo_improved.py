from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from frogger_env import FroggerEnv

# Tạo môi trường render
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])

# Gói thêm FrameStack (cùng số frame như khi train, ví dụ 4)
env = VecFrameStack(env, n_stack=4)

# Load mô hình framestack đã train
model = PPO.load("frogger_ppo_model_framestack_final")

# Reset env
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    # Nếu xong 1 episode thì reset
    if done:
        obs = env.reset()
