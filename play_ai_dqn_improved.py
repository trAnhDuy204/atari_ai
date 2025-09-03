from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from frogger_env import FroggerEnv

# Tạo môi trường render với FrameStack
n_stack = 4
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])
env = VecFrameStack(env, n_stack=n_stack, channels_order='last')

# Load mô hình DQN cải tiến
model = DQN.load("frogger_dqn_model_improved")

# Chạy thử nghiệm với AI
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)  # deterministic để ít random
    obs, reward, done, info = env.step(action)
    env.envs[0].render()

    if done:
        obs = env.reset()
