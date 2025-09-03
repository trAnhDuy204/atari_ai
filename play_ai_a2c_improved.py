from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from frogger_env import FroggerEnv

# Tạo môi trường render với FrameStack
n_stack = 4
env = DummyVecEnv([lambda: FroggerEnv(render_mode="human")])
env = VecFrameStack(env, n_stack=n_stack, channels_order='last')

# Load mô hình A2C cải tiến
model = A2C.load("frogger_a2c_model_improved")

# Chạy thử nghiệm với AI
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.envs[0].render()

    if done:
        obs = env.reset()
