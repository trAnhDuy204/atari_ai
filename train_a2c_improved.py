from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval với FrameStack
n_stack = 4
env = DummyVecEnv([lambda: FroggerEnv()])
env = VecFrameStack(env, n_stack=n_stack, channels_order='last')

eval_env = DummyVecEnv([lambda: FroggerEnv()])
eval_env = VecFrameStack(eval_env, n_stack=n_stack, channels_order='last')

# Callback để lưu best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_a2c_improved/",
    log_path="./logs_a2c_improved/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Khởi tạo mô hình A2C cải tiến
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=20,          # tăng từ 5 lên 20 để giảm noise
    gamma=0.99,
    ent_coef=0.02,       # tăng entropy để exploration tốt hơn
    vf_coef=0.5,
    verbose=1,
    tensorboard_log="./a2c_frogger_tensorboard_improved/"
)

# Huấn luyện
model.learn(total_timesteps=500_000, callback=eval_callback)

# Lưu mô hình cải tiến
model.save("frogger_a2c_model_improved")

env.close()
eval_env.close()
