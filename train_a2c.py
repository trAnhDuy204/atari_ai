from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval
env = DummyVecEnv([lambda: FroggerEnv()])
eval_env = DummyVecEnv([lambda: FroggerEnv()])

# Callback để lưu best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_a2c/",
    log_path="./logs_a2c/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Khởi tạo mô hình A2C
# n_steps: số bước rollout trước khi update
# gamma: discount factor
# learning_rate: tốc độ học
# ent_coef: hệ số entropy để khuyến khích exploration
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.5,
    verbose=1,
    tensorboard_log="./a2c_frogger_tensorboard/"
)

# Huấn luyện mô hình
model.learn(total_timesteps=200_000, callback=eval_callback)

# Lưu model cuối cùng
model.save("frogger_a2c_model")

# Đóng môi trường
env.close()
eval_env.close()
