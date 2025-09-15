from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval
env = DummyVecEnv([lambda: FroggerEnv()])
eval_env = DummyVecEnv([lambda: FroggerEnv()])

# Callback để lưu best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/eval_logs/",
    eval_freq=10_000,        # đánh giá mỗi 10k bước
    deterministic=True,
    render=False,
)

# Callback để lưu checkpoint định kỳ
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,        # lưu mỗi 50k bước
    save_path="./checkpoints/",
    name_prefix="frogger_ppo"
)

# Khởi tạo PPO với hyperparameters tốt hơn
# Chạy TensorBoard: tensorboard --logdir ./ppo_frogger_tensorboard/
# Mở tại: http://localhost:6006
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_frogger_tensorboard/",
    learning_rate=3e-4,      # tốc độ học
    n_steps=1024,            # số bước rollout trước khi update
    batch_size=64,           # minibatch
    ent_coef=0.01,           # khuyến khích exploration
    gae_lambda=0.95,
    gamma=0.99,
    clip_range=0.2,
    n_epochs=5
)


# Huấn luyện
model.learn(
    total_timesteps=1_000_000, 
    callback=[eval_callback, checkpoint_callback]
)

# Lưu model cuối cùng
model.save("frogger_ppo_model_final")

# Đóng môi trường
env.close()
eval_env.close()
