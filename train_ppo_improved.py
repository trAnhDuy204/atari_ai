from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval
train_env = DummyVecEnv([lambda: FroggerEnv()])
train_env = VecFrameStack(train_env, n_stack=4)  # Frame stacking: ghép 4 trạng thái liên tiếp

eval_env = DummyVecEnv([lambda: FroggerEnv()])
eval_env = VecFrameStack(eval_env, n_stack=4)

# Callback để lưu best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_framestack/best_model/",
    log_path="./logs_framestack/eval_logs/",
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# Callback checkpoint
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints_framestack/",
    name_prefix="frogger_ppo"
)

# Khởi tạo PPO với ent_coef cao hơn (khuyến khích exploration)
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_frogger_tensorboard_framestack/",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.02,    # tăng entropy để agent khám phá nhiều hơn
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
model.save("frogger_ppo_model_framestack_final")

# Đóng môi trường
train_env.close()
eval_env.close()
