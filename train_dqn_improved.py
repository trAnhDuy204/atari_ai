from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval với FrameStack
n_stack = 4
train_env = DummyVecEnv([lambda: FroggerEnv(render_mode="rgb_array")])
train_env = VecFrameStack(train_env, n_stack=n_stack, channels_order='last')

# Bọc Video Recorder để quay lại agent trong quá trình train
train_env = VecVideoRecorder(
    train_env,
    "videos_dqn/",
    record_video_trigger=lambda step: step % 100000 == 0,  # quay mỗi 100k steps
    video_length=1000,  # số bước trong mỗi video
    name_prefix="dqn_frogger"
)

eval_env = DummyVecEnv([lambda: FroggerEnv(render_mode="rgb_array")])
eval_env = VecFrameStack(eval_env, n_stack=n_stack, channels_order='last')

# Callback để lưu best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_dqn_improved/",
    log_path="./logs_dqn_improved/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints_dqn_improved/",
    name_prefix="frogger_dqn"
)

# Khởi tạo mô hình DQN cải tiến
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=200000,
    learning_starts=10000,
    batch_size=64, 
    tau=1.0,                       # hard update (có thể thử tau=0.05 = soft update)
    gamma=0.99,
    train_freq=4,
    target_update_interval=2000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./dqn_frogger_tensorboard_improved/"
)

# Huấn luyện
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, checkpoint_callback]
)

# Lưu mô hình cải tiến
model.save("frogger_dqn_model_improved")

train_env.close()
eval_env.close()
