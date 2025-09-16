from stable_baselines3 import A2C
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
    "videos_a2c/",
    record_video_trigger=lambda step: step % 100000 == 0,  # quay mỗi 100k steps
    video_length=1000,  # số bước trong mỗi video
    name_prefix="a2c_frogger"
)

eval_env = DummyVecEnv([lambda: FroggerEnv(render_mode="rgb_array")])
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

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints_a2c_improved/",
    name_prefix="frogger_a2c"
)

# Khởi tạo mô hình A2C cải tiến
model = A2C(
    "MlpPolicy",
    train_env,
    learning_rate=7e-4,
    n_steps=40,          # tăng steps để giảm noise
    gamma=0.99,
    ent_coef=0.01,       # hệ số entropy
    vf_coef=0.5,         # hệ số value function
    verbose=1,
    tensorboard_log="./a2c_frogger_tensorboard_improved/"
)

# Huấn luyện
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, checkpoint_callback]
)

# Lưu mô hình cải tiến
model.save("frogger_a2c_model_improved")

train_env.close()
eval_env.close()
