from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from frogger_env import FroggerEnv

# Tạo môi trường train & eval
env = DummyVecEnv([lambda: FroggerEnv()])
eval_env = DummyVecEnv([lambda: FroggerEnv()])

# Callback để lưu best model và log
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs_dqn/",
    log_path="./logs_dqn/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Khởi tạo mô hình DQN
# Một số tham số quan trọng:
# - learning_rate: tốc độ học (nhỏ để DQN ổn định)
# - buffer_size: số lượng mẫu lưu trong Replay Buffer
# - learning_starts: bắt đầu học sau khi có đủ dữ liệu
# - batch_size: số mẫu dùng trong mỗi lần cập nhật
# - gamma: hệ số chiết khấu (discount factor)
# - target_update_interval: số bước trước khi update target network
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    tensorboard_log="./dqn_frogger_tensorboard/"
)


# Huấn luyện mô hình
# Chạy TensorBoard bằng:
# tensorboard --logdir ./dqn_frogger_tensorboard/
# Mở http://localhost:6006 để xem kết quả
model.learn(total_timesteps=200_000, callback=eval_callback)

# Lưu mô hình sau khi train xong
model.save("frogger_dqn_model")

# Đóng môi trường
env.close()
eval_env.close()
