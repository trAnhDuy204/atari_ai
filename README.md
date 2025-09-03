# 🐸 Frogger AI với PPO (Stable-Baselines3)

Dự án này huấn luyện AI chơi game **Frogger** tự viết bằng `gymnasium` + `pygame`.  
Thuật toán sử dụng là **PPO (Proximal Policy Optimization)** từ thư viện `stable-baselines3`.

---

## 🚀 Cài đặt môi trường

1. Clone repo hoặc copy mã nguồn về máy.  
2. Tạo virtual environment (khuyến khích):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
3. Cài đặt thư viện cần thiết:
   pip install -r requirements.txt

## 🎮 Huấn luyện AI
1. Train AI cơ bản (logging + checkpoint + TensorBoard)

Huấn luyện theo thuật toán PPO chạy file:
    ```bash
    python train_ppo.py
Kết quả: sinh ra file frogger_ppo_model_final.zip.

Huấn luyện theo thuật toán A2C chạy file:
    ```bash
    python train_a2c.py
Kết quả: sinh ra file frogger_a2c_model.zip.

Huấn luyện theo thuật toán DQN chạy file:
    ```bash
    python train_dqn.py
Kết quả: sinh ra file frogger_dqn_model.zip.

2. Train AI cải tiến (logging + checkpoint + TensorBoard)

Huấn luận theo thuật toán PPO chạy file:
    ```bash
    python train_ppo_improved.py
Lưu model cuối cùng: frogger_ppo_model_final.zip
Lưu best model: logs_framestack/best_model/best_model.zip
Log TensorBoard: ppo_frogger_tensorboard/

Huấn luận theo thuật toán A2C chạy file:
    ```bash
    python train_a2c_improved.py
Lưu model cuối cùng: frogger_a2c_model_improved.zip
Lưu best model: logs_a2c_improved/best_model.zip
Log TensorBoard: a2c_frogger_tensorboard_improved/

Huấn luận theo thuật toán DQN chạy file:
    ```bash
    python train_dqn_improved.py
Lưu model cuối cùng: frogger_dqn_model_improved.zip
Lưu best model: logs_dqn_improved/best_model.zip
Log TensorBoard: dqn_frogger_tensorboard_improved/

Mở TensorBoard để theo dõi quá trình huấn luyện:
    ```bash
    tensorboard --logdir_spec=ppo:./ppo_frogger_tensorboard,ppo_imp:./ppo_frogger_tensorboard_framestack,dqn:./dqn_frogger_tensorboard,dqn_imp:./dqn_frogger_tensorboard_improved,a2c:./a2c_fro
    gger_tensorboard,a2c_imp:./a2c_frogger_tensorboard_improved
Rồi truy cập: http://localhost:6006

## 🎥 Chạy thử AI
Sau khi train xong, chạy theo PPO:
    ```bash
    python play_ai_ppo.py
    ```bash
    python play_ai_ppo_improved.py

Sau khi train xong, chạy theo A2C:
    ```bash
    python play_ai_a2c.py
    ```bash
    python play_ai_a2c_improved.py

Sau khi train xong, chạy theo DQN:
    ```bash
    python play_ai_dqn.py
    ```bash
    python play_ai_dqn_improved.py

Một cửa sổ Pygame mở ra, AI sẽ điều khiển chú ếch.
Nếu ếch chết → episode reset và chơi lại từ đầu.

## 📌 Yêu cầu hệ thống
Python 3.9+

GPU (khuyến khích, nhưng CPU vẫn chạy được)

Windows/Linux/Mac