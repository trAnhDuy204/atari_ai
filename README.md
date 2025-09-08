# 🐸 Frogger AI (Stable-Baselines3)

Dự án này huấn luyện AI chơi game **Frogger** tự viết bằng `gymnasium` + `pygame`.  
Thuật toán sử dụng: **PPO**, **A2C**, **DQN** từ thư viện `stable-baselines3`.

---

## 🚀 Cài đặt môi trường

1. Clone repo hoặc copy mã nguồn về máy.  
2. Tạo virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Cài đặt thư viện cần thiết:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Huấn luyện AI

### 1️. Huấn luyện cơ bản (logging + checkpoint + TensorBoard)

- PPO:
  ```bash
  python train_ppo.py
  ```
  → Lưu model: `frogger_ppo_model_final.zip`

- A2C:
  ```bash
  python train_a2c.py
  ```
  → Lưu model: `frogger_a2c_model.zip`

- DQN:
  ```bash
  python train_dqn.py
  ```
  → Lưu model: `frogger_dqn_model.zip`

---

### 2️. Huấn luyện cải tiến (frame stacking + entropy + logging)

- PPO Improved:
  ```bash
  python train_ppo_improved.py
  ```
  → Model cuối: `frogger_ppo_model_final.zip`  
  → Best model: `logs_framestack/best_model/best_model.zip`  
  → TensorBoard: `./ppo_frogger_tensorboard_framestack/`

- A2C Improved:
  ```bash
  python train_a2c_improved.py
  ```
  → Model cuối: `frogger_a2c_model_improved.zip`  
  → Best model: `logs_a2c_improved/best_model.zip`  
  → TensorBoard: `a2c_frogger_tensorboard_improved/`

- DQN Improved:
  ```bash
  python train_dqn_improved.py
  ```
  → Model cuối: `frogger_dqn_model_improved.zip`  
  → Best model: `logs_dqn_improved/best_model.zip`  
  → TensorBoard: `dqn_frogger_tensorboard_improved/`

---

### 📊 Theo dõi TensorBoard

Mở TensorBoard để so sánh tất cả log:

```bash
tensorboard --logdir_spec ppo:./ppo_frogger_tensorboard/,ppo_imp:./ppo_frogger_tensorboard_framestack/,a2c:./a2c_frogger_tensorboard/,a2c_imp:./a2c_frogger_tensorboard_improved/,dqn:./dqn_frogger_tensorboard/,dqn_imp:./dqn_frogger_tensorboard_improved/
```

Mở trình duyệt tại: [http://localhost:6006](http://localhost:6006)

---

## 🎥 Chạy thử AI

- PPO:
  ```bash
  python play_ai_ppo.py
  python play_ai_ppo_improved.py
  ```

- A2C:
  ```bash
  python play_ai_a2c.py
  python play_ai_a2c_improved.py
  ```

- DQN:
  ```bash
  python play_ai_dqn.py
  python play_ai_dqn_improved.py
  ```

👉 Một cửa sổ Pygame sẽ mở ra, AI điều khiển chú ếch.  
Nếu ếch chết → episode reset và chơi lại từ đầu.

---

## 📌 Yêu cầu hệ thống

- Python 3.9+  
- GPU (khuyến khích, CPU vẫn chạy được)  
- Hệ điều hành: Windows / Linux / Mac

---
