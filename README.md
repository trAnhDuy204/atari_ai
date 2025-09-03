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
1. Train nhanh (không log, không checkpoint)

Chạy file:
    python train_ai.py

Kết quả: sinh ra file frogger_ppo_model.zip.
2. Train nâng cao (logging + checkpoint + TensorBoard)

Chạy file:
    python train_with_logging.py

Lưu model cuối cùng: frogger_ppo_model_final.zip

Lưu best model: logs/best_model.zip

Log TensorBoard: ppo_frogger_tensorboard/

Mở TensorBoard để theo dõi quá trình huấn luyện:
    tensorboard --logdir ./ppo_frogger_tensorboard/
Rồi truy cập: http://localhost:6006

## 🎥 Chạy thử AI
Sau khi train xong, chạy:
    python play_ai.py

Một cửa sổ Pygame mở ra, AI sẽ điều khiển chú ếch.
Nếu ếch chết → episode reset và chơi lại từ đầu.

## 📌 Yêu cầu hệ thống
Python 3.9+

GPU (khuyến khích, nhưng CPU vẫn chạy được)

Windows/Linux/Mac