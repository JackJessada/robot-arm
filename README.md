# GRPO-SmolVLA for LIBERO

โปรเจกต์นี้เป็นการใช้ **Group Relative Policy Optimization (GRPO)** เพื่อทำการ Fine-tuning โมเดล **SmolVLA** (Vision-Language-Action model) บนชุดการทดสอบแขนหุ่นยนต์ **LIBERO**

## โครงสร้างโปรเจกต์

- `grpo_smolvla/`: โค้ดหลักของอัลกอริทึม
  - `grpo.py`: การคำนวณ GRPO Loss และ Advantage
  - `train_grpo.py`: ไฟล์หลักสำหรับการรันการฝึกสอน (Training Loop)
  - `rewards.py`: การกำหนดฟังก์ชันรางวัล (Reward Function)
  - `flow_utils.py`: เครื่องมือช่วยจัดการ Flow Matching Denoising
  - `env_utils.py`: ตัวเชื่อมต่อกับ LIBERO Environment
  - `evaluate.py`: สคริปต์สำหรับการประเมินโมเดล
  - `visualize.py`: สคริปต์สำหรับสร้างภาพเคลื่อนไหว (Visualization)
- `configs/`: ไฟล์ตั้งค่า (Configuration)
  - `grpo_config.yaml`: ตั้งค่า Hyperparameters ทั้งหมด
- `LIBERO/`: ชุด benchmark และโค้ดสภาพแวดล้อมหุ่นยนต์

## วิธีการใช้งาน (Usage)

### 1. การตั้งค่าสภาพแวดล้อม (Setup)
แนะนำให้ใช้ Conda environment:
```bash
conda activate robot-proj
# ติดตั้ง dependencies หลัก
uv pip install -r requirements.txt
# ติดตั้ง lerobot พร้อม smolvla
uv pip install "lerobot[smolvla]"
```

### 2. การฝึกสอนโมเดล (Training)
เริ่มการเทรนด้วย GRPO:
```bash
python -m grpo_smolvla.train_grpo --config configs/grpo_config.yaml
```

### 3. การประเมินผล (Evaluation)
ประเมินโมเดลที่เทรนแล้วบน LIBERO suites:
```bash
python -m grpo_smolvla.evaluate --checkpoint checkpoints/grpo_smolvla/step_5000 --suites libero_10 --n_episodes 20
```

### 4. การแสดงผล (Visualization)
สร้างภาพ GIF แสดงการทำงานของหุ่นยนต์:
```bash
python -m grpo_smolvla.visualize --suite libero_10 --task_id 5 --n_rollouts 3 --output vis_result.png
```

## ไฟล์ที่สามารถปรับแก้ได้ (Where to modify)

| สิ่งที่ต้องการทำ | ไฟล์ที่ต้องแก้ไข |
|-----------------|-----------------|
| เปลี่ยน Hyperparameters (LR, Batch size, Steps) | `configs/grpo_config.yaml` |
| แก้ไขวิธีการคำนวณรางวัล (Reward Logic) | `grpo_smolvla/rewards.py` |
| ปรับปรุงสูตร GRPO หรือ clipping | `grpo_smolvla/grpo.py` |
| เปลี่ยนชุดงานที่ใช้เทรน (Task Suites) | แก้ไข `task_suite` ใน `configs/grpo_config.yaml` |
| ปรับปรุงการจัดการภาพหรือ state input | `grpo_smolvla/env_utils.py` |

---
**หมายเหตุ:** โปรเจกต์นี้เน้นการเพิ่มความสามารถในการ Generalization ของหุ่นยนต์โดยใช้ RL ต่อยอดจากโมเดลที่ผ่านการทำ SFT (Supervised Fine-Tuning) มาแล้ว
