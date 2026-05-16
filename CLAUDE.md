# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## โปรเจกต์นี้คืออะไร

ML project สำหรับ image classification โดยใช้ภาพถ่ายจากมุม top และ side มี 2 class: `cos` และ `gok`

## โครงสร้าง Dataset

รูปภาพอยู่ใน `raw/<class>/` ตั้งชื่อตาม convention:
```
{CLASS}{ID}_{GROUP}_{DAY}_{CONDITION}_{VIEW}.jpg
```
- CLASS: `COS` หรือ `GOK`
- GROUP: `A` หรือ `B`
- DAY: `D0` = วันแรก, `D1` = วันถัดไป
- CONDITION: `E` = ปกติ, `M` = อื่นๆ
- VIEW: `top` = มุมบน, `side` = มุมข้าง

## Knowledge Base

`knowledge/` มีเนื้อหา ML Zoomcamp + PyTorch workshop เป็นภาษาไทย ใช้ผ่าน skill `ml-knowledge` — เมื่อถามเรื่อง ML ให้ดึงไฟล์ที่เกี่ยวข้องมาอ่านก่อนตอบเสมอ

| ไฟล์ | หัวข้อ |
|------|--------|
| `01-intro.md` | ML basics, NumPy, Pandas |
| `02-regression.md` | Linear regression |
| `03-classification.md` | Logistic regression, churn |
| `04-evaluation.md` | Accuracy, ROC, AUC, cross-validation |
| `05-deployment.md` | Flask, Docker, AWS |
| `06-trees.md` | Decision tree, Random Forest, XGBoost |
| `08-deep-learning.md` | CNN, Keras/TensorFlow, Xception |
| `09-serverless.md` | TF Lite, AWS Lambda |
| `10-kubernetes.md` | TF Serving, Kubernetes |
| `11-kserve.md` | KServe |
| `workshop-pytorch-deep-learning.md` | PyTorch, MobileNetV2 |

## แนวทางการพัฒนา

- โปรเจกต์นี้เน้น deep learning / image classification — ให้พิจารณา transfer learning (MobileNetV2, Xception) เป็นจุดเริ่มต้น
- dataset มี 2 view (top + side) ต่อ sample — อาจ combine หรือ train แยกก็ได้
- ตอบและ comment เป็นภาษาไทย เว้นแต่โค้ดและชื่อตัวแปร
