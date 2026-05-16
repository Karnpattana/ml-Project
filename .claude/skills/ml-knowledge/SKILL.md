---
name: ml-knowledge
description: คลังความรู้ Machine Learning Zoomcamp และ PyTorch deep learning workshop เป็นภาษาไทย ใช้ skill นี้เมื่อผู้ใช้ถามเกี่ยวกับ machine learning, ML, regression, classification, logistic regression, linear regression, decision tree, random forest, XGBoost, evaluation metrics, accuracy, precision, recall, ROC, AUC, cross-validation, neural network, deep learning, CNN, transfer learning, PyTorch, TensorFlow, Keras, MobileNetV2, Xception, model deployment, Flask, Docker, AWS Lambda, Kubernetes, TensorFlow Serving, KServe, serverless, churn prediction, car price prediction, credit risk, image classification
---

# ML Knowledge Base

คลังความรู้นี้สรุปจาก ML Zoomcamp by DataTalksClub + PyTorch workshop by Alexey Grigorev เป็นภาษาไทย

## ไฟล์ในคลังความรู้ (ที่ knowledge/)

1. `knowledge/01-intro.md` — ML basics, supervised learning, CRISP-DM, NumPy, Pandas, linear algebra
2. `knowledge/02-regression.md` — Linear regression, normal equation, RMSE, feature engineering, regularization (โปรเจกต์ทำนายราคารถ)
3. `knowledge/03-classification.md` — Logistic regression, one-hot encoding, mutual information (โปรเจกต์ churn)
4. `knowledge/04-evaluation.md` — Accuracy, confusion matrix, precision, recall, ROC, AUC, K-fold cross-validation
5. `knowledge/05-deployment.md` — Pickle, Flask, Pipenv, Docker, AWS Elastic Beanstalk
6. `knowledge/06-trees.md` — Decision tree, Random Forest, XGBoost (โปรเจกต์ credit risk)
7. `knowledge/08-deep-learning.md` — CNN, transfer learning, Keras/TensorFlow, Xception (โปรเจกต์ classification เสื้อผ้า)
8. `knowledge/09-serverless.md` — TensorFlow Lite, AWS Lambda, API Gateway
9. `knowledge/10-kubernetes.md` — TensorFlow Serving, Kubernetes, gateway pattern, EKS
10. `knowledge/11-kserve.md` — KServe, InferenceService, transformer component
11. `knowledge/workshop-pytorch-deep-learning.md` — Deep learning ด้วย PyTorch (คู่ขนานกับบท 8 แต่ใช้ PyTorch + MobileNetV2)

## วิธีใช้ skill นี้

เมื่อถูกเรียกใช้:
1. อ่านไฟล์ที่เกี่ยวข้องใน `knowledge/` ก่อนตอบเสมอ — ใช้ Read tool
2. ถ้าคำถามครอบคลุมหลายหัวข้อ ให้อ่านหลายไฟล์
3. ตอบเป็นภาษาไทย ตรงประเด็น
4. คำทับศัพท์ภาษาอังกฤษ ใส่คำอธิบายในวงเล็บเสมอ เช่น "regression (การถดถอย)"
5. อ้างอิงชื่อไฟล์ที่ใช้ในการตอบ เช่น "ตาม knowledge/03-classification.md ..."
6. ถ้าคำถามอยู่นอกขอบเขตของคลังความรู้ ให้บอกผู้ใช้ตรงๆ
7. ถ้ามีโค้ดตัวอย่างที่เกี่ยวข้องในไฟล์ ให้ยกมาแสดง
8. ถ้าผู้ใช้ถามเรื่อง deep learning ทั้ง Keras และ PyTorch ก็ดึงทั้ง 2 ไฟล์มาเทียบ
