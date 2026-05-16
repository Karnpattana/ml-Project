# บทที่ 4: Evaluation Metrics for Classification
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> ต่อจากบท 3 (churn project) — มาดูวิธีวัดผล classification

---

## 4.1 Overview

Accuracy อย่างเดียวไม่พอ โดยเฉพาะเมื่อ **class imbalance (คลาสไม่สมดุล)**

Metrics ที่จะเรียน:
- Accuracy
- Confusion Matrix
- Precision & Recall
- ROC Curve & AUC
- Cross-Validation

---

## 4.2 Accuracy

**Accuracy (ความแม่นยำ)** = สัดส่วนที่ทำนายถูก

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred >= 0.5)
```

ลอง threshold ต่างๆ:
```python
thresholds = np.linspace(0, 1, 21)
scores = []
for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    scores.append(score)
```

**ปัญหา**: ถ้า class imbalance เช่น 27% churn, 73% stay
- แค่ทำนายว่า "no churn ทุกคน" ก็ได้ accuracy = 73%
- → accuracy ดูสูงแต่โมเดลไม่มีประโยชน์

นี่คือ **dummy baseline** ที่ต้องเอาชนะให้ได้

---

## 4.3 Confusion Matrix (ตารางความสับสน)

ตาราง 2×2 สำหรับ binary classification:

|                | Predicted Negative | Predicted Positive |
|----------------|-------------------:|-------------------:|
| **Actual Negative** | True Negative (TN)  | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP)  |

- **TP (จริง+ทาย+)** — churn จริง และทายว่า churn → ถูก
- **TN (จริง−ทาย−)** — stay จริง และทายว่า stay → ถูก
- **FP (จริง−ทาย+)** — stay จริง แต่ทายว่า churn → **เตือนผิด**
- **FN (จริง+ทาย−)** — churn จริง แต่ทายว่า stay → **พลาด**

```python
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
predict_positive = (y_pred >= 0.5)
predict_negative = (y_pred < 0.5)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()
```

---

## 4.4 Precision & Recall

**Precision (ความแม่นยำ)** = ของที่ทายว่า + จริงๆ + กี่ %
$$\text{Precision} = \frac{TP}{TP + FP}$$
→ ทาย churn 100 คน จริงๆ churn กี่คน

**Recall (ความครอบคลุม)** = ของจริง + จับได้กี่ %
$$\text{Recall} = \frac{TP}{TP + FN}$$
→ ลูกค้าที่ churn จริง 100 คน จับได้กี่คน

**Trade-off**: เพิ่ม recall มักลด precision และกลับกัน

**F1 Score** = harmonic mean ของทั้งคู่:
$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

---

## 4.5 ROC Curve

**ROC (Receiver Operating Characteristic)** — กราฟแสดงประสิทธิภาพที่ทุก threshold

แกน:
- **TPR (True Positive Rate)** = Recall = TP / (TP + FN)
- **FPR (False Positive Rate)** = FP / (FP + TN)

ที่ threshold ต่างกัน → ได้จุดต่างกันบนกราฟ

**กราฟอ้างอิง**:
- **Random model** = เส้นทแยงมุม (FPR = TPR)
- **Ideal model** = ขึ้นไปมุมซ้ายบน (TPR=1, FPR=0)

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])   # random baseline
```

---

## 4.6 AUC (Area Under Curve)

**AUC** = พื้นที่ใต้กราฟ ROC

- 0.5 = random
- 1.0 = perfect
- ยิ่งสูง โมเดลยิ่งดี

**ความหมาย**: AUC = ความน่าจะเป็นที่โมเดลให้คะแนน positive sample สูงกว่า negative sample (เมื่อสุ่มมาคู่หนึ่ง)

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_pred)
```

ข้อดี: ไม่ขึ้นกับ threshold, ทนต่อ class imbalance

---

## 4.7 Cross-Validation

แทนที่จะแบ่ง train/val ครั้งเดียว → แบ่งเป็น K folds แล้วเทรน K ครั้ง

**K-Fold Cross-Validation**:
1. แบ่ง full_train เป็น K folds
2. ฝึก K ครั้ง: แต่ละครั้งใช้ K-1 folds เทรน, 1 fold val
3. เฉลี่ย score ทั้ง K ครั้ง

```python
from sklearn.model_selection import KFold

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)
    scores.append(roc_auc_score(y_val, y_pred))

print(f'AUC = {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
```

ใช้ tune **regularization C** ของ logistic regression:
- `C` ใหญ่ = regularization น้อย
- `C` เล็ก = regularization มาก

---

## สรุปบทที่ 4

- **Accuracy** ใช้ได้แต่ระวัง class imbalance
- **Confusion Matrix** บอก TP/TN/FP/FN
- **Precision** = ทายถูกในของที่ทาย+ / **Recall** = จับได้ของจริง+
- **ROC curve** = TPR vs FPR ทุก threshold
- **AUC** = ตัวเลขรวมประสิทธิภาพ ไม่ขึ้นกับ threshold (สำคัญที่สุดบทนี้)
- **K-Fold Cross-Validation** ช่วยประเมินโมเดลแม่นยำขึ้น และเลือก hyperparameter
