# บทที่ 3: Machine Learning for Classification
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> **โปรเจกต์**: ทำนาย Customer Churn (ลูกค้าจะเลิกใช้บริการไหม)

---

## 3.1 Churn Prediction Project

**Churn** = ลูกค้าเลิกใช้บริการ (1 = churn, 0 = stay)

เป็น **Binary Classification (จำแนก 2 คลาส)**

Output ของโมเดล = **probability** ที่ลูกค้าจะ churn

---

## 3.2 Data Preparation

```python
import pandas as pd
import numpy as np

df = pd.read_csv('telco_churn.csv')

# ทำให้ชื่อคอลัมน์/ค่าเป็นตัวพิมพ์เล็ก
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# TotalCharges มี string ปนตัวเลข → แก้
tc = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = tc.fillna(0)

# Encode target: yes/no → 1/0
df.churn = (df.churn == 'yes').astype(int)
```

---

## 3.3 Validation Framework

ใช้ `train_test_split` จาก sklearn:
```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
# → train 60%, val 20%, test 20%

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```

---

## 3.4 EDA

```python
df.churn.value_counts(normalize=True)
# 0    0.73
# 1    0.27  → 27% churn rate

# Missing values
df.isnull().sum()

# Unique values ในแต่ละ categorical
for c in categorical:
    print(c, df[c].nunique())
```

---

## 3.5 Feature Importance: Churn Rate และ Risk Ratio

ดูว่าแต่ละกลุ่ม churn ต่างจากค่าเฉลี่ยรวมไหม:

```python
global_mean = df_train.churn.mean()

# Churn rate ของผู้หญิง vs ผู้ชาย
df_train.groupby('gender').churn.mean()

# Risk ratio = group_rate / global_rate
# > 1 → กลุ่มนี้ churn มากกว่าเฉลี่ย
# < 1 → กลุ่มนี้ churn น้อยกว่าเฉลี่ย
```

ตัวอย่าง: `partner=no` มี churn rate สูงกว่า `partner=yes` → เป็น signal ที่ดี

---

## 3.6 Mutual Information

**Mutual Information (MI)** = วัดความสัมพันธ์ระหว่าง categorical variable กับ target (ยิ่งสูงยิ่งสำคัญ)

```python
from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_train.churn)

mi = df_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
```

---

## 3.7 Correlation (Numerical Features)

สำหรับ **numerical features** ใช้ **Pearson correlation** กับ target:

```python
df_train[numerical].corrwith(df_train.churn).abs()
```

ค่าใกล้ 1 = สัมพันธ์เชิงเส้นมาก, ใกล้ 0 = ไม่สัมพันธ์

---

## 3.8 One-Hot Encoding (OHE)

ใช้ **DictVectorizer** จาก sklearn:

```python
from sklearn.feature_extraction import DictVectorizer

train_dict = df_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
# DictVectorizer ทำ OHE ให้อัตโนมัติสำหรับ categorical
```

---

## 3.9 Logistic Regression

สูตร: 
$$g(x_i) = \text{Sigmoid}(w_0 + w_1 x_1 + ... + w_n x_n)$$

$$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

**Sigmoid function** แปลง score ใดๆ → ความน่าจะเป็น [0, 1]

ต่างจาก linear regression:
- Linear: output เป็นตัวเลขจริงใดๆ
- Logistic: output เป็น probability 0-1

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

## 3.10 Training Logistic Regression with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Probability
y_pred = model.predict_proba(X_val)[:, 1]   # column 1 = prob ของ churn

# Hard prediction (threshold 0.5)
churn_decision = (y_pred >= 0.5)

# Accuracy
(y_val == churn_decision).mean()
```

---

## 3.11 Model Interpretation

ดู weights ของแต่ละ feature:
```python
model.intercept_   # w_0 (bias)
model.coef_[0]     # weights ของแต่ละ feature

# จับคู่กับชื่อ feature
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))
```

- **Weight บวก** → เพิ่มโอกาส churn
- **Weight ลบ** → ลดโอกาส churn

---

## 3.12 Using the Model

```python
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    # ...
}

X_new = dv.transform([customer])
model.predict_proba(X_new)[0, 1]   # probability churn

# ถ้า > 0.5 → ส่งโปรโมชั่นให้ลูกค้าคนนี้
```

---

## สรุปบทที่ 3

- **Classification** ต่างจาก regression: ทำนายหมวด ไม่ใช่ตัวเลข
- **Logistic regression** = linear regression + sigmoid → ได้ probability
- **One-Hot Encoding** แปลง categorical → ตัวเลข (ใช้ DictVectorizer)
- **Mutual Information** วัดความสำคัญของ categorical features
- **Correlation** วัดความสำคัญของ numerical features
- โมเดลให้ probability → ใช้ threshold (default 0.5) ตัดสิน
- ดู weights เพื่อตีความ feature importance
