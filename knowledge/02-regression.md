# บทที่ 2: Machine Learning for Regression
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> **โปรเจกต์**: ทำนายราคารถมือสอง (Car Price Prediction)

---

## 2.1 Car Price Prediction Project

ปัญหา: เว็บไซต์ขายรถมือสองอยากช่วยผู้ใช้ตั้งราคาที่เหมาะสม

แนวทาง:
1. ดาวน์โหลด dataset ราคารถ
2. ทำ EDA (Exploratory Data Analysis, สำรวจข้อมูล)
3. ใช้ **linear regression (การถดถอยเชิงเส้น)** เป็น baseline
4. ประเมินด้วย **RMSE (Root Mean Squared Error)**
5. ทำ feature engineering และ regularization

---

## 2.2 Data Preparation

ขั้นตอนเตรียมข้อมูล:
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# ทำให้ชื่อคอลัมน์เป็นมาตรฐาน
df.columns = df.columns.str.lower().str.replace(' ', '_')

# ทำให้ค่า string ใน columns เป็นตัวพิมพ์เล็กและ underscore
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
```

---

## 2.3 Exploratory Data Analysis (EDA)

ดูการกระจายของ target variable (msrp = ราคา)

```python
import seaborn as sns
sns.histplot(df.msrp, bins=50)
```

**ปัญหา**: ราคามี long-tail distribution (กระจายยาวด้านขวา) → โมเดลทำนายแย่

**วิธีแก้: Log transformation**
```python
price_logs = np.log1p(df.msrp)   # log(1+x) กัน log(0)
sns.histplot(price_logs, bins=50)  # ดูเป็น normal distribution มากขึ้น
```

ตอนทำนายเสร็จต้องแปลงกลับด้วย `np.expm1()` (= exp(x) - 1)

---

## 2.4 Validation Framework

แบ่งข้อมูลเป็น 3 ส่วน (60/20/20):

```python
n = len(df)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - n_val - n_test

# Shuffle ข้อมูลก่อนแบ่ง
idx = np.arange(n)
np.random.seed(2)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

# ลบ target ออกจาก features
del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
```

---

## 2.5–2.6 Linear Regression

**Linear Regression** สูตร:
$$g(x_i) = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n$$

- `w_0` = **bias term (จุดตัดแกน)**
- `w_1...w_n` = **weights (น้ำหนัก)** ของแต่ละ feature

ในรูปแบบ vector:
$$g(x_i) = w_0 + x_i^T \cdot w$$

ถ้าใส่ `x_0 = 1` เข้าไป จะได้:
$$g(x_i) = x_i^T \cdot w$$

โค้ดตัวอย่าง:
```python
def linear_regression(xi):
    w0 = 7.17
    w = [0.01, 0.04, 0.002]
    pred = w0
    for j in range(len(xi)):
        pred += w[j] * xi[j]
    return pred
```

---

## 2.7 Training Linear Regression (Normal Equation)

แก้สมการเชิงเส้นด้วย **Normal Equation**:
$$w = (X^T X)^{-1} X^T y$$

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])  # เพิ่มคอลัมน์ของ 1 (bias)
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]   # bias, weights
```

---

## 2.8 Baseline Model

เลือก features ที่เป็นตัวเลข (numerical):
```python
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 
        'city_mpg', 'popularity']

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)   # เติม missing ด้วย 0
    return df_num.values

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)
y_pred = w0 + X_train.dot(w)
```

---

## 2.9 RMSE (Root Mean Squared Error)

วัดความแม่นยำของ regression model:
$$RMSE = \sqrt{\frac{1}{n} \sum (y_{pred} - y_{actual})^2}$$

```python
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
```

ค่า RMSE ยิ่งน้อยยิ่งดี

---

## 2.10 Validation

```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

print(rmse(y_val, y_pred))
```

---

## 2.11 Feature Engineering

สร้าง feature ใหม่จากที่มี เช่น **อายุของรถ** จากปี:
```python
df['age'] = 2017 - df['year']
```

ช่วยให้โมเดลแม่นขึ้น

---

## 2.12 Categorical Variables

**Categorical (ตัวแปรเชิงหมวดหมู่)** เช่น brand, fuel type

วิธีแปลงเป็นตัวเลข: **One-Hot Encoding (OHE)**

ตัวอย่างง่ายๆ ทำเอง:
```python
for v in [2, 3, 4]:
    df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')

# ค่าที่พบบ่อย
makes = list(df.make.value_counts().head().index)
for m in makes:
    df['make_%s' % m] = (df.make == m).astype('int')
```

⚠️ ถ้าใช้ categorical เยอะเกินไปแล้วโมเดลพัง อาจเกิด **multicollinearity** → ต้องใช้ regularization

---

## 2.13 Regularization

ปัญหา: เมื่อมี columns ที่ correlate กันมาก เมทริกซ์ `X^T X` จะ near-singular → invert ไม่ได้แม่นยำ → weights ใหญ่ผิดปกติ

**วิธีแก้**: บวก `r * I` (diagonal regularization) เข้าไป:
$$w = (X^T X + rI)^{-1} X^T y$$

```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])  # regularization
    
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]
```

`r` ใหญ่ขึ้น → weights เล็กลง → underfit
`r` เล็กไป → เหมือนไม่มี regularization

---

## 2.14 Tuning the Model

ลอง `r` หลายค่าแล้วเลือกตัวที่ RMSE ต่ำสุดบน validation:
```python
for r in [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w0 + X_val.dot(w)
    print(r, rmse(y_val, y_pred))
```

---

## 2.15 Using the Model

หลังเลือก `r` ที่ดีที่สุด:
1. รวม train + val แล้วเทรนใหม่
2. ทดสอบบน test set
3. ใช้กับข้อมูลใหม่:

```python
ad = {'city_mpg': 18, 'engine_hp': 268, ...}
df_test_ad = pd.DataFrame([ad])
X_test = prepare_X(df_test_ad)
y_pred = w0 + X_test.dot(w)
suggestion = np.expm1(y_pred)  # แปลงกลับจาก log
```

---

## สรุปบทที่ 2

- **Linear regression** = หา weights ที่ทำให้ `w_0 + Xw ≈ y`
- ใช้ **Normal equation** หา weights แบบ closed-form
- **RMSE** วัดความแม่นยำ
- **Log transform** ช่วยจัดการ target ที่กระจายเบ้
- **Feature engineering** + **OHE** เพิ่มข้อมูลให้โมเดล
- **Regularization (r)** ป้องกัน weights ระเบิด
- เลือก `r` จาก validation, ยืนยันบน test
