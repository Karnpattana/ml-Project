# บทที่ 1: Introduction to Machine Learning
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) (Free, MIT License)
> สอนโดย Alexey Grigorev — สรุปและเรียบเรียงเป็นภาษาไทย

---

## 1.1 What is Machine Learning?

**Machine Learning (ML, การเรียนรู้ของเครื่อง)** คือกระบวนการ "สกัดรูปแบบ (patterns) จากข้อมูล"

ตัวอย่าง: ทำนายราคารถ
- **Features (ฟีเจอร์, คุณลักษณะ)** = ข้อมูลของวัตถุ เช่น ปี รถยี่ห้อ ระยะทาง
- **Target (เป้าหมาย)** = สิ่งที่จะทำนาย เช่น ราคา

โมเดล (model) จะเรียนรู้ความสัมพันธ์จากข้อมูลที่มีเฉลย แล้วนำมาทำนายข้อมูลใหม่ที่ยังไม่รู้คำตอบ

```
Features + Target  →  Train  →  Model
Features (ใหม่)    →  Model  →  Predictions
```

---

## 1.2 ML vs Rule-Based Systems

เปรียบเทียบโดยใช้ตัวอย่าง **spam filter (ตัวกรองสแปม)**

**Rule-Based (ระบบกฎเกณฑ์)**:
- เขียน if/else ตามคำสำคัญ ความยาวอีเมล ฯลฯ
- พอสแปมเปลี่ยนรูปแบบ ต้องอัปเดตกฎไปเรื่อยๆ → ดูแลยาก

**ML Approach**:
1. **เก็บข้อมูล** — อีเมลจาก spam folder + inbox
2. **สร้าง features** — แปลงกฎเดิมเป็น features เช่น มีคำว่า "free"? ความยาว? ผู้ส่งใหม่?
3. **เทรนโมเดล** — โมเดลให้ output เป็น **probability (ความน่าจะเป็น)** แล้วเอา threshold มาตัดสิน

---

## 1.3 Supervised Machine Learning

**Supervised Learning (การเรียนแบบมีผู้สอน)** — มี label หรือเฉลยให้

สูตรทั่วไป: `g(X) = y`
- `X` = feature matrix (เมทริกซ์ของฟีเจอร์)
- `y` = target vector
- `g` = โมเดล

**ประเภทของ supervised tasks:**
| ประเภท | ความหมาย | ตัวอย่าง |
|---|---|---|
| **Regression** | ทำนายตัวเลขต่อเนื่อง | ราคารถ, อุณหภูมิ |
| **Classification** | ทำนายหมวดหมู่ | สแปม/ไม่สแปม, แมว/หมา/นก |
| **Ranking** | จัดอันดับ | เครื่องมือค้นหา, recommendation |

Classification แบ่งเป็น **Binary (2 คลาส)** และ **Multiclass (หลายคลาส)**

---

## 1.4 CRISP-DM

**CRISP-DM** (Cross-Industry Standard Process for Data Mining, มาตรฐานกระบวนการทำงาน ML)

6 ขั้นตอน:
1. **Business Understanding** — เข้าใจปัญหาธุรกิจ ตั้งเป้าหมายที่วัดได้ ตัดสินใจว่าต้องใช้ ML ไหม
2. **Data Understanding** — สำรวจข้อมูล ดูคุณภาพ ความน่าเชื่อถือ
3. **Data Preparation** — clean, transform, สร้าง pipeline
4. **Modeling** — เทรนหลายโมเดล เลือกตัวที่ดีสุด
5. **Evaluation** — ประเมินว่าโมเดลตอบโจทย์ธุรกิจไหม
6. **Deployment** — เอาขึ้น production, monitor, maintain

เป็น iterative process (วนซ้ำหลายรอบ) เริ่มจากเล็กแล้วขยาย

---

## 1.5 Model Selection Process

**ปัญหา Multiple Comparison Problem**: ถ้าทดสอบหลายโมเดลกับชุดข้อมูลเดียว อาจเจอตัวที่ "บังเอิญดี"

**วิธีแก้: แบ่งข้อมูลเป็น 3 ส่วน**
- **Training set (~60%)** — เทรน
- **Validation set (~20%)** — เลือกโมเดล/jhyperparameter
- **Test set (~20%)** — ทดสอบครั้งสุดท้าย เพื่อเช็คว่าไม่ใช่บังเอิญ

ลำดับ:
1. Split data → train, val, test
2. Train โมเดลต่างๆ บน training set
3. ประเมินบน validation set → เลือกตัวดีที่สุด
4. ทดสอบบน test set → ยืนยันผล

---

## 1.6 Environment Setup

เครื่องมือที่ใช้:
- **Python 3** (แนะนำผ่าน Anaconda)
- **NumPy** — คำนวณเชิงตัวเลข
- **Pandas** — จัดการ tabular data
- **Scikit-Learn** — ML algorithms
- **Matplotlib / Seaborn** — กราฟ
- **Jupyter Notebook** — เขียนโค้ดเชิงโต้ตอบ

ติดตั้ง Anaconda → ได้ครบทุกตัว
หรือใช้ cloud: AWS, GCP, Saturn Cloud

---

## 1.7 NumPy Basics

**NumPy (Numerical Python)** — ทำงานกับ array หลายมิติได้เร็วกว่า list ปกติ

```python
import numpy as np

# สร้าง array
np.zeros(10)            # array ของศูนย์ 10 ตัว
np.ones(10)             # array ของหนึ่ง 10 ตัว
np.full(10, 2.5)        # array ค่า 2.5 ทั้งหมด 10 ตัว
np.arange(3, 10)        # [3,4,5,6,7,8,9]
np.linspace(0, 1, 11)   # 11 ตัวเลขจาก 0 ถึง 1 ระยะเท่ากัน

# 2D array (matrix)
n = np.array([[1,2,3], [4,5,6], [7,8,9]])
n.shape                 # (3, 3)
n[0]                    # แถวแรก
n[:, 0]                 # คอลัมน์แรก

# Random
np.random.seed(2)
np.random.rand(5, 2)    # ค่าสุ่ม 0-1
np.random.randn(5, 2)   # normal distribution

# Element-wise operations
a + b, a * b, a / b     # ทำทีละตัว
```

---

## 1.8 Linear Algebra Refresher

ทบทวนพีชคณิตเชิงเส้น (linear algebra)

**Vector-vector multiplication (dot product):**
```
u·v = Σ u[i]*v[i]
```

```python
def vector_vector_multiplication(u, v):
    n = u.shape[0]
    result = 0.0
    for i in range(n):
        result += u[i] * v[i]
    return result

# NumPy
u.dot(v)
```

**Matrix-vector multiplication:**
- U (m×n) × v (n×1) = ผลลัพธ์ (m×1)
- คือ dot product ของแต่ละแถวของ U กับ v

**Matrix-matrix multiplication:** U (m×n) × V (n×k) = (m×k)

**Identity matrix (I)** — เมทริกซ์เอกลักษณ์ I·A = A
**Inverse (A⁻¹)** — A·A⁻¹ = I (มีเฉพาะเมทริกซ์จัตุรัสที่ไม่ singular)

```python
np.linalg.inv(V)  # หาเมทริกซ์ผกผัน
```

---

## 1.9 Pandas Basics

**Pandas** — จัดการข้อมูลแบบตาราง

```python
import pandas as pd

# สร้าง DataFrame
df = pd.DataFrame(data, columns=columns)
df.head()              # 5 แถวแรก
df.columns             # ชื่อคอลัมน์
df.dtypes              # ประเภทข้อมูล
df.index               # index

# เลือกข้อมูล
df['column_name']      # เลือก column
df[['col1', 'col2']]   # หลาย columns
df.iloc[0]             # เลือก row ตามตำแหน่ง
df.loc[0]              # เลือก row ตาม label

# กรอง
df[df['Year'] >= 2015]

# สถิติพื้นฐาน
df.describe()
df['col'].mean()
df['col'].nunique()    # นับค่าไม่ซ้ำ
df['col'].value_counts()  # นับแต่ละค่า

# Missing values
df.isnull().sum()
df.fillna(0)
```

---

## สรุปบทที่ 1 (Summary)

- **ML** = เครื่องเรียนรู้รูปแบบจากข้อมูล (features → target)
- เหมาะกับปัญหาที่ rules เปลี่ยนบ่อย/ซับซ้อน
- **Supervised ML** = เทรนด้วยข้อมูลที่มี label
- **CRISP-DM** = กรอบงาน ML 6 ขั้น
- แบ่ง data เป็น train/val/test เพื่อหลีกเลี่ยง multiple comparison problem
- เครื่องมือหลัก: NumPy, Pandas, Scikit-Learn, Jupyter
