# บทที่ 6: Decision Trees and Ensemble Learning
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> **โปรเจกต์**: Credit Risk Scoring — ทำนายว่าลูกค้าจะผิดนัดชำระไหม

---

## 6.1 Credit Risk Scoring Project

ปัญหา: ธนาคารต้องประเมินความเสี่ยงก่อนปล่อยกู้
- Target: `default` (0/1)
- Dataset: German credit data

วิธี: ใช้ **tree-based models** (Decision Tree, Random Forest, XGBoost)

---

## 6.2 Data Preparation

```python
df = pd.read_csv('credit_data.csv')

# Clean column names
df.columns = df.columns.str.lower()

# แปลง categorical → ตัวเลขผ่าน mapping (หรือใช้ DictVectorizer)
# จัดการ missing values

# Split train/val/test (60/20/20)
from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)
```

---

## 6.3 Decision Trees

**Decision Tree (ต้นไม้ตัดสินใจ)** = ลำดับของ if/else แตกเป็น branches

**ปัญหา**: **Overfitting (โมเดลจำขึ้นใจ)** เพราะต้นไม้ลึกเกินจะจำทุก noise

วิธีคุม:
- `max_depth` = ความลึกสูงสุด
- ลึก 1 = **decision stump** (split ครั้งเดียว)

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

# ดูกฎ
print(export_text(dt, feature_names=dv.get_feature_names_out()))
```

---

## 6.4 Decision Tree Learning Algorithm

อัลกอริทึมเลือก split ที่ "ดีที่สุด" ทีละ node

**Impurity (ความปนเปื้อน)** ที่ใช้วัด:
- **Misclassification rate** = ฝั่งไหนพลาดน้อยสุด
- **Gini impurity**
- **Entropy**

ขั้นตอน Recursive Split:
1. สำหรับทุก feature และทุก threshold ที่เป็นไปได้
2. คำนวณ weighted impurity ของ left + right
3. เลือก split ที่ลด impurity มากสุด
4. ทำซ้ำกับ child nodes
5. หยุดเมื่อ: ถึง max_depth, หรือ node เล็กกว่า min_samples, หรือ pure แล้ว

---

## 6.5 Decision Tree Parameter Tuning

Hyperparameters สำคัญ:
- **max_depth** — ความลึก
- **min_samples_leaf** — จำนวนตัวอย่างขั้นต่ำใน leaf
- **min_samples_split** — จำนวนตัวอย่างขั้นต่ำที่จะ split

```python
for d in [1, 2, 3, 4, 5, 6, 10, 15, 20, None]:
    for s in [1, 5, 10, 15, 20, 50, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
        dt.fit(X_train, y_train)
        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        # บันทึก
```

ทำเป็น heatmap ดูคู่ที่ดีสุด

---

## 6.6 Ensemble Learning & Random Forest

**Ensemble Learning** = รวมหลายโมเดล → ผลดีกว่าตัวเดี่ยว

**Random Forest** = ป่าของ decision trees
- เทรนหลายต้น แต่ละต้นเห็นข้อมูล + features ไม่เหมือนกัน (สุ่ม)
- รวมผลด้วย average (regression) หรือ voting (classification)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # จำนวนต้น
    max_depth=10,            # ความลึกของแต่ละต้น
    min_samples_leaf=5,
    random_state=1,
    n_jobs=-1                # ใช้ทุก CPU core
)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_val)[:, 1]
```

**ข้อดี**: ทน overfitting มากกว่า single tree, ไม่ต้อง tune มาก

Tune `n_estimators`: ลองค่าต่างๆ ดูที่ AUC คงที่ → ใช้ค่านั้น

---

## 6.7 Gradient Boosting & XGBoost

**Boosting** = สร้างต้นไม้ทีละต้น แต่ละต้นแก้ข้อผิดพลาดของก่อนหน้า

**XGBoost (eXtreme Gradient Boosting)** = library ยอดนิยม เร็ว แม่นยำ

```python
import xgboost as xgb

# เตรียมข้อมูลใน format ของ XGBoost
features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

xgb_params = {
    'eta': 0.3,              # learning rate
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  evals=watchlist, verbose_eval=10)

y_pred = model.predict(dval)
```

---

## 6.8 XGBoost Parameter Tuning

**Hyperparameters หลัก**:
| พารามิเตอร์ | ความหมาย |
|---|---|
| `eta` (learning_rate) | step size, ปกติ 0.01-0.3 |
| `max_depth` | ความลึกต้น 3-10 |
| `min_child_weight` | จำนวนตัวอย่างขั้นต่ำใน leaf |
| `subsample` | สุ่มแถว % ต่อรอบ |
| `colsample_bytree` | สุ่ม columns % ต่อต้น |
| `num_boost_round` | จำนวนต้น |

**Order ของการ tune**:
1. tune `eta` ก่อน
2. tune `max_depth`
3. tune `min_child_weight`
4. (อาจเพิ่ม) `subsample`, `colsample_bytree`

ใช้ **early stopping** ป้องกัน overfit:
```python
model = xgb.train(xgb_params, dtrain, num_boost_round=500,
                  evals=watchlist, early_stopping_rounds=10)
```

---

## 6.9 Selecting the Final Model

เปรียบเทียบ 3 โมเดล:
- Decision Tree (tuned)
- Random Forest (tuned)
- XGBoost (tuned)

เลือกตัวที่ AUC บน validation สูงสุด → train บน full_train → test บน test set

---

## สรุปบทที่ 6

- **Decision Tree** = if/else recursive, overfit ง่าย → tune `max_depth`, `min_samples_leaf`
- **Random Forest** = หลายต้นสุ่ม, รวมผล → ทน overfit
- **XGBoost** = gradient boosting, แม่นที่สุดสำหรับ tabular data
- Tree-based models **ไม่ต้อง scale features** (ต่างจาก linear models)
- XGBoost มักชนะใน Kaggle สำหรับข้อมูลตาราง
