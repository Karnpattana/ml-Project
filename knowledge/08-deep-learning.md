# บทที่ 8: Neural Networks and Deep Learning
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> **โปรเจกต์**: Image Classification (จำแนกประเภทเสื้อผ้า)
> *หมายเหตุ: ไม่มีบท 7 ในคอร์ส*

---

## 8.1 Fashion Classification Project

ปัญหา: จำแนกรูปเสื้อผ้า 10 หมวด (เสื้อยืด, รองเท้า, กระเป๋า, ...)
- Input: รูปภาพ
- Output: หมวด (multi-class classification)

ใช้ **Deep Learning (การเรียนรู้เชิงลึก)** เพราะ tabular models เจอภาพไม่ไหว

---

## 8.2 TensorFlow & Keras

**TensorFlow** = library ของ Google สำหรับ deep learning
**Keras** = API ระดับสูงของ TensorFlow (เขียนง่ายขึ้น)

```python
import tensorflow as tf
from tensorflow import keras

# โหลดภาพ
from tensorflow.keras.preprocessing.image import load_img
img = load_img('image.jpg', target_size=(299, 299))

# แปลงเป็น array
import numpy as np
x = np.array(img)
```

---

## 8.3 Pre-trained Models

**Pre-trained model** = โมเดลที่ฝึกมาแล้วจาก dataset ใหญ่ (เช่น **ImageNet**)

ใช้ทันที (ไม่ต้องเทรนเอง):
```python
from tensorflow.keras.applications.xception import (
    Xception, preprocess_input, decode_predictions
)

model = Xception(weights='imagenet', input_shape=(299, 299, 3))

# Preprocess (Xception ต้องการ pixel ใน [-1, 1])
X = np.array([x])
X = preprocess_input(X)

pred = model.predict(X)
decode_predictions(pred)   # ดูคลาสและ probability
```

**ปัญหา**: Xception รู้จัก 1000 คลาสของ ImageNet ไม่ใช่ 10 หมวดเสื้อผ้าของเรา → ต้องทำ **transfer learning**

---

## 8.4 Convolutional Neural Networks (CNN)

**CNN (เครือข่ายประสาทคอนโวลูชัน)** = โครงสร้างที่เหมาะกับภาพ

ส่วนประกอบหลัก:
- **Convolutional layer** — ตรวจจับ pattern (เส้น, ขอบ, รูปทรง)
- **Pooling layer** — ลดขนาด, ดึงเฉพาะ feature เด่น
- **Dense layer** — ชั้น fully-connected ปกติ (ที่ตอนท้ายของโมเดล)

**Filter (ตัวกรอง)** = matrix เล็กๆ (เช่น 3×3) ที่ slide ทั้งภาพ
- เลเยอร์แรก: ตรวจ edges, lines
- เลเยอร์ลึก: ตรวจ shapes, parts, objects

---

## 8.5 Transfer Learning

แทนที่จะเทรน CNN ใหม่หมด → ใช้ **base model** (Xception, ResNet) แล้วเปลี่ยนเฉพาะชั้นบนสุด

```python
base_model = Xception(
    weights='imagenet',
    include_top=False,        # ตัดชั้น dense ทิ้ง
    input_shape=(150, 150, 3)
)
base_model.trainable = False   # freeze base — ไม่เทรนซ้ำ

# ต่อด้วยชั้นใหม่ของเราเอง
inputs = keras.Input(shape=(150, 150, 3))
base = base_model(inputs, training=False)
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(10)(vectors)   # 10 คลาส
model = keras.Model(inputs, outputs)

# Compile
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss = keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
```

**Image Data Generator** สำหรับโหลดภาพเป็น batch:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    'data/train', target_size=(150, 150), batch_size=32
)

# เทรน
model.fit(train_ds, epochs=10, validation_data=val_ds)
```

---

## 8.6 Learning Rate

**Learning Rate (อัตราการเรียนรู้)** = ขนาดก้าวเวลาอัปเดต weights

- ใหญ่ไป → ก้าวเกินจุดต่ำสุด, loss กระโดด
- เล็กไป → เทรนช้า, ค้างที่ local minimum

ลองหลายค่า:
```python
for lr in [0.0001, 0.001, 0.01, 0.1]:
    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    # plot learning curve
```

มักดี: 0.001 หรือ 0.01

---

## 8.7 Model Checkpointing

บันทึกโมเดลที่ดีที่สุดระหว่างเทรน (validation accuracy สูงสุด)

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])
```

---

## 8.8 Adding More Layers

เพิ่ม **inner dense layer** ระหว่าง GlobalAveragePooling กับ output:

```python
vectors = keras.layers.GlobalAveragePooling2D()(base)
inner = keras.layers.Dense(100, activation='relu')(vectors)
outputs = keras.layers.Dense(10)(inner)
```

**Activation function (ฟังก์ชันกระตุ้น)**:
- **ReLU** — `max(0, x)` ใช้บ่อยสุด
- **Sigmoid** — binary output
- **Softmax** — multi-class probability (ตอน output)

---

## 8.9 Dropout (Regularization)

**Dropout** = สุ่มปิด neurons บางตัวระหว่างเทรน → กัน overfit

```python
inner = keras.layers.Dense(100, activation='relu')(vectors)
drop = keras.layers.Dropout(0.5)(inner)   # ปิด 50% neurons
outputs = keras.layers.Dense(10)(drop)
```

Dropout rate ปกติ: 0.2–0.5

---

## 8.10 Data Augmentation

**Augmentation (การเพิ่มข้อมูล)** = เปลี่ยนภาพต้นฉบับเล็กน้อย (พลิก, หมุน, ซูม) → ได้ข้อมูลมากขึ้น

```python
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10.0,
    height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)
```

⚠️ ใช้กับ training set เท่านั้น, ไม่ใช้กับ validation/test

---

## 8.11 Training a Larger Model

ลองเปลี่ยนขนาด input image จาก 150×150 → 299×299 (ขนาดเดิมของ Xception)
- ใช้ทรัพยากรมากขึ้น
- มักได้ accuracy ดีขึ้น

---

## 8.12 Using the Model

โหลดโมเดลที่บันทึกไว้ + ทำนายภาพใหม่:
```python
model = keras.models.load_model('xception_v4_large_08_0.894.h5')

img = load_img('pants.jpg', target_size=(299, 299))
x = np.array(img)
X = np.array([x])
X = preprocess_input(X)

pred = model.predict(X)
classes = ['dress', 'hat', 'longsleeve', 'outwear',
           'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
dict(zip(classes, pred[0]))
```

---

## สรุปบทที่ 8

- **Deep Learning** เหมาะกับภาพ/เสียง/ข้อความ (unstructured data)
- **CNN** = layer คอนโวลูชันสำหรับภาพ
- **Transfer Learning** = ใช้ pre-trained model + เทรนเฉพาะชั้นบน
- **Learning rate** สำคัญสุด — ต้อง tune
- **Checkpointing** เก็บโมเดลที่ดีที่สุด
- **Dropout** + **Augmentation** ลด overfit
- Larger image / inner layers อาจช่วยแต่ใช้ทรัพยากรเพิ่ม
