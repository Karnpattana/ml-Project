# Workshop: Deep Learning ด้วย PyTorch
> ที่มา: [alexeygrigorev/workshops — mlzoomcamp-deep-learning](https://github.com/alexeygrigorev/workshops/blob/main/mlzoomcamp-deep-learning/README.md)
> โดย Alexey Grigorev — [Video](https://www.youtube.com/watch?v=Ne25VujHRLA) · [Notebook (Colab)](https://colab.research.google.com/drive/1nCA4Q0f8DVFiLpfXdXvZtYUh-yYDy5i_)
> เป็นเนื้อหาคู่ขนานกับ **บท 8 ของ ML Zoomcamp** แต่ใช้ **PyTorch** แทน TensorFlow/Keras

---

## ภาพรวม

**โปรเจกต์**: Image Classification เสื้อผ้า 10 หมวด (dress, hat, longsleeve, outwear, pants, shirt, shoes, shorts, skirt, t-shirt)
**Dataset**: `clothing-dataset-small`
**Pre-trained model**: **MobileNetV2** (เบากว่า Xception ที่ใช้ในบท 8)
**Run บน**: Google Colab (ไม่ต้องตั้งค่าอะไร)

ดาวน์โหลด dataset:
```bash
git clone https://github.com/alexeygrigorev/clothing-dataset-small.git
```

---

## 1. Introduction to PyTorch

**PyTorch** = framework deep learning ของ Facebook AI Research

**จุดเด่น**:
- **Dynamic computation graphs (define-by-run)** — สร้าง graph ขณะรัน → debug ง่ายกว่า static
- **Pythonic** — เขียนเหมือน Python ปกติ
- รองรับ GPU
- ecosystem ใหญ่ (torchvision, torchaudio, ...)

**ความต่างหลักจาก TensorFlow/Keras**:

| TensorFlow/Keras | PyTorch |
|---|---|
| `model.fit()` | เขียน training loop เอง |
| `ImageDataGenerator` | `Dataset` + `DataLoader` + `transforms` |
| `keras.layers.Dense()` | `nn.Linear()` |
| `keras.Model` | `nn.Module` |
| `.h5` / `.keras` | `.pth` / `.pt` |
| Device อัตโนมัติ | ต้อง `.to(device)` เอง |

---

## 2. การโหลดและเตรียมภาพ

**ภาพ** = array 3 มิติ
- Height × Width × Channels
- 3 channels = R, G, B
- แต่ละ pixel 0-255 (8 bits ต่อ channel)

```python
from PIL import Image
import numpy as np

img = Image.open('clothing-dataset-small/train/pants/xxx.jpg')
img = img.resize((224, 224))
x = np.array(img)
print(x.shape)  # (224, 224, 3)
```

---

## 3. Pre-trained Models

**ทำไมใช้โมเดลที่ฝึกแล้ว (pre-trained)?**
- ใช้ความรู้จาก ImageNet (1.4M ภาพ, 1000 คลาส)
- เลเยอร์ต้นๆ จับ edge/texture/shape ได้ดีอยู่แล้ว
- ใช้ได้แม้ data เราน้อย
- ประหยัดเวลาเทรน

### ใช้ MobileNetV2

```python
import torch
import torchvision.models as models
from torchvision import transforms

# โหลด pre-trained
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
model.eval()   # eval mode — สำคัญ! ปิด dropout, batchnorm

# Preprocessing มาตรฐานของ MobileNetV2
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

ทำนาย:
```python
img = Image.open('pants.jpg')
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)   # เพิ่ม batch dimension

with torch.no_grad():    # ปิดการคำนวณ gradient ตอน inference
    output = model(batch_t)

_, indices = torch.sort(output, descending=True)
```

**สิ่งที่ต้องจำ**:
- Input MobileNetV2 = 224×224 (Xception = 299×299)
- ต้อง **normalize** ด้วย mean/std ของ ImageNet
- รูปร่าง tensor = `(batch_size, channels, height, width)` เช่น `(1, 3, 224, 224)`
- ⚠️ PyTorch ใช้ **NCHW** (channels นำหน้า) ต่างจาก Keras ที่ใช้ **NHWC**

---

## 4. Convolutional Neural Networks (CNN)

**ส่วนประกอบหลัก**:

1. **Convolutional Layer** — ใช้ filter (3×3, 5×5) เลื่อนทั่วภาพ จับ pattern → ได้ **feature maps**
2. **ReLU Activation** — `f(x) = max(0, x)` ทำให้โมเดล non-linear
3. **Pooling Layer** — ลดขนาด, **Max pooling** เก็บค่ามากสุดในพื้นที่
4. **Fully Connected (Dense) Layer** — flatten 2D → 1D แล้วต่อเข้า output

**Flow**:
```
Input → [Conv + ReLU → Pool] × N → Flatten → Dense → Output
```

---

## 5. Transfer Learning

**Transfer Learning (การเรียนข้ามงาน)** = เอาโมเดลที่เทรนจากงานหนึ่ง (ImageNet) มาประยุกต์งานอื่น (clothing classification)

**ขั้นตอน**:
1. โหลด pre-trained model (ใช้เป็น feature extractor)
2. ตัด classification head เดิมทิ้ง
3. **Freeze (แช่แข็ง)** conv layers — ไม่เทรนซ้ำ
4. เพิ่ม dense layers ใหม่ของเรา
5. เทรนเฉพาะ layers ใหม่

### Custom Dataset Class

PyTorch ต้องเขียน class โหลดข้อมูลเอง (ไม่มี ImageDataGenerator):

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

3 method ที่ต้องมี: `__init__`, `__len__`, `__getitem__`

### Preprocessing Transforms

```python
from torchvision import transforms

input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

### DataLoader

```python
from torch.utils.data import DataLoader

train_dataset = ClothingDataset('./clothing-dataset-small/train', train_transforms)
val_dataset = ClothingDataset('./clothing-dataset-small/validation', val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

- `shuffle=True` สำหรับ train (สำคัญ — กันโมเดลจำ order)
- `shuffle=False` สำหรับ val

### สร้างโมเดล

```python
import torch.nn as nn
import torchvision.models as models

class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # Freeze base
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # ตัด classifier เดิม
        self.base_model.classifier = nn.Identity()
        
        # ชั้นใหม่
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x
```

### Training Loop (เขียนเองทั้งหมด)

```python
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClothingClassifierMobileNet(num_classes=10)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()    # → เปิด dropout, batchnorm update
    running_loss = 0.0
    correct, total = 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()         # ⚠️ สำคัญ! เคลียร์ gradient เดิม
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()               # backward pass
        optimizer.step()              # อัปเดต weights
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Validation phase
    model.eval()     # → ปิด dropout, batchnorm ใช้สถิติเดิม
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch+1}: train_acc={correct/total:.3f}, val_acc={val_correct/val_total:.3f}')
```

**3 จุดสำคัญที่ต้องเข้าใจ**:

1. **`optimizer.zero_grad()`** — PyTorch สะสม gradient โดย default ถ้าไม่เคลียร์ก่อน gradient ของ batch ก่อนหน้าจะบวกเข้า batch ปัจจุบัน → อัปเดตผิด
2. **`model.train()` vs `model.eval()`** — ควบคุมพฤติกรรมของ Dropout, BatchNorm
3. **`with torch.no_grad():`** — ไม่บันทึก gradient ตอน inference → เร็วขึ้น ประหยัด RAM

---

## 6. Tuning Learning Rate

**Learning rate** = ขนาดก้าวอัปเดต weights — hyperparameter สำคัญที่สุด

**เปรียบเทียบกับการอ่านหนังสือ**:
- เร็วไป → ข้ามรายละเอียด → ไม่ converge
- ช้าไป → อ่านไม่จบสักที
- พอดี → เข้าใจดีและเร็ว

**วิธีหา**:
```python
for lr in [0.0001, 0.001, 0.01, 0.1]:
    model, optimizer = make_model(learning_rate=lr)
    train_and_evaluate(model, optimizer, ...)
```

→ ดูที่ **validation accuracy สูงสุด** และ **ช่อง train/val น้อย**
ใน workshop ค่าที่ดีที่สุด: `lr = 0.001` (val acc ~0.815)

---

## 7. Model Checkpointing

บันทึกโมเดลที่ดีที่สุดระหว่างเทรน:

```python
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    # ... training + validation ...
    
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        checkpoint_path = f'mobilenet_v2_{epoch+1:02d}_{val_acc:.3f}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
```

**เทียบกับ Keras**:
- Keras: `ModelCheckpoint` callback ทำให้
- PyTorch: เขียนเอง (มีอิสระเลือก condition ใดก็ได้)

`model.state_dict()` = dict ที่เก็บ weights ทั้งหมด

---

## 8. Adding Inner Layers

เพิ่ม **inner dense layer** ระหว่าง feature extractor กับ output:

```python
class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, num_classes=10):
        super().__init__()
        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.classifier = nn.Identity()
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.inner = nn.Linear(1280, size_inner)    # NEW
        self.relu = nn.ReLU()                        # NEW
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x
```

**ลอง**: `size_inner = [10, 100, 1000]`
- ใหญ่ → จุได้มาก แต่อาจ overfit
- เล็ก → เร็ว แต่อาจ underfit

⚠️ **Output layer ไม่ใส่ activation** — เพราะ `CrossEntropyLoss` apply softmax ภายในเองอยู่แล้ว (ค่า output เรียกว่า **logits**)

---

## 9. Dropout Regularization

**Dropout** = สุ่มปิด neurons ระหว่างเทรน → กัน overfitting

**กลไก**:
- **Training**: ปิด neurons แบบสุ่ม `droprate%`
- **Inference**: ใช้ neurons ทั้งหมด (Dropout ปิดอัตโนมัติเมื่อ `model.eval()`)
- เหมือนสร้าง ensemble ของ sub-networks

```python
class ClothingClassifierMobileNet(nn.Module):
    def __init__(self, size_inner=100, droprate=0.2, num_classes=10):
        super().__init__()
        # ... โค้ดเดิม ...
        self.inner = nn.Linear(1280, size_inner)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(droprate)        # NEW
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)                        # NEW
        x = self.output_layer(x)
        return x
```

ลอง: `droprate = [0.0, 0.2, 0.5, 0.8]` — ค่าปกติ 0.2-0.5
ใน workshop ค่าที่ดีที่สุด: `0.2`

---

## 10. Data Augmentation

**Data Augmentation** = สร้างภาพปลอมจากภาพจริง (พลิก, หมุน, ซูม) เพื่อเพิ่มความหลากหลาย

```python
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),                          # หมุนสุ่ม ±10°
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),   # ซูมสุ่ม
    transforms.RandomHorizontalFlip(),                      # พลิกซ้าย-ขวา
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Validation — ไม่ augment!
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
```

**กฎสำคัญ**:
- ✅ ใช้กับ **training data เท่านั้น**
- ❌ **ห้าม** augment validation/test data

**ควรใช้เมื่อ**:
- Dataset เล็ก
- เสี่ยง overfit
- ภาพในโลกจริงมีหลายมุม

**ทิป**:
- เลือก augmentation ที่สมเหตุสมผล (เสื้อกลับหัวไม่สมเหตุสมผล)
- มากไปอาจแย่ลง
- มักต้องเทรนนานขึ้น
- ถ้าหลัง 20 epochs ไม่ดีขึ้น ลองเอาออก

---

## 11. การใช้งานโมเดล

### โหลดโมเดลที่บันทึก

```python
import glob, os

list_of_files = glob.glob('mobilenet_v2_*.pth')
latest_file = max(list_of_files, key=os.path.getctime)

model = ClothingClassifierMobileNet(size_inner=32, droprate=0.2, num_classes=10)
model.load_state_dict(torch.load(latest_file))
model.to(device)
model.eval()
```

### ทำนายภาพใหม่

```python
from keras_image_helper import create_preprocessor

def preprocess_pytorch_style(X):
    # X: shape (1, 224, 224, 3), values 0-255
    X = X / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    X = X.transpose(0, 3, 1, 2)         # NHWC → NCHW
    X = (X - mean) / std
    return X.astype(np.float32)

preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(224, 224))

url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)
X = torch.Tensor(X).to(device)

with torch.no_grad():
    pred = model(X).cpu().numpy()[0]

classes = ["dress", "hat", "longsleeve", "outwear", "pants",
           "shirt", "shoes", "shorts", "skirt", "t-shirt"]
result = dict(zip(classes, pred.tolist()))
```

---

## 12. Export เป็น ONNX

**ONNX (Open Neural Network Exchange)** = format กลางสำหรับโมเดล ML

**ข้อดี**:
- Deploy ได้หลาย platform
- ใช้ ONNX Runtime — เร็วกว่า PyTorch ปกติ
- ไม่ผูกกับ Python (รัน C++, JS ก็ได้)
- เหมาะกับ serverless

```python
dummy_input = torch.randn(1, 3, 224, 224).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "clothing_classifier_mobilenet_v2.onnx",
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},      # batch size ยืดหยุ่นได้
        'output': {0: 'batch_size'}
    }
)
```

ใช้ใน serverless module ต่อ

---

## TensorFlow/Keras vs PyTorch — Quick Reference

| Concept | TensorFlow/Keras | PyTorch |
|---|---|---|
| Framework | High-level (Keras) | Low-level, control เอง |
| Data Loading | `ImageDataGenerator` | `Dataset` + `DataLoader` |
| Transforms | `preprocessing_function` | `transforms.Compose()` |
| Model | Functional / Sequential | `nn.Module` class |
| Layers | `keras.layers.Dense()` | `nn.Linear()` |
| Training | `model.fit()` | Training loop เอง |
| Loss | `CategoricalCrossentropy` | `CrossEntropyLoss` |
| Optimizer | `keras.optimizers.Adam` | `optim.Adam` |
| Saving | `.h5` / `.keras` | `.pth` / `.pt` |
| Checkpointing | `ModelCheckpoint` callback | Manual ใน loop |
| Device | อัตโนมัติ | ต้อง `.to(device)` |
| Tensor layout | NHWC (channels last) | NCHW (channels first) |

---

## สรุป Key Concepts

1. **Transfer Learning** — ใช้ pre-trained model ประหยัดเวลาเทรน
2. **CNN** — Conv + Pool + Dense
3. **Learning Rate** — hyperparameter สำคัญสุด — tune ก่อน
4. **Dropout** — กัน overfit
5. **Augmentation** — เพิ่มข้อมูลปลอม (เฉพาะ training)
6. **Checkpointing** — เก็บโมเดลที่ดีที่สุด
7. **PyTorch Workflow**: Dataset → DataLoader → nn.Module → Training Loop

## Best Practices

1. เริ่มจาก pre-trained model เสมอ
2. Freeze conv layers ก่อน เทรนเฉพาะ head
3. ใช้ normalization ตามที่ pre-trained ใช้
4. ปรับ hyperparameter ทีละตัว
5. ดูช่อง train/val gap เพื่อจับ overfit
6. ใช้ checkpointing
7. Augment เฉพาะ train
8. ถ้าใช้ dropout + augmentation → เทรนนานขึ้น
9. ใช้ GPU ถ้ามี: `torch.cuda.is_available()`
