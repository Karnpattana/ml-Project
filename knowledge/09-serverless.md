# บทที่ 9: Serverless Deep Learning
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> Deploy โมเดล deep learning จากบท 8 ไป AWS Lambda

---

## 9.1 Intro to Serverless

**Serverless** = รัน code โดยไม่ต้องจัดการ server เอง
- จ่ายตามที่ใช้จริง (per request)
- Auto-scale
- ไม่มี server ค้างไว้

**AWS Lambda** = serverless service ของ Amazon
- เหมาะกับ workload ไม่บ่อย (เทียบกับ EC2 ที่เปิดทั้งวัน)

---

## 9.2 AWS Lambda

**Lambda Function** = function ที่ AWS รันให้เมื่อ trigger
- ตั้งค่า memory, timeout
- รองรับ Python, Node.js, ...
- ลิมิตขนาด deployment package

ตัวอย่างเล็กที่สุด:
```python
def lambda_handler(event, context):
    print('parameters:', event)
    return {
        'statusCode': 200,
        'body': 'hello world'
    }
```

---

## 9.3 TensorFlow Lite

**TensorFlow Lite (TFLite)** = TensorFlow เวอร์ชันเล็กสำหรับ mobile/edge/serverless

ทำไมใช้ TFLite:
- TensorFlow เต็มตัว ~1.7 GB → ใส่ Lambda ไม่ได้
- TFLite ~50 MB
- ใช้แค่ inference (ไม่ train)

แปลง model `.h5` → `.tflite`:
```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('clothing-model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)
```

ใช้งาน TFLite:
```python
import tensorflow.lite as tflite
# หรือ: from tflite_runtime import interpreter as tflite (เบากว่า)

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
```

---

## 9.4 Preparing Lambda Code

`lambda_function.py`:
```python
import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO
from urllib import request
import numpy as np

classes = ['dress', 'hat', 'longsleeve', 'outwear',
           'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    return Image.open(stream)

def prepare_image(img, target_size):
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    img = prepare_image(img, (299, 299))
    
    x = np.array(img, dtype='float32')
    X = np.array([x])
    X /= 127.5
    X -= 1.0   # preprocess: pixel ใน [-1, 1]
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return dict(zip(classes, preds[0].tolist()))

def lambda_handler(event, context):
    url = event['url']
    return predict(url)
```

---

## 9.5 Docker Image for Lambda

AWS Lambda ตอนนี้รองรับ **container image** (สูงสุด 10 GB) — เหมาะกับ ML

`Dockerfile`:
```dockerfile
FROM public.ecr.aws/lambda/python:3.8

RUN pip install --no-cache-dir \
    https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl \
    pillow

COPY clothing-model.tflite .
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
```

**Build & Test Local:**
```bash
docker build -t clothing-model .
docker run -it --rm -p 8080:8080 clothing-model
```

ทดสอบ:
```python
import requests
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
data = {'url': 'https://example.com/pants.jpg'}
result = requests.post(url, json=data).json()
```

---

## 9.6 Creating the Lambda Function

1. **Push image** ไป **ECR (Elastic Container Registry)**:
```bash
aws ecr create-repository --repository-name lambda-images
$(aws ecr get-login --no-include-email)
docker tag clothing-model:latest <account>.dkr.ecr.<region>.amazonaws.com/lambda-images:clothing-model
docker push <account>.dkr.ecr.<region>.amazonaws.com/lambda-images:clothing-model
```

2. **Create Lambda function** ใน AWS Console:
   - Container image
   - เลือก image จาก ECR
   - ตั้ง memory (1024 MB+) และ timeout (30s+)

3. **Test** ใน console ด้วย event:
```json
{"url": "https://example.com/pants.jpg"}
```

---

## 9.7 API Gateway

ทำให้ Lambda เรียกได้ผ่าน HTTP ปกติ

ขั้นตอน:
1. AWS Console → API Gateway → REST API → Build
2. สร้าง resource เช่น `/predict`
3. สร้าง method `POST` → Integration type: **Lambda Function**
4. Deploy API → ได้ URL

ใช้งาน:
```python
url = 'https://xxxxx.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://example.com/pants.jpg'}
result = requests.post(url, json=data).json()
```

---

## สรุปบทที่ 9

Pipeline serverless:
1. **Convert** model `.h5` → `.tflite` (เล็กลง)
2. **Code** Lambda function (TFLite + PIL + numpy)
3. **Containerize** ด้วย Docker (image พิเศษของ Lambda)
4. **Push** ไป ECR
5. **Create Lambda** จาก container image
6. **Expose** ผ่าน API Gateway

ข้อดี: ไม่ต้องดู server, จ่ายเฉพาะที่ใช้
ข้อจำกัด: cold start, timeout 15 นาที, RAM สูงสุด ~10 GB
