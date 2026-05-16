# บทที่ 10: Kubernetes and TensorFlow Serving
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> Deploy โมเดล deep learning ด้วย Kubernetes (k8s) + TensorFlow Serving

---

## 10.1 Overview

**Kubernetes (k8s, คูเบอร์เนติส)** = orchestration platform สำหรับ container
- รัน container หลายตัวพร้อมกัน
- Auto-scaling, load balancing, self-healing
- มาตรฐานอุตสาหกรรมสำหรับ container ใน production

**Architecture ของระบบในบทนี้**:
```
Client → Gateway (Flask) → TF-Serving (gRPC) → ผลลัพธ์
```
- **Gateway** — รับ HTTP, ดาวน์โหลดภาพ, preprocess, ส่งต่อให้ TF-Serving
- **TF-Serving** — รัน TensorFlow model เฉพาะ inference

---

## 10.2 TensorFlow Serving

**TensorFlow Serving** = service ของ Google สำหรับ deploy TF models
- Optimized C++
- รองรับ gRPC + REST API
- ต้องใช้ format `saved_model` (ไม่ใช่ `.h5`)

**แปลง .h5 → saved_model:**
```python
import tensorflow as tf
model = tf.keras.models.load_model('clothing-model.h5')
tf.saved_model.save(model, 'clothing-model')
```

ดูโครงสร้าง model:
```bash
saved_model_cli show --dir clothing-model --all
```

**รัน TF-Serving ด้วย Docker:**
```bash
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/clothing-model:/models/clothing-model/1" \
    -e MODEL_NAME="clothing-model" \
    tensorflow/serving:2.7.0
```

---

## 10.3 Preprocessing Service (Gateway)

`gateway.py`:
```python
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from flask import Flask, request, jsonify
from keras_image_helper import create_preprocessor

host = 'localhost:8500'
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
preprocessor = create_preprocessor('xception', target_size=(299, 299))

classes = ['dress', 'hat', 'longsleeve', 'outwear',
           'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

def np_to_protobuf(data):
    return tf.make_tensor_proto(data, shape=data.shape)

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X))
    return pb_request

def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    preds = pb_response.outputs['dense_7'].float_val
    return dict(zip(classes, preds))

app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    return jsonify(predict(url))
```

---

## 10.4 Docker Compose

Run 2 containers พร้อมกันด้วย **docker-compose**

`docker-compose.yaml`:
```yaml
version: "3.9"
services:
  clothing-model:
    image: clothing-model:xception-v4-001
    ports:
      - "8500:8500"
  gateway:
    image: clothing-model-gateway:001
    environment:
      - TF_SERVING_HOST=clothing-model:8500
    ports:
      - "9696:9696"
```

```bash
docker-compose up
docker-compose down
```

---

## 10.5 Kubernetes Intro

**ส่วนประกอบหลักของ k8s**:
- **Node** — เครื่อง (VM/server) ในคลัสเตอร์
- **Pod** — หน่วยเล็กที่สุด (มัก = 1 container)
- **Deployment** — กลุ่มของ Pod ตามแม่แบบเดียวกัน (มี replica)
- **Service** — exposed endpoint คงที่ → กระจายโหลดให้ pods
  - **ClusterIP** — เข้าถึงภายใน cluster เท่านั้น
  - **LoadBalancer** — exposed สู่ภายนอก
- **Ingress** — routing HTTP จากภายนอก

**สถาปัตยกรรม**:
```
Internet → LoadBalancer Service → Gateway Pods (Deployment)
                                       ↓ (ClusterIP)
                                  TF-Serving Pods (Deployment)
```

---

## 10.6 Kubernetes — Simple Service

ติดตั้ง **Kind** (Kubernetes in Docker) สำหรับเทสต์ local:
```bash
kind create cluster
kubectl cluster-info
```

ตัวอย่าง deployment (`deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ping-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ping
  template:
    metadata:
      labels:
        app: ping
    spec:
      containers:
      - name: ping-pod
        image: ping:v001
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
        ports:
        - containerPort: 9696
```

Service (`service.yaml`):
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ping
spec:
  type: LoadBalancer
  selector:
    app: ping
  ports:
  - port: 80
    targetPort: 9696
```

Apply:
```bash
kind load docker-image ping:v001        # โหลดเข้า kind cluster
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

kubectl get deployments
kubectl get pods
kubectl get services
```

---

## 10.7 Kubernetes — TF-Serving

Deploy 2 services:
1. **clothing-model** (TF-Serving) — ClusterIP, port 8500
2. **gateway** (Flask) — LoadBalancer, port 80 → 9696

ไฟล์ `model-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-serving-clothing-model
spec:
  selector:
    matchLabels:
      app: tf-serving-clothing-model
  template:
    metadata:
      labels:
        app: tf-serving-clothing-model
    spec:
      containers:
      - name: tf-serving-clothing-model
        image: zoomcamp-10-model:xception-v4-001
        ports:
        - containerPort: 8500
---
apiVersion: v1
kind: Service
metadata:
  name: tf-serving-clothing-model
spec:
  type: ClusterIP
  selector:
    app: tf-serving-clothing-model
  ports:
  - port: 8500
    targetPort: 8500
```

Gateway แบบเดียวกันแต่ type: LoadBalancer

ใน gateway env: `TF_SERVING_HOST=tf-serving-clothing-model.default.svc.cluster.local:8500`

---

## 10.8 EKS (Elastic Kubernetes Service)

**EKS** = managed Kubernetes ของ AWS

สร้างคลัสเตอร์ด้วย `eksctl`:
```bash
# config file
cat > eks-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: mlzoomcamp-eks
  region: eu-west-1
nodeGroups:
  - name: ng-m5-xlarge
    instanceType: m5.xlarge
    desiredCapacity: 1
EOF

eksctl create cluster -f eks-config.yaml
```

Push images ไป ECR แล้ว apply deployment/service files เดิม

ลบเมื่อเลิกใช้: `eksctl delete cluster --name mlzoomcamp-eks`

---

## สรุปบทที่ 10

- **Kubernetes** = orchestrate container ใน production
- **TensorFlow Serving** = service ที่ optimized สำหรับ TF models (ใช้ gRPC)
- **Gateway pattern** = แยก preprocessing/HTTP จาก model serving
- **Docker Compose** ใช้ทดสอบหลาย container ก่อนขึ้น k8s
- **k8s objects**: Pod → Deployment → Service (ClusterIP / LoadBalancer)
- **Kind** สำหรับ local, **EKS** สำหรับ AWS
