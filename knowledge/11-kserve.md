# บทที่ 11: KServe
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> Deploy โมเดลด้วย KServe (เดิมชื่อ KFServing)

---

## 11.1 Overview

**KServe** = platform deploy ML models บน Kubernetes
- ลด boilerplate code (ไม่ต้องเขียน Flask app)
- รองรับหลาย framework: scikit-learn, TensorFlow, PyTorch, XGBoost
- มี **autoscaling**, **canary deployment**, **GPU support**

**เทียบกับวิธีบท 10**:
| บท 10 (Manual) | บท 11 (KServe) |
|---|---|
| เขียน Flask gateway เอง | KServe จัดให้ |
| Deployment + Service ทำเอง | InferenceService 1 ไฟล์ |
| ต้องเขียน preprocessing | Transformer แยก component |

---

## 11.2 Running KServe Locally

KServe ต้องการ Kubernetes cluster

ใช้ **Kind** local + Quickstart script:
```bash
kind create cluster
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.9/hack/quick_install.sh" | bash
```

ตรวจสอบ:
```bash
kubectl get pods -n kserve
```

---

## 11.3 Deploying Scikit-Learn Model

**ตัวอย่างง่ายสุด** — Iris classifier

เทรน + save:
```python
from sklearn import svm, datasets
from joblib import dump

iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

dump(clf, 'model.joblib')
```

อัปโหลด `model.joblib` ขึ้นที่เก็บ (Google Cloud Storage, S3, หรือ HTTP server)

`sklearn-iris.yaml`:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```

Apply:
```bash
kubectl apply -f sklearn-iris.yaml
kubectl get inferenceservices
```

เรียกใช้:
```bash
SERVICE_HOSTNAME=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" \
  http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/sklearn-iris:predict \
  -d @./input.json
```

`input.json`:
```json
{"instances": [[6.8, 2.8, 4.8, 1.4], [6.0, 3.4, 4.5, 1.6]]}
```

---

## 11.4 Custom Image for KServe

ถ้า model ใช้ library/version ที่ KServe ไม่รองรับ → ใช้ **custom image**

```dockerfile
FROM python:3.8.12-slim

RUN pip install kserve scikit-learn==1.0.2

COPY model.joblib /mnt/models/model.joblib
COPY serve.py .

CMD ["python", "serve.py"]
```

`serve.py`:
```python
import kserve
from joblib import load

class IrisModel(kserve.Model):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.model = None
        self.load()
    
    def load(self):
        self.model = load('/mnt/models/model.joblib')
    
    def predict(self, payload, headers=None):
        instances = payload['instances']
        result = self.model.predict(instances).tolist()
        return {'predictions': result}

if __name__ == '__main__':
    model = IrisModel('iris-model')
    kserve.ModelServer().start([model])
```

InferenceService สำหรับ custom image:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: custom-iris
spec:
  predictor:
    containers:
    - name: kserve-container
      image: your-registry/custom-iris:v1
```

---

## 11.5 Serving TensorFlow Models with KServe

KServe มี built-in serving สำหรับ TensorFlow:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: clothing-model
spec:
  predictor:
    tensorflow:
      storageUri: "gs://your-bucket/clothing-model/"
```

อัปโหลด `saved_model` (จากบท 10) ไป GCS หรือ S3

---

## 11.6 KServe Transformers

**Transformer** = component แยกสำหรับ preprocessing/postprocessing
- รัน pod แยกจาก predictor
- เหมาะกับ image preprocessing, text tokenization

ตัวอย่าง: รับ URL → download → preprocess → ส่ง pixel ให้ TF predictor

```python
import kserve
from typing import Dict
import logging
from keras_image_helper import create_preprocessor

class ImageTransformer(kserve.Model):
    def __init__(self, name, predictor_host):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.preprocessor = create_preprocessor('xception', target_size=(299, 299))
    
    def preprocess(self, inputs: Dict) -> Dict:
        url = inputs['instances'][0]['url']
        X = self.preprocessor.from_url(url)
        return {'instances': X.tolist()}
    
    def postprocess(self, response: Dict) -> Dict:
        classes = ['dress', 'hat', 'longsleeve', 'outwear',
                   'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
        preds = response['predictions'][0]
        result = dict(zip(classes, preds))
        return result
```

InferenceService รวม transformer + predictor:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: clothing-model
spec:
  transformer:
    containers:
    - image: your-registry/image-transformer:v1
      name: kserve-container
  predictor:
    tensorflow:
      storageUri: "gs://your-bucket/clothing-model/"
```

**Flow**: Client → Transformer (preprocess) → Predictor → Transformer (postprocess) → Client

---

## 11.7 Deploying to EKS

ติดตั้ง KServe บน EKS (production):
1. สร้าง EKS cluster ด้วย `eksctl`
2. ติดตั้ง dependencies: Istio, Cert-Manager, Knative
3. ติดตั้ง KServe
4. ตั้งค่า S3/ECR สำหรับ models และ images
5. Apply InferenceService YAML

ลบเมื่อเลิกใช้:
```bash
eksctl delete cluster --name <cluster-name>
```

---

## สรุปบทที่ 11

- **KServe** = ลด boilerplate ของการ deploy ML บน k8s
- **InferenceService** = 1 YAML แทน Deployment + Service หลายไฟล์
- รองรับ **built-in frameworks** (sklearn, tensorflow, xgboost, pytorch) → แค่ชี้ไปยัง `storageUri`
- ถ้าต้อง logic พิเศษ ใช้ **custom image** หรือ **transformer**
- รองรับ canary, autoscaling, GPU
- เปรียบเทียบบท 9 (serverless) vs บท 10 (manual k8s) vs บท 11 (KServe) — แต่ละแบบมีข้อดีต่างกัน
