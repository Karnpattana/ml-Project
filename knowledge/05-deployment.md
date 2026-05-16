# บทที่ 5: Deploying Machine Learning Models
> ที่มา: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
> เอาโมเดล churn จากบท 3 ไป deploy ใช้งานจริง

---

## 5.1 Intro to Deployment

จากโมเดลใน Jupyter Notebook → **Web Service** ที่คนอื่นเรียกใช้ได้

Stack ที่ใช้:
1. **Pickle** — บันทึกโมเดล
2. **Flask** — Python web framework
3. **Pipenv** — จัดการ Python dependencies
4. **Docker** — แพ็คทุกอย่างใส่ container
5. **AWS Elastic Beanstalk** — deploy ขึ้น cloud

---

## 5.2 Saving and Loading the Model (Pickle)

**Pickle** — serialize Python objects ลงไฟล์

```python
import pickle

# บันทึก
output_file = 'model_C=1.0.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

# โหลด
with open('model_C=1.0.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
```

⚠️ ตอนโหลดต้องมี library เดียวกัน (sklearn version ฯลฯ)

---

## 5.3 Intro to Flask

**Flask** = Python framework สร้าง web service ง่ายๆ

ตัวอย่างเล็กที่สุด `ping.py`:
```python
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'PONG'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

เรียกใช้: `curl http://localhost:9696/ping` → `PONG`

---

## 5.4 Deploying Model with Flask

`predict.py`:
```python
import pickle
from flask import Flask, request, jsonify

with open('model_C=1.0.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

เรียกใช้:
```python
import requests
url = 'http://localhost:9696/predict'
customer = {'gender': 'female', 'tenure': 1, ...}
response = requests.post(url, json=customer).json()
```

**Production**: ใช้ **gunicorn (WSGI server)** แทน Flask dev server
```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```

(Windows ใช้ **waitress** แทน gunicorn)

---

## 5.5 Pipenv (Virtual Environment)

**Pipenv** = จัดการ dependencies แบบ reproducible

ติดตั้ง:
```bash
pip install pipenv
```

ใช้งาน:
```bash
pipenv install numpy scikit-learn flask gunicorn

# จะสร้าง Pipfile + Pipfile.lock
# Pipfile.lock = บันทึก version ที่ใช้แน่นอน

pipenv shell        # เข้า virtual env
pipenv run python predict.py
```

ตอน deploy ที่อื่น:
```bash
pipenv install     # ติดตั้งจาก Pipfile.lock อย่างเดียว
```

---

## 5.6 Docker

**Docker** = container ที่รวมแอป + dependencies + OS environment

`Dockerfile`:
```dockerfile
FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
```

**Build & Run:**
```bash
docker build -t churn-prediction .
docker run -it --rm -p 9696:9696 churn-prediction
```

- `-p 9696:9696` = map port host:container
- `--rm` = ลบ container เมื่อหยุด

---

## 5.7 AWS Elastic Beanstalk

**Elastic Beanstalk (EB)** = AWS service ที่ deploy app อัตโนมัติ (จัดการ EC2, load balancer ให้)

ติดตั้ง CLI:
```bash
pipenv install awsebcli --dev
```

Workflow:
```bash
# 1. Init project
eb init -p docker -r us-east-1 churn-serving

# 2. ทดสอบ local
eb local run --port 9696

# 3. สร้าง environment (deploy)
eb create churn-serving-env

# จะได้ URL: churn-serving-env.xxxxx.elasticbeanstalk.com
```

จัดการ:
```bash
eb terminate churn-serving-env   # ลบเมื่อไม่ใช้
```

⚠️ EB เปิดต่อโลก → ใน production ควรปิดด้วย security group/auth

---

## สรุปบทที่ 5

Pipeline การ deploy:
1. **Train + Save** → pickle file
2. **Web Service** → Flask app + gunicorn
3. **Dependencies** → Pipenv (Pipfile.lock)
4. **Containerize** → Docker
5. **Cloud Deploy** → AWS Elastic Beanstalk

แต่ละชั้นเพิ่ม **reproducibility** และ **portability**
