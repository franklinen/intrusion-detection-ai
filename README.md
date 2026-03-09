# intrusion-detection-ai
AI-Powered Network Intrusion Detection System using LSTM Autoencoder with Real-Time Streaming and Explainable AI


## Overview

This project implements a **deep learning–based network intrusion detection system (IDS)** using the **UNSW-NB15 cybersecurity dataset**. The system uses an **LSTM Autoencoder** to learn patterns of normal network traffic and detect anomalies that may indicate cyberattacks.

The project demonstrates an **end-to-end machine learning pipeline**, including data preprocessing, sequence modeling, anomaly detection, explainable AI, real-time streaming detection, and containerized deployment.

This project is designed as a **portfolio-level machine learning and cybersecurity engineering project**, showcasing skills in **deep learning, MLOps, streaming data systems, and model deployment**.

---

## Key Features

* **Deep Learning Intrusion Detection**

  * LSTM Autoencoder for unsupervised anomaly detection in network traffic.

* **Explainable AI**

  * SHAP analysis to interpret which features contribute to anomaly detection decisions.

* **Real-Time Detection**

  * Kafka-based streaming pipeline for processing live network traffic logs.

* **API Deployment**

  * FastAPI service for real-time anomaly detection via REST endpoints.

* **Containerization**

  * Dockerized deployment for portability and reproducibility.

---

## Project Architecture

```
Network Traffic Logs
        │
        ▼
Data Preprocessing
        │
        ▼
Sequence Builder
        │
        ▼
LSTM Autoencoder
        │
        ▼
Reconstruction Error
        │
        ├── Normal Traffic
        └── Anomaly (Potential Attack)
                │
                ▼
        SHAP Explainability
                │
                ▼
    Streaming Detection (Kafka)
                │
                ▼
        FastAPI Detection API
                │
                ▼
            Docker Deployment
```

---

## Project Structure

```
intrusion-detection-ai/
│
├── data/                # Dataset
├── models/              # Trained models and scalers
│
├── src/                 # Core ML pipeline
│   ├── preprocess.py
│   ├── sequence_builder.py
│   ├── train_autoencoder.py
│   └── detect_anomaly.py
│
├── explainability/      # SHAP model explainability
│   └── shap_explain.py
│
├── streaming/           # Kafka real-time pipeline
│   ├── kafka_producer.py
│   └── kafka_consumer.py
│
├── api/                 # FastAPI inference service
│   └── app.py
│
├── docker/              # Container configuration
│   └── Dockerfile
│
└── requirements.txt
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* SHAP (Explainable AI)
* Apache Kafka (Streaming pipeline)
* FastAPI (Model API service)
* Docker (Containerization)

---

## Dataset

The project uses the **UNSW-NB15 dataset**, a modern network intrusion dataset containing both normal and malicious network traffic.

The dataset includes features such as:

* Network protocol
* Connection duration
* Packet sizes
* Service types
* Traffic statistics

These features enable training models to detect anomalies in network activity.

---

## Running the Project

### Install dependencies

```
pip install -r requirements.txt
```

### Train the model

```
python src/train_autoencoder.py
```

### Run the detection API

```
uvicorn api.app:app --reload
```

API documentation will be available at:

```
http://localhost:8000/docs
```

---

## Docker Deployment

Build the container:

```
docker build -t intrusion-detection .
```

Run the container:

```
docker run -p 8000:8000 intrusion-detection
```

---

## Future Improvements

* Attention-based LSTM architecture
* Transformer-based intrusion detection models
* Real-time monitoring dashboards
* Kubernetes deployment for scalable production systems
* Online learning for adaptive threat detection

---

## Author

**Frankline Ononiwu**
Data Scientist | Machine Learning Engineer | AI Systems Builder

---

## License

This project is for educational and research purposes.

