# ML Predictive Maintenance System

An enterprise-grade solution engineered for high performance.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-VisionQuantech%20Custom-orange)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## 🚀 Overview

Welcome to the **ML Predictive Maintenance System** repository. This project is built to deliver a robust and scalable solution tailored to modern development standards.

The system exposes a machine-learning inference service over HTTP, designed to classify incoming sensor/equipment telemetry data and return a predicted class (e.g., fault type or maintenance category) along with a confidence score. It is packaged as a containerized FastAPI microservice, making it straightforward to deploy on any laptop, server, or cloud environment.

## ✨ Features

- **High Performance:** Optimized for speed and efficiency.
- **Scalable Architecture:** Designed to grow with your needs.
- **Clean Codebase:** Follows best practices and industry standards.
- **Secure by Default:** Engineered with security in mind.
- **REST Inference API:** A `/predict` endpoint that accepts JSON payloads and returns a predicted `class_id` and `confidence` score.
- **Health Check Endpoint:** A `/` endpoint reporting service status and model version (`v2.4.1`) for monitoring and load-balancer probes.
- **ML Stack Ready:** Dependencies include `numpy`, `pandas`, `scikit-learn`, and `torch` for real model integration.
- **Containerized:** Ships with a `Dockerfile` for one-command, reproducible deployment.

## 🏗️ Architecture / How It Works

The current codebase is a lightweight FastAPI microservice with the following flow:

```
Client Request (JSON)
        │
        ▼
┌─────────────────────────────┐
│   FastAPI App (main.py)     │
│  ┌───────────────────────┐  │
│  │ GET /                 │  │ ──► Health check: {"status": "operational", "model_version": "v2.4.1"}
│  ├───────────────────────┤  │
│  │ POST /predict         │  │ ──► Accepts arbitrary JSON dict
│  │  • Generate feature   │  │     vector (currently simulated:
│  │    vector (128-dim)   │  │     np.random.rand(128))
│  │  • argmax → class_id  │  │
│  │  • max → confidence   │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
        │
        ▼
Response: {"class_id": int, "confidence": float}
```

**Key components:**

| File | Purpose |
|---|---|
| `main.py` | FastAPI application with `/` (health) and `/predict` (inference) endpoints. |
| `requirements.txt` | Python dependencies: `fastapi`, `uvicorn`, `numpy`, `pandas`, `scikit-learn`, `torch`. |
| `Dockerfile` | Builds a `python:3.9-slim` image, installs dependencies, and serves the app via Uvicorn on port 8000. |
| `LICENSE` | VisionQuantech Custom Commercial License (see [License](#-license)). |

**Intended data flow (production):** sensor/telemetry data → preprocessing (pandas/scikit-learn) → trained model inference (torch/sklearn) → predicted maintenance class + confidence. Currently, the inference step is **simulated** with a random 128-dimensional vector (see [Workability Assessment](#-workability-assessment)).

## 🛠️ Prerequisites

Ensure you have the following installed in your environment before proceeding:
- **Python 3.9+** (if running locally), or
- **Docker** (recommended — no local Python setup required)
- Standard development tools (`git`)

## 📦 Installation

Follow standard installation steps for `Python` to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/ml-predictive-maintenance-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-predictive-maintenance-system
   ```
3. Install dependencies according to the standard `Python` ecosystem:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

Run the project using standard execution commands for `Python`. Ensure all environment variables and configurations are set prior to execution.

**Start the API server locally:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Health check:**
```bash
curl http://localhost:8000/
# {"status":"operational","model_version":"v2.4.1"}
```

**Run a prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sensor_1": 0.82, "vibration": 12.4, "temperature": 71.3}'
# {"class_id": 37, "confidence": 0.91}
```

Interactive API documentation is automatically available at `http://localhost:8000/docs` (Swagger UI).

## 🐳 Running with Docker

The repository includes a `Dockerfile`, so you can run the entire service on any laptop or server with Docker installed — no local Python environment needed.

**1. Build the image:**
```bash
docker build -t ml-predictive-maintenance .
```

**2. Run the container:**
```bash
docker run -d -p 8000:8000 --name pdm-service ml-predictive-maintenance
```

**3. Verify it is running:**
```bash
curl http://localhost:8000/
```

**4. Stop and clean up:**
```bash
docker stop pdm-service && docker rm pdm-service
```

**Optional — docker-compose:** A `docker-compose.yml` is not included in the repository, but the following minimal file works out of the box if you prefer `docker-compose up`:

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
```

Then simply run:
```bash
docker-compose up --build
```

## 🔍 Workability Assessment

In the interest of full transparency, here is an honest evaluation of the repository's current state:

**What works:**
- ✅ The FastAPI application runs correctly both locally and in Docker.
- ✅ Endpoints respond as documented; the health check is suitable for uptime probes.
- ✅ The Docker image builds cleanly and the service is deployable anywhere.
- ✅ The dependency list (`torch`, `scikit-learn`, `pandas`) anticipates real model integration.

**What is NOT production-ready:**
- ⚠️ **No trained model exists.** The `/predict` endpoint generates a *random* 128-dimensional vector (`np.random.rand(128)`) and returns its argmax — predictions are meaningless placeholders. A serialized model (e.g., `.pt`, `.pkl`) and loading logic must be added.
- ⚠️ **Input data is ignored.** The `data: dict` payload is accepted but never used in inference. There is no validation schema (e.g., Pydantic models), feature preprocessing, or pipeline.
- ⚠️ **No tests, CI/CD, or linting** configuration is present.
- ⚠️ **No persistent storage or telemetry ingestion** — despite the predictive-maintenance framing, there is no connection to sensor data sources, databases, or streaming systems.
- ⚠️ **Heavy dependencies for current code:** `torch` (~2GB) is installed in the Docker image but unused, inflating image size and build time.
- ⚠️ **Docker port exposure** relies on Uvicorn's default port 8000; adding an explicit `EXPOSE 8000` to the Dockerfile would improve clarity.

**Verdict:** This repository is a **solid architectural skeleton / proof-of-concept** for an ML inference microservice, but it is **not production-ready**. It requires, at minimum, a trained model artifact, input validation, preprocessing logic, and a test suite before it can deliver genuine predictive-maintenance value.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page. High-impact contributions include: integrating a real trained model, adding Pydantic request schemas, writing tests, and providing a `docker-compose.yml`.

## 📝 License

This project is **not** under a standard open-source license. It is governed by the **VisionQuantech Custom Commercial License** (Copyright © 2026 Shivay00001 / VisionQuantech):

- **Personal / educational / non-earning use:** Free.
- **Individual revenue-generating use:** Requires a **15–30% revenue share** of gross earnings derived from the Software.
- **Business / enterprise use:** **Prohibited** without a separate commercial license — contact **visionquantech@proton.me**.

The Software is provided "AS IS", without warranty of any kind. See the [LICENSE](LICENSE) file for full terms.