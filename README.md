# Cloud Native MLOps Platform for Text Classification

End-to-end text classification MlOps project.

# Project Guide

This guide provides a comprehensive overview of the MLOps text classification project, detailing its structure and the sequence of operations for data processing, model training, serving, and continuous integration/deployment. It also explains the rationale behind each technical decision and potential alternatives.

## 1. Project Structure

The project is organized efficiently to separate data, code, configuration, and artifacts. Below is a description of **every** file and directory in the project root.

| Directory/File | Description |
| :--- | :--- |
| **`.dvc/`** | Internal directory for DVC usage (stores config, cache settings, and local remote details). |
| **`.dvcignore`** | Specifies files/directories for DVC to ignore (similar to `.gitignore` but for data). |
| **`.env`** | Stores environment variables (e.g., API keys, secrets) for local development. **Never** commit this. |
| **`.git/`** | Internal directory for Git version control history. |
| **`.github/`** | Directory for GitHub-specific configurations. |
| &nbsp;&nbsp;`workflows/ci.yaml` | Defines the CI/CD pipeline (GitHub Actions) for automated testing and deployment. |
| **`.gitignore`** | Specifies files that Git should ignore (e.g., `__pycache__`, large data files, virtual envs). |
| **`Dockerfile`** | Instructions to build the Docker image for the Flask application. |
| **`LICENSE`** | Legal license file defining how the project code can be used. |
| **`Makefile`** | Utility script to run common tasks (e.g., `make data`, `make lint`) using simple commands. |
| **`README.md`** | Top-level documentation for developers, explaining the project goal and organization. |
| **`Grafana Dashboard.json`** | JSON export of the Grafana dashboard for monitoring model metrics. |
| **`data/`** | Stores datasets at various stages. Managed by DVC. |
| &nbsp;&nbsp;`raw/` | Original immutable data dump (downloaded via `data_ingestion.py`). |
| &nbsp;&nbsp;`interim/` | Transformed data (cleaned text) used for feature engineering. |
| &nbsp;&nbsp;`external/` | Data from third-party sources (empty or placeholder in this project). |
| &nbsp;&nbsp;`processed/` | Final canonical datasets (vectorized) ready for modeling. |
| **`deployment.yaml`** | Kubernetes manifest file defining the Deployment and Service for the EKS cluster. |
| **`docs/`** | Documentation files (Sphinx project) for generating project manuals. |
| **`dvc.lock`** | Auto-generated file by DVC that captures the exact state (hash) of the pipeline and data. |
| **`dvc.yaml`** | Defines the data pipeline stages (`data_ingestion`, `model_building`, etc.) and dependencies. |
| **`flask_app/`** | Contains the code for the web application. |
| &nbsp;&nbsp;`app.py` | The main entry point for the Flask web server. |
| &nbsp;&nbsp;`load_model_test.py` | Script to test loading the model locally. |
| &nbsp;&nbsp;`preprocessing_utility.py` | Helper functions for cleaning text during serving. |
| &nbsp;&nbsp;`templates/` | HTML templates for the web UI (`index.html`). |
| &nbsp;&nbsp;`requirements.txt` | Python dependencies specific to the Flask app container. |
| **`local_S3/`** | Directory used to simulate S3 storage locally (likely for testing DVC remotes). |
| **`logs/`** | Stores log files generated during pipeline execution. |
| **`mlruns/`** | Local directory for MLflow runs (metrics, params, artifacts) if not using a remote server. |
| **`models/`** | Directory for trained models and serialization artifacts. |
| &nbsp;&nbsp;`model.pkl` | The trained Logistic Regression model. |
| &nbsp;&nbsp;`vectorizer.pkl` | The fitted CountVectorizer object. |
| **`notebooks/`** | Jupyter notebooks for experimentation and exploration (e.g., `1.0-jqp-initial-data-exploration.ipynb`). |
| **`params.yaml`** | Central configuration file for pipeline hyperparameters (e.g., `max_features`, `test_size`). |
| **`references/`** | Explanatory materials, data dictionaries, or manuals. |
| **`reports/`** | Generated analysis and metrics. |
| &nbsp;&nbsp;`metrics.json` | JSON file containing model performance metrics(Accuracy, Precision, etc.). |
| &nbsp;&nbsp;`experiment_info.json` | Stores `run_id` and model path for handoff between pipeline steps. |
| &nbsp;&nbsp;`figures/` | Generated graphics/plots used in reporting. |
| **`requirements.txt`** | List of Python packages required to run the project (for `pip install`). |
| **`scripts/`** | Standalone helper scripts used in the CI/CD pipeline. |
| &nbsp;&nbsp;`promote_model.py` | Script to promote a model from Staging to Production in MLflow Registry. |
| **`setup.py`** | Script to make the project pip-installable so `src` can be imported as a module. |
| **`src/`** | Source code for use in this project. |
| &nbsp;&nbsp;`__init__.py` | Makes `src` a Python module. |
| &nbsp;&nbsp;`logger/` | Logging configuration utility. |
| &nbsp;&nbsp;`connections/` | Connection helpers (`s3_connection.py` for S3). |
| &nbsp;&nbsp;`data/` | Scripts to download or generate data (`data_ingestion.py`, `data_preprocessing.py`). |
| &nbsp;&nbsp;`features/` | Scripts to turn raw data into features (`feature_engineering.py`). |
| &nbsp;&nbsp;`model/` | Scripts to train and evaluate models (`model_building.py`, `model_evaluation.py`). |
| &nbsp;&nbsp;`visualization/` | Scripts to create visualizations (`visualize.py`). |
| **`test_environment.py`** | Script to check if the Python environment is set up correctly (Python version). |
| **`tests/`** | Directory containing unit tests. |
| &nbsp;&nbsp;`test_model.py` | Tests for model logic. |
| &nbsp;&nbsp;`test_flask_app.py` | Tests for the web application endpoints. |
| **`tox.ini`** | Configuration file for `tox` (or `flake8`) to automate testing and linting. |

---

## 2. Sequence of Execution (Data Pipeline)

The project uses **DVC (Data Version Control)** to manage the local manufacturing pipeline. The sequence of events is defined in `dvc.yaml`.

### Step 1: Data Ingestion
- **Script**: `src/data/data_ingestion.py`
- **Action**: Downloads raw data (CSV), performs initial cleaning, and splits into train/test sets.
- **Why**: To bring external data into a controlled, versioned local environment. Splitting early ensures no data leakage during testing.
- **Alternatives**:
    -   *Manual Download*: Hard to reproduce and automate.
    -   *Direct Database Connection*: Harder to version control datasets specific to model training runs.

### Step 2: Data Preprocessing
- **Script**: `src/data/data_preprocessing.py`
- **Action**: Cleans text (removes URLs, numbers, punctuation, stopwords) and lemmatizes it.
- **Why**: Raw text contains noise that degrades model performance. Lemmatization reduces word forms to a common base (e.g., "running" -> "run"), reducing dimensionality.
- **Alternatives**:
    -   *Stemming*: Faster but less accurate (cuts off ends of words) than lemmatization.
    -   *Spacy*: A more modern NLP library than NLTK, often faster but larger dependency.

### Step 3: Feature Engineering
- **Script**: `src/features/feature_engineering.py`
- **Action**: Applies **TF-IDF Vectorizer** to convert text to numerical vectors.
- **Why**: Machine learning models require numerical input. TF-IDF weighs words by importance, helping to filter out common but less meaningful terms.
- **Alternatives**:
    -   *Bag of Words (CountVectorizer)*: Simple frequency-based approach, but ignores word importance.
    -   *Word Embeddings (Word2Vec, GloVe)*: Captures semantic meaning but requires more data/computation.
    -   *Transformer Embeddings (BERT)*: State-of-the-art context awareness but computationally expensive.

### Step 4: Model Building
- **Script**: `src/model/model_building.py`
- **Action**: Trains a **Logistic Regression** model.
- **Why**: Logistic Regression is interpretable, fast to train, and works well as a baseline for binary classification (Positive/Negative).
- **Alternatives**:
    -   *Naive Bayes*: Very fast, effective for text, assumes feature independence.
    -   *SVM (Support Vector Machine)*: Good for high-dimensional spaces like text, but slower.
    -   *Deep Learning (LSTM/Transformers)*: Better for complex patterns but overkill for simple sentiment tasks.

### Step 5: Model Evaluation
- **Script**: `src/model/model_evaluation.py`
- **Action**: Evaluates the model and logs metrics/artifacts to **MLflow/Dagshub**.
- **Why**: To quantitatively assess performance and track experiments over time. MLflow provides a centralized dashboard to compare runs.
- **Alternatives**:
    -   *Local JSON logs*: Hard to visualize and compare multiple runs.
    -   *Weights & Biases (W&B)*: A popular alternative to MLflow with rich visualization features.
    -   *TensorBoard*: Good for deep learning, less suited for general experiment tracking.

### Step 6: Model Registration
- **Script**: `src/model/register_model.py`
- **Action**: Registers the trained model in MLflow Registry and tags it as "Staging".
- **Why**: decoupling the *storage* of an artifact from its *lifecycle state*. It creates a clear hand-off point for CI/CD.
- **Alternatives**:
    -   *File naming conventions*: e.g., `model_v1_final.pkl`. Error-prone and messy.
    -   *Git LFS*: Versions the file but doesn't manage "Staging/Production" lifecycle metadata easily.

---

## 3. CI/CD Pipeline (GitHub Actions)

The `ci.yaml` file defines the automation pipeline that triggers on every `push`.

### Phase 1: Pipeline Execution & Testing
- **Action**: Runs `dvc repro` and unit tests (`python -m unittest`).
- **Why**: Ensures that changes to code or data don't break the pipeline and that the model can be successfully reproduced.
- **Alternatives**:
    -   *Jenkins / GitLab CI*: Other popular CI tools. GitHub Actions is chosen for its tight integration with the repository.

### Phase 2: Model Promotion
- **Action**: Moves the model from "Staging" to "Production" in MLflow.
- **Why**: Automated promotion after successful tests removes human bottleneck and ensures only validated models reach production.
- **Alternatives**:
    -   *Manual Approval*: A human clicks a button to promote. Safer but slower.
    -   *Canary Deployment*: Releasing to a small % of users first (more complex infrastructure).

### Phase 3: Deployment (AWS ECR & EKS)

#### Step A: Build & Push to AWS ECR
- **Action**: Docker image is built and pushed to Elastic Container Registry (ECR).
- **Why**:
    -   **Docker**: Packages code + dependencies together, ensuring it runs the same on your laptop as it does in the cloud.
    -   **ECR**: A secure, private place to store Docker images. It integrates natively with AWS IAM for security.
- **Alternatives**:
    -   *Docker Hub*: Public/Private registry. Less integrated with AWS permissions.
    -   *JFrog Artifactory*: Enterprise artifact management.
    -   *GitHub Container Registry (GHCR)*: Good if staying within GitHub ecosystem.

#### Step B: Deploy to EKS (Kubernetes)
- **Action**: Updates the Kubernetes cluster to use the new Docker image.
- **Why**: **Kubernetes (EKS)** handles scaling (running multiple copies), self-healing (restarting crashed containers), and zero-downtime updates (Rolling Updates).
- **Alternatives**:
    -   *AWS Lambda*: Serverless, cheaper for low traffic, but "cold starts" can rely latency.
    -   *AWS ECS*: Simpler container orchestration than Kubernetes, good for smaller teams.
    -   *EC2 (Virtual Machine)*: You manage the OS and server manually. Harder to scale and maintain.
    -   *App Runner / Heroku*: PaaS solutions that abstract away the infrastructure completely. Easiest, but less control.

---

## 4. Model Serving (Flask App)

Once deployed, the `flask_app` serves the "Production" model.

### Sequence of Serving:
1.  **Startup**: Fetches "Production" model from MLflow.
2.  **Prediction**: Receives text -> Preprocesses -> Vectorizes -> Predicts.
3.  **Why Flask?**: It's a lightweight, simple web framework that is easy to set up for microservices.
    -   *Alternatives*:
        -   **FastAPI**: Modern, faster (async), auto-generates documentation (Swagger). Often preferred over Flask for new projects.
        -   **TensorFlow Serving / TorchServe**: Specialized high-performance servers for TF/PyTorch models (complex to set up).
        -   **BentoML**: Framework specifically for packaging and serving ML models.

---

## 5. Monitoring (Prometheus & Grafana)

The project includes built-in observability to track model performance and system health in production.

### Workflow:
1.  **Metrics Exposure (Flask App)**:
    -   The `flask_app/app.py` uses the `prometheus_client` library to define and expose custom metrics.
    -   **Endpoint**: `/metrics` (Standard Prometheus scraping endpoint).
    -   **Key Metrics**:
        -   `app_request_count`: Total number of requests (broken down by method/endpoint).
        -   `app_request_latency_seconds`: Histogram of response times.
        -   `app_preprocessing_latency_seconds`: Latency of preprocessing steps (e.g., text cleaning, vectorization).
        -   `app_inference_latency_seconds`: Latency of the model inference logic itself.
        -   `app_input_length_words`: Histogram of input text length (number of words) to monitor data drift.
        -   `model_prediction_count`: Tracks the distribution of predictions (e.g., how many "Positive" vs "Negative").
2.  **Metric Scraping (Prometheus)**: 
    -   A Prometheus server (deployed separately in the cluster) scrapes the `/metrics` endpoint of the Flask Pods at regular intervals.
3.  **Visualization (Grafana)**:
    -   Grafana queries the Prometheus server to display dashboards.
    -   **Use Case**: Visualize real-time traffic, latency spikes, or data drift (e.g., sudden increase in "Negative" predictions).

### Why Prometheus & Grafana?
-   **Standardization**: They are the de facto standard for Kubernetes monitoring.
-   **Pull Model**: Prometheus "pulls" metrics, meaning your app doesn't crash if the monitoring server goes down.
-   **Alerting**: Easy to set up alerts (e.g., "Latency > 500ms for 5 minutes").

### Alternatives:
-   **ELK Stack (Elasticsearch, Logstash, Kibana)**: Better for *logs* (text search) but heavier for *metrics* (time-series).
-   **Datadog / New Relic**: Full-featured commercial SaaS platforms. Easier to set up but can be expensive.
-   **CloudWatch**: AWS-native monitoring. Good integration with AWS services but less flexible for custom app metrics than Prometheus.

--------

> Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).#cookiecutterdatascience
