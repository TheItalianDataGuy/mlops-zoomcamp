![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Progress-Week_5-blueviolet)
![MLflow Screenshot](images/mlflow-ui.png)
![Grafana Screenshot](images/grafana-dashboard.png)

# 🛠️ MLOps Zoomcamp – Personal Implementation

This repository contains my personal implementation of the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) by DataTalksClub. It includes weekly exercises, code labs, and project configurations that cover the complete machine learning operations (MLOps) lifecycle from experimentation to production monitoring.

---

## 🚀 Goals

- Gain hands-on experience with real-world MLOps tools and workflows
- Build production-ready ML pipelines
- Learn best practices for reproducibility, deployment, and monitoring

---

## 📆 Weekly Breakdown

### Week 1: Introduction & Environment
- Overview of the ML lifecycle
- Setting up virtual environments and project structure
- Baseline NYC Taxi trip duration model

### Week 2: Experiment Tracking
- MLflow for logging metrics and artifacts
- Creating and comparing multiple experiment runs
- Registering and storing best models

### Week 3: Workflow Orchestration
- Using Prefect 2.0 for DAG-based workflows
- Scheduling data ingestion and preprocessing pipelines
- Parameterization and environment separation

### Week 4: Model Deployment
- Serving models with Flask and Gunicorn
- Packaging model files with `joblib`
- Dockerizing the API and exposing it locally

### Week 5: Monitoring
- Logging metrics to PostgreSQL
- Real-time visualization with Grafana
- Using Evidently for data quality and drift detection
- Custom dashboard saved in `05-monitoring/dashboards/taxi_dashboard.json`

### ⏳ Week 6 & 7: Best Practices & Final Project
- CI/CD pipelines (GitHub Actions, pre-commit, black, pylint)
- Final project combining all concepts (WIP)

---

## 📁 Repo Structure

```bash
.
├── 01-intro                  # Environment setup and baseline model
├── 02-experiment-tracking   # MLflow experiments and tracking
├── 03-orchestration         # Prefect-based workflows
├── 04-deployment            # Flask API + Docker
├── 05-monitoring            # PostgreSQL + Grafana + Evidently
├── 06-best-practices        # CI/CD and testing (TBD)
├── 07-project               # Final pipeline and deployment (WIP)
```

---

## 🪨 Tech Stack

- **Python 3.11**
- **MLflow** – experiment tracking
- **Prefect 2.0** – orchestration
- **Docker & Docker Compose** – containerization
- **PostgreSQL** – metrics backend
- **Grafana** – real-time dashboard
- **Evidently** – drift and data quality reports
- **Flask** – model serving

---

## 👤 About Me

**Andrea / TheItalianDataGuy**  
Aspiring MLOps Engineer | MSc Data Science | Passionate about building robust ML systems

---

## 📅 Todo / Coming Soon

- [ ] Week 6: CI/CD and testing
- [ ] Week 7: Final deployment pipeline
- [ ] Add unit tests and coverage reports
- [ ] Improve README with architecture diagrams and GIFs

---

## 📅 Acknowledgements

Huge thanks to the [DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp) team for the amazing open-source MLOps Zoomcamp course and the vibrant Discord community!

---

## 📢 License

This project is licensed under the MIT License. Feel free to use and adapt!