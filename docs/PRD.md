**Product Requirement Document (PRD)**
*Full-stack MLOps repository for AI-driven CKD risk prediction*

---

### 1. Purpose & Scope

| Item                 | Description                                                                                                                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Business problem** | Provide nephrologists and CKD patients with reliable, up-to-date probabilities of (a) starting renal replacement therapy (RRT) and (b) all-cause mortality over 1- to 5-year horizons.                                 |
| **Outcome**          | A reproducible MLOps repository that ingests longitudinal electronic-health-record data, trains deep-learning survival models, deploys the best model to production, and continuously monitors real-world performance. |
| **Target users**     | • Nephrologists in Hong Kong public hospitals • CKD clinic nurses • Researchers developing new survival models • DevOps / MLOps engineers maintaining the pipeline                                                     |

---

### 2. Success Metrics

| Category     | KPI / Acceptance criterion                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Model**    | c-index ≥ 0.80 for mortality and ≥ 0.95 for RRT on hold-out set; integrated Brier score < 0.10                      |
| **Ops**      | ⩽ 1 h end-to-end training time per weekly retraining run; full provenance (code + data + params) captured in MLflow |
| **Clinical** | ≥ 90 % of prediction requests served < 200 ms (P95); dashboard adopted in ≥ 3 renal clinics within 6 months         |

---

### 3. Technical Stack & Environment

| Layer                          | Technology / Version                                                     |
| ------------------------------ | ------------------------------------------------------------------------ |
| Language                       | Python 3.11.8                                                            |
| Core DL                        | PyTorch 2.4.1                                                            |
| MLOps orchestration            | **ZenML 0.82.1** (pipelines, step caching, secrets)                      |
| Experiment tracking & registry | **MLflow 2.22.0**                                                        |
| Data validation                | Great Expectations + Pandera                                             |
| Workflow runner                | Docker + GitHub Actions (CI)                                             |
| Serving                        | ZenML **Model Deployer** → MLflow server (REST/gRPC)                     |
| Monitoring                     | Prometheus + Grafana; MLflow metrics + custom ZenML post-execution hooks |

---

### 4. System Architecture

```
CDARS ➜ Data Lake (Parquet on S3/minio)
         │
         ▼  (ZenML Data Engineering pipeline)
Clean Feature Store  ─▶ Split Registry  ─▶ Training Dataset
         │                                   │
         │                     (ZenML Model Engineering pipeline)
         ▼                                   ▼
   Great Expectations tests          PyTorch models (DeepSurv, DeepHit,
         │                           ensemble meta-learner) → MLflow artifacts
         ▼                                   │
Feature Service (Feast/duckdb) ◀─────────────┘
         │
         ▼  (ZenML Deployment pipeline)
 MLflow Model Server (prod)  ⟷  CKD-Risk REST API  ⟷  EMR / Web dashboard
         │
         ▼
Prometheus-Grafana monitoring  +  MLflow performance logging
```

---

### 5. Functional Requirements

#### 5.1 Data-Engineering Pipeline

| Step                         | Detail                                                                                                                     | ZenML implementation                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Ingestion**                | Pull pseudonymised CDARS data (demographic, labs, meds, ICD-10), 2009-2023                                                 | `@step` ➔ connector to SFTP / SQL           |
| **Exploration & validation** | Automatic schema profiling; GE expectations for ranges, nulls, units; Pandera type checks                                  | `expectations_step`                         |
| **Wrangling**                | Imputation (MICE), log/MinMax scaling, one-hot encoding; CKD-EPI eGFR & KDIGO A/G labels                                   | `transform_step` (re-uses `dataloader2.py`) |
| **Labeling**                 | Derive time-to-event targets:<br>• Event 1 = earliest of persistent eGFR < 10 or RRT start<br>• Event 2 = all-cause death  | `label_step`                                |
| **Splitting**                | Temporal split (≤ 31-12-2019 train, 2020-2023 temporal test) + external HKU-SZH split for geo-valid.                       | `split_step` with deterministic hash        |

Outputs are written to an immutable **ZenML Artifact Store** and registered in MLflow as **dataset versions**.

#### 5.2 ML-Model-Engineering Pipeline

| Phase          | Detail                                                                                                                                                               |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training**   | DeepSurv and DeepHit variants (ANN & LSTM) trained on sequences of 5 previous visits; hyper-params tuned with Optuna via Ray Tune; GPU-accelerated using A100 nodes. |
| **Evaluation** | Metrics: time-dependent c-index, integrated Brier, IPA, NLL; bootstrapped 95 % CIs.                                                                                  |
| **Testing**    | Temporal & geographical validation; competing-risk Fine-Gray comparison.                                                                                             |
| **Packaging**  | Best model ensemble (simple average or super-learner) exported to MLflow flavor (`pyfunc`), including requirements.txt and preprocessing code.                       |

ZenML service connector registers every run in MLflow Experiment **“ckd-risk”**; successful runs trigger the *Model Deployer* pipeline.

#### 5.3 Code-Engineering Requirements

* **Repository layout**

```
repo/
├── zenml.yaml              # stack & pipeline settings
├── pipelines/
│   ├── data_eng.py
│   ├── model_eng.py
│   └── deploy.py
├── steps/                  # atomic ZenML steps
├── src/
│   ├── ckdml/              # domain code (feature builders, models)
│   └── utils/
├── notebooks/              # EDA & ad-hoc analysis
├── tests/                  # pytest + coverage ≥ 80 %
└── .github/workflows/ci.yml
```

* **CI/CD**

  * Pre-commit (black, ruff, isort, mypy).
  * GitHub Actions: run unit tests, build Docker image, trigger ZenML server to execute **staging** pipeline, promote if tests pass.

* **Documentation** — MkDocs-Material + automated API docs from docstrings.

#### 5.4 Deployment & Monitoring Pipeline

| Component               | Requirement                                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Serving**             | MLflow Model Server behind FastAPI adapter; exposes `/predict` accepting JSON or Pandas records.                                                             |
| **Performance logging** | ZenML post-prediction hook pushes latency, c-index drift, volume counters to Prometheus; predictions + ground truth streamed to MLflow **evaluation store**. |
| **Monitoring**          | Grafana dashboard: latency histogram, weekly c-index, KS-stat for feature drift; alert to Slack when c-index drops > 3 %.                                    |
| **Retaining**           | Weekly retraining job; automatic canary deploy → baseline comparison → promote if KPI met.                                                                   |

---

### 6. Non-Functional Requirements

| Aspect            | Spec                                                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Security**      | Data encrypted at rest (S3 SSE-S3) and in transit (TLS 1.2); RBAC on ZenML & MLflow; audit logs retained 7 years to satisfy HA policy. |
| **Compliance**    | Local IRB approval and HA data-governance rules; no direct identifiers stored.                                                         |
| **Scalability**   | Pipelines must run on-prem GPU cluster (Slurm) *and* fallback to cloud (GCP) via ZenML stack profiles.                                 |
| **Observability** | End-to-end lineage: ZenML → MLflow → Docker image digest → Git SHA.                                                                    |

---

### 7. Milestones & Timeline

| Date      | Deliverable                                                |
| --------- | ---------------------------------------------------------- |
| **M + 1** | Repository skeleton, ZenML stack config, data-ingest PoC   |
| **M + 2** | Complete data-engineering pipeline; GE data-quality report |
| **M + 3** | Baseline DeepSurv/DeepHit models; automated evaluation     |
| **M + 4** | Ensemble + packaging; MLflow registry integration          |
| **M + 5** | Deployment pipeline, REST API, Grafana dashboard           |
| **M + 6** | Pilot rollout to 3 renal clinics; performance review       |

---

### 8. Risks & Mitigations

| Risk                       | Impact                     | Mitigation                                   |
| -------------------------- | -------------------------- | -------------------------------------------- |
| Data drift post-deployment | Incorrect risk predictions | Weekly monitoring & automatic retraining     |
| GPU cluster downtime       | Pipeline failure           | ZenML cloud stack profile fallback           |
| Regulatory changes         | Deployment delay           | Maintain compliance matrix & periodic review |

---

### 9. Glossary

| Term            | Meaning                                                                                  |
| --------------- | ---------------------------------------------------------------------------------------- |
| **RRT**         | Renal Replacement Therapy (dialysis or transplantation)                                  |
| **c-index**     | Concordance index, discrimination metric for survival models                             |
| **Brier score** | Calibration error measure for probabilistic forecasts                                    |
| **ZenML stack** | Re-usable bundle of orchestrator + artifact store + container registry + secrets manager |

---

### 10. Appendix – Example Commands

```bash
# 1️⃣  Set up stack (local docker example)
zenml stack register local_mlop_stack \
    -o default \
    -a default \
    -o orchestrator=local \
    -a artifact_store=local_artifacts \
    -m container_registry=local_registry \
    -x secrets_manager=local_secrets

# 2️⃣  Run data pipeline
zenml pipeline run data_eng.py

# 3️⃣  Train models
zenml pipeline run model_eng.py --config configs/train.yaml

# 4️⃣  Deploy best model
zenml pipeline run deploy.py --model_name ckd_risk_best
```

---

**Approval**
Product owner: *Ka Chun Leung*  Date: \_\_\_\_\_\_\_\_\_\_\_\_\_\_
