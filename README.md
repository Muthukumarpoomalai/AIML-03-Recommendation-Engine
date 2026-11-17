# AIML-03: Movie Recommendation Engine

## Project Overview
An end-to-end movie recommendation system built using Apache Spark, TensorFlow Recommenders, FAISS, Deep & Cross Network (DCN), NVIDIA Triton, Redis, FastAPI, and Apache Airflow.

## Architecture
1. MovieLens 25M dataset ingestion and preprocessing.
2. Chronological train, validation, and test split.
3. Apache Spark feature engineering.
4. TensorFlow Recommenders Two-Tower retrieval model.
5. FAISS nearest-neighbor movie retrieval.
6. DCN-based movie ranking.
7. NVIDIA Triton model inference serving.
8. Redis recommendation caching.
9. FastAPI recommendation endpoints.
10. Cold-start recommendations using popularity and genre similarity.
11. Apache Airflow DAG for scheduled retraining orchestration.

## Technology Stack
- Python
- Apache Spark / PySpark
- TensorFlow / TensorFlow Recommenders
- FAISS
- NVIDIA Triton Inference Server
- Redis
- FastAPI
- Apache Airflow
- MovieLens 25M

## Dataset
MovieLens 25M dataset.

Chronological split:
- Training: 19,988,226 interactions
- Validation: 2,505,893 interactions
- Testing: 2,505,976 interactions

## Model Evaluation

Warm-start offline evaluation on 95 users:

| Metric | FAISS Retrieval | DCN Final |
|---|---:|---:|
| Recall@10 | 0.019557 | 0.010428 |
| NDCG@10 | 0.026232 | 0.035931 |

Candidate Recall@1000: 0.186325

These are offline evaluation results, not live CTR or online A/B testing results.

## API Endpoints
- `GET /` – API information
- `GET /health` – Service health
- `GET /recommend/{user_id}` – Personalized or popularity-based recommendations
- `GET /cold-start/movie/{movie_id}` – Genre-based similar movie recommendations
- `GET /api-info` – API details

## Cold-Start Handling
- New users: popularity-based recommendations.
- New movies: genre-based similarity recommendations.
- Redis: caches recommendation results.

## Running the API
Start Redis and Triton services, then run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

API documentation: `http://localhost:8080/docs`

## Airflow
DAG: `movie_recommendation_retraining`

Tasks:
1. train_two_tower
2. rebuild_faiss_index
3. train_dcn

Note: DAG structure has been created and loaded. End-to-end scheduled task execution has not yet been validated.

## Limitations
- Evaluation is offline and uses a sampled user set.
- Live randomized A/B testing and CTR measurement are not yet completed.
- Airflow DAG execution requires a compatible training environment with the required ML dependencies.

## Future Improvements
- Live randomized A/B testing.
- Improved retrieval and ranking performance.
- Production-scale monitoring.
- Resource-optimized scheduled retraining.

## Author
Muthukumar
