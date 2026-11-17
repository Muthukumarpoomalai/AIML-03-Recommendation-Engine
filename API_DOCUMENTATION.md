# API Documentation – Movie Recommendation Engine

## 1. Overview

This project provides REST APIs for personalized movie recommendations, popularity-based cold-start recommendations, and genre-based similar movie recommendations.

**Base URL:** `http://localhost:8080`

**API Framework:** FastAPI

## 2. API Endpoints

### 2.1 Root Endpoint

- **Method:** GET
- **URL:** `/`
- **Purpose:** Returns API information.

### 2.2 Health Check

- **Method:** GET
- **URL:** `/health`
- **Purpose:** Checks API health and service connectivity, including Redis and Triton.

### 2.3 Personalized Movie Recommendations

- **Method:** GET
- **URL:** `/recommend/{user_id}`
- **Parameter:** `user_id` – User identifier.
- **Purpose:** Generates personalized movie recommendations using Two-Tower retrieval, FAISS and DCN ranking.
- **Cold-start behavior:** Unknown users receive popularity-based recommendations.

### 2.4 Similar Movie Recommendations

- **Method:** GET
- **URL:** `/cold-start/movie/{movie_id}`
- **Parameter:** `movie_id` – Movie identifier.
- **Purpose:** Generates similar movie recommendations using genre-based similarity.

### 2.5 API Information

- **Method:** GET
- **URL:** `/api-info`
- **Purpose:** Provides API details and available endpoint information.

## 3. Recommendation Workflow

1. Receive a user ID through the recommendation endpoint.
2. Generate user embeddings using the Two-Tower model.
3. Retrieve candidate movies using FAISS.
4. Rank candidate movies using the DCN model through NVIDIA Triton.
5. Return the top recommendations.
6. Cache recommendation results using Redis.

## 4. Cold-Start Strategy

**New Users:** Popularity-based movie recommendations.

**New Movies:** Genre-based similarity recommendations.

## 5. Technologies Used

- FastAPI – REST API development.
- TensorFlow Recommenders – Two-Tower retrieval.
- FAISS – Similarity search.
- DCN – Recommendation ranking.
- NVIDIA Triton – Model serving.
- Redis – Recommendation caching.

## 6. Running the API

Start the required Redis and Triton services, then execute:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 7. Interactive API Testing

Open the following URL after starting the API:

`http://localhost:8080/docs`

FastAPI Swagger UI provides interactive endpoint testing.

## 8. Evaluation

The recommendation system is evaluated using:

- Recall@K
- NDCG@K
- Candidate Recall@1000

The current reported results are offline warm-start evaluation results on 95 users. Live CTR and online A/B testing have not been validated.

## 9. Notes

- Redis and Triton must be available for the personalized recommendation pipeline.
- Model and dataset files must be available locally.
- API response schemas can be inspected through the interactive Swagger documentation.
