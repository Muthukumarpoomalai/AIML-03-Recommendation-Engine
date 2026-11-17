
# Movie Recommendation Engine – Architecture

```mermaid
flowchart TD
    A["MovieLens 25M Dataset"] --> B["Data Cleaning and Chronological Split"]
    B --> C["Apache Spark Feature Engineering"]

    C --> D["TensorFlow Recommenders - Two Tower"]
    D --> E["Movie Embeddings"]
    E --> F["FAISS Vector Index"]

    G["User ID"] --> H["User Tower"]
    H --> I["User Embedding"]
    I --> F

    F --> J["Top 100 Movie Candidates"]
    J --> K["NVIDIA Triton Inference Server"]
    K --> L["DCN Ranking Model"]
    L --> M["Top 10 Recommendations"]

    M --> N["Redis Cache"]
    N --> O["FastAPI Recommendation API"]

    P["New User"] --> Q["Popularity Based Fallback"]
    R["New Movie"] --> S["Genre Based Similarity"]
    Q --> O
    S --> O

    T["Apache Airflow DAG"] -.-> D
    T -.-> F
    T -.-> L

    O --> U["Offline Evaluation - Recall@K and NDCG@K"]
```

## Architecture Components

1. **Dataset:** MovieLens 25M.
2. **Data Processing:** Cleaning, chronological splitting and Spark feature engineering.
3. **Retrieval:** TensorFlow Recommenders Two-Tower model.
4. **Candidate Search:** FAISS vector index.
5. **Ranking:** Deep & Cross Network (DCN).
6. **Model Serving:** NVIDIA Triton Inference Server.
7. **Caching:** Redis.
8. **API:** FastAPI.
9. **Cold Start:** Popularity-based and genre-based recommendations.
10. **Orchestration:** Apache Airflow DAG (created; execution not yet validated).
11. **Evaluation:** Recall@K and NDCG@K.

## Evaluation Results

- Candidate Recall@1000: 0.186325
- FAISS Recall@10: 0.019557
- FAISS NDCG@10: 0.026232
- DCN Recall@10: 0.010428
- DCN NDCG@10: 0.035931

*Results are from offline warm-start evaluation on 95 users.*
