
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from recommendation_pipeline import recommend, cache, triton

from cold_start import (
    is_known_user,
    recommend_popular_movies,
    recommend_similar_movies
)


app = FastAPI(
    title="AI Movie Recommendation Engine",
    description=(
        "Two-Tower retrieval, FAISS, DCN ranking, "
        "Triton serving, Redis caching and cold-start fallback."
    ),
    version="1.0.0"
)


# =====================================
# RESPONSE MODELS
# =====================================

class MovieRecommendation(BaseModel):
    movie_id: str
    title: str
    genres: str
    predicted_rating: Optional[float] = None
    retrieval_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: str
    recommendation_type: str
    total_recommendations: int
    recommendations: List[MovieRecommendation]


# =====================================
# HOME
# =====================================

@app.get("/")
def home():
    return {
        "message": "AI Movie Recommendation Engine API",
        "status": "running",
        "docs": "/docs"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health_check():

    redis_status = False
    triton_status = False

    try:
        redis_status = bool(cache.ping())
    except Exception:
        pass

    try:
        triton_status = (
            triton.is_server_live()
            and triton.is_server_ready()
            and triton.is_model_ready("dcn_ranking")
        )
    except Exception:
        pass

    overall_status = redis_status and triton_status

    return {
        "status": "healthy" if overall_status else "degraded",
        "redis": "connected" if redis_status else "disconnected",
        "triton": "ready" if triton_status else "unavailable"
    }


# =====================================
# USER RECOMMENDATIONS
# =====================================

@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse
)
def get_recommendations(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=20)
):

    try:

        # Cold-start: New user
        if not is_known_user(user_id):

            recommendations = recommend_popular_movies(
                top_n=top_k
            )

            recommendation_type = "popularity_based_cold_start"

        # Existing user: Two-Tower + FAISS + DCN
        else:

            recommendations = recommend(
                user_id=user_id,
                top_k=top_k
            )

            recommendation_type = "personalized"

        return {
            "user_id": user_id,
            "recommendation_type": recommendation_type,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(error)}"
        )


# =====================================
# NEW MOVIE COLD-START
# =====================================

@app.get(
    "/cold-start/movie/{movie_id}",
    response_model=List[MovieRecommendation]
)
def get_similar_movies(
    movie_id: str,
    top_k: int = Query(default=10, ge=1, le=20)
):

    try:

        return recommend_similar_movies(
            new_movie_id=movie_id,
            top_n=top_k
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Cold-start recommendation failed: {str(error)}"
        )


# =====================================
# API INFORMATION
# =====================================

@app.get("/api-info")
def api_info():

    return {
        "retrieval": "Two-Tower + FAISS",
        "ranking": "DCN",
        "model_serving": "NVIDIA Triton",
        "caching": "Redis",
        "cold_start": [
            "Popularity-based recommendations for new users",
            "Genre-based recommendations for new movies"
        ],
        "default_recommendations": 10
    }
