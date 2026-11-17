
import redis
import json

# =====================================
# 1. CONNECT TO REDIS
# =====================================

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# Test connection
try:
    print("Redis connection:", r.ping())
except redis.ConnectionError:
    print("Redis server is not running!")
    raise

# =====================================
# 2. CACHE RECOMMENDATIONS
# =====================================

def cache_recommendations(user_id, recommendations):

    key = f"recommendations:user:{user_id}"

    r.setex(
        key,
        3600,
        json.dumps(recommendations)
    )

    print("Recommendations cached successfully!")


# =====================================
# 3. GET CACHED RECOMMENDATIONS
# =====================================

def get_cached_recommendations(user_id):

    key = f"recommendations:user:{user_id}"

    cached_data = r.get(key)

    if cached_data:
        print("Cache HIT!")
        return json.loads(cached_data)

    print("Cache MISS!")
    return None


# =====================================
# 4. TEST CACHE
# =====================================

sample_recommendations = [
    "Forrest Gump (1994)",
    "Pulp Fiction (1994)",
    "The Matrix (1999)"
]

cache_recommendations("2", sample_recommendations)

result = get_cached_recommendations("2")

print("\nCached recommendations:")
print(result)

print("\nREDIS CACHE TEST COMPLETED!")
