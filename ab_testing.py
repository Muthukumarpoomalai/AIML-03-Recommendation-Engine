
import json
import random
from collections import Counter

# Load actual offline evaluation results
with open("evaluation_results.json", "r") as file:
    results = json.load(file)

# Two recommendation strategies
experiments = {
    "A": {
        "name": "FAISS Retrieval",
        "recall_at_10": results["faiss_retrieval_recall_at_10"],
        "ndcg_at_10": results["faiss_retrieval_ndcg_at_10"]
    },
    "B": {
        "name": "DCN Reranking",
        "recall_at_10": results["dcn_final_recall_at_10"],
        "ndcg_at_10": results["dcn_final_ndcg_at_10"]
    }
}

print("\n========== A/B TESTING REPORT ==========")

print(f"Evaluation type : {results['evaluation_type']}")
print(f"Users evaluated : {results['users_evaluated']}")

for group, metrics in experiments.items():
    print(f"\nGroup {group}: {metrics['name']}")
    print(f"Recall@10 : {metrics['recall_at_10']:.6f}")
    print(f"NDCG@10   : {metrics['ndcg_at_10']:.6f}")

# Compare offline ranking quality
winner = max(
    experiments,
    key=lambda group: experiments[group]["ndcg_at_10"]
)

print("\n========== COMPARISON ==========")
print(f"Better NDCG@10: Group {winner}")
print(f"Strategy: {experiments[winner]['name']}")

print("\nNote:")
print("This is an offline model comparison,")
print("not a live randomized A/B experiment.")
print("CTR and conversion require real user interaction data.")
