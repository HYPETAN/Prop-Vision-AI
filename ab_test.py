import numpy as np
from scipy import stats

print("--- STARTING OFFLINE A/B TEST SIMULATION ---")
print("Comparing Model A (Keyword Search) vs Model B (CLIP Multi-Modal Search)\n")

# 1. SETUP THE SIMULATION METRICS
# We evaluate using Mean Reciprocal Rank (MRR). 
# A score of 1.0 means the model found the correct image on the 1st try.
# A score of 0.5 means it was 2nd, 0.33 means 3rd. 0.0 means it failed completely.

# Let's simulate running 50 search queries on both models.
num_test_queries = 50

# Model A (Keyword Search): Often fails on synonyms (e.g., searching "couch" when the caption says "sofa")
np.random.seed(42) # For reproducible results
mrr_model_a = np.random.choice([1.0, 0.5, 0.33, 0.0], size=num_test_queries, p=[0.2, 0.2, 0.2, 0.4])

# Model B (Your CLIP Model): Understands context better, so it finds the right image faster on average.
mrr_model_b = np.random.choice([1.0, 0.5, 0.33, 0.0], size=num_test_queries, p=[0.45, 0.3, 0.15, 0.1])

print(f"Executed {num_test_queries} test queries on both models...")
print(f"Model A (Keyword) Average MRR:  {mrr_model_a.mean():.3f}")
print(f"Model B (CLIP) Average MRR:     {mrr_model_b.mean():.3f}\n")

# 2. THE STATISTICAL TEST
# We use a Paired T-Test because we ran the exact same 50 queries through both models.
print("--- CALCULATING STATISTICAL SIGNIFICANCE ---")
t_statistic, p_value = stats.ttest_rel(mrr_model_b, mrr_model_a)

print(f"T-Statistic: {t_statistic:.3f}")
print(f"P-Value:     {p_value:.5f}")

# 3. THE BUSINESS OUTCOME
if p_value < 0.05:
    print("\n[RESULT]: SUCCESS!")
    print("The improvement is STATISTICALLY SIGNIFICANT (p < 0.05).")
    print("Model B outperforms Model A")
else:
    print("\n[RESULT]: FAILED.")