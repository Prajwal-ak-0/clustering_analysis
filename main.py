import pandas as pd
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

CLUSTER_CSV = "cluster.csv"
SOURCE_CSV = "source.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
CLUSTER_SIZE = 16
NUM_ITERATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05
OUTPUT_SCORES_CSV = "random_cluster_scores.csv"
SUMMARY_STATS_FILE = "summary_stats.txt"
DISTRIBUTION_PLOT = "similarity_distribution.png"

print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Reading cluster.csv...")
cluster_df = pd.read_csv(CLUSTER_CSV)
code_snippets = cluster_df["code_snippet"].tolist()
embeddings = model.encode(code_snippets, convert_to_tensor=True)

score_sum = 0.0
pair_count = 0
for idx1, idx2 in itertools.combinations(range(len(code_snippets)), 2):
    score_sum += util.cos_sim(embeddings[idx1], embeddings[idx2]).item() * 100
    pair_count += 1
cluster_avg = score_sum / pair_count
print(f"Cluster.csv average similarity score: {cluster_avg:.2f}")

print("Reading source.csv...")
source_df = pd.read_csv(SOURCE_CSV)
random_cluster_avgs = []

for i in range(NUM_ITERATIONS):
    sample_df = source_df.sample(n=CLUSTER_SIZE)
    code_snippets = sample_df["code_snippet"].tolist()
    embeddings = model.encode(code_snippets, convert_to_tensor=True)
    score_sum = 0.0
    pair_count = 0
    for idx1, idx2 in itertools.combinations(range(len(code_snippets)), 2):
        score_sum += util.cos_sim(embeddings[idx1], embeddings[idx2]).item() * 100
        pair_count += 1
    random_cluster_avgs.append(score_sum / pair_count)
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} iterations out of {NUM_ITERATIONS}")

random_cluster_avgs = np.array(random_cluster_avgs)
overall_mean = np.mean(random_cluster_avgs)
overall_std = np.std(random_cluster_avgs)
median_score = np.median(random_cluster_avgs)

plt.figure(figsize=(10, 6))
sns.histplot(random_cluster_avgs, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.axvline(overall_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {overall_mean:.2f}')
plt.axvline(median_score, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_score:.2f}')
plt.title("Distribution of Average Similarity Scores (Random Clusters)")
plt.xlabel("Average Similarity Score (scaled by 100)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(DISTRIBUTION_PLOT)
plt.close()

print("Performing hypothesis test...")
t_stat, two_tailed_p = stats.ttest_1samp(random_cluster_avgs, popmean=cluster_avg)
one_tailed_p = two_tailed_p / 2 if t_stat < 0 else 1 - (two_tailed_p / 2)

if one_tailed_p < SIGNIFICANCE_LEVEL:
    hypothesis_conclusion = (
        f"The average similarity score from cluster.csv is significantly higher than what is obtained by random sampling (p < {SIGNIFICANCE_LEVEL:.2f})."
    )
else:
    hypothesis_conclusion = (
        f"There is no significant difference between the cluster similarity and random clusters (p >= {SIGNIFICANCE_LEVEL:.2f})."
    )

print("FINAL CONCLUSION:")
print(hypothesis_conclusion)

pd.DataFrame(random_cluster_avgs, columns=["average_similarity"]).to_csv(OUTPUT_SCORES_CSV, index=False)

with open(SUMMARY_STATS_FILE, "w") as f:
    f.write(
        f"Cluster.csv average similarity score: {cluster_avg:.2f}\n"
        f"Random clusters (from source.csv) over {NUM_ITERATIONS} iterations:\n"
        f"Mean similarity: {overall_mean:.2f}\n"
        f"Standard deviation: {overall_std:.2f}\n"
        f"Median similarity: {median_score:.2f}\n"
        f"T-test t-statistic: {t_stat:.4f}\n"
        f"One-tailed p-value: {one_tailed_p:.4f}\n"
        f"Hypothesis Conclusion: {hypothesis_conclusion}\n"
    )
