# %%
# MODEL COMPARISON
import matplotlib.pyplot as plt
import numpy as np
import os

# Models and metrics
models = ["SVD", "SGD", "ALS", "NCF"]
metrics = ["MAE", "RMSE", "Precision@10", "NDCG@10"]

# Correct results
results = np.array([
    [0.7694, 1.0634, 0.2064, 0.9651],  # SVD
    [0.7040, 1.0948, 0.8878, 0.9819],  # SGD
    [0.7635, 1.1078, 0.7952, 0.9821],  # ALS
    [0.7382, 1.0878, 0.7952, 0.9820],  # NCF
])

# Create output folder if not exist
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# --- Plotting ---
x = np.arange(len(metrics))
bar_width = 0.2

plt.figure(figsize=(8, 5))
for i, model in enumerate(models):
    plt.bar(x + i * bar_width, results[i], width=bar_width, label=model)

plt.xticks(x + bar_width * 1.5, metrics)
plt.ylabel("Score")
plt.title("Model Comparison Across Metrics")
plt.legend()
plt.tight_layout()

# Save the plot
save_path = os.path.join(output_folder, "model_comparison.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Plot saved to: {save_path}")


# %%
# TRUE RATING & PREDICTION
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample or collect small subset for visualization

pdf = pd.read_parquet("results/ranked_predictions.parquet")

plt.figure(figsize=(8, 5))
sns.histplot(pdf["rating"], color="blue", label="True Rating", kde=True, bins=5)
sns.histplot(pdf["pred_rating"], color="orange", label="Predicted Rating", kde=True, bins=20, alpha=0.6)

plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.title("ALS Distribution of True vs Predicted Ratings")
plt.legend()
plt.tight_layout()

plt.savefig("images/rating_distribution.png", dpi=300)
plt.close()
print("✅ Saved plot to images/rating_distribution.png")

