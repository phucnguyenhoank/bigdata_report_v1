# %%
# visualize_als_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("images", exist_ok=True)

# Load the result
results_file = "results/als_grid_results.csv"
df = pd.read_csv(results_file)

print("Loaded data:")
print(df.head())

# ---------------------------------------------------
# 1️⃣  Biểu đồ so sánh RMSE train vs valid theo maxIter
# ---------------------------------------------------
plt.figure(figsize=(8, 5))

# Gom 2 cột train/valid vào 1 để plot dễ hơn
df_long = df.melt(
    id_vars=["regParam", "maxIter"],
    value_vars=["RMSE_train", "RMSE_valid"],
    var_name="Dataset",
    value_name="RMSE"
)

sns.lineplot(
    data=df_long,
    x="maxIter",
    y="RMSE",
    hue="Dataset",
    style="regParam",
    markers=True,
    dashes=False
)

plt.title("Train vs Validation RMSE across ALS model Iterations")
plt.xlabel("maxIter")
plt.ylabel("RMSE")
plt.legend(title="Dataset / regParam")
plt.tight_layout()
plt.savefig("images/als_train_valid_rmse_vs_iter.png", dpi=300)
plt.close()

# ---------------------------------------------------
# 2️⃣  Heatmap: Validation RMSE (vẫn giữ nguyên)
# ---------------------------------------------------
pivot_valid = df.pivot_table(index="regParam", columns="maxIter", values="RMSE_valid")
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_valid, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("Validation RMSE Heatmap of the ALS model")
plt.tight_layout()
plt.savefig("images/als_valid_rmse_heatmap.png", dpi=300)
plt.close()

# ---------------------------------------------------
# 3️⃣  Heatmap: Train RMSE (để so sánh overfitting)
# ---------------------------------------------------
pivot_train = df.pivot_table(index="regParam", columns="maxIter", values="RMSE_train")
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_train, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Training RMSE Heatmap of the ALS model")
plt.tight_layout()
plt.savefig("images/als_train_rmse_heatmap.png", dpi=300)
plt.close()

print("✅ Saved visualizations to images/ folder:")
print(" - als_train_valid_rmse_vs_iter.png")
print(" - als_valid_rmse_heatmap.png")
print(" - als_train_rmse_heatmap.png")
