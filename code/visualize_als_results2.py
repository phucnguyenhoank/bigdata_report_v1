import matplotlib.pyplot as plt
import pandas as pd

# --- Your data ---
data = {
    "Dataset": ["Train", "Valid", "Test"],
    "RMSE": [0.9713, 1.2499, 1.1078],
    "MAE": [0.6615, 0.8396, 0.7635]
}
df = pd.DataFrame(data)

# --- Plot ---
plt.figure(figsize=(7, 5))
bar_width = 0.35
x = range(len(df))

bars_rmse = plt.bar(x, df["RMSE"], width=bar_width, label="RMSE", color="skyblue")
bars_mae = plt.bar([i + bar_width for i in x], df["MAE"], width=bar_width, label="MAE", color="lightcoral")

# --- Add text labels ---
for bar in bars_rmse:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', ha='center', va='bottom')

for bar in bars_mae:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.3f}', ha='center', va='bottom')

plt.xticks([i + bar_width / 2 for i in x], df["Dataset"])
plt.ylabel("Error")
plt.title("ALS Performance Comparison (Train / Valid / Test)")
plt.legend()
plt.tight_layout()
plt.savefig("images/als_train_valid_test_metrics.png", dpi=300)
plt.show()
