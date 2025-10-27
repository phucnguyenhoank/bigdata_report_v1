import os
import sys
import pyspark

# -----------------------------
# 1. Setup Spark
# -----------------------------

# Set PYSPARK_PYTHON to the current Python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
print("PYSPARK_PYTHON:", os.environ["PYSPARK_PYTHON"])
print(pyspark.__version__)


from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("AmazonReviewsPreprocess")
    .master("local[*]")
    .config("spark.driver.extraJavaOptions", "-Xss40m") # Increase JVM stack size
    .getOrCreate()
)

# ============ 0. LOAD DỮ LIỆU ============

# Read train, validation, and test CSVs (gzip)
file_name = "All_Beauty"
train_df = spark.read.csv(f"{file_name}.train.csv.gz", header=True, inferSchema=True)
valid_df = spark.read.csv(f"{file_name}.valid.csv.gz", header=True, inferSchema=True)
test_df = spark.read.csv(f"{file_name}.test.csv.gz", header=True, inferSchema=True)

train_df.show(3)
print(f"Số dòng train + valid + test: {train_df.count()} + {valid_df.count()} + {test_df.count()} = {train_df.count() + valid_df.count() + test_df.count()}")


from pyspark.sql import functions as F
import os
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# ============ 1. TẠO MAPPING USER/ITEM ============

# Lấy tất cả user và item xuất hiện trong bất kỳ tập nào
all_users = train_df.select("user_id").union(valid_df.select("user_id")).union(test_df.select("user_id")).distinct()
all_items = train_df.select("parent_asin").union(valid_df.select("parent_asin")).union(test_df.select("parent_asin")).distinct()

# Tạo index (monotonically_increasing_id đảm bảo unique ID, không cần thứ tự)
user_index = all_users.withColumn("userIndex", row_number().over(Window.orderBy("user_id")) - 1)
item_index = all_items.withColumn("itemIndex", row_number().over(Window.orderBy("parent_asin")) - 1)

# Lưu mapping ra file
os.makedirs("mappings_svd", exist_ok=True)
user_index.write.mode("overwrite").parquet("mappings_svd/user_index.parquet")
item_index.write.mode("overwrite").parquet("mappings_svd/item_index.parquet")

print("✅ Saved user_index & item_index mappings_svd.")


# ============ 2. JOIN MAPPING VÀ CHUẨN HÓA ============

def map_and_normalize(df, user_index, item_index, item_means=None, global_mean=None):
    # Gán index cho user và item
    df = df.join(user_index, on="user_id", how="left")
    df = df.join(item_index, on="parent_asin", how="left")

    # Nếu chưa có item_means (train phase)
    if item_means is None:
        item_means = df.groupBy("itemIndex").agg(F.avg("rating").alias("item_mean"))

    # Gộp item_mean vào df
    df = df.join(item_means, on="itemIndex", how="left")

    # Nếu có item mới trong valid/test → dùng global_mean
    if global_mean is not None:
        df = df.withColumn(
            "item_mean",
            F.when(F.col("item_mean").isNull(), F.lit(global_mean)).otherwise(F.col("item_mean"))
        )

    # Chuẩn hóa rating theo item mean
    df = df.withColumn("rating_norm", F.col("rating") - F.col("item_mean"))

    return df, item_means



# ============ 3. ÁP DỤNG CHO TRAIN/VALID/TEST ============

# Tải lại mapping từ file (đảm bảo có thể dùng ở session khác)
user_index = spark.read.parquet("mappings_svd/user_index.parquet")
item_index = spark.read.parquet("mappings_svd/item_index.parquet")

# Tính global mean rating
global_mean = train_df.select(F.avg("rating")).first()[0]

# Chuẩn hóa train (tính item_means)
train_df_norm, item_means = map_and_normalize(train_df, user_index, item_index)

# Chuẩn hóa valid/test (dùng lại item_means và global_mean)
valid_df_norm, _ = map_and_normalize(valid_df, user_index, item_index, item_means, global_mean)
test_df_norm, _ = map_and_normalize(test_df, user_index, item_index, item_means, global_mean)

# Lưu item_means
os.makedirs("preprocessed_svd", exist_ok=True)
item_means.write.mode("overwrite").parquet("preprocessed_svd/item_means.parquet")

# Lưu dữ liệu đã chuẩn hóa
cols_to_save = ['itemIndex', 'parent_asin', 'userIndex', 'user_id', 'rating', 'rating_norm', 'item_mean']
train_df_norm.select(*cols_to_save).write.mode("overwrite").parquet("preprocessed_svd/train_norm.parquet")
valid_df_norm.select(*cols_to_save).write.mode("overwrite").parquet("preprocessed_svd/valid_norm.parquet")
test_df_norm.select(*cols_to_save).write.mode("overwrite").parquet("preprocessed_svd/test_norm.parquet")

print("✅ Saved normalized data and item mean vectors.")


