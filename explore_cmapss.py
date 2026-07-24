import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Force Pandas to display ALL columns and rows without truncating with '...'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Define C-MAPSS Column Headers
index_cols = ['unit_nr', 'time_cycles']
setting_cols = ['setting_1', 'setting_2', 'setting_3']
sensor_cols = [f's_{i}' for i in range(1, 22)]  # 21 sensors (s_1 to s_21)
col_names = index_cols + setting_cols + sensor_cols

# 2. Load Dataset (Replace 'train_FD001.txt' with your file path if needed)
# C-MAPSS uses spaces as separators
df = pd.read_csv('archive/train_FD001.txt', sep=r'\s+', header=None, names=col_names)

# 3. Basic Inspection
print("=== DATASET SHAPE ===")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}\n")

print("=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA TYPES & MISSING VALUES ===")
print(df.info())

print("\n=== STATISTICAL SUMMARY ===")
print(df.describe().T) # Transposed for easier reading
# 1. Define a variance threshold (drops std <= 0.01)
threshold = 0.01

# 2. Calculate standard deviation for all 21 sensors
sensor_stds = df[sensor_cols].std()

# 3. Separate informative sensors from flat/near-flat sensors
useful_sensors = sensor_stds[sensor_stds > threshold].index.tolist()
dropped_sensors = sensor_stds[sensor_stds <= threshold].index.tolist()

print("=== FEATURE SELECTION RESULTS ===")
print(f"Dropped {len(dropped_sensors)} uninformative sensors: {dropped_sensors}")
print(f"Kept {len(useful_sensors)} informative sensors: {useful_sensors}\n")

# 4. Create the cleaned DataFrame
keep_columns = index_cols + setting_cols + useful_sensors
df_clean = df[keep_columns]

print("=== CLEANED DATASET SHAPE ===")
print(f"Original Shape: {df.shape}")
print(f"Cleaned Shape:  {df_clean.shape}")
print("\nFirst 5 rows of cleaned data:")
print(df_clean.head())