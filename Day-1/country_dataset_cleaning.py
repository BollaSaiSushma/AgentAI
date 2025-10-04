import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\saisu\OneDrive\Desktop\AgentAI\AgentAI\Day-1\country_wise_latest.csv")
print(df.head(10))

# 1. Drop duplicate rows
df = df.drop_duplicates()

# 2. Drop completely empty columns
df = df.dropna(axis=1, how="all")

# 3. Fill missing numeric values with median, categorical with mode
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 4. Strip spaces in column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 5. Save cleaned dataset
df.to_csv("country_wise_latest_cleaned.csv", index=False)

print("✅ Data cleaned and saved as country_wise_latest_cleaned.csv")
print(df.head(10))
